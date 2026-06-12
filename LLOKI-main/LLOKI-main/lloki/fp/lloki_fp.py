import torch
from lloki.fp.feature_propagation import FeaturePropagation
from lloki.fp.graph_construction import create_spatially_weighted_knn_graph
import numpy as np
import json
from scipy.sparse import issparse
from scipy.interpolate import interp1d
import os

def calculate_sparsity(adata):
    """Calculate sparsity per cell for an AnnData object."""
    return (
        1 - (adata.X.getnnz(axis=1) / adata.X.shape[1])
        if issparse(adata.X)
        else np.mean(adata.X == 0, axis=1)
    ).flatten()

def ensure_gene_symbols(adata, mapping_file=None):
    if mapping_file is None:
        mapping_file = os.path.join(os.path.dirname(__file__), "ensembl_to_symbol.json")
    var_names = adata.var_names.tolist()
    ensembl_ids = [g for g in var_names if g.startswith("ENS")]
    
    if not ensembl_ids:
        return adata

    with open(mapping_file) as f:
        mapping = json.load(f)

    new_names = [mapping.get(g, g) for g in var_names]
    adata.var_names = new_names
    return adata

def compute_empirical_cdf(sparsity_values):
    """Compute the empirical CDF for sorted sparsity values."""
    sorted_vals = np.sort(sparsity_values)
    return sorted_vals, np.linspace(0, 1, len(sorted_vals))


def map_sparsity_values(cdf_src, sorted_src, cdf_tgt, sorted_tgt):
    """Map sparsity values from source to target distribution using interpolation.

    Implements s* = F^{-1}_{tgt}(F_{src}(s)):
    given source CDF values (cdf_src) evaluated at sorted_sr2c,
    interpolate the inverse target CDF and apply it.
    """
    return interp1d(
        cdf_tgt,
        sorted_tgt,
        bounds_error=False,
        fill_value=(sorted_tgt[0], sorted_tgt[-1]),
    )(cdf_src)


def propagate(adata, adata_scrna, args):
    device = args.device
    obsm_spatial_key = getattr(args, "obsm_spatial_key", "spatial")

    adata = ensure_gene_symbols(adata)
    adata = adata[:, adata.var_names.isin(adata_scrna.var_names)]
    cell_data = adata.X.toarray() if issparse(adata.X) else adata.X.copy()

    sparsity_scrna = calculate_sparsity(adata_scrna)
    sparsity_st = calculate_sparsity(adata)

    # OT sparsity mapping: s* = F^{-1}_{scRNA}(F_{ST}(s))  [Eq. 4 in paper]
    # Sort ST sparsity values and get their CDF, then map through inverse scRNA CDF
    sorted_sparsity_st, cdf_st = compute_empirical_cdf(sparsity_st)
    sorted_sparsity_scrna, cdf_scrna = compute_empirical_cdf(sparsity_scrna)

    # Apply F_ST to get uniform quantiles, then apply F^{-1}_{scRNA}
    # map_sparsity_values(cdf_src=cdf_st, sorted_src=sorted_st,
    #                     cdf_tgt=cdf_scrna, sorted_tgt=sorted_scrna)
    # returns F^{-1}_{scRNA}(cdf_st) evaluated at sorted_st points
    mapped_sparsity_sorted = map_sparsity_values(
        cdf_st, sorted_sparsity_st, cdf_scrna, sorted_sparsity_scrna
    )

    # mapped_sparsity_sorted is in sorted order of ST sparsity values.
    # Use the inverse permutation (argsort of argsort) to map back to
    # original cell order.
    inv_sort = np.argsort(np.argsort(sparsity_st))
    target_sparsity = mapped_sparsity_sorted[inv_sort]

    # Compute max additional genes to impute per cell:
    # N_max = ceil((1 - s*) * G) gives target nonzero count;
    # subtract existing nonzero to get how many more can be imputed.
    num_genes = adata.X.shape[1]
    existing_nonzero = (
        adata.X.getnnz(axis=1)
        if issparse(adata.X)
        else np.count_nonzero(adata.X, axis=1)
    )
    max_impute_per_cell = torch.tensor(
        np.clip(
            np.ceil((1 - target_sparsity) * num_genes) - existing_nonzero, 0, None
        ),
        device=device,
        dtype=torch.int32,
    )

    # Normalize data and create KNN graph for hard (masked) propagation
    cell_data = torch.Tensor(cell_data).to(device)
    cell_data = cell_data / (torch.max(cell_data) + 1e-8)

    edge_index, edge_weight = create_spatially_weighted_knn_graph(
        cell_data,
        adata,
        args.k,
        device,
        gcn_norm=getattr(args, "fp_gcn_norm", True),
        sym=getattr(args, "fp_sym", True),
        avg_dist_samples=getattr(args, "fp_avg_dist_samples", 100),
        seed=getattr(args, "seed", None),
        obsm_spatial_key=obsm_spatial_key,
    )

    # Hard propagation: impute zero entries up to target sparsity (Eq. 2)
    # alpha_hard=0.0 means full graph propagation for missing values (mask=True)
    model = FeaturePropagation(
        num_iterations=args.iter,
        adata=adata,
        mask=True,
        alpha=getattr(args, "fp_hard_alpha", 0.0),
        max_imputation_per_cell=max_impute_per_cell,
        early_stopping=getattr(args, "fp_early_stopping", True),
        tol=getattr(args, "fp_tol", 1e-6),
        patience=getattr(args, "fp_patience", 15),
        device=device,
    ).to(device)
    denoised_matrix = model(
        cell_data, edge_index.to(device), edge_weight.to(device)
    )
    adata.obsm["denoised"] = denoised_matrix.cpu().numpy()

    # Soft propagation (feature diffusion) with new graph built on imputed data (Eq. 5)
    # X_denoised = alpha * A_imputed * X^(0) + (1 - alpha) * X^(0)
    # alpha = args.alpha = 0.05 (self-weight = 0.95), single iteration
    print("Starting Soft Feature Propagation...")
    edge_index, edge_weight = create_spatially_weighted_knn_graph(
        denoised_matrix,
        adata,
        args.k,
        device,
        gcn_norm=getattr(args, "fp_gcn_norm", True),
        sym=getattr(args, "fp_sym", True),
        avg_dist_samples=getattr(args, "fp_avg_dist_samples", 100),
        seed=getattr(args, "seed", None),
        obsm_spatial_key=obsm_spatial_key,
    )
    model = FeaturePropagation(
        num_iterations=getattr(args, "fp_soft_iter", getattr(args, "soft_iter", 1)),
        adata=adata,
        mask=False,
        alpha=args.alpha,
        early_stopping=getattr(args, "fp_early_stopping", True),
        tol=getattr(args, "fp_tol", 1e-6),
        patience=getattr(args, "fp_patience", 15),
        device=device,
    ).to(device)
    soft_input = (
        denoised_matrix
        if getattr(args, "fp_soft_use_imputed_input", True)
        else cell_data
    )
    denoised_matrix = model(soft_input, edge_index.to(device), edge_weight.to(device))
    adata.obsm["denoised"] = denoised_matrix.detach().cpu().numpy()

    return adata