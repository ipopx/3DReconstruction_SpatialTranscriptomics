import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
import ot 
from anndata import AnnData
from tqdm import tqdm
import time
import scvi
from scvi.model import SCVI
import json
import scipy.sparse as sp
import scanpy as sc

def generate_spatial_coord_wasserstein(
    adata1,
    adata2,
    alpha=0.5,
    device="cpu",
    n_cell=None,
    n_mag=1.0,
    lr=1e5,
    nb_iter_max=100,
    seed=42,
    num_projections=80,
    verbose=True,
    time_logger=None,
):
    if verbose:
        print("Begin to generate spatial coordinates......")

    start_time = time.time()
    coor1_torch = torch.tensor(adata1.obsm["spatial"], dtype=torch.float).to(device=device)
    coor2_torch = torch.tensor(adata2.obsm["spatial"], dtype=torch.float).to(device=device)

    local_n_cell = n_cell
    if local_n_cell is None:
        local_n_cell = int((alpha * adata1.n_obs + (1 - alpha) * adata2.n_obs) * n_mag)

    n_cell1 = int(local_n_cell * alpha)
    n_cell2 = local_n_cell - n_cell1

    sampled_indices1 = np.linspace(0, coor1_torch.shape[0] - 1, n_cell1, dtype=int)
    sampled_indices2 = np.linspace(0, coor2_torch.shape[0] - 1, n_cell2, dtype=int)

    Coor_init1 = coor1_torch[sampled_indices1].cpu().numpy()
    Coor_init2 = coor2_torch[sampled_indices2].cpu().numpy()
    Coor_init = np.concatenate([Coor_init1, Coor_init2], axis=0)

    Coor_torch = torch.tensor(Coor_init, dtype=torch.float32, device=device).requires_grad_(True)
    if time_logger is not None:
        time_logger("coordinate initialization", start_time)
    elif verbose:
        print(f"coordinate initialization time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    gen = torch.Generator(device=device).manual_seed(seed)
    for i in range(nb_iter_max):
        loss = (
            alpha
            * ot.sliced_wasserstein_distance(
                Coor_torch, coor1_torch, n_projections=num_projections, seed=gen
            )
            + (1 - alpha)
            * ot.sliced_wasserstein_distance(
                Coor_torch, coor2_torch, n_projections=num_projections, seed=gen
            )
        )
        loss.backward()

        with torch.no_grad():
            Coor_torch -= Coor_torch.grad * lr
            Coor_torch.grad.zero_()
        if verbose and i % 1000 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    if time_logger is not None:
        time_logger("Ot optimization", start_time)
    elif verbose:
        print(f"Ot optimization time: {time.time() - start_time:.2f} seconds")

    Coor_final = Coor_torch.detach().cpu().numpy()
    var_data = pd.DataFrame(index=adata1.var_names)
    adata3 = AnnData(
        X=np.zeros((Coor_final.shape[0], adata1.n_vars), dtype=np.float32),
        var=var_data,
        dtype=np.float32,
    )
    adata3.obsm["spatial"] = Coor_final

    return Coor_final, adata3


def generate_cell_type_knn(
    adata1,
    adata2,
    adata3,
    Coor_final,
    k_ct=1,
    cell_type_key="cell_type",
    adata1_id="above",
    adata2_id="below",
    add_obs_list=None,
    verbose=True,
    time_logger=None,
):
    if verbose:
        print("Begin to determine cell types......")

    start_time = time.time()
    nn_adata1 = NearestNeighbors(n_neighbors=k_ct).fit(adata1.obsm["spatial"])
    nn_adata2 = NearestNeighbors(n_neighbors=k_ct).fit(adata2.obsm["spatial"])
    distances_1, indices_1 = nn_adata1.kneighbors(Coor_final)
    distances_2, indices_2 = nn_adata2.kneighbors(Coor_final)

    epsilon = 0.1

    sim_celltype = []
    closest_indices = []
    for i in range(Coor_final.shape[0]):
        types_1 = adata1.obs.iloc[indices_1[i]][cell_type_key].values
        types_2 = adata2.obs.iloc[indices_2[i]][cell_type_key].values

        weights_1 = 1 / (distances_1[i] + epsilon)
        weights_2 = 1 / (distances_2[i] + epsilon)
        all_types = np.concatenate([types_1, types_2])
        all_weights = np.concatenate([weights_1, weights_2])

        type_weights = pd.Series(all_weights, index=all_types).groupby(level=0).sum()
        dominant_type = type_weights.idxmax() if not type_weights.empty else None

        if dominant_type:
            min_dist_1 = (
                np.min(distances_1[i][types_1 == dominant_type])
                if dominant_type in types_1
                else np.inf
            )
            min_dist_2 = (
                np.min(distances_2[i][types_2 == dominant_type])
                if dominant_type in types_2
                else np.inf
            )
            if min_dist_1 <= min_dist_2:
                closest_index = adata1.obs_names[
                    indices_1[i][np.where(types_1 == dominant_type)[0][0]]
                ]
            else:
                closest_index = adata2.obs_names[
                    indices_2[i][np.where(types_2 == dominant_type)[0][0]]
                ]
        else:
            closest_dist_1 = np.argmin(distances_1[i])
            closest_dist_2 = np.argmin(distances_2[i])
            if distances_1[i][closest_dist_1] < distances_2[i][closest_dist_2]:
                dominant_type = types_1[closest_dist_1]
                closest_index = adata1.obs_names[indices_1[i][closest_dist_1]]
            else:
                dominant_type = types_2[closest_dist_2]
                closest_index = adata2.obs_names[indices_2[i][closest_dist_2]]

        sim_celltype.append(dominant_type)
        closest_indices.append(closest_index)

    adata3.obs[cell_type_key] = sim_celltype

    if time_logger is not None:
        time_logger("Cell type determination", start_time)
    elif verbose:
        print(f"Cell type determination time: {time.time() - start_time:.2f} seconds")

    if add_obs_list is not None:
        start_time = time.time()
        if verbose:
            print("Begin to transfer the attribute......")
        for obs_key in add_obs_list:
            adata3.obs[obs_key] = [
                adata1.obs[obs_key][index] if adata1_id in index else adata2.obs[obs_key][index]
                for index in closest_indices
            ]

        if time_logger is not None:
            time_logger("Transfer the attribute", start_time)
        elif verbose:
            print(f"Transfer the attribute time: {time.time() - start_time:.2f} seconds")

    return sim_celltype, closest_indices, distances_1, distances_2, indices_1, indices_2, adata3


def generate_gex_scvi(
    adata1,
    adata2,
    adata3,
    Coor_final,
    sim_celltype,
    distances_1,
    distances_2,
    indices_1,
    indices_2,
    k_gex=3,
    cell_type_key="cell_type",
    verbose=True,
    time_logger=None,
    scvi_model=None,
    query_adata=None,
    dataset=None,
    symbol_to_ensembl=None,
):
    start_time = time.time()
    if verbose:
        print("Begin to synthesize gene expression with scVI......")

    if scvi_model is None:
        raise ValueError("scvi_model must be provided")
    if symbol_to_ensembl is None:
        raise ValueError("symbol_to_ensembl mapping must be provided")

    n_cells = Coor_final.shape[0]
    n_genes = query_adata.n_vars
    X_new = np.zeros((n_cells, n_genes), dtype=np.float32)
    n_adata1 = adata1.n_obs
    model_adata = query_adata

    # Precompute latent representations + log-library for the query cells (used for decoding averaged latent)
    Z_all = scvi_model.get_latent_representation()
    if torch.is_tensor(Z_all):
        Z_all = Z_all.detach().cpu().numpy()
    Z_all = np.asarray(Z_all, dtype=np.float32)
    Xc = model_adata.X
    lib = np.asarray(Xc.sum(axis=1)).reshape(-1)
    lib_log = np.log1p(lib).astype(np.float32, copy=False)
    module_device = next(scvi_model.module.parameters()).device
    
    batch_mapping = scvi_model.adata_manager.registry['field_registries']['batch']['state_registry']['categorical_mapping']
    assay_mapping = scvi_model.adata_manager.registry['field_registries']['extra_categorical_covs']['state_registry']['mappings']['assay']
    if isinstance(batch_mapping, np.ndarray):
        batch_mapping = batch_mapping.tolist()
    if isinstance(assay_mapping, np.ndarray):
        assay_mapping = assay_mapping.tolist()

    for i in range(n_cells):
        # Take k nearest neighbours across BOTH adjacent slices, filtered to the same cell type
        ctype = sim_celltype[i]
        idxs1 = np.asarray(indices_1[i])
        idxs2 = np.asarray(indices_2[i])
        dists1 = np.asarray(distances_1[i])
        dists2 = np.asarray(distances_2[i])

        neighbor_sources: list[tuple[str, int]] = []
        neighbor_dists: list[float] = []

        # Only keep neighbours of the same predicted cell type
        types_1 = adata1.obs.iloc[idxs1][cell_type_key].values
        types_2 = adata2.obs.iloc[idxs2][cell_type_key].values

        for j, t in enumerate(types_1):
            if t == ctype:
                neighbor_sources.append(("adata1", int(idxs1[j])))
                neighbor_dists.append(float(dists1[j]))
        for j, t in enumerate(types_2):
            if t == ctype:
                neighbor_sources.append(("adata2", int(idxs2[j])))
                neighbor_dists.append(float(dists2[j]))

        # Fallback: if no same-type neighbours found, use the nearest regardless of type
        if len(neighbor_sources) == 0:
            for j in range(len(idxs1)):
                neighbor_sources.append(("adata1", int(idxs1[j])))
                neighbor_dists.append(float(dists1[j]))
            for j in range(len(idxs2)):
                neighbor_sources.append(("adata2", int(idxs2[j])))
                neighbor_dists.append(float(dists2[j]))

        order = np.argsort(neighbor_dists)[:k_gex]

        # Map neighbor indices back to the concatenated AnnData that scVI was trained on
        combined_indices: list[int] = []
        for o in order:
            src, idx = neighbor_sources[int(o)]
            combined_indices.append(idx if src == "adata1" else (n_adata1 + idx))

        # Encode neighbours -> average latent z -> decode averaged z
        z_mean = np.mean(Z_all[np.asarray(combined_indices, dtype=int)], axis=0, keepdims=True)  # (1, n_latent)

        # Use mean log-library from neighbours as a decoding context
        lib_mean = float(np.mean(lib_log[np.asarray(combined_indices, dtype=int)]))

        z_t = torch.tensor(z_mean, device=module_device)
        library_t = torch.tensor([[lib_mean]], device=module_device)
        
        # Get the categorical values for this observation
        donor_id_str = 'new_donor'
        assay_str = dataset.upper()
        donor_idx = batch_mapping.index(donor_id_str)
        assay_idx = assay_mapping.index(assay_str)
        batch_index_t = torch.tensor([[donor_idx]], device=module_device, dtype=torch.long)
        cat_covs_t = torch.tensor([[assay_idx]], device=module_device, dtype=torch.long)
    
        
        with torch.no_grad():
            gen_out = scvi_model.module.generative(
                z=z_t,
                library=library_t,
                batch_index=batch_index_t,
                cat_covs=cat_covs_t
            )
            px_rate = gen_out.get("px_rate", None)
            if px_rate is None:
                px = gen_out.get("px", None)
                if px is None:
                    raise RuntimeError(
                        "scVI generative() did not return 'px_rate' or 'px'. "
                        f"Available keys: {list(gen_out.keys())}"
                    )
                if hasattr(px, "rate"):
                    px_rate = px.rate
                elif hasattr(px, "mean"):
                    px_rate = px.mean
                else:
                    raise RuntimeError(
                        "scVI generative() returned 'px' but it has no .rate or .mean attribute."
                    )
            X_new[i, :] = (
                px_rate.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            )


    # Get the Ensembl names from the scVI model (query_adata)
    model_ensembl_genes = model_adata.var_names.tolist()
    
    # For each gene in adata3 (symbol), find its position in the scVI output
    target_symbols = adata3.var_names.tolist()
    X_aligned = np.zeros((X_new.shape[0], len(target_symbols)), dtype=X_new.dtype)
    
    for i, symbol in enumerate(target_symbols):
        ensembl = symbol_to_ensembl.get(symbol, None)
        if ensembl and ensembl in model_ensembl_genes:
            # Find the index of this Ensembl gene in the scVI output
            ensembl_idx = model_ensembl_genes.index(ensembl)
            X_aligned[:, i] = X_new[:, ensembl_idx]
        # else: gene not in scVI model, leave as zeros

    adata3.X = X_aligned
    if time_logger is not None:
        time_logger("Gene expression synthesis with scVI", start_time)
    elif verbose:
        print(
            f"Gene expression synthesis with scVI time: {time.time() - start_time:.2f} seconds"
        )
    return adata3

def prepare_adata_for_scvi(adata, dataset, symbol_to_ensembl):
    adata_ensbl = adata.copy()
    ensmbl_adata = [symbol_to_ensembl[symbol] for symbol in adata.var_names]
    adata_ensbl.var_names = ensmbl_adata
    adata_ensbl.obs['donor_id'] = 'new_donor'
    adata_ensbl.obs['assay'] = dataset.upper()
    return adata_ensbl

def expand_to_genes(adata: AnnData, ref_genes) -> AnnData:
    ref_genes = pd.Index(ref_genes)
    q_genes = pd.Index(adata.var_names)

    pos = ref_genes.get_indexer(q_genes)
    keep = pos >= 0

    n, p = adata.n_obs, len(ref_genes)
    if sp.issparse(adata.X):
        X_full = sp.csr_matrix((n, p), dtype=adata.X.dtype)
        Xq = adata.X[:, keep]
        cols = pos[keep]
        X_full[:, cols] = Xq
    else:
        X_full = np.zeros((n, p), dtype=adata.X.dtype)
        X_full[:, pos[keep]] = adata.X[:, keep]

    out = AnnData(X=X_full, obs=adata.obs.copy(), var=pd.DataFrame(index=ref_genes))
    return out

def query_scvi(adata, scvi_model):
    scvi.model.SCVI.prepare_query_anndata(adata, scvi_model)
    scvi_query = scvi.model.SCVI.load_query_data(
        adata,
        scvi_model,
    )
    scvi_query.train(max_epochs=15, plan_kwargs=dict(weight_decay=0.0))
    return scvi_query


def generate_pretrained_scvi(adata1, adata2, adata1_id='above', adata2_id='below',
                      alpha=0.5, device='auto', n_cell=None, k_ct=1,
                      n_mag=1.0, lr=1e5, nb_iter_max=100, seed=42,
                      num_projections=80, cell_type_key='cell_type', k_gex=3, add_obs_list= None, verbose=True, dataset = 'starmap'):
    """
    The generate_pretrained_scvi function is designed to integrate spatial coordinates and gene expression from two AnnData objects and generate a new AnnData object. 
    """
    def print_time(message, start):
        if verbose:
            print(f"{message} time: {time.time() - start:.2f} seconds")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)

    # Check required fields in AnnData
    if 'spatial' not in adata1.obsm or 'spatial' not in adata2.obsm:
        raise ValueError("Both adata1 and adata2 must have 'spatial' coordinates in obsm.")
    if cell_type_key not in adata1.obs or cell_type_key not in adata2.obs:
        raise ValueError(f"Both adata1 and adata2 must have '{cell_type_key}' information in obs.")

    # Adjusting indices with unique identifiers
    adata1.obs_names = [f"{name}_{adata1_id}" for name in adata1.obs_names]
    adata2.obs_names = [f"{name}_{adata2_id}" for name in adata2.obs_names]

    # Load pre-trained scVI model and prepare query adata
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        # Set weights_only=False if not explicitly specified
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    
    # Load symbol to Ensembl mapping ONCE
    with open(f"./data/{dataset}/{dataset}_symbol_to_ensembl.json", "r") as f:
        symbol_to_ensembl = json.load(f)
    
    with open(f"./data/large_merfish/ref_genes.txt", "r") as f:
        ref_genes = f.read().splitlines()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    adata1_ensbl = prepare_adata_for_scvi(adata1, dataset, symbol_to_ensembl)
    adata1_ensbl = expand_to_genes(adata1_ensbl, ref_genes)
    adata2_ensbl = prepare_adata_for_scvi(adata2, dataset, symbol_to_ensembl)
    adata2_ensbl = expand_to_genes(adata2_ensbl, ref_genes)
    
    # Integrate new batch
    adata_query = adata1_ensbl.concatenate(adata2_ensbl)
    load_adata = sc.read_h5ad("./data/large_merfish/animal4_int_5cells.h5ad")
    scvi_model = scvi.model.SCVI.load("./models/scvi_subset_model_4March_16latent_2layers_128hidden_8epochs", adata=load_adata)
    scvi_model.to_device(device)  
    scvi_query = query_scvi(adata_query, scvi_model)
    

    Coor_final, adata3 = generate_spatial_coord_wasserstein(
        adata1=adata1,
        adata2=adata2,
        alpha=alpha,
        device=device,
        n_cell=n_cell,
        n_mag=n_mag,
        lr=lr,
        nb_iter_max=nb_iter_max,
        seed=seed,
        num_projections=num_projections,
        verbose=verbose,
        time_logger=print_time,
    )

    (
        sim_celltype,
        closest_indices,
        distances_1,
        distances_2,
        indices_1,
        indices_2,
        adata3,
    ) = generate_cell_type_knn(
        adata1=adata1,
        adata2=adata2,
        adata3=adata3,
        Coor_final=Coor_final,
        k_ct=k_ct,
        cell_type_key=cell_type_key,
        adata1_id=adata1_id,
        adata2_id=adata2_id,
        add_obs_list=add_obs_list,
        verbose=verbose,
        time_logger=print_time,
    )

    # Pass symbol_to_ensembl mapping to generate_gex_scvi
    adata3 = generate_gex_scvi(
        adata1=adata1,
        adata2=adata2,
        adata3=adata3,
        Coor_final=Coor_final,
        sim_celltype=sim_celltype,
        distances_1=distances_1,
        distances_2=distances_2,
        indices_1=indices_1,
        indices_2=indices_2,
        k_gex=k_gex,
        cell_type_key=cell_type_key,
        verbose=verbose,
        time_logger=print_time,
        scvi_model=scvi_query,
        query_adata=adata_query,
        dataset=dataset,
        symbol_to_ensembl=symbol_to_ensembl,
    )

    return adata3

def generate_multiple_scvi(adata1, adata2, num_sim, adata1_id='above', adata2_id='below',
                               device='auto', n_cell=None, n_mag=1.0, lr=1e5, nb_iter_max=3000, seed=42, num_projections=80,
                               cell_type_key='cell_type',syn_mode= 'default', k_gex=3, micro_env_key = 'mender', Beta = 100, add_obs_list=None, verbose=True,
                               include_raw=True, dataset='starmap'):
    """
    The generate_multiple_scvi function extends the capabilities of the Generate_spatialz by generating multiple integrated AnnData objects. 
    """
    sim_adatas = []
    num_sim = num_sim + 1

    # Optionally include raw adata1 at the beginning
    if include_raw:
        adata1.obs['slice_id'] = f"{adata1_id}"
        adata1.obs['data_type'] = 'real'
        sim_adatas.append(adata1.copy())

    #for i in range(1, num_sim):  # Start from 1 to exclude alpha=1 and end at num_sim to exclude alpha=0
    for i in tqdm(range(1, num_sim), desc="Generating simulations"): 
        alpha = 1 - i / num_sim
        #print(alpha)
        sim_adata = generate_pretrained_scvi(adata1, adata2, adata1_id=adata1_id, adata2_id=adata2_id,
                                      alpha=alpha, device=device, n_cell=n_cell, n_mag=n_mag, lr=lr,
                                      nb_iter_max=nb_iter_max, seed=seed, num_projections=num_projections,
                                      cell_type_key=cell_type_key, k_gex=k_gex, add_obs_list=add_obs_list, verbose=True, dataset=dataset
                                      )
        # Create slice_id
        #slice_id = f"{adata1_id}-{adata2_id}-{alpha:.2f}"
        slice_id = f"{adata1_id}-{adata2_id}-{i}"
        sim_adata.obs['slice_id'] = slice_id
        sim_adata.obs['data_type'] = 'synthetic'
        sim_adatas.append(sim_adata)
        if verbose:
            print(f"Completed {slice_id} generated!")

    # Optionally include raw adata2 at the end
    if include_raw:
        adata2.obs['slice_id'] = f"{adata2_id}"
        adata2.obs['data_type'] = 'real'
        sim_adatas.append(adata2.copy())

    # Concatenate all generated AnnData objects
    concatenated_adata = AnnData.concatenate(*sim_adatas, batch_key='slice_id', batch_categories=[s.obs['slice_id'][0] for s in sim_adatas])
    return concatenated_adata