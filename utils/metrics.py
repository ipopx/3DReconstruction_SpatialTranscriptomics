from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
from pathlib import Path


def _build_spatial_weights(
    coords: np.ndarray,
    n_neighbors: int = 8,
) -> np.ndarray:
    """
    Build a symmetric k-NN spatial weight matrix (binary adjacency).
    """
    n_cells = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords, return_distance=True)

    W = np.zeros((n_cells, n_cells), dtype=float)
    for i in range(n_cells):
        for j in indices[i, 1:]:
            W[i, j] = 1.0
            W[j, i] = 1.0

    return W


def _moran_i(x: np.ndarray, W: np.ndarray) -> float:
    """
    Compute Moran's I for a single variable x given spatial weights W.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n != W.shape[0] or W.shape[0] != W.shape[1]:
        raise ValueError("Dimensions of x and W are incompatible")
    x_mean = x.mean()
    x_dev = x - x_mean
    denom = np.sum(x_dev**2)
    if denom == 0:
        return np.nan
    w_sum = W.sum()
    num = 0.0
    num = np.sum(W * np.outer(x_dev, x_dev))

    return (n / w_sum) * (num / denom)


def _geary_c(x: np.ndarray, W: np.ndarray) -> float:
    """
    Compute Geary's C for a single variable x given spatial weights W.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n != W.shape[0] or W.shape[0] != W.shape[1]:
        raise ValueError("Dimensions of x and W are incompatible")
    x_mean = x.mean()
    x_dev = x - x_mean
    denom = np.sum(x_dev**2)
    if denom == 0:
        return np.nan
    w_sum = W.sum()
    diff = x[:, None] - x[None, :]
    num = np.sum(W * (diff**2))
    return ((n - 1) / (2 * w_sum)) * (num / denom)


def _coords_from_adata(
    adata: ad.AnnData,
    coord_keys: Sequence[str]| None = ("x", "y"),
) -> np.ndarray:
    """
    Extract coordinates from AnnData.
    """
    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"], dtype=float)
        return spatial[:, :2]

    if coord_keys is not None:
        return np.column_stack([np.asarray(adata.obs[k].values, dtype=float) for k in coord_keys])
    
    raise ValueError("No spatial coordinates found (expected obsm['spatial'] or obs['x'], obs['y'])")   


def _density_image(
    adata: ad.AnnData,
    coord_keys: Sequence[str] = ("x", "y"),
    grid_size: int = 64,
) -> np.ndarray:
    """
    Construct a simple 2D density image from cell coordinates, as a proxy for pixel-level spatial structure.
    """
    coords = _coords_from_adata(adata, coord_keys=coord_keys)
    x, y = coords[:, 0], coords[:, 1]
    # Normalize to [0, 1] then bin
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

    xi = np.clip((x_norm * (grid_size - 1)).astype(int), 0, grid_size - 1)
    yi = np.clip((y_norm * (grid_size - 1)).astype(int), 0, grid_size - 1)

    img = np.zeros((grid_size, grid_size), dtype=float)
    for i, j in zip(xi, yi):
        img[j, i] += 1.0

    # Normalize image intensity
    if img.max() > 0:
        img /= img.max()
    return img


def _density_image_weighted(
    coords: np.ndarray,
    values: np.ndarray,
    grid_size: int = 64,
) -> np.ndarray:
    """
    Construct a 2D density image where each cell contributes its expression
    value to the corresponding pixel instead of a unit count.
    """
    x, y = coords[:, 0], coords[:, 1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

    xi = np.clip((x_norm * (grid_size - 1)).astype(int), 0, grid_size - 1)
    yi = np.clip((y_norm * (grid_size - 1)).astype(int), 0, grid_size - 1)

    img = np.zeros((grid_size, grid_size), dtype=float)
    for i, j, v in zip(xi, yi, values):
        img[j, i] += float(v)

    if img.max() > 0:
        img /= img.max()
    return img


def compute_spatial_autocorrelation_metrics(
    true_adata: ad.AnnData,
    sim_adata: ad.AnnData,
    *,
    coord_keys: Sequence[str] = ("x", "y"),
    n_neighbors: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Moran's I and Geary's C per gene for true and simulated data.
    """
    # Use intersection of genes in same order
    common_genes = np.intersect1d(true_adata.var_names, sim_adata.var_names)

    X_true = true_adata[:, common_genes].X.A if hasattr(true_adata[:, common_genes].X, "A") else np.asarray(
        true_adata[:, common_genes].X
    )
    X_sim = sim_adata[:, common_genes].X.A if hasattr(sim_adata[:, common_genes].X, "A") else np.asarray(
        sim_adata[:, common_genes].X
    )

    coords_true = _coords_from_adata(true_adata, coord_keys=coord_keys)
    coords_sim = _coords_from_adata(sim_adata, coord_keys=coord_keys)
    W_true = _build_spatial_weights(coords_true, n_neighbors=n_neighbors)
    W_sim = _build_spatial_weights(coords_sim, n_neighbors=n_neighbors)

    n_genes = common_genes.size
    moran_true = np.empty(n_genes, dtype=float)
    moran_sim = np.empty(n_genes, dtype=float)
    geary_true = np.empty(n_genes, dtype=float)
    geary_sim = np.empty(n_genes, dtype=float)

    for g in range(n_genes):
        x_t = X_true[:, g]
        x_s = X_sim[:, g]
        moran_true[g] = _moran_i(x_t, W_true)
        moran_sim[g] = _moran_i(x_s, W_sim)
        geary_true[g] = _geary_c(x_t, W_true)
        geary_sim[g] = _geary_c(x_s, W_sim)

    return moran_true, moran_sim, geary_true, geary_sim


def compute_domain_label_metrics(
    true_adata: ad.AnnData,
    sim_adata: ad.AnnData,
    *,
    label_key: str = "cell_class",
    coord_keys: Sequence[str] = ("x", "y"),
) -> Tuple[float, float]:
    """
    Compute ARI and NMI between spatial domain labels in true and simulated data.

    Since the real and simulated slices do not share identical cells,
    we match cells in the true slice to their nearest neighbor in the
    simulated slice based on spatial coordinates, and then compare the
    corresponding labels.
    """
    for adata in (true_adata, sim_adata):
        if label_key not in adata.obs:
            raise KeyError(f"'{label_key}' not found in adata.obs")

    # Match each true cell to its nearest simulated cell in space
    coords_true = _coords_from_adata(true_adata, coord_keys=coord_keys)
    coords_sim = _coords_from_adata(sim_adata, coord_keys=coord_keys)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(coords_sim)
    _, indices = nn.kneighbors(coords_true, return_distance=True)

    y_true = true_adata.obs[label_key].to_numpy()
    y_sim = sim_adata.obs[label_key].to_numpy()[indices[:, 0]]

    ari = adjusted_rand_score(y_true, y_sim)
    nmi = normalized_mutual_info_score(y_true, y_sim)
    return ari, nmi


def compute_ssim_between_densities(
    true_adata: ad.AnnData,
    sim_adata: ad.AnnData,
    *,
    coord_keys: Sequence[str] = ("x", "y"),
    grid_size: int = 64,
    include_gene_expression: bool = False,
    show_ssim_plots: bool = True,
    plot_save_dir: str = "inference/ssim",
    model_name: str = "Model",
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute SSIM between 2D density maps derived from true and simulated data.

    Returns
    -------
    occupancy_ssim : float
        SSIM score between simple occupancy density maps.
    gene_ssim : Optional[np.ndarray]
        Per-gene SSIM scores between gene-weighted density maps
        (or None if include_gene_expression is False).
    ssim_genes : Optional[np.ndarray]
        Names of genes corresponding to gene_ssim
        (or None if include_gene_expression is False).
    """
    slice_id = (
                str(true_adata.obs["slice_id"].iloc[0])
                if "slice_id" in true_adata.obs and len(true_adata.obs) > 0
                else "NA"
            )
    
    # Occupancy-based density SSIM
    img_true = _density_image(true_adata, coord_keys=coord_keys, grid_size=grid_size)
    img_sim = _density_image(sim_adata, coord_keys=coord_keys, grid_size=grid_size)
    occupancy_score = float(ssim(img_true, img_sim, data_range=1.0))

    # Visualize the occupancy-based density maps
    if show_ssim_plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_true, cmap="Greys", origin="lower")
        axes[0].set_title(f"True occupancy (slice_id={slice_id})")
        axes[0].axis("off")

        axes[1].imshow(img_sim, cmap="Greys", origin="lower")
        axes[1].set_title(f"Simulated occupancy (slice_id={slice_id})")
        axes[1].axis("off")
        fig.suptitle(f"Occupancy SSIM = {occupancy_score:.3f}", y=0.98)
        fig.tight_layout()
        save_dir = Path(plot_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{model_name}_ssim_occopancy_{slice_id}.png"
        plt.savefig(save_path)
        plt.show()
    
    if not include_gene_expression:
        return occupancy_score, None, None

    # Gene expression–weighted density SSIM per gene
    common_genes = np.intersect1d(true_adata.var_names, sim_adata.var_names)
    X_true = true_adata[:, common_genes].X.A if hasattr(true_adata[:, common_genes].X, "A") else np.asarray(true_adata[:, common_genes].X)
    X_sim = sim_adata[:, common_genes].X.A if hasattr(sim_adata[:, common_genes].X, "A") else np.asarray(sim_adata[:, common_genes].X)

    coords_true = _coords_from_adata(true_adata, coord_keys=coord_keys)
    coords_sim = _coords_from_adata(sim_adata, coord_keys=coord_keys)

    n_genes = common_genes.size
    gene_ssim = np.empty(n_genes, dtype=float)
    for g in range(n_genes):
        img_true_g = _density_image_weighted(coords_true, X_true[:, g], grid_size=grid_size)
        img_sim_g = _density_image_weighted(coords_sim, X_sim[:, g], grid_size=grid_size)
        gene_ssim[g] = float(ssim(img_true_g, img_sim_g, data_range=1.0))

    # Visualize the gene-density maps for the gene with the largest SSIM (ignoring NaNs)
    if np.isfinite(gene_ssim).any() and show_ssim_plots:
        best_g = int(np.nanargmax(gene_ssim))
        gene_name = str(common_genes[best_g])
        best_score = float(gene_ssim[best_g])

        img_true_best = _density_image_weighted(coords_true, X_true[:, best_g], grid_size=grid_size)
        img_sim_best = _density_image_weighted(coords_sim, X_sim[:, best_g], grid_size=grid_size)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_true_best, cmap="Purples", origin="lower")
        axes[0].set_title(f"True: {gene_name} (slice_id={slice_id})")
        axes[0].axis("off")

        axes[1].imshow(img_sim_best, cmap="Purples", origin="lower")
        axes[1].set_title(f"Simulated: {gene_name} (slice_id={slice_id})")
        axes[1].axis("off")

        fig.suptitle(f"Best gene SSIM = {best_score:.3f}", y=0.98)
        fig.tight_layout()
        save_dir = Path(plot_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{model_name}_ssim_best_gene_{gene_name}_{slice_id}.png"
        plt.savefig(save_path)
        plt.show()

        top_k = 5
        finite_mask = np.isfinite(gene_ssim)
        finite_idx = np.flatnonzero(finite_mask)
        if finite_idx.size:
            order = np.argsort(gene_ssim[finite_mask])[::-1]
            top_idx = finite_idx[order[:top_k]]
            print(f"Top {min(top_k, top_idx.size)} genes by SSIM (slice_id={slice_id}):")
            for rank, gi in enumerate(top_idx, start=1):
                print(f"  {rank}. {common_genes[int(gi)]}: {float(gene_ssim[int(gi)]):.4f}")
    
    return occupancy_score, gene_ssim, common_genes


def summarize_and_plot_metrics(
    true_adata: ad.AnnData,
    sim_adata: ad.AnnData,
    *,
    coord_keys: Sequence[str] = ("x", "y"),
    include_ari: bool = True,
    include_spatial_autocorrelation: bool = True,
    include_ssim: bool = True,
    include_ssim_gene_expression: bool = True,
    include_soft_metrics: bool = True,
    ari_label_key: str = "domain_label",
    autocorrelation_n_neighbors: int = 8,
    ssim_grid_size: int = 20,
    soft_radius: float = 10.0,
    show: bool = True,
    plot_name: str = "Metrics Plot",
    plot_save_dir: str = "output/metrics_plots",
    plot_save_name: str = "metrics_plot.png",
    model_name: str = "Model",
) -> Mapping[str, float]:
    """
    SSIM grid size 20, gives 400 grid cells, which means there should be on average 10 cells per pixel. 
    Metrics (all optional via flags):
    - Moran's I per gene (true vs. sim) + correlation
    - Geary's C per gene (true vs. sim) + correlation
    - ARI and NMI for label_key labels (domain_label by default)
    - SSIM for 2D density maps (occupancy only or also per-gene density)

    Returns
    -------
    dict with scalar summary statistics (e.g. correlations, ARI, NMI, SSIM).
    """
    # Initialize containers
    moran_true = moran_sim = geary_true = geary_sim = None
    moran_corr = geary_corr = np.nan
    ari = nmi = np.nan
    ssim_score = np.nan
    gene_ssim = None
    ssim_genes = None
    soft_spearman = soft_f1 = None
    soft_spearman_mean = soft_f1_mean = np.nan

    # Spatial autocorrelation metrics
    if include_spatial_autocorrelation:
        (
            moran_true,
            moran_sim,
            geary_true,
            geary_sim,
        ) = compute_spatial_autocorrelation_metrics(
            true_adata,
            sim_adata,
            coord_keys=coord_keys,
            n_neighbors=autocorrelation_n_neighbors,
        )

        valid_moran = np.isfinite(moran_true) & np.isfinite(moran_sim)
        valid_geary = np.isfinite(geary_true) & np.isfinite(geary_sim)
        moran_corr = (
            float(np.corrcoef(moran_true[valid_moran], moran_sim[valid_moran])[0, 1])
            if valid_moran.any()
            else np.nan
        )
        geary_corr = (
            float(np.corrcoef(geary_true[valid_geary], geary_sim[valid_geary])[0, 1])
            if valid_geary.any()
            else np.nan
        )

    # Domain label metrics
    if include_ari:
        ari, nmi = compute_domain_label_metrics(
            true_adata,
            sim_adata,
            label_key=ari_label_key,
            coord_keys=coord_keys,
        )

    # SSIM metrics (occupancy + optional per-gene)
    if include_ssim:
        ssim_score, gene_ssim, ssim_genes = compute_ssim_between_densities(
            true_adata,
            sim_adata,
            coord_keys=coord_keys,
            grid_size=ssim_grid_size,
            show_ssim_plots=show,
            include_gene_expression=include_ssim_gene_expression,
            plot_save_dir=plot_save_dir,
            model_name=model_name
        )

    # Soft neighbourhood metrics
    if include_soft_metrics:
        (
            soft_spearman,
            soft_f1,
            soft_spearman_mean,
            soft_f1_mean,
        ) = compute_soft_metrics(
            true_adata,
            sim_adata,
            coord_keys=coord_keys,
            radius=soft_radius
        )

    avg_gene_ssim = np.nanmean(gene_ssim) if gene_ssim is not None else np.nan

    summary: dict[str, float] = {}
    if include_spatial_autocorrelation:
        summary["moran_corr"] = moran_corr
        summary["geary_corr"] = geary_corr
    if include_ari:
        summary["ari"] = float(ari)
        summary["nmi"] = float(nmi)
    if include_ssim:
        summary["ssim"] = float(ssim_score)
        if include_ssim_gene_expression:
            summary["avg_gene_ssim"] = float(avg_gene_ssim)
    if include_soft_metrics:
        summary["soft_spearman_mean"] = float(soft_spearman_mean)
        summary["soft_f1_mean"] = float(soft_f1_mean)
    

    # Text summary
    print("Spatial metrics summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: NaN")

    # Visualization
    if show:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Moran's I scatter
        if include_spatial_autocorrelation and moran_true is not None and moran_sim is not None:
            ax0 = axes[0, 0]
            ax0.scatter(moran_true, moran_sim, s=5, alpha=0.7)
            lims = [
                np.nanmin([moran_true, moran_sim]),
                np.nanmax([moran_true, moran_sim]),
            ]
            ax0.plot(lims, lims, "k--", linewidth=1)
            ax0.set_xlabel("Moran's I (true)")
            ax0.set_ylabel("Moran's I (sim)")
            ax0.set_title(f"Moran's I per gene\ncorr={moran_corr:.2f}")
        else:
            axes[0, 0].axis("off")

        # Geary's C scatter
        if include_spatial_autocorrelation and geary_true is not None and geary_sim is not None:
            ax1 = axes[0, 1]
            ax1.scatter(geary_true, geary_sim, s=5, alpha=0.7)
            lims = [
                np.nanmin([geary_true, geary_sim]),
                np.nanmax([geary_true, geary_sim]),
            ]
            ax1.plot(lims, lims, "k--", linewidth=1)
            ax1.set_xlabel("Geary's C (true)")
            ax1.set_ylabel("Geary's C (sim)")
            ax1.set_title(f"Geary's C per gene\ncorr={geary_corr:.2f}")
        else:
            axes[0, 1].axis("off")

        # Boxplot panel for soft metrics (if requested)
        ax2 = axes[1, 0]
        if include_soft_metrics and soft_spearman is not None and soft_f1 is not None:
            data = [soft_spearman, soft_f1]
            labels = ["Soft Spearman", "Soft F1"]
            bp = ax2.boxplot(
                data,
                vert=True,
                positions=[1, 2],
                widths=0.6,
                patch_artist=True,
            )
            colors = ["#a1d99b", "#9ecae1"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set(facecolor=color, alpha=0.6, edgecolor="#636363", linewidth=1.5)
            for whisker in bp["whiskers"]:
                whisker.set(color="#636363", linewidth=1.2)
            for cap in bp["caps"]:
                cap.set(color="#636363", linewidth=1.2)
            for median in bp["medians"]:
                median.set(color="#252525", linewidth=1.5)
            ax2.set_xticks([1, 2])
            ax2.set_xticklabels(labels, rotation=0)
            ax2.set_ylabel("Score")
            ax2.set_title("Soft neighbourhood metrics")
        else:
            ax2.axis("off")

        # Boxplot of per-gene SSIM with overlaid points (if requested)
        ax3 = axes[1, 1]
        if include_ssim and include_ssim_gene_expression and gene_ssim is not None and ssim_genes is not None:
            # Single-method boxplot at x=1
            bp = ax3.boxplot(
                gene_ssim,
                vert=True,
                positions=[1],
                widths=0.5,
                patch_artist=True,
            )
            # Prettier style
            for box in bp["boxes"]:
                box.set(facecolor="#9ecae1", alpha=0.6, edgecolor="#3182bd", linewidth=1.5)
            for whisker in bp["whiskers"]:
                whisker.set(color="#3182bd", linewidth=1.2)
            for cap in bp["caps"]:
                cap.set(color="#3182bd", linewidth=1.2)
            for median in bp["medians"]:
                median.set(color="#08519c", linewidth=1.5)

            # Overlay jittered points
            x_center = 1.0
            jitter = (np.random.rand(gene_ssim.size) - 0.5) * 0.15
            ax3.scatter(
                np.full_like(gene_ssim, x_center) + jitter,
                gene_ssim,
                color="#08519c",
                alpha=0.7,
                s=15,
                edgecolor="white",
                linewidth=0.3,
            )

            ax3.set_xlim(0.5, 1.5)
            ax3.set_xticks([1])
            ax3.set_xticklabels([model_name])
            ax3.set_ylabel("SSIM")
            ax3.set_title("Per-gene density SSIM")
        else:
            ax3.axis("off")

        fig.tight_layout()
        fig.suptitle(plot_name)

        # Always save the figure before showing/closing it to avoid blank images
        plot_save_path = Path(plot_save_dir) / plot_save_name
        plot_save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_save_path.as_posix(), bbox_inches="tight")
        
        if show:
            plt.show()

    return summary





def compute_soft_metrics(
    true_adata: ad.AnnData,
    sim_adata: ad.AnnData,
    *,
    coord_keys: Sequence[str] = ("x", "y"),
    radius: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Soft metrics based on local neighbourhoods around true cells.

    For each true cell, we:
    - Average gene expression in a spatial neighbourhood in the true slice.
    - Consider the true cell's coordinates in the simulated slice coordinate
      system, and directly aggregate over all simulated cells within the same
      spatial neighbourhood (no intermediate nearest-cell matching).

    Using these local averages:
    - Soft Spearman: Spearman correlation per gene between true and simulated
      local averages, summarized by the mean correlation.
    - Soft F1: for each gene, binarize local averages at >0 (active / inactive)
      and compute an F1 score between true and simulated activation; summarized
      by the mean F1 across genes.
    """
    # Use intersection of genes in same order
    common_genes = np.intersect1d(true_adata.var_names, sim_adata.var_names)
    X_true = true_adata[:, common_genes].X.A if hasattr(true_adata[:, common_genes].X, "A") else np.asarray(
        true_adata[:, common_genes].X
    )
    X_sim = sim_adata[:, common_genes].X.A if hasattr(sim_adata[:, common_genes].X, "A") else np.asarray(
        sim_adata[:, common_genes].X
    )
    coords_true = _coords_from_adata(true_adata, coord_keys=coord_keys)
    coords_sim = _coords_from_adata(sim_adata, coord_keys=coord_keys)
    n_true, n_genes = X_true.shape

    # Local neighbourhoods in true slice
    nn_true = NearestNeighbors(radius=radius, metric="euclidean")
    nn_true.fit(coords_true)
    indices_true = nn_true.radius_neighbors(coords_true, return_distance=False)

    local_true = np.zeros_like(X_true, dtype=float)
    for i in range(n_true):
        neigh = indices_true[i]
        if neigh.size == 0:
            neigh = np.array([i], dtype=int)
        vals = X_true[neigh]
        if hasattr(vals, "A"):
            vals = vals.A
        local_true[i] = vals.mean(axis=0)

    # Local neighbourhoods in simulated slice, queried around each true cell
    nn_sim = NearestNeighbors(radius=radius, metric="euclidean")
    nn_sim.fit(coords_sim)
    # For each true cell, find all simulated cells within radius of its location
    indices_sim_for_true = nn_sim.radius_neighbors(coords_true, return_distance=False)

    local_sim = np.zeros_like(X_true, dtype=float)
    for i in range(n_true):
        neigh = indices_sim_for_true[i]
        # Fallback: if no simulated neighbours within radius, use the closest one
        if neigh.size == 0:
            _, nn_idx = nn_sim.kneighbors(coords_true[i : i + 1], n_neighbors=1, return_distance=True)
            neigh = nn_idx[0]
        vals = X_sim[neigh]
        if hasattr(vals, "A"):
            vals = vals.A
        local_sim[i] = vals.mean(axis=0)

    # Soft Spearman: per-gene correlation
    soft_spearman = np.empty(n_genes, dtype=float)
    for g in range(n_genes):
        v_true = local_true[:, g]
        v_sim = local_sim[:, g]
        if np.allclose(v_true, v_true[0]) or np.allclose(v_sim, v_sim[0]):
            soft_spearman[g] = np.nan
        else:
            soft_spearman[g] = spearmanr(v_true, v_sim).correlation

    soft_spearman_mean = float(np.nanmean(soft_spearman)) if np.isfinite(soft_spearman).any() else np.nan

    # Soft F1: per-gene F1 on binary activation (local mean > 0)
    soft_f1 = np.empty(n_genes, dtype=float)
    for g in range(n_genes):
        y_true_bin = (local_true[:, g] > 0).astype(int)
        y_sim_bin = (local_sim[:, g] > 0).astype(int)
        # If both are all-zero or all-one, f1_score is well-defined but not informative
        if y_true_bin.sum() == 0 and y_sim_bin.sum() == 0:
            soft_f1[g] = np.nan
        else:
            soft_f1[g] = f1_score(y_true_bin, y_sim_bin)

    soft_f1_mean = float(np.nanmean(soft_f1)) if np.isfinite(soft_f1).any() else np.nan

    # Nicely formatted textual summary
    print("Soft neighbourhood metrics:")
    if np.isfinite(soft_spearman_mean):
        print(f"  Soft Spearman (mean per gene): {soft_spearman_mean:.4f}")
    else:
        print("  Soft Spearman (mean per gene): NaN")
    if np.isfinite(soft_f1_mean):
        print(f"  Soft F1 (mean per gene)      : {soft_f1_mean:.4f}")
    else:
        print("  Soft F1 (mean per gene)      : NaN")

    return soft_spearman, soft_f1, soft_spearman_mean, soft_f1_mean
