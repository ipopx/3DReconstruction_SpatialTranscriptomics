from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_spatial_coords(adata: ad.AnnData) -> np.ndarray:
    """
    Returns (n_cells, 2) float array of x/y coordinates.
    Prefers `.obsm["spatial"]` and falls back to `.obs[["x","y"]]`.
    """
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError('Expected `adata.obsm["spatial"]` to be (n,2+) array.')
        return coords[:, :2].astype(float, copy=False)

    if "x" in adata.obs and "y" in adata.obs:
        return adata.obs[["x", "y"]].to_numpy(dtype=float)

    raise KeyError('Could not find coordinates in `.obsm["spatial"]` or `.obs[["x","y"]]`.')


def _is_categorical(values) -> bool:
    dtype_name = getattr(values.dtype, "name", "")
    return dtype_name == "category" or values.dtype == object


def _plot_slice_categorical(
    ax: plt.Axes,
    adata: ad.AnnData,
    *,
    color_label: str,
    title: str,
    category_to_code: dict[str, int],
    n_categories: int,
) -> None:
    coords = _get_spatial_coords(adata)
    x = coords[:, 0]
    y = coords[:, 1]

    if color_label not in adata.obs:
        raise KeyError(f'color_label "{color_label}" not found in adata.obs')

    values = adata.obs[color_label].astype(str).to_numpy()
    codes = np.array([category_to_code.get(v, -1) for v in values], dtype=int)
    ax.scatter(
        x,
        y,
        c=codes,
        s=6,
        alpha=0.9,
        cmap="tab20",
        vmin=-0.5,
        vmax=max(n_categories - 0.5, 0.5),
        linewidths=0,
    )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_slice_numeric(
    ax: plt.Axes,
    adata: ad.AnnData,
    *,
    color_label: str,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    coords = _get_spatial_coords(adata)
    x = coords[:, 0]
    y = coords[:, 1]

    if color_label not in adata.obs:
        raise KeyError(f'color_label "{color_label}" not found in adata.obs')

    values = np.asarray(adata.obs[color_label], dtype=float)
    sc = ax.scatter(
        x,
        y,
        c=values,
        s=6,
        alpha=0.9,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return sc


def _get_vector_for_color(adata: ad.AnnData, key: str) -> np.ndarray:
    """
    Returns a per-cell numeric vector for coloring.
    Prefers `adata.obs[key]`, otherwise falls back to gene expression `adata[:, key].X`.
    """
    if key in adata.obs:
        return np.asarray(adata.obs[key], dtype=float)

    if key in adata.var_names:
        x = adata[:, key].X
        # (n_cells, 1) -> (n_cells,)
        if hasattr(x, "A"):
            x = x.A
        x = np.asarray(x).reshape(-1)
        return x.astype(float, copy=False)

    raise KeyError(f'"{key}" not found in adata.obs or adata.var_names')


def _plot_slice_numeric_vector(
    ax: plt.Axes,
    adata: ad.AnnData,
    *,
    values: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> plt.Collection:
    coords = _get_spatial_coords(adata)
    x = coords[:, 0]
    y = coords[:, 1]
    sc = ax.scatter(
        x,
        y,
        c=values,
        s=6,
        alpha=0.9,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return sc


def main() -> None:
    cfg_path = Path(__file__).with_name("inference_config.json")
    cfg = _load_json(cfg_path)

    true_path = Path(cfg["true_adata_path"])
    sim_path = Path(cfg["sim_adata_path"])
    model_name = str(cfg.get("model_name", "spatialz"))
    dataset = str(cfg.get("dataset", "merfish"))
    
    cfg = cfg.get("visualize_parameters", {})
    color_label = str(cfg.get("color_label", "cell_class"))
    
    if not true_path.exists():
        raise FileNotFoundError(f"true_adata_path not found: {true_path}")
    if not sim_path.exists():
        raise FileNotFoundError(f"sim_adata_path not found: {sim_path}")

    true_adata = ad.read_h5ad(true_path)
    sim_adata = ad.read_h5ad(sim_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    if color_label not in true_adata.obs or color_label not in sim_adata.obs:
        missing = []
        if color_label not in true_adata.obs:
            missing.append(f"true missing '{color_label}'")
        if color_label not in sim_adata.obs:
            missing.append(f"sim missing '{color_label}'")
        raise KeyError(", ".join(missing))

    v_true = true_adata.obs[color_label]
    v_sim = sim_adata.obs[color_label]

    if _is_categorical(v_true) or _is_categorical(v_sim):
        # Build a shared category -> color code mapping for both plots
        cats_true = v_true.astype("category").cat.categories.astype(str).to_list()
        cats_sim = v_sim.astype("category").cat.categories.astype(str).to_list()
        all_cats = list(dict.fromkeys([*cats_true, *cats_sim]))  # preserve order
        category_to_code = {c: i for i, c in enumerate(all_cats)}

        _plot_slice_categorical(
            axes[0],
            true_adata,
            color_label=color_label,
            title="True",
            category_to_code=category_to_code,
            n_categories=len(all_cats),
        )
        _plot_slice_categorical(
            axes[1],
            sim_adata,
            color_label=color_label,
            title="Simulated",
            category_to_code=category_to_code,
            n_categories=len(all_cats),
        )

        # Single legend outside (right side)
        cmap = plt.get_cmap("tab20")
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=cmap(i % cmap.N),
                markersize=6,
            )
            for i in range(len(all_cats))
        ]
        fig.legend(
            handles,
            all_cats,
            title=color_label,
            loc="center right",
            frameon=False,
            fontsize=7,
            title_fontsize=8,
        )
        # Reserve space for legend (right) and suptitle (top)
        fig.subplots_adjust(right=0.85, top=0.88, wspace=0.25)
    else:
        # Numeric: shared scale + single colorbar
        vals_all = np.concatenate([np.asarray(v_true, dtype=float), np.asarray(v_sim, dtype=float)])
        vals_all = vals_all[np.isfinite(vals_all)]
        vmin = float(np.nanmin(vals_all)) if vals_all.size else 0.0
        vmax = float(np.nanmax(vals_all)) if vals_all.size else 1.0
        sc0 = _plot_slice_numeric(axes[0], true_adata, color_label=color_label, title="True", vmin=vmin, vmax=vmax)
        sc1 = _plot_slice_numeric(axes[1], sim_adata, color_label=color_label, title="Simulated", vmin=vmin, vmax=vmax)
        # Use the second scatter for the colorbar (same norm/cmap)
        cb = fig.colorbar(sc1, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        cb.set_label(color_label)
        # Reserve a bit of space for the suptitle
        fig.subplots_adjust(top=0.88, wspace=0.2)

    slice_id = (
        str(true_adata.obs["slice_id"].values[0])
        if "slice_id" in true_adata.obs and len(true_adata.obs) > 0
        else "NA"
    )
    fig.suptitle(f"Slice {slice_id} visualization (color: {color_label}) - {model_name}", y=0.98)

    save_dir = Path("inference") / "visualize"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{dataset}_{model_name}_slice_{slice_id}.png"
    plt.savefig(save_path)
    plt.show()
    
    genes_to_visualize = cfg.get("genes_to_visualize", [])
    for gene in genes_to_visualize:
        v_true_gene = _get_vector_for_color(true_adata, str(gene))
        v_sim_gene = _get_vector_for_color(sim_adata, str(gene))
        vals_all = np.concatenate([v_true_gene, v_sim_gene])
        vals_all = vals_all[np.isfinite(vals_all)]
        vmin_g = float(np.nanmin(vals_all)) if vals_all.size else 0.0
        vmax_g = float(np.nanmax(vals_all)) if vals_all.size else 1.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
        sc1 = _plot_slice_numeric_vector(
            axes[0],
            true_adata,
            values=v_true_gene,
            title="True",
            vmin=vmin_g,
            vmax=vmax_g,
        )
        sc2 = _plot_slice_numeric_vector(
            axes[1],
            sim_adata,
            values=v_sim_gene,
            title="Simulated",
            vmin=vmin_g,
            vmax=vmax_g,
        )
        cb = fig.colorbar(sc2, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        cb.set_label(str(gene))
        fig.subplots_adjust(top=0.88, wspace=0.2)
        fig.suptitle(f"Slice {slice_id} visualization (color: {gene}) - {model_name}", y=0.98)
        save_path = save_dir / f"{dataset}_{model_name}_slice_{slice_id}_gene_{gene}.png"
        plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    main()
