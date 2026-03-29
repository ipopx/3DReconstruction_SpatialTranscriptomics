from __future__ import annotations

from typing import Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  - needed for 3D projection


def _get_xyz(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract x, y, z coordinates from AnnData.obs."""
    try:
        x = np.asarray(adata.obs["x"].values, dtype=float)
        y = np.asarray(adata.obs["y"].values, dtype=float)
        z = np.asarray(adata.obs["z"].values, dtype=float)
    except KeyError as e:
        raise KeyError("AnnData.obs must contain 'x', 'y', and 'z' columns for 3D plotting") from e
    return x, y, z


def _scatter_3d(
    ax: Axes,
    adata: ad.AnnData,
    color_key: str,
    title: str,
    cmap: str = "tab20",
    s: float = 3.0,
    z_override: float | None = None,
    codes: np.ndarray | None = None,
    categories: Sequence[object] | None = None,
    show_legend: bool = True,
    add_colorbar: bool = True,
) -> None:
    """Helper to draw a single 3D scatter plot."""
    x, y, z = _get_xyz(adata)

    if z_override is not None:
        z = np.full_like(z, fill_value=float(z_override))

    if color_key not in adata.obs:
        raise KeyError(f"'{color_key}' not found in adata.obs")

    color_vals = adata.obs[color_key]

    # If explicit codes/categories are provided, treat as categorical with global mapping
    if codes is not None and categories is not None:
        scatter = ax.scatter(x, y, z, c=codes, cmap=cmap, s=s, linewidths=0)

        if show_legend:
            handles = []
            for code, cat in enumerate(categories):
                handles.append(
                    plt.Line2D(
                        [],
                        [],
                        linestyle="",
                        marker="o",
                        label=str(cat),
                        color=scatter.cmap(scatter.norm(code)),
                    )
                )
            ax.legend(handles=handles, title=color_key, loc="upper right", fontsize="xx-small")
    else:
        # Fallback: infer from dtype
        if not np.issubdtype(color_vals.dtype, np.number):
            cats = color_vals.astype("category")
            local_codes = cats.cat.codes.to_numpy()
            scatter = ax.scatter(x, y, z, c=local_codes, cmap=cmap, s=s, linewidths=0)

            if show_legend:
                handles = []
                for code, cat in enumerate(cats.cat.categories):
                    handles.append(
                        plt.Line2D(
                            [],
                            [],
                            linestyle="",
                            marker="o",
                            label=str(cat),
                            color=scatter.cmap(scatter.norm(code)),
                        )
                    )
                ax.legend(handles=handles, title=color_key, loc="upper right", fontsize="xx-small")
        else:
            scatter = ax.scatter(x, y, z, c=color_vals, cmap=cmap, s=s, linewidths=0)
            if add_colorbar:
                cb = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.02)
                cb.set_label(color_key)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_left_middle_right_slices(
    left: ad.AnnData,
    middle: ad.AnnData,
    right: ad.AnnData,
    *,
    color_key: str = "cell_class",
    figsize: tuple[float, float] = (15.0, 5.0),
    s: float = 3.0,
    display_mode: str = "split",
) -> plt.Figure:
    """
    Plot three slices (left, middle, right) in 3D.

    All cells within each slice are projected onto a single z‑plane given by
    the mean of that slice's z‑values.

    Parameters
    ----------
    left, middle, right
        AnnData objects for the three slices. Each must have 'x', 'y', 'z'
        in ``.obs`` for spatial coordinates.
    color_key
        Column in ``.obs`` used for coloring points (default: ``'cell_class'``).
    figsize
        Size of the matplotlib figure.
    s
        Marker size for the scatter plots.
    display_mode
        Either ``'split'`` (three side‑by‑side subplots) or ``'stacked'``
        (all slices drawn in a single 3D plot).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if display_mode not in {"split", "stacked"}:
        raise ValueError("display_mode must be 'split' or 'stacked'")

    # Ensure color_key exists and build a global categorical mapping for consistency
    for adata_slice in (left, middle, right):
        if color_key not in adata_slice.obs:
            raise KeyError(f"'{color_key}' not found in adata.obs")

    # Concatenate all values to get a global category ordering
    all_vals = pd.concat(
        [
            left.obs[color_key].astype("category"),
            middle.obs[color_key].astype("category"),
            right.obs[color_key].astype("category"),
        ],
        axis=0,
    )
    all_cats = all_vals.astype("category").cat.categories

    def _codes_for(adata_slice: ad.AnnData) -> np.ndarray:
        series = adata_slice.obs[color_key].astype("category")
        series = series.cat.set_categories(all_cats)
        return series.cat.codes.to_numpy()

    # Compute constant z for each slice (mean z of that slice)
    _, _, z_left = _get_xyz(left)
    _, _, z_mid = _get_xyz(middle)
    _, _, z_right = _get_xyz(right)

    z_left_mean = float(np.mean(z_left))
    z_mid_mean = float(np.mean(z_mid))
    z_right_mean = float(np.mean(z_right))

    if display_mode == "split":
        fig = plt.figure(figsize=figsize)
        axes: Sequence[Axes] = []
        for i, title in enumerate(["Left slice", "Middle slice", "Right slice"], start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            axes.append(ax)

        _scatter_3d(
            axes[0],
            left,
            color_key=color_key,
            title=f"Left slice (z≈{z_left_mean:.2f})",
            s=s,
            z_override=z_left_mean,
            codes=_codes_for(left),
            categories=all_cats,
        )
        _scatter_3d(
            axes[1],
            middle,
            color_key=color_key,
            title=f"Middle slice (z≈{z_mid_mean:.2f})",
            s=s,
            z_override=z_mid_mean,
            codes=_codes_for(middle),
            categories=all_cats,
        )
        _scatter_3d(
            axes[2],
            right,
            color_key=color_key,
            title=f"Right slice (z≈{z_right_mean:.2f})",
            s=s,
            z_override=z_right_mean,
            codes=_codes_for(right),
            categories=all_cats,
        )

        fig.tight_layout()
        return fig

    # Stacked mode: draw all slices on a single 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    _scatter_3d(
        ax,
        left,
        color_key=color_key,
        title="Left / Middle / Right (stacked)",
        s=s,
        z_override=z_left_mean,
        codes=_codes_for(left),
        categories=all_cats,
        # we'll add a single legend after plotting all slices
        show_legend=False,
    )
    _scatter_3d(
        ax,
        middle,
        color_key=color_key,
        title="Left / Middle / Right (stacked)",
        s=s,
        z_override=z_mid_mean,
        codes=_codes_for(middle),
        categories=all_cats,
        show_legend=False,
    )
    _scatter_3d(
        ax,
        right,
        color_key=color_key,
        title="Left / Middle / Right (stacked)",
        s=s,
        z_override=z_right_mean,
        codes=_codes_for(right),
        categories=all_cats,
        show_legend=False,
    )

    # Build a single combined legend on the stacked axis
    handles = []
    # Use colormap and normalization from the last scatter call
    # (consistent across slices because codes/categories are shared)
    cmap = plt.get_cmap("tab20")
    for code, cat in enumerate(all_cats):
        handles.append(
            plt.Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                label=str(cat),
                color=cmap(code / max(1, len(all_cats) - 1)),
            )
        )
    ax.legend(handles=handles, title=color_key, loc="upper right", fontsize="xx-small")

    fig.tight_layout()
    return fig


def get_spatial_coords(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """
    Get 2D (x, y) coordinates from AnnData.

    Preference order:
    - ``adata.obsm['spatial']`` if present (common in spatial protocols)
    - ``adata.obs['x']`` / ``adata.obs['y']`` otherwise
    """
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        return coords[:, 0], coords[:, 1]
    if "x" in adata.obs and "y" in adata.obs:
        return (
            np.asarray(adata.obs["x"].values, dtype=float),
            np.asarray(adata.obs["y"].values, dtype=float),
        )
    raise ValueError("No spatial coordinates found (expected obsm['spatial'] or obs['x'], obs['y'])")


def plot_spatial_locations(
    true_adata: ad.AnnData,
    sim_adata: ad.AnnData,
    *,
    cell_type_color: bool = False,
    label_key: str = "cell_class",
) -> None:
    """
    Plot spatial locations of true and simulated data side by side.

    Parameters
    ----------
    true_adata, sim_adata
        AnnData objects with spatial coordinates.
    cell_type_color
        If True, color points by ``label_key`` (e.g. cell type).
        If False, use simple solid colors.
    label_key
        Column in ``obs`` used for coloring when ``cell_type_color=True``.
    """
    x1, y1 = get_spatial_coords(true_adata)
    x2, y2 = get_spatial_coords(sim_adata)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if cell_type_color and label_key in true_adata.obs and label_key in sim_adata.obs:
        # Use a shared categorical mapping to keep colors consistent
        cats_true = true_adata.obs[label_key].astype("category")
        cats_sim = sim_adata.obs[label_key].astype("category")
        all_cats = pd.concat([cats_true, cats_sim]).astype("category").cat.categories

        def _codes_for(adata_slice: ad.AnnData) -> np.ndarray:
            series = adata_slice.obs[label_key].astype("category")
            series = series.cat.set_categories(all_cats)
            return series.cat.codes.to_numpy()

        codes_true = _codes_for(true_adata)
        codes_sim = _codes_for(sim_adata)

        scatter0 = axes[0].scatter(x1, y1, c=codes_true, s=1, alpha=0.7, cmap="tab20")
        axes[0].set_title("True (cell type)")
        axes[0].set_aspect("equal")

        scatter1 = axes[1].scatter(x2, y2, c=codes_sim, s=1, alpha=0.7, cmap="tab20")
        axes[1].set_title("Simulated (cell type)")
        axes[1].set_aspect("equal")

        # Single colorbar for cell types
        plt.colorbar(scatter1, ax=axes, location="right", pad=0.1, label=label_key)
    else:
        # Simple location-only view
        axes[0].scatter(x1, y1, s=1, alpha=0.5, c="steelblue")
        axes[0].set_title("True (locations)")
        axes[0].set_aspect("equal")

        axes[1].scatter(x2, y2, s=1, alpha=0.5, c="coral")
        axes[1].set_title("Simulated (locations)")
        axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.show()


def plot_cell_type_distribution(
    true_adata: ad.AnnData,
    sim_adata: ad.AnnData,
    *,
    label_key: str = "cell_class",
) -> None:
    """
    Plot cell type distribution of true and simulated data as side‑by‑side bars.
    """
    if label_key not in true_adata.obs or label_key not in sim_adata.obs:
        raise KeyError(f"'{label_key}' must be present in both true_adata.obs and sim_adata.obs")

    ct_true = true_adata.obs[label_key].value_counts().sort_index()
    ct_sim = sim_adata.obs[label_key].value_counts().sort_index()

    all_types = sorted(
        set(ct_true.index) | set(ct_sim.index),
        key=lambda x: (int(x) if str(x).isdigit() else 999, str(x)),
    )

    n = len(all_types)
    x = np.arange(n)
    w = 0.35
    v_true = [ct_true.get(t, 0) for t in all_types]
    v_sim = [ct_sim.get(t, 0) for t in all_types]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), 4))
    ax.bar(x - w / 2, v_true, w, label="True", color="steelblue")
    ax.bar(x + w / 2, v_sim, w, label="Simulated", color="coral")
    ax.set_xlabel(f"Cell type ({label_key})")
    ax.set_ylabel("Count")
    ax.set_title("Cell type distribution")
    ax.legend()

    stride = max(1, n // 15)
    ax.set_xticks(x[::stride])
    ax.set_xticklabels([all_types[i] for i in range(0, n, stride)], rotation=45, ha="right")

    plt.tight_layout()
    plt.show()
