from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import mannwhitneyu, wilcoxon
except Exception:  # pragma: no cover
    mannwhitneyu = None  # type: ignore[assignment]
    wilcoxon = None  # type: ignore[assignment]

from .metrics import (
    compute_domain_label_metrics,
    compute_soft_metrics,
    compute_spatial_autocorrelation_metrics,
    compute_ssim_between_densities,
)


def _pvalue_two_models(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute a p-value comparing model A vs model B.

    Prefers a paired non-parametric test (Wilcoxon) when lengths match; otherwise
    falls back to an unpaired Mann-Whitney U test. NaNs are ignored.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if a.size == 0 or b.size == 0:
        return np.nan

    if a.size == b.size and wilcoxon is not None:
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return np.nan
        try:
            return float(wilcoxon(a[mask], b[mask]).pvalue)
        except Exception:
            return np.nan

    if mannwhitneyu is None:
        return np.nan
    a2 = a[np.isfinite(a)]
    b2 = b[np.isfinite(b)]
    if a2.size < 3 or b2.size < 3:
        return np.nan
    try:
        return float(mannwhitneyu(a2, b2, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return f"p={p:.1e}"
    return f"p={p:.2g}"


def _add_pvalue_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k")
    ax.text((x1 + x2) * 0.5, y + h, text, ha="center", va="bottom", fontsize=8)


def _align_by_gene(
    vals_a: np.ndarray,
    genes_a: np.ndarray,
    vals_b: np.ndarray,
    genes_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align two per-gene vectors to the same gene set and ordering.

    Returns
    -------
    genes : np.ndarray
        The common genes (sorted as in `np.intersect1d`).
    a_aligned : np.ndarray
        `vals_a` restricted/reordered to `genes`.
    b_aligned : np.ndarray
        `vals_b` restricted/reordered to `genes`.
    """
    genes_a = np.asarray(genes_a).astype(str)
    genes_b = np.asarray(genes_b).astype(str)
    vals_a = np.asarray(vals_a, dtype=float).reshape(-1)
    vals_b = np.asarray(vals_b, dtype=float).reshape(-1)

    common = np.intersect1d(genes_a, genes_b)
    if common.size == 0:
        return common, np.array([], dtype=float), np.array([], dtype=float)

    idx_a = {g: i for i, g in enumerate(genes_a)}
    idx_b = {g: i for i, g in enumerate(genes_b)}
    a_aligned = np.array([vals_a[idx_a[g]] for g in common], dtype=float)
    b_aligned = np.array([vals_b[idx_b[g]] for g in common], dtype=float)
    return common, a_aligned, b_aligned


def compute_comparative_metrics(
    true_adata: ad.AnnData,
    sim_adata_1: ad.AnnData,
    sim_adata_2: ad.AnnData,
    *,
    coord_keys: Sequence[str] = ("x", "y"),
    include_ari: bool = True,
    include_spatial_autocorrelation: bool = True,
    include_ssim: bool = True,
    include_ssim_gene_expression: bool = True,
    include_soft_metrics: bool = True,
    ari_label_key: str = "domain_label",
    autocorrelation_n_neighbors: int = 8,
    ssim_grid_size: int = 64,
    soft_radius: float = 10.0,
    show: bool = True,
    model_names: Sequence[str] = ("Model 1", "Model 2"),
    plot_save_path: Union[str, Path, None] = None,
    dataset_name: str | None = None,
) -> Mapping[str, Mapping[str, float]]:
    """
    Compute and *plot* comparative spatial metrics for two simulated datasets
    against the same ground-truth slice.

    This mirrors the metrics used in `summarize_and_plot_metrics` but produces
    a single figure where both simulations are shown together.

    Returns
    -------
    A nested dict:
        {
          "model_1": {metric_name: value, ...},
          "model_2": {metric_name: value, ...},
        }
    """
    if len(model_names) != 2:
        raise ValueError("model_names must have length 2 (for sim_adata_1 and sim_adata_2).")

    sims = [sim_adata_1, sim_adata_2]

    summaries: dict[str, dict[str, float]] = {"model_1": {}, "model_2": {}}

    moran_true_list = [None, None]
    moran_sim_list = [None, None]
    geary_true_list = [None, None]
    geary_sim_list = [None, None]
    moran_corrs = [np.nan, np.nan]
    geary_corrs = [np.nan, np.nan]

    ari_vals = [np.nan, np.nan]
    nmi_vals = [np.nan, np.nan]

    ssim_scores = [np.nan, np.nan]
    gene_ssim_list = [None, None]
    gene_ssim_genes_list = [None, None]

    soft_spearman_list = [None, None]
    soft_f1_list = [None, None]
    soft_spearman_means = [np.nan, np.nan]
    soft_f1_means = [np.nan, np.nan]

    for idx, sim_adata in enumerate(sims):
        key = "model_1" if idx == 0 else "model_2"

        # Spatial autocorrelation
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
            moran_true_list[idx] = moran_true
            moran_sim_list[idx] = moran_sim
            geary_true_list[idx] = geary_true
            geary_sim_list[idx] = geary_sim

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
            moran_corrs[idx] = moran_corr
            geary_corrs[idx] = geary_corr
            summaries[key]["moran_corr"] = moran_corr
            summaries[key]["geary_corr"] = geary_corr

        # ARI / NMI
        if include_ari:
            ari, nmi = compute_domain_label_metrics(
                true_adata,
                sim_adata,
                label_key=ari_label_key,
                coord_keys=coord_keys,
            )
            ari_vals[idx] = float(ari)
            nmi_vals[idx] = float(nmi)
            summaries[key]["ari"] = float(ari)
            summaries[key]["nmi"] = float(nmi)

        # SSIM
        if include_ssim:
            ssim_score, gene_ssim, ssim_genes = compute_ssim_between_densities(
                true_adata,
                sim_adata,
                coord_keys=coord_keys,
                grid_size=ssim_grid_size,
                include_gene_expression=include_ssim_gene_expression,
            )
            ssim_scores[idx] = float(ssim_score)
            summaries[key]["ssim"] = float(ssim_score)
            if include_ssim_gene_expression and gene_ssim is not None:
                gene_ssim_list[idx] = gene_ssim
                gene_ssim_genes_list[idx] = ssim_genes
                avg_gene_ssim = float(np.nanmean(gene_ssim)) if np.isfinite(gene_ssim).any() else np.nan
                summaries[key]["avg_gene_ssim"] = avg_gene_ssim

        # Soft metrics
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
                radius=soft_radius,
            )
            soft_spearman_list[idx] = soft_spearman
            soft_f1_list[idx] = soft_f1
            soft_spearman_means[idx] = soft_spearman_mean
            soft_f1_means[idx] = soft_f1_mean
            summaries[key]["soft_spearman_mean"] = float(soft_spearman_mean)
            summaries[key]["soft_f1_mean"] = float(soft_f1_mean)

    # Console summary
    print("Comparative spatial metrics summary:")
    for key, name in zip(("model_1", "model_2"), model_names):
        print(f"  {name}:")
        for k, v in summaries[key].items():
            print(f"    {k}: {v:.4f}" if np.isfinite(v) else f"    {k}: NaN")

    # Single comparative figure (optional show and/or save to output/compare)
    if show or plot_save_path is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        colors = ["#3182bd", "#e6550d"]
        ds = dataset_name or "dataset"
        fig.suptitle(f"{ds}: {model_names[0]} vs {model_names[1]}", y=1.02)

        # Moran's I
        ax0 = axes[0, 0]
        if include_spatial_autocorrelation and all(m is not None for m in moran_true_list):
            for idx in range(2):
                ax0.scatter(
                    moran_true_list[idx],
                    moran_sim_list[idx],
                    s=5,
                    alpha=0.7,
                    color=colors[idx],
                    label=f"{model_names[idx]} (corr={moran_corrs[idx]:.2f})",
                )
            all_moran_true = np.concatenate(moran_true_list)  # type: ignore[arg-type]
            all_moran_sim = np.concatenate(moran_sim_list)  # type: ignore[arg-type]
            lims = [
                np.nanmin([all_moran_true, all_moran_sim]),
                np.nanmax([all_moran_true, all_moran_sim]),
            ]
            ax0.plot(lims, lims, "k--", linewidth=1)
            ax0.set_xlabel("Moran's I (true)")
            ax0.set_ylabel("Moran's I (sim)")
            ax0.set_title("Moran's I per gene")
            ax0.legend(frameon=False, fontsize=8)
        else:
            ax0.axis("off")

        # Geary's C
        ax1 = axes[0, 1]
        if include_spatial_autocorrelation and all(g is not None for g in geary_true_list):
            for idx in range(2):
                ax1.scatter(
                    geary_true_list[idx],
                    geary_sim_list[idx],
                    s=5,
                    alpha=0.7,
                    color=colors[idx],
                    label=f"{model_names[idx]} (corr={geary_corrs[idx]:.2f})",
                )
            all_geary_true = np.concatenate(geary_true_list)  # type: ignore[arg-type]
            all_geary_sim = np.concatenate(geary_sim_list)  # type: ignore[arg-type]
            lims = [
                np.nanmin([all_geary_true, all_geary_sim]),
                np.nanmax([all_geary_true, all_geary_sim]),
            ]
            ax1.plot(lims, lims, "k--", linewidth=1)
            ax1.set_xlabel("Geary's C (true)")
            ax1.set_ylabel("Geary's C (sim)")
            ax1.set_title("Geary's C per gene")
            ax1.legend(frameon=False, fontsize=8)
        else:
            ax1.axis("off")

        # Soft metrics (boxplots) — same blue/orange as other plots, model names on x
        ax2 = axes[1, 0]
        if include_soft_metrics and all(s is not None for s in soft_spearman_list) and all(
            f is not None for f in soft_f1_list
        ):
            # Order: model1 Spearman, model2 Spearman, model1 F1, model2 F1
            data = [
                soft_spearman_list[0],
                soft_spearman_list[1],
                soft_f1_list[0],
                soft_f1_list[1],
            ]
            positions = [1, 2, 4, 5]
            bp = ax2.boxplot(
                data,
                vert=True,
                positions=positions,
                widths=0.6,
                patch_artist=True,
            )
            # Blue for model 1, orange for model 2 (match colors list and per-gene SSIM)
            box_colors = ["#9ecae1", "#fdae6b", "#9ecae1", "#fdae6b"]
            edge_colors = ["#3182bd", "#e6550d", "#3182bd", "#e6550d"]
            for patch, fc, ec in zip(bp["boxes"], box_colors, edge_colors):
                patch.set(facecolor=fc, alpha=0.6, edgecolor=ec, linewidth=1.5)
            for whisker in bp["whiskers"]:
                whisker.set(color="#636363", linewidth=1.2)
            for cap in bp["caps"]:
                cap.set(color="#636363", linewidth=1.2)
            for median in bp["medians"]:
                median.set(color="#252525", linewidth=1.5)

            ax2.set_xticks(positions)
            ax2.set_xticklabels(
                [model_names[0], model_names[1], model_names[0], model_names[1]]
            )
            ax2.set_ylabel("Score")
            ax2.set_title("Soft neighbourhood metrics (per-gene): Spearman (left), F1 (right)")

            # P-values between models for each soft metric
            print("Soft metrics p-value:")
            p_spear = _pvalue_two_models(np.asarray(soft_spearman_list[0]), np.asarray(soft_spearman_list[1]))
            p_f1 = _pvalue_two_models(np.asarray(soft_f1_list[0]), np.asarray(soft_f1_list[1]))

            # Place brackets above each pair
            y_max_spear = float(
                np.nanmax(np.concatenate([np.asarray(soft_spearman_list[0]), np.asarray(soft_spearman_list[1])]))
            )
            y_max_f1 = float(
                np.nanmax(np.concatenate([np.asarray(soft_f1_list[0]), np.asarray(soft_f1_list[1])]))
            )
            y0 = max(y_max_spear, y_max_f1)
            if np.isfinite(y0):
                span = 0.08 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])
                _add_pvalue_bracket(ax2, 1, 2, y_max_spear + span * 0.2, span * 0.4, _format_p(p_spear))
                _add_pvalue_bracket(ax2, 4, 5, y_max_f1 + span * 0.2, span * 0.4, _format_p(p_f1))
        else:
            ax2.axis("off")

        # Per-gene SSIM comparison
        ax3 = axes[1, 1]
        if include_ssim and include_ssim_gene_expression and all(
            g is not None for g in gene_ssim_list
        ):
            # Align per-gene SSIM vectors by gene name (robust pairing)
            if gene_ssim_genes_list[0] is not None and gene_ssim_genes_list[1] is not None:
                _, g0, g1 = _align_by_gene(
                    np.asarray(gene_ssim_list[0]),
                    np.asarray(gene_ssim_genes_list[0]),
                    np.asarray(gene_ssim_list[1]),
                    np.asarray(gene_ssim_genes_list[1]),
                )
                data = [g0, g1]
            else:
                data = [np.asarray(gene_ssim_list[0]), np.asarray(gene_ssim_list[1])]
            positions = [1, 2]
            bp = ax3.boxplot(
                data,
                vert=True,
                positions=positions,
                widths=0.5,
                patch_artist=True,
            )
            box_colors = ["#9ecae1", "#fdae6b"]
            edge_colors = ["#3182bd", "#e6550d"]
            for patch, fc, ec in zip(bp["boxes"], box_colors, edge_colors):
                patch.set(facecolor=fc, alpha=0.6, edgecolor=ec, linewidth=1.5)
            for whisker in bp["whiskers"]:
                whisker.set(color="#636363", linewidth=1.2)
            for cap in bp["caps"]:
                cap.set(color="#636363", linewidth=1.2)
            for median in bp["medians"]:
                median.set(color="#252525", linewidth=1.5)

            ax3.set_xticks(positions)
            ax3.set_xticklabels(model_names)
            ax3.set_ylabel("SSIM")
            ax3.set_title("Per-gene density SSIM")

            # Overlay jittered points (one dot per gene)
            for pos, vals, c in zip(positions, data, edge_colors):
                vals_arr = np.asarray(vals, dtype=float)
                vals_arr = vals_arr[np.isfinite(vals_arr)]
                if vals_arr.size == 0:
                    continue
                jitter = (np.random.rand(vals_arr.size) - 0.5) * 0.18
                ax3.scatter(
                    np.full(vals_arr.size, pos, dtype=float) + jitter,
                    vals_arr,
                    color=c,
                    alpha=0.35,
                    s=10,
                    edgecolor="none",
                    zorder=3,
                )

            # P-value between models (paired per gene when possible)
            p_gene = _pvalue_two_models(np.asarray(data[0]), np.asarray(data[1]))
            y_max = float(np.nanmax(np.concatenate([np.asarray(data[0]), np.asarray(data[1])])))
            if np.isfinite(y_max):
                span = 0.08 * (ax3.get_ylim()[1] - ax3.get_ylim()[0])
                _add_pvalue_bracket(ax3, 1, 2, y_max + span * 0.2, span * 0.4, _format_p(p_gene))
        else:
            ax3.axis("off")

        fig.tight_layout()
        save_path = (
            Path(plot_save_path)
            if plot_save_path is not None
            else Path("output/compare/comparative_metrics.png")
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

    return summaries
