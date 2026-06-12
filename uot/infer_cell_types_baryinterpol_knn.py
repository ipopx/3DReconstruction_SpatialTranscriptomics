#!/usr/bin/env python3
"""
Infer cell types for UOT-generated middle slices via KNN in gene space.

Reference slices (bottom + top) supply labels from a configurable ``obs`` column
(default from dataset config, e.g. ``leiden`` for STARmap). Labels are written
to the same column on the generated middle slice.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Union

import numpy as np

from uot.adapter import _dense_X
from uot.data import parse_triplets_file
from uot.geometry import norm_cae_min_max, union_cae_min_max
from uot.interpolation_genespace import h5ad_file_lock, inferred_path_from_middle


DEFAULT_TAG = "baryinterpol_l100_reg0p005_regm0p1"


def _read_genes_and_labels(
    h5ad_path: str,
    *,
    label_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    try:
        if label_col not in adata.obs.columns:
            raise KeyError(f"{h5ad_path}: missing obs[{label_col!r}]")
        genes = _dense_X(adata)
        labels = np.asarray(adata.obs[label_col].astype(str).values)
        var_names = np.asarray(adata.var_names.astype(str))
        if genes.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{h5ad_path}: n_obs mismatch genes={genes.shape[0]} labels={labels.shape[0]}"
            )
        return genes, labels, var_names
    finally:
        del adata


def _genes_and_labels_from_adata(adata, *, label_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if label_col not in adata.obs.columns:
        raise KeyError(f"missing obs[{label_col!r}]")
    genes = _dense_X(adata)
    labels = np.asarray(adata.obs[label_col].astype(str).values)
    var_names = np.asarray(adata.var_names.astype(str))
    return genes, labels, var_names


def _vote_with_tie_break(
    neighbor_labels: np.ndarray,
    neighbor_dists: np.ndarray,
) -> str:
    counts = Counter(map(str, neighbor_labels.tolist()))
    best = max(counts.values())
    tied = {lab for lab, c in counts.items() if c == best}
    if len(tied) == 1:
        return next(iter(tied))
    for lab in neighbor_labels.tolist():
        lab = str(lab)
        if lab in tied:
            return lab
    return str(neighbor_labels[0])


def infer_cell_types_for_middle(
    *,
    bottom,
    top,
    query_genes: np.ndarray,
    var_names: np.ndarray,
    label_col: str,
    k: int = 7,
) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    ref_genes_bottom, ref_y_bottom, var_bottom = _genes_and_labels_from_adata(bottom, label_col=label_col)
    ref_genes_top, ref_y_top, var_top = _genes_and_labels_from_adata(top, label_col=label_col)
    if not np.array_equal(var_bottom, var_top):
        raise ValueError("Gene names differ between bottom and top reference slices")
    if not np.array_equal(var_bottom, var_names):
        raise ValueError("Gene names differ between reference slices and query")

    gene_min, gene_max = union_cae_min_max(ref_genes_bottom, ref_genes_top)
    ref_z = np.vstack(
        [
            norm_cae_min_max(ref_genes_bottom, gene_min, gene_max),
            norm_cae_min_max(ref_genes_top, gene_min, gene_max),
        ]
    )
    ref_y = np.concatenate([ref_y_bottom, ref_y_top], axis=0)
    q_z = norm_cae_min_max(query_genes, gene_min, gene_max)

    k_eff = int(min(max(1, int(k)), ref_z.shape[0]))
    knn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    knn.fit(ref_z)
    dists, idx = knn.kneighbors(q_z, return_distance=True)

    out = np.empty(q_z.shape[0], dtype=object)
    for i in range(q_z.shape[0]):
        out[i] = _vote_with_tie_break(ref_y[idx[i]], dists[i])
    return out


def infer_gex_cell_types_for_middle_file(
    *,
    bottom_h5ad: str,
    top_h5ad: str,
    generated_middle_h5ad: str,
    label_col: str,
    k: int,
) -> np.ndarray:
    import anndata as ad
    import pandas as pd

    bottom = ad.read_h5ad(bottom_h5ad)
    top = ad.read_h5ad(top_h5ad)
    gen = ad.read_h5ad(generated_middle_h5ad)
    labels = infer_cell_types_for_middle(
        bottom=bottom,
        top=top,
        query_genes=_dense_X(gen),
        var_names=np.asarray(gen.var_names.astype(str)),
        label_col=label_col,
        k=k,
    )
    gen.obs[label_col] = pd.Series(labels, index=gen.obs_names, dtype=str)
    with h5ad_file_lock(generated_middle_h5ad, exclusive=True):
        gen.write_h5ad(generated_middle_h5ad)
    return labels


def _load_json_config(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tag",
        type=str,
        default=DEFAULT_TAG,
        help=f"Baryinterpol output tag (default: {DEFAULT_TAG}).",
    )
    p.add_argument(
        "--triplets",
        type=str,
        default="triplets_genespace.tsv",
        help="TSV: bottom top middle paths.",
    )
    p.add_argument("--k", type=int, default=7, help="KNN neighbors (default: 7).")
    p.add_argument(
        "--config",
        type=str,
        default="utils/config.json",
        help="Config JSON with data.*.cell_class_key for label column.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="starmap",
        help="Dataset key under config.data for cell_class_key.",
    )
    args = p.parse_args()

    config = _load_json_config(args.config)
    data_cfg = config.get("data", {}).get(args.dataset, {})
    label_col = str(data_cfg.get("cell_class_key", "leiden"))

    tag = str(args.tag).strip()
    if not tag:
        p.error("--tag must be non-empty")

    triplets_path = os.path.abspath(args.triplets)
    if not os.path.isfile(triplets_path):
        raise FileNotFoundError(f"Triplets file not found: {triplets_path}")

    triplets = parse_triplets_file(triplets_path)
    if not triplets:
        print(f"No triplets in {triplets_path}", flush=True)
        return

    k = max(1, int(args.k))

    print(
        f"KNN cell-type inference for baryinterpol middles  tag={tag!r}  "
        f"triplets={len(triplets)}  k={k}  label_col={label_col!r}",
        flush=True,
    )

    n_ok = 0
    n_skip_exists = 0
    n_skip_missing = 0
    n_fail = 0

    for i, (bottom, top, middle) in enumerate(triplets, start=1):
        generated = inferred_path_from_middle(middle, tag)
        stem = Path(middle).stem

        print(f"\n--- [{i}/{len(triplets)}] {stem} ---", flush=True)

        if not os.path.isfile(bottom) or not os.path.isfile(top) or not os.path.isfile(generated):
            print(
                f"Skip (missing files) bottom={os.path.isfile(bottom)} "
                f"top={os.path.isfile(top)} generated={os.path.isfile(generated)}",
                flush=True,
            )
            n_skip_missing += 1
            continue

        try:
            import anndata as ad

            existing = ad.read_h5ad(generated, backed="r")
            try:
                if label_col in existing.obs.columns and existing.obs[label_col].notna().all():
                    print(f"Skip (already has obs[{label_col!r}]): {generated}", flush=True)
                    n_skip_exists += 1
                    continue
            finally:
                try:
                    existing.file.close()
                except Exception:
                    pass

            infer_gex_cell_types_for_middle_file(
                bottom_h5ad=bottom,
                top_h5ad=top,
                generated_middle_h5ad=generated,
                label_col=label_col,
                k=k,
            )
            print(f"Wrote obs[{label_col!r}] into {generated}", flush=True)
            n_ok += 1
        except Exception as e:
            print(f"FAILED {stem}: {e}", flush=True)
            n_fail += 1

    print(
        f"\nDone. ok={n_ok} skipped_exists={n_skip_exists} "
        f"skipped_missing={n_skip_missing} failed={n_fail} tag={tag!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
