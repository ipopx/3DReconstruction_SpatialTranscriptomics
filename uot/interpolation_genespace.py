#!/usr/bin/env python3
"""
Batch slice-level unbalanced OT (UOT) between bottom and top, then barycentric linear
interpolation at t (xy from X_CCF z,y,x + gene expression from adata.X).

UOT always uses source = larger slice, target = smaller slice (tie -> bottom is source).
Subsample n_mid = round(t * n_bottom + (1-t) * n_top) source anchors with highest transport
mass; interpolate in physical space (OT cost in normalized [0,1] features).

Reads triplets from a TSV. Writes:
  {animal}_slice_{j}_baryinterpol_l{lambda}_reg{ot_reg}_regm{ot_reg_m}.h5ad
No metrics or plots.

Concurrent runs (e.g. different hyperparameters in parallel) use advisory file locks on
each input/output .h5ad so HDF5 reads/writes do not collide. Disable with
``INTERPOLATION_GENESPACE_NO_FILE_LOCK=1``. Lock files live under
``INTERPOLATION_GENESPACE_LOCK_DIR`` (default: ``$TMPDIR/interpolation_genespace_locks``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

# In-process guards (threads within one job); cross-process uses fcntl on lock files.
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()

from uot.data import feature_ids_from_adata, parse_triplets_file
from uot.geometry import (
    ccf_zyx_to_xy,
    infer_z_mid,
    norm_cae_min_max,
    norm_xy_min_max,
    union_cae_min_max,
    union_xy_min_max,
    xy_to_ccf_zyx,
)
from uot.ot_matching import alignment_cost_matrix


@dataclass
class SliceGeneArrays:
    xyz: np.ndarray
    genes: np.ndarray
    feature_id: np.ndarray
    var_names: np.ndarray


def _file_locks_enabled() -> bool:
    return os.environ.get("INTERPOLATION_GENESPACE_NO_FILE_LOCK", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _lock_dir() -> str:
    base = os.environ.get(
        "INTERPOLATION_GENESPACE_LOCK_DIR",
        os.path.join(os.environ.get("TMPDIR", "/tmp"), "interpolation_genespace_locks"),
    )
    os.makedirs(base, exist_ok=True)
    return base


def _lock_file_path(h5ad_path: str) -> str:
    digest = hashlib.sha256(os.path.abspath(h5ad_path).encode()).hexdigest()[:24]
    name = Path(h5ad_path).name.replace(os.sep, "_")
    return os.path.join(_lock_dir(), f"{digest}_{name}.lock")


def _thread_lock_for(h5ad_path: str) -> threading.RLock:
    key = os.path.abspath(h5ad_path)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def h5ad_file_lock(h5ad_path: str, *, exclusive: bool = False):
    """
    Serialize access to one .h5ad across processes (fcntl) and threads (RLock).

    Readers use shared locks; writers use exclusive locks.
    """
    if not _file_locks_enabled():
        yield
        return

    tlock = _thread_lock_for(h5ad_path)
    tlock.acquire()
    lock_fp = None
    try:
        if _HAVE_FCNTL:
            lock_fp = open(_lock_file_path(h5ad_path), "a+")
            fcntl.flock(
                lock_fp.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
        yield
    finally:
        if lock_fp is not None and _HAVE_FCNTL:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            lock_fp.close()
        tlock.release()


def _to_float32_2d(arr: np.ndarray, *, label: str, path: str) -> np.ndarray:
    """Coerce a 2D array to float32; raise a clear error on ragged/object gene data."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path}: {label} must be 2D, got shape {arr.shape}")

    if arr.dtype == np.float32:
        return np.ascontiguousarray(arr)

    if np.issubdtype(arr.dtype, np.number):
        return np.ascontiguousarray(arr, dtype=np.float32)

    if arr.dtype == object:
        # Per-row numeric vectors (ragged) or nested sequences trigger this path.
        try:
            return np.ascontiguousarray(arr, dtype=np.float32)
        except (ValueError, TypeError) as e:
            row0 = arr[0, 0] if arr.size else None
            raise ValueError(
                f"{path}: {label} has dtype=object and is not a uniform numeric matrix "
                f"(e.g. sample [0,0]={row0!r}). Expected scipy sparse CSR/CSC or dense float."
            ) from e

    return np.ascontiguousarray(arr, dtype=np.float32)


def _dense_X(adata, path: str = "") -> np.ndarray:
    """Materialize in-memory ``adata.X`` as (n_obs, n_vars) float32."""
    from scipy import sparse

    x = adata.X
    if x is None:
        raise ValueError(f"{path}: adata.X is None")

    if sparse.issparse(x):
        return _to_float32_2d(x.toarray(), label="adata.X", path=path)

    return _to_float32_2d(np.asarray(x), label="adata.X", path=path)


def _as_xyz(x_ccf, path: str) -> np.ndarray:
    if hasattr(x_ccf, "toarray"):
        x_ccf = x_ccf.toarray()
    xyz = np.asarray(x_ccf)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"{path}: obsm['X_CCF'] must be (n, 3+), got {xyz.shape}")
    return np.ascontiguousarray(xyz[:, :3], dtype=np.float32)


def _index_as_str(index, path: str, label: str) -> np.ndarray:
    """Safe str conversion for obs/var names (bytes, object, categoricals)."""
    try:
        return np.asarray(index.astype(str))
    except (ValueError, TypeError):
        pass
    try:
        return np.asarray([str(v) for v in index], dtype=str)
    except Exception as e:
        raise ValueError(f"{path}: could not convert {label} to strings") from e


def read_slice_genespace(path: str) -> SliceGeneArrays:
    import gc

    import anndata as ad

    with h5ad_file_lock(path, exclusive=False):
        adata = ad.read_h5ad(path)
        try:
            if "X_CCF" not in adata.obsm:
                raise KeyError(f"{path}: missing obsm['X_CCF']")
            xyz = _as_xyz(adata.obsm["X_CCF"], path)
            genes = _dense_X(adata, path=path)
            fid = _index_as_str(feature_ids_from_adata(adata, path=path), path, "feature_id")
            if fid.shape[0] != xyz.shape[0] or genes.shape[0] != xyz.shape[0]:
                raise ValueError(
                    f"{path}: n_obs mismatch xyz={xyz.shape[0]} genes={genes.shape[0]} "
                    f"feature_id={fid.shape[0]}"
                )
            var_names = _index_as_str(adata.var_names, path, "var_names")
            if genes.shape[1] != var_names.shape[0]:
                raise ValueError(f"{path}: X cols {genes.shape[1]} != n_vars {var_names.shape[0]}")
            return SliceGeneArrays(
                xyz=xyz.copy(),
                genes=genes.copy(),
                feature_id=fid.copy(),
                var_names=var_names.copy(),
            )
        finally:
            del adata
            gc.collect()


def _param_tag(value: float) -> str:
    v = float(value)
    if v == int(v):
        return str(int(v))
    return str(v).replace(".", "p")


def inferred_param_tag(*, lambda_xy: float, ot_reg: float, ot_reg_m: float) -> str:
    return (
        f"l{_param_tag(lambda_xy)}"
        f"_reg{_param_tag(ot_reg)}"
        f"_regm{_param_tag(ot_reg_m)}"
    )


def inferred_out_path(
    middle_h5ad: str,
    *,
    lambda_xy: float,
    ot_reg: float,
    ot_reg_m: float,
) -> str:
    p = Path(middle_h5ad)
    return str(p.parent / f"{p.stem}_baryinterpol_{inferred_param_tag(lambda_xy=lambda_xy, ot_reg=ot_reg, ot_reg_m=ot_reg_m)}.h5ad")


def inferred_path_from_middle(middle_h5ad: str, tag: str) -> str:
    p = Path(middle_h5ad)
    return str(p.parent / f"{p.stem}_{tag}.h5ad")


def output_exists_for_middle(
    middle_h5ad: str,
    *,
    lambda_xy: float,
    ot_reg: float,
    ot_reg_m: float,
) -> str | None:
    """Return output path if tagged baryinterpol file already exists, else None."""
    out_path = inferred_out_path(
        middle_h5ad,
        lambda_xy=lambda_xy,
        ot_reg=ot_reg,
        ot_reg_m=ot_reg_m,
    )
    return out_path if os.path.isfile(out_path) else None


def unbalanced_ot_coupling(
    xy_a: np.ndarray,
    gene_a: np.ndarray,
    xy_b: np.ndarray,
    gene_b: np.ndarray,
    *,
    lambda_xy: float,
    reg: float,
    reg_m: float,
) -> np.ndarray:
    try:
        import ot
    except ImportError as e:
        raise RuntimeError("interpolation_genespace.py requires POT (pip install POT).") from e

    n_a, n_b = int(xy_a.shape[0]), int(xy_b.shape[0])
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float64), 1.0

    n_cost = n_a * n_b
    mem_gb = n_cost * 4 / (1024**3)
    print(
        f"UOT: building cost matrix {n_a} x {n_b} (~{mem_gb:.2f} GiB float32)",
        flush=True,
    )
    if mem_gb > 48.0:
        raise MemoryError(
            f"Slice-level OT cost would be ~{mem_gb:.1f} GiB. Use more RAM or subsample cells."
        )

    cost = alignment_cost_matrix(xy_a, gene_a, xy_b, gene_b, lambda_xy=float(lambda_xy))
    cost = np.asarray(cost, dtype=np.float32)

    cost_scale = float(np.median(cost))
    if cost_scale > 0:
        cost = cost / cost_scale
    
    a = np.ones(n_a, dtype=np.float64) / n_a
    b = np.ones(n_b, dtype=np.float64) / n_b
    print(
        f"UOT: median cost scale={cost_scale:.4g} (Sinkhorn reg={reg} reg_m={reg_m})",
        flush=True,
    )
    pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
        a,
        b,
        cost,
        float(reg),
        float(reg_m),
        reg_type="kl",
        numItermax=2000,
        stopThr=1e-7,
    )
    pi = np.asarray(pi, dtype=np.float64)
    row_mass = pi.sum(axis=1)
    col_mass = pi.sum(axis=0)
    print(
        f"UOT: pi_sum={float(pi.sum()):.6g} "
        f"active_source={int((row_mass > 1e-12).sum())}/{n_a} "
        f"active_target={int((col_mass > 1e-12).sum())}/{n_b}",
        flush=True,
    )
    return pi, cost_scale


def _soft_partner_counts(
    pi_rows: np.ndarray,
    *,
    mass_eps: float,
    min_frac_of_row_max: float = 0.05,
) -> np.ndarray:
    """Per row: number of target cells with coupling >= max(eps, frac * row_max)."""
    pi_rows = np.asarray(pi_rows, dtype=np.float64)
    if pi_rows.size == 0:
        return np.zeros(0, dtype=np.int32)
    row_max = pi_rows.max(axis=1, keepdims=True)
    thresh = np.maximum(float(mass_eps), float(min_frac_of_row_max) * row_max)
    counts = (pi_rows >= thresh).sum(axis=1).astype(np.int32)
    inactive = row_max.ravel() <= float(mass_eps)
    counts[inactive] = 0
    return counts


def _uot_anchor_diagnostics(
    pi: np.ndarray,
    anchor_idx: np.ndarray,
    *,
    mass_eps: float,
    min_frac_of_row_max: float = 0.05,
) -> dict:
    """Summary stats for selected source anchors (soft OT partners per anchor)."""
    if anchor_idx.size == 0:
        return {
            "mean_soft_partners": 0.0,
            "median_soft_partners": 0.0,
            "min_soft_partners": 0,
            "max_soft_partners": 0,
            "mean_row_mass": 0.0,
            "median_row_mass": 0.0,
        }
    pi_a = pi[anchor_idx]
    partner_counts = _soft_partner_counts(
        pi_a,
        mass_eps=mass_eps,
        min_frac_of_row_max=min_frac_of_row_max,
    )
    row_mass = pi_a.sum(axis=1)
    return {
        "mean_soft_partners": float(partner_counts.mean()),
        "median_soft_partners": float(np.median(partner_counts)),
        "min_soft_partners": int(partner_counts.min()),
        "max_soft_partners": int(partner_counts.max()),
        "mean_row_mass": float(row_mass.mean()),
        "median_row_mass": float(np.median(row_mass)),
    }


def _print_slice_progress(
    slice_tag: str,
    *,
    stage: str,
    uot_source: str | None = None,
    cost_scale: float | None = None,
    anchor_diag: dict | None = None,
    n_mid_requested: int | None = None,
    n_mid_written: int | None = None,
) -> None:
    parts = [f"[{slice_tag}] {stage}"]
    if uot_source is not None:
        parts.append(f"uot_source={uot_source}")
    if cost_scale is not None:
        parts.append(f"cost_median_scale={cost_scale:.4g}")
    if n_mid_requested is not None:
        parts.append(f"n_mid_req={n_mid_requested}")
    if n_mid_written is not None:
        parts.append(f"n_mid_out={n_mid_written}")
    if anchor_diag is not None:
        parts.append(
            "mean_soft_partners="
            f"{anchor_diag['mean_soft_partners']:.2f}"
            f" median={anchor_diag['median_soft_partners']:.0f}"
            f" range=[{anchor_diag['min_soft_partners']},{anchor_diag['max_soft_partners']}]"
        )
        parts.append(
            "anchor_row_mass_mean="
            f"{anchor_diag['mean_row_mass']:.4g}"
            f" median={anchor_diag['median_row_mass']:.4g}"
        )
    print("  ".join(parts), flush=True)


def _select_source_anchor_indices(
    pi: np.ndarray,
    n_mid: int,
    *,
    mass_eps: float,
) -> np.ndarray:
    """Top ``n_mid`` source rows with largest transport mass (row sum of pi)."""
    row_mass = pi.sum(axis=1)
    valid = np.where(row_mass > mass_eps)[0]
    if valid.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = valid[np.argsort(row_mass[valid])[::-1]]
    return order[: min(int(n_mid), order.size)]


def _barycentric_partners_physical(
    pi: np.ndarray,
    target_xy: np.ndarray,
    target_genes: np.ndarray,
    source_idx: np.ndarray,
    *,
    mass_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted average of target features in physical space for each source row."""
    pi_sub = pi[source_idx]
    mass = pi_sub.sum(axis=1, keepdims=True)
    if np.any(mass.ravel() <= mass_eps):
        bad = source_idx[mass.ravel() <= mass_eps]
        raise RuntimeError(f"Zero-mass source anchors in barycenter step: {bad[:5]}")
    bary_xy = ((pi_sub @ target_xy) / mass).astype(np.float32)
    bary_genes = ((pi_sub @ target_genes) / mass).astype(np.float32)
    return bary_xy, bary_genes


def _synthetic_feature_ids(n: int, *, prefix: str = "synth") -> np.ndarray:
    return np.asarray([f"{prefix}_{i:08d}" for i in range(int(n))], dtype=str)


def build_interpolated_middle_slice(
    bottom: SliceGeneArrays,
    top: SliceGeneArrays,
    *,
    t_interp: float,
    lambda_xy: float,
    ot_reg: float,
    ot_reg_m: float,
    mass_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if bottom.genes.shape[1] != top.genes.shape[1]:
        raise ValueError(
            f"Gene dimension mismatch: bottom n_vars={bottom.genes.shape[1]} "
            f"top n_vars={top.genes.shape[1]}"
        )

    n_bottom = int(bottom.xyz.shape[0])
    n_top = int(top.xyz.shape[0])
    t_val = float(t_interp)
    n_mid = int(round(t_val * n_bottom + (1.0 - t_val) * n_top))
    n_mid = max(0, n_mid)

    b_xy = ccf_zyx_to_xy(bottom.xyz)
    t_xy = ccf_zyx_to_xy(top.xyz)

    # UOT: source = larger slice (tie -> bottom), target = smaller slice.
    if n_top > n_bottom:
        uot_source = "top"
        src_xy, src_genes = t_xy, top.genes
        tgt_xy, tgt_genes = b_xy, bottom.genes
    else:
        uot_source = "bottom"
        src_xy, src_genes = b_xy, bottom.genes
        tgt_xy, tgt_genes = t_xy, top.genes

    xy_min, xy_max = union_xy_min_max(b_xy, t_xy)
    gene_min, gene_max = union_cae_min_max(bottom.genes, top.genes)

    src_xy_n = norm_xy_min_max(src_xy, xy_min, xy_max)
    tgt_xy_n = norm_xy_min_max(tgt_xy, xy_min, xy_max)
    src_gene_n = norm_cae_min_max(src_genes, gene_min, gene_max)
    tgt_gene_n = norm_cae_min_max(tgt_genes, gene_min, gene_max)

    slice_tag = f"{uot_source}_src{n_top if uot_source == 'top' else n_bottom}"

    print(
        f"UOT: coupling {n_top if uot_source == 'top' else n_bottom} source x "
        f"{n_bottom if uot_source == 'top' else n_top} target",
        flush=True,
    )
    pi, cost_scale = unbalanced_ot_coupling(
        src_xy_n,
        src_gene_n,
        tgt_xy_n,
        tgt_gene_n,
        lambda_xy=lambda_xy,
        reg=ot_reg,
        reg_m=ot_reg_m,
    )

    anchor_idx = _select_source_anchor_indices(pi, n_mid, mass_eps=mass_eps)
    n_matched = int(anchor_idx.size)
    if n_matched == 0:
        raise RuntimeError("No source cells with non-zero UOT transport mass.")

    anchor_diag = _uot_anchor_diagnostics(pi, anchor_idx, mass_eps=mass_eps)
    _print_slice_progress(
        slice_tag,
        stage="anchors selected",
        uot_source=uot_source,
        cost_scale=cost_scale,
        anchor_diag=anchor_diag,
        n_mid_requested=n_mid,
        n_mid_written=n_matched,
    )

    if n_matched < n_mid:
        print(
            f"WARNING: requested n_mid={n_mid} anchors but only {n_matched} source cells "
            f"have transport mass > {mass_eps} (uot_source={uot_source})",
            flush=True,
        )

    bary_xy, bary_genes = _barycentric_partners_physical(
        pi, tgt_xy, tgt_genes, anchor_idx, mass_eps=mass_eps
    )
    src_xy_a = src_xy[anchor_idx]
    src_genes_a = src_genes[anchor_idx]

    if uot_source == "bottom":
        # mid = (1-t) * bottom + t * bary_top
        xy_phys = ((1.0 - t_val) * src_xy_a + t_val * bary_xy).astype(np.float32)
        genes_phys = ((1.0 - t_val) * src_genes_a + t_val * bary_genes).astype(np.float32)
    else:
        # mid = (1-t) * bary_bottom + t * top
        xy_phys = ((1.0 - t_val) * bary_xy + t_val * src_xy_a).astype(np.float32)
        genes_phys = ((1.0 - t_val) * bary_genes + t_val * src_genes_a).astype(np.float32)

    z_mid = infer_z_mid(bottom.xyz, top.xyz)
    xyz = xy_to_ccf_zyx(np.full(xy_phys.shape[0], z_mid, dtype=np.float32), xy_phys)
    fids = _synthetic_feature_ids(xy_phys.shape[0])

    n_genes = bottom.genes.shape[1]
    stats = {
        "n_cells": int(xyz.shape[0]),
        "n_bottom": n_bottom,
        "n_top": n_top,
        "n_mid_requested": int(n_mid),
        "n_mid_written": int(xyz.shape[0]),
        "n_source_matched": n_matched,
        "uot_source": uot_source,
        "t_interp": float(t_interp),
        "lambda_xy": float(lambda_xy),
        "ot_reg": float(ot_reg),
        "ot_reg_m": float(ot_reg_m),
        "cost_median_scale": float(cost_scale),
        "z_mid": float(z_mid),
        "coupling_level": "slice",
        "feature_space": "gene_expression",
        "interpolation_space": "physical",
        "n_vars": int(n_genes),
        **anchor_diag,
    }
    return xyz, genes_phys, fids, stats


def write_middle_h5ad(
    out_path: str,
    xyz: np.ndarray,
    genes: np.ndarray,
    feature_ids: np.ndarray,
    var_names: np.ndarray,
    *,
    bottom_path: str,
    top_path: str,
    middle_path: str,
    stats: dict,
) -> None:
    import anndata as ad
    import pandas as pd

    n = int(xyz.shape[0])
    out = ad.AnnData(
        X=np.asarray(genes, dtype=np.float32),
        obs=pd.DataFrame({"feature_id": np.asarray(feature_ids, dtype=str)}),
        var=pd.DataFrame(index=np.asarray(var_names, dtype=str)),
    )
    out.obsm["X_CCF"] = np.asarray(xyz, dtype=np.float32)
    out.uns["interpolation"] = {
        "method": "slice_uot_barycentric_linear_genespace_subsampled",
        "bottom": os.path.abspath(bottom_path),
        "top": os.path.abspath(top_path),
        "middle_template": os.path.abspath(middle_path),
        **stats,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with h5ad_file_lock(out_path, exclusive=True):
        out.write_h5ad(out_path)


def _parse_animal_slice(stem: str) -> tuple[str, int] | None:
    m = re.match(r"^(a[1-4])_slice_(\d+)$", stem, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower(), int(m.group(2))


def process_triplet(
    bottom_path: str,
    top_path: str,
    middle_path: str,
    *,
    lambda_xy: float,
    t_interp: float,
    ot_reg: float,
    ot_reg_m: float,
    mass_eps: float,
    overwrite: bool,
) -> str | None:
    param_tag = inferred_param_tag(lambda_xy=lambda_xy, ot_reg=ot_reg, ot_reg_m=ot_reg_m)
    out_path = inferred_out_path(
        middle_path,
        lambda_xy=lambda_xy,
        ot_reg=ot_reg,
        ot_reg_m=ot_reg_m,
    )
    if not overwrite and os.path.isfile(out_path):
        print(
            f"Skip (output exists, tag={param_tag!r}): {os.path.abspath(out_path)}",
            flush=True,
        )
        return None

    print(f"Generating tag={param_tag!r} -> {out_path}", flush=True)
    print(f"Loading bottom={bottom_path}", flush=True)
    print(f"Loading top={top_path}", flush=True)
    bottom = read_slice_genespace(bottom_path)
    top = read_slice_genespace(top_path)

    mid_stem = Path(middle_path).stem
    parsed = _parse_animal_slice(mid_stem)
    tag = f"{parsed[0]} slice {parsed[1]}" if parsed else mid_stem

    n_mid = int(round(float(t_interp) * bottom.xyz.shape[0] + (1.0 - float(t_interp)) * top.xyz.shape[0]))
    print(
        f"{tag}: n_bottom={bottom.xyz.shape[0]} n_top={top.xyz.shape[0]} n_mid={n_mid} "
        f"n_vars={bottom.genes.shape[1]} t={t_interp} lambda_xy={lambda_xy} "
        f"reg={ot_reg} reg_m={ot_reg_m}",
        flush=True,
    )

    print(f"{tag}: running UOT + barycentric interpolation...", flush=True)
    mid_xyz, mid_genes, mid_fid, stats = build_interpolated_middle_slice(
        bottom,
        top,
        t_interp=float(t_interp),
        lambda_xy=float(lambda_xy),
        ot_reg=float(ot_reg),
        ot_reg_m=ot_reg_m,
        mass_eps=float(mass_eps),
    )
    _print_slice_progress(
        tag,
        stage="interpolation done",
        uot_source=stats.get("uot_source"),
        cost_scale=stats.get("cost_median_scale"),
        anchor_diag={
            k: stats[k]
            for k in (
                "mean_soft_partners",
                "median_soft_partners",
                "min_soft_partners",
                "max_soft_partners",
                "mean_row_mass",
                "median_row_mass",
            )
            if k in stats
        },
        n_mid_requested=stats.get("n_mid_requested"),
        n_mid_written=stats.get("n_mid_written"),
    )

    var_names = bottom.var_names
    if not np.array_equal(var_names, top.var_names):
        print(
            f"WARNING {tag}: bottom/top var_names differ; using bottom var index",
            flush=True,
        )

    write_middle_h5ad(
        out_path,
        mid_xyz,
        mid_genes,
        mid_fid,
        var_names,
        bottom_path=bottom_path,
        top_path=top_path,
        middle_path=middle_path,
        stats=stats,
    )
    print(f"Wrote {os.path.abspath(out_path)}  n_cells={stats['n_cells']}", flush=True)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch UOT barycentric interpolation (xy + gene expression) from triplets TSV."
    )
    p.add_argument(
        "--triplets",
        type=str,
        default="triplets_genespace.tsv",
        help="TSV with columns: bottom_h5ad top_h5ad middle_h5ad.",
    )
    p.add_argument(
        "--lambda_xy",
        type=float,
        default=100.0,
        help="Weight on xy MSE in OT cost (gene MSE weight is 1).",
    )
    p.add_argument("--t", type=float, default=0.5, help="Interpolation time in [0, 1].")
    p.add_argument("--ot_reg", type=float, default=0.005, help="Entropic regularization.")
    p.add_argument("--ot_reg_m", type=float, default=0.1, help="Unbalanced marginal KL penalty.")
    p.add_argument("--mass_eps", type=float, default=1e-9, help="Min transport mass for a source anchor.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite baryinterpol outputs; default skips if tagged output exists.",
    )
    args = p.parse_args()

    if not 0.0 <= float(args.t) <= 1.0:
        p.error("--t must be in [0, 1]")

    triplets_path = os.path.abspath(args.triplets)
    if not os.path.isfile(triplets_path):
        raise FileNotFoundError(f"Triplets file not found: {triplets_path}")

    triplets = parse_triplets_file(triplets_path)
    if not triplets:
        print(f"No triplets in {triplets_path}")
        return

    param_tag = inferred_param_tag(
        lambda_xy=float(args.lambda_xy),
        ot_reg=float(args.ot_reg),
        ot_reg_m=float(args.ot_reg_m),
    )
    print(
        f"Processing {len(triplets)} triplets from {triplets_path}  output_tag={param_tag!r}",
        flush=True,
    )

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for i, (bottom, top, middle) in enumerate(triplets, start=1):
        print(f"\n--- Triplet {i}/{len(triplets)} ---", flush=True)
        for label, path in (("bottom", bottom), ("top", top), ("middle", middle)):
            if not os.path.isfile(path):
                print(f"FAILED: missing {label} file: {path}", flush=True)
                n_fail += 1
                break
        else:
            existing = None if args.overwrite else output_exists_for_middle(
                middle,
                lambda_xy=float(args.lambda_xy),
                ot_reg=float(args.ot_reg),
                ot_reg_m=float(args.ot_reg_m),
            )
            if existing is not None:
                print(
                    f"Skip (tag={param_tag!r}, no --overwrite): {existing}",
                    flush=True,
                )
                n_skip += 1
                continue
            try:
                out = process_triplet(
                    bottom,
                    top,
                    middle,
                    lambda_xy=float(args.lambda_xy),
                    t_interp=float(args.t),
                    ot_reg=float(args.ot_reg),
                    ot_reg_m=float(args.ot_reg_m),
                    mass_eps=float(args.mass_eps),
                    overwrite=bool(args.overwrite),
                )
                if out is None:
                    n_skip += 1
                else:
                    n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"FAILED {middle}: {e}", flush=True)
                if os.environ.get("INTERPOLATION_GENESPACE_TRACEBACK"):
                    import traceback

                    traceback.print_exc()

    print(
        f"\nDone. wrote={n_ok} skipped={n_skip} failed={n_fail} "
        f"lambda_xy={args.lambda_xy} t={args.t}",
        flush=True,
    )


if __name__ == "__main__":
    main()
