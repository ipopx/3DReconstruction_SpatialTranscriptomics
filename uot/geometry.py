"""CCF coordinate transforms and xy min-max normalization."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def xy_from_X_CCF(x_ccf: np.ndarray) -> np.ndarray:
    """
    Physical (x, y) from obsm['X_CCF'] stored as (z, y, x).

    This is the read path paired with ``xy_to_ccf_zyx`` / ``physical_xy_to_ccf`` in
    ``transformer.inference`` (and training slice I/O).
    """
    return ccf_zyx_to_xy(x_ccf)


def ccf_zyx_to_xy(zyx: np.ndarray) -> np.ndarray:
    """
    Map obsm['X_CCF'] rows (z, y, x) to an (n, 2) array [physical_x, physical_y]
    for the horizontal / vertical slice plane.
    """
    zyx = np.asarray(zyx, dtype=np.float32)
    if zyx.ndim != 2 or zyx.shape[1] < 3:
        raise ValueError(f"Expected X_CCF shape (n, 3) with columns (z, y, x), got {zyx.shape}")
    x = zyx[:, 2]
    y = zyx[:, 1]
    return np.stack([x, y], axis=1).astype(np.float32)


def xy_to_ccf_zyx(z_mid: float | np.ndarray, xy: np.ndarray) -> np.ndarray:
    """xy is (n, 2) as [x, y]; output (n, 3) as (z, y, x) for obsm['X_CCF']."""
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected xy shape (n, 2) as [x, y], got {xy.shape}")
    z = np.asarray(z_mid, dtype=np.float32)
    if z.ndim == 0:
        z = np.full((xy.shape[0],), float(z), dtype=np.float32)
    elif z.shape[0] != xy.shape[0]:
        raise ValueError("z_mid as array must have same length as xy")
    return np.stack([z, xy[:, 1], xy[:, 0]], axis=1).astype(np.float32)


def union_xy_min_max(bottom_xy: np.ndarray, top_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Global per-axis min/max for xy over bottom ∪ top (matches inference)."""
    u = np.vstack([bottom_xy, top_xy]).astype(np.float32)
    xy_min = u.min(axis=0).astype(np.float32)
    xy_max = u.max(axis=0).astype(np.float32)
    span = np.maximum(xy_max - xy_min, 1e-8).astype(np.float32)
    xy_max = (xy_min + span).astype(np.float32)
    return xy_min, xy_max


def norm_xy_min_max(xy: np.ndarray, xy_min: np.ndarray, xy_max: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1] per axis; xy_min/xy_max are (2,) broadcastable."""
    xy = np.asarray(xy, dtype=np.float32)
    lo = np.asarray(xy_min, dtype=np.float32)
    hi = np.asarray(xy_max, dtype=np.float32)
    span = np.maximum(hi - lo, 1e-8).astype(np.float32)
    return ((xy - lo) / span).astype(np.float32)


def denorm_xy(
    xy_normalized: torch.Tensor,
    xy_min: torch.Tensor,
    xy_max: torch.Tensor,
) -> torch.Tensor:
    """Inverse min-max: map [0, 1] normalized xy back to physical coordinates."""
    span = (xy_max - xy_min).clamp_min(1e-8)
    return xy_normalized * span + xy_min


def union_cae_min_max(bottom_cae: np.ndarray, top_cae: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dimension min/max for CAE over bottom ∪ top (slice-level, like xy bounds)."""
    u = np.vstack([bottom_cae, top_cae]).astype(np.float32)
    cae_min = u.min(axis=0).astype(np.float32)
    cae_max = u.max(axis=0).astype(np.float32)
    span = np.maximum(cae_max - cae_min, 1e-8).astype(np.float32)
    cae_max = (cae_min + span).astype(np.float32)
    return cae_min, cae_max


def norm_cae_min_max(cae: np.ndarray, cae_min: np.ndarray, cae_max: np.ndarray) -> np.ndarray:
    """Min-max each CAE dimension to [0, 1] using slice-level bounds."""
    cae = np.asarray(cae, dtype=np.float32)
    lo = np.asarray(cae_min, dtype=np.float32)
    hi = np.asarray(cae_max, dtype=np.float32)
    span = np.maximum(hi - lo, 1e-8).astype(np.float32)
    return ((cae - lo) / span).astype(np.float32)


def denorm_cae_min_max(cae_normalized: np.ndarray, cae_min: np.ndarray, cae_max: np.ndarray) -> np.ndarray:
    """Inverse per-dimension min-max back to latent space."""
    zn = np.asarray(cae_normalized, dtype=np.float32)
    lo = np.asarray(cae_min, dtype=np.float32).reshape(1, -1)
    hi = np.asarray(cae_max, dtype=np.float32).reshape(1, -1)
    span = np.maximum(hi - lo, 1e-8)
    return (zn * span + lo).astype(np.float32)


def infer_z_mid(b_xyz: np.ndarray, t_xyz: np.ndarray) -> float:
    zb = float(np.asarray(b_xyz[:, 0], dtype=np.float32).mean()) if b_xyz.shape[1] >= 3 else 0.0
    zt = float(np.asarray(t_xyz[:, 0], dtype=np.float32).mean()) if t_xyz.shape[1] >= 3 else 0.0
    return 0.5 * (zb + zt)
