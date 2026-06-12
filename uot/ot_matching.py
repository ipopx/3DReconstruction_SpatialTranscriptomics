"""Partial optimal transport alignments (xy + CAE cost) via rectangular LAP."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def feature_id_to_index(feature_ids: np.ndarray) -> dict[str, int]:
    ids = np.asarray(feature_ids).astype(str)
    return {fid: i for i, fid in enumerate(ids)}


def _squared_euclidean_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    C[i, j] = ||a[i] - b[j]||^2 without materializing (n, m, d).

    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b  -> output shape (n, m) only.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"Expected (n, d) and (m, d) with same d, got {a.shape} and {b.shape}")
    sq_a = np.einsum("nd,nd->n", a, a, dtype=np.float32)
    sq_b = np.einsum("nd,nd->n", b, b, dtype=np.float32)
    cross = a @ b.T
    out = sq_a[:, None] + sq_b[None, :] - 2.0 * cross
    return np.maximum(out, 0.0, dtype=np.float32)


def alignment_cost_matrix(
    xy_a: np.ndarray,
    cae_a: np.ndarray,
    xy_b: np.ndarray,
    cae_b: np.ndarray,
    *,
    lambda_xy: float = 1.0,
) -> np.ndarray:
    """
    C[i, j] = lambda_xy * ||xy_a[i] - xy_b[j]||^2 + ||cae_a[i] - cae_b[j]||^2
    on normalized features. Memory O(n * m), not O(n * m * d).
    """
    xy_d = _squared_euclidean_cost(xy_a, xy_b)
    cae_d = _squared_euclidean_cost(cae_a, cae_b)
    return float(lambda_xy) * xy_d + cae_d


def partial_ot_match(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimum-cost partial matching (rectangular LAP).

    Returns (row_ind, col_ind) with length min(n_rows, n_cols); unmatched rows/cols dropped.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind.astype(np.int64), col_ind.astype(np.int64)


def ot_match_middle_to_picked(
    mid_xy: np.ndarray,
    mid_cae: np.ndarray,
    picked_xy: np.ndarray,
    picked_cae: np.ndarray,
    *,
    lambda_xy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mid_row_indices, picked_col_indices) into the input arrays, length n_work."""
    cost = alignment_cost_matrix(mid_xy, mid_cae, picked_xy, picked_cae, lambda_xy=lambda_xy)
    row_ind, col_ind = partial_ot_match(cost)
    return row_ind, col_ind


def ot_match_picked_slots_to_other(
    picked_xy: np.ndarray,
    picked_cae: np.ndarray,
    other_xy: np.ndarray,
    other_cae: np.ndarray,
    *,
    lambda_xy: float = 1.0,
) -> np.ndarray:
    """
    For each picked slot i in 0..n_picked-1, other index or -1 if unmatched.

    Partial OT: min(n_picked, n_other) pairs; each other cell used at most once.
    """
    n_picked = int(picked_xy.shape[0])
    out = np.full(n_picked, -1, dtype=np.int64)
    if n_picked == 0 or other_xy.shape[0] == 0:
        return out
    cost = alignment_cost_matrix(picked_xy, picked_cae, other_xy, other_cae, lambda_xy=lambda_xy)
    row_ind, col_ind = partial_ot_match(cost)
    out[row_ind] = col_ind
    return out
