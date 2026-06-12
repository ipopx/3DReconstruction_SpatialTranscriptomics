"""Patch-level triplet preprocessing, alignment cache, and token batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np
import torch

from transformer_flow_matching.constants import CAE_DIM, DEFAULT_OT_LAMBDA_XY
from transformer_flow_matching.data import (
    OT_OTHER_PARTNER_COL,
    OT_PATCH_ID_COL,
    OT_PICKED_PARTNER_COL,
    OT_SLOT_COL,
    SliceArrays,
    middle_has_ot_alignment_cache,
    read_middle_ot_alignment_cache,
    read_slice_from_h5ad,
    write_middle_ot_alignment_cache,
)
from transformer_flow_matching.geometry import (
    ccf_zyx_to_xy,
    denorm_cae_min_max,
    norm_cae_min_max,
    norm_xy_min_max,
    union_cae_min_max,
    union_xy_min_max,
    xy_to_ccf_zyx,
)
from transformer_flow_matching.ot_matching import (
    feature_id_to_index,
    ot_match_middle_to_picked,
    ot_match_picked_slots_to_other,
)
from transformer_flow_matching.patching import assign_patch_ids, subsample_first_n, target_n_mid_per_patch


PickedSide = Literal["bottom", "top"]


@dataclass
class TripletSlices:
    bottom: SliceArrays
    top: SliceArrays
    middle: SliceArrays
    b_xy: np.ndarray = field(init=False)
    t_xy: np.ndarray = field(init=False)
    m_xy: np.ndarray = field(init=False)
    xy_min: np.ndarray = field(init=False)
    xy_max: np.ndarray = field(init=False)
    cae_min: np.ndarray = field(init=False)
    cae_max: np.ndarray = field(init=False)
    picked_side: PickedSide = field(init=False)
    b_patch_ids: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=np.int32))
    t_patch_ids: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=np.int32))
    m_patch_ids: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=np.int32))
    p_patch_ids: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=np.int32))
    o_patch_ids: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=np.int32))

    def __post_init__(self) -> None:
        self.b_xy = ccf_zyx_to_xy(self.bottom.xyz)
        self.t_xy = ccf_zyx_to_xy(self.top.xyz)
        self.m_xy = ccf_zyx_to_xy(self.middle.xyz)
        self.xy_min, self.xy_max = union_xy_min_max(self.b_xy, self.t_xy)
        self.cae_min, self.cae_max = union_cae_min_max(self.bottom.cae, self.top.cae)
        if self.bottom.xyz.shape[0] > self.top.xyz.shape[0]:
            self.picked_side = "bottom"
        elif self.top.xyz.shape[0] > self.bottom.xyz.shape[0]:
            self.picked_side = "top"
        else:
            self.picked_side = "bottom"  # tie-break

    @property
    def picked(self) -> SliceArrays:
        return self.bottom if self.picked_side == "bottom" else self.top

    @property
    def other(self) -> SliceArrays:
        return self.top if self.picked_side == "bottom" else self.bottom

    @property
    def picked_xy(self) -> np.ndarray:
        return self.b_xy if self.picked_side == "bottom" else self.t_xy

    @property
    def other_xy(self) -> np.ndarray:
        return self.t_xy if self.picked_side == "bottom" else self.b_xy


@dataclass
class PatchTrainingBatch:
    ctx_tokens: torch.Tensor
    mid_tokens: torch.Tensor
    xy_init_norm: np.ndarray
    cae_init_norm: np.ndarray
    patch_id: int
    n_work: int


def _norm_slice(xy: np.ndarray, xy_min: np.ndarray, xy_max: np.ndarray) -> np.ndarray:
    return norm_xy_min_max(xy, xy_min, xy_max)


def _norm_cae_slice(cae: np.ndarray, cae_min: np.ndarray, cae_max: np.ndarray) -> np.ndarray:
    return norm_cae_min_max(cae, cae_min, cae_max)


def _tokens_from_norm_xy(zn: np.ndarray, cae: np.ndarray, device: torch.device) -> torch.Tensor:
    cae = np.asarray(cae, dtype=np.float32)
    if cae.shape[1] != CAE_DIM:
        raise ValueError(f"Expected CAE dim {CAE_DIM}, got {cae.shape[1]}")
    h = np.concatenate([np.asarray(zn, dtype=np.float32), cae], axis=1)
    return torch.from_numpy(h).to(device)


def _ctx_tokens_picked_other(
    *,
    picked_norm: np.ndarray,
    picked_cae_norm: np.ndarray,
    other_norm: np.ndarray | None,
    other_cae_norm: np.ndarray | None,
    device: torch.device,
) -> torch.Tensor:
    """Context = [picked] or [picked, other] when the other slice has cells in this patch."""
    picked_t = _tokens_from_norm_xy(picked_norm, picked_cae_norm, device)
    if other_norm is None or other_cae_norm is None:
        return picked_t
    return torch.cat(
        [picked_t, _tokens_from_norm_xy(other_norm, other_cae_norm, device)],
        dim=0,
    )


def _indices_in_patch(n_cells: int, patch_ids: np.ndarray, patch_id: int) -> np.ndarray:
    return np.where(patch_ids[:n_cells] == patch_id)[0].astype(np.int64)


def _patch_bottom_top_counts(triplet: TripletSlices, patch_id: int) -> tuple[int, int]:
    n_b = int(_indices_in_patch(triplet.bottom.xyz.shape[0], triplet.b_patch_ids, patch_id).size)
    n_t = int(_indices_in_patch(triplet.top.xyz.shape[0], triplet.t_patch_ids, patch_id).size)
    return n_b, n_t


def _patch_picked_side(n_bottom: int, n_top: int) -> PickedSide:
    """Per-patch picked side (tie -> bottom), same rule as inference."""
    if n_bottom >= n_top:
        return "bottom"
    return "top"


def compute_patch_alignment(
    triplet: TripletSlices,
    patch_id: int,
    grid_n: int,
    *,
    lambda_xy: float = DEFAULT_OT_LAMBDA_XY,
) -> tuple[int, PickedSide, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partial OT middle↔picked (drop unmatched), then OT picked slots↔other.

    Returns (n_work, picked_side, mid_idx, picked_aligned, other_per_slot).
    ``other_per_slot[i]`` is global other index or -1.
    """
    _ = grid_n
    m_all = _indices_in_patch(triplet.middle.xyz.shape[0], triplet.m_patch_ids, patch_id)
    b_all = _indices_in_patch(triplet.bottom.xyz.shape[0], triplet.b_patch_ids, patch_id)
    t_all = _indices_in_patch(triplet.top.xyz.shape[0], triplet.t_patch_ids, patch_id)
    n_b, n_t = b_all.size, t_all.size

    empty = np.zeros(0, dtype=np.int64)
    if m_all.size == 0:
        return 0, "bottom", empty, empty, empty

    picked_side = _patch_picked_side(n_b, n_t)
    if picked_side == "bottom":
        p_all, o_all = b_all, t_all
        picked_xy, other_xy = triplet.b_xy, triplet.t_xy
        picked_cae, other_cae = triplet.bottom.cae, triplet.top.cae
    else:
        p_all, o_all = t_all, b_all
        picked_xy, other_xy = triplet.t_xy, triplet.b_xy
        picked_cae, other_cae = triplet.top.cae, triplet.bottom.cae

    if p_all.size == 0:
        return 0, picked_side, empty, empty, empty

    mid_xy_n = _norm_slice(triplet.m_xy[m_all], triplet.xy_min, triplet.xy_max)
    mid_cae_n = _norm_cae_slice(triplet.middle.cae[m_all], triplet.cae_min, triplet.cae_max)
    picked_xy_n = _norm_slice(picked_xy[p_all], triplet.xy_min, triplet.xy_max)
    picked_cae_n = _norm_cae_slice(picked_cae[p_all], triplet.cae_min, triplet.cae_max)

    row_ind, col_ind = ot_match_middle_to_picked(
        mid_xy_n, mid_cae_n, picked_xy_n, picked_cae_n, lambda_xy=lambda_xy
    )
    n_work = int(row_ind.size)
    if n_work == 0:
        return 0, picked_side, empty, empty, empty

    mid_idx = m_all[row_ind]
    picked_aligned = p_all[col_ind]
    picked_norm_slots = picked_xy_n[col_ind]
    picked_cae_slots = picked_cae_n[col_ind]

    other_per_slot = np.full(n_work, -1, dtype=np.int64)
    if o_all.size > 0:
        other_xy_n = _norm_slice(other_xy[o_all], triplet.xy_min, triplet.xy_max)
        other_cae_n = _norm_cae_slice(other_cae[o_all], triplet.cae_min, triplet.cae_max)
        local_other = ot_match_picked_slots_to_other(
            picked_norm_slots,
            picked_cae_slots,
            other_xy_n,
            other_cae_n,
            lambda_xy=lambda_xy,
        )
        for i in range(n_work):
            if local_other[i] >= 0:
                other_per_slot[i] = o_all[local_other[i]]

    return n_work, picked_side, mid_idx, picked_aligned, other_per_slot


def build_patch_training_batch(
    triplet: TripletSlices,
    patch_id: int,
    device: torch.device,
    *,
    picked_is_bottom: bool,
    mid_idx: np.ndarray,
    picked_aligned: np.ndarray,
    other_aligned: np.ndarray,
) -> PatchTrainingBatch | None:
    n_work = int(mid_idx.shape[0])
    if n_work == 0:
        return None

    if picked_is_bottom:
        picked_xy, other_xy = triplet.b_xy, triplet.t_xy
        picked_cae, other_cae = triplet.bottom.cae, triplet.top.cae
    else:
        picked_xy, other_xy = triplet.t_xy, triplet.b_xy
        picked_cae, other_cae = triplet.top.cae, triplet.bottom.cae

    mid_norm = _norm_slice(triplet.m_xy[mid_idx], triplet.xy_min, triplet.xy_max)
    picked_norm = _norm_slice(picked_xy[picked_aligned], triplet.xy_min, triplet.xy_max)
    picked_cae_norm = _norm_cae_slice(picked_cae[picked_aligned], triplet.cae_min, triplet.cae_max)
    mid_cae_norm = _norm_cae_slice(triplet.middle.cae[mid_idx], triplet.cae_min, triplet.cae_max)

    other_per_slot = np.asarray(other_aligned, dtype=np.int64)
    other_norm = other_cae_norm = None
    if other_per_slot.size > 0:
        mask = other_per_slot >= 0
        if np.any(mask):
            o_idx = other_per_slot[mask]
            other_norm = _norm_slice(other_xy[o_idx], triplet.xy_min, triplet.xy_max)
            other_cae_norm = _norm_cae_slice(other_cae[o_idx], triplet.cae_min, triplet.cae_max)

    xy_residual = (mid_norm - picked_norm).astype(np.float32)
    cae_residual = (mid_cae_norm - picked_cae_norm).astype(np.float32)
    xy_init_norm = picked_norm.astype(np.float32)
    cae_init_norm = picked_cae_norm.astype(np.float32)

    ctx_tokens = _ctx_tokens_picked_other(
        picked_norm=picked_norm,
        picked_cae_norm=picked_cae_norm,
        other_norm=other_norm,
        other_cae_norm=other_cae_norm,
        device=device,
    )
    mid_tokens = _tokens_from_norm_xy(xy_residual, cae_residual, device)

    return PatchTrainingBatch(
        ctx_tokens=ctx_tokens,
        mid_tokens=mid_tokens,
        xy_init_norm=xy_init_norm,
        cae_init_norm=cae_init_norm,
        patch_id=int(patch_id),
        n_work=n_work,
    )


def init_triplet_patch_ids(triplet: TripletSlices, grid_n: int) -> None:
    triplet.b_patch_ids = assign_patch_ids(triplet.b_xy, triplet.xy_min, triplet.xy_max, grid_n)
    triplet.t_patch_ids = assign_patch_ids(triplet.t_xy, triplet.xy_min, triplet.xy_max, grid_n)
    triplet.m_patch_ids = assign_patch_ids(triplet.m_xy, triplet.xy_min, triplet.xy_max, grid_n)
    # Per-patch picked/other assignment is done in compute_patch_alignment / inference (not global).
    triplet.p_patch_ids = triplet.b_patch_ids
    triplet.o_patch_ids = triplet.t_patch_ids


def load_triplet_slices(
    bottom_path: str,
    top_path: str,
    middle_path: str,
    *,
    grid_n: int,
) -> TripletSlices:
    triplet = TripletSlices(
        bottom=read_slice_from_h5ad(bottom_path),
        top=read_slice_from_h5ad(top_path),
        middle=read_slice_from_h5ad(middle_path),
    )
    init_triplet_patch_ids(triplet, grid_n)
    return triplet


def load_cached_alignment_from_middle(
    middle_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return read_middle_ot_alignment_cache(middle_path)


def _empty_slice_arrays() -> SliceArrays:
    return SliceArrays(
        xyz=np.zeros((0, 3), dtype=np.float32),
        cae=np.zeros((0, CAE_DIM), dtype=np.float32),
        feature_id=np.zeros(0, dtype=str),
    )


def release_triplet_side_slices(triplet: TripletSlices) -> None:
    """Drop bottom/top from RAM after patch training; keep middle and m_* arrays."""
    triplet.bottom = _empty_slice_arrays()
    triplet.top = _empty_slice_arrays()
    triplet.b_xy = np.zeros((0, 2), dtype=np.float32)
    triplet.t_xy = np.zeros((0, 2), dtype=np.float32)
    triplet.b_patch_ids = np.zeros(0, dtype=np.int32)
    triplet.t_patch_ids = np.zeros(0, dtype=np.int32)
    triplet.p_patch_ids = np.zeros(0, dtype=np.int32)
    triplet.o_patch_ids = np.zeros(0, dtype=np.int32)


def rebuild_patch_from_cache(
    triplet: TripletSlices,
    patch_id: int,
    device: torch.device,
    *,
    patch_slot: np.ndarray,
    picked_partner: np.ndarray,
    other_partner: np.ndarray,
) -> PatchTrainingBatch | None:
    """Rebuild batch using cached partners on middle (epoch >= 2)."""
    sel = np.where(
        (triplet.m_patch_ids == patch_id)
        & (patch_slot >= 0)
        & (picked_partner != "")
    )[0]
    if sel.size == 0:
        return None

    order = np.argsort(patch_slot[sel])
    mid_idx = sel[order]
    n_work = mid_idx.shape[0]

    n_b, n_t = _patch_bottom_top_counts(triplet, patch_id)
    picked_is_bottom = _patch_picked_side(n_b, n_t) == "bottom"
    if picked_is_bottom:
        picked_fid_index = feature_id_to_index(triplet.bottom.feature_id)
        other_fid_index = feature_id_to_index(triplet.top.feature_id)
    else:
        picked_fid_index = feature_id_to_index(triplet.top.feature_id)
        other_fid_index = feature_id_to_index(triplet.bottom.feature_id)

    picked_aligned = np.empty(n_work, dtype=np.int64)
    other_aligned = np.full(n_work, -1, dtype=np.int64)
    for i, mi in enumerate(mid_idx):
        pf = picked_partner[mi]
        of = str(other_partner[mi])
        if pf not in picked_fid_index:
            return None
        picked_aligned[i] = picked_fid_index[pf]
        if of:
            if of not in other_fid_index:
                return None
            other_aligned[i] = other_fid_index[of]

    return build_patch_training_batch(
        triplet,
        patch_id,
        device,
        picked_is_bottom=picked_is_bottom,
        mid_idx=mid_idx,
        picked_aligned=picked_aligned,
        other_aligned=other_aligned,
    )


def iter_patch_training_batches(
    triplet: TripletSlices,
    device: torch.device,
    *,
    grid_n: int,
    use_cache: bool,
    lambda_xy: float = DEFAULT_OT_LAMBDA_XY,
    patch_slot: np.ndarray | None = None,
    picked_partner: np.ndarray | None = None,
    other_partner: np.ndarray | None = None,
) -> Iterator[PatchTrainingBatch]:
    """Yield one training batch per non-empty patch."""
    for patch_id in range(grid_n * grid_n):
        if use_cache and patch_slot is not None:
            batch = rebuild_patch_from_cache(
                triplet,
                patch_id,
                device,
                patch_slot=patch_slot,
                picked_partner=picked_partner,
                other_partner=other_partner,
            )
        else:
            n_work, picked_side, mid_idx, picked_aligned, other_aligned = compute_patch_alignment(
                triplet, patch_id, grid_n, lambda_xy=lambda_xy
            )
            if n_work == 0:
                continue
            batch = build_patch_training_batch(
                triplet,
                patch_id,
                device,
                picked_is_bottom=(picked_side == "bottom"),
                mid_idx=mid_idx,
                picked_aligned=picked_aligned,
                other_aligned=other_aligned,
            )
        if batch is not None:
            yield batch


def build_and_cache_triplet_alignments(
    triplet: TripletSlices,
    middle_path: str,
    *,
    grid_n: int,
    lambda_xy: float = DEFAULT_OT_LAMBDA_XY,
) -> None:
    """Epoch 1: OT alignments -> OT_* obs columns on middle h5ad."""
    n_m = triplet.middle.xyz.shape[0]
    patch_id = triplet.m_patch_ids.copy()
    patch_slot = np.full(n_m, -1, dtype=np.int32)
    picked_partner = np.array([""] * n_m, dtype=object)
    other_partner = np.array([""] * n_m, dtype=object)

    for pid in range(grid_n * grid_n):
        n_work, picked_side, mid_idx, picked_aligned, other_per_slot = compute_patch_alignment(
            triplet, pid, grid_n, lambda_xy=lambda_xy
        )
        if n_work == 0:
            continue
        if picked_side == "bottom":
            picked_fids = triplet.bottom.feature_id
            other_fids = triplet.top.feature_id
        else:
            picked_fids = triplet.top.feature_id
            other_fids = triplet.bottom.feature_id
        for slot, mi, pi in zip(range(n_work), mid_idx, picked_aligned):
            patch_slot[mi] = int(slot)
            picked_partner[mi] = str(picked_fids[pi])
            oi = int(other_per_slot[slot])
            other_partner[mi] = str(other_fids[oi]) if oi >= 0 else ""

    release_triplet_side_slices(triplet)
    write_middle_ot_alignment_cache(
        middle_path,
        patch_id=patch_id,
        patch_slot=patch_slot,
        ot_picked_partner=picked_partner,
        ot_other_partner=other_partner,
    )


# --- Inference ----------------------------------------------------------------


@dataclass
class PatchInferenceBatch:
    ctx_tokens: torch.Tensor
    xy_init_norm: np.ndarray
    cae_init_norm: np.ndarray
    patch_id: int
    n_work: int
    picked_is_bottom: bool


def compute_patch_inference_alignment(
    triplet: TripletSlices,
    patch_id: int,
    *,
    lambda_xy: float = DEFAULT_OT_LAMBDA_XY,
) -> tuple[int, PickedSide, np.ndarray, np.ndarray] | None:
    """
    OT picked-slot ↔ other; generate ``n_infer = target_n_mid`` middle cells.

    Returns (n_infer, picked_side, picked_idx_for_slots, other_per_slot).
  other_per_slot[i] is global other index or -1 (length = n_slots = min(n_infer, |p_all|)).
    """
    b_all = _indices_in_patch(triplet.bottom.xyz.shape[0], triplet.b_patch_ids, patch_id)
    t_all = _indices_in_patch(triplet.top.xyz.shape[0], triplet.t_patch_ids, patch_id)
    n_b, n_t = b_all.size, t_all.size
    n_infer = target_n_mid_per_patch(n_b, n_t)
    if n_infer == 0:
        return None

    picked_side = _patch_picked_side(n_b, n_t)
    if picked_side == "bottom":
        p_all, o_all = b_all, t_all
        picked_xy, other_xy = triplet.b_xy, triplet.t_xy
        picked_cae, other_cae = triplet.bottom.cae, triplet.top.cae
    else:
        p_all, o_all = t_all, b_all
        picked_xy, other_xy = triplet.t_xy, triplet.b_xy
        picked_cae, other_cae = triplet.top.cae, triplet.bottom.cae

    if p_all.size == 0:
        return None

    n_slots = int(min(n_infer, p_all.size))
    picked_idx = p_all[subsample_first_n(n_slots)]
    picked_norm = _norm_slice(picked_xy[picked_idx], triplet.xy_min, triplet.xy_max)
    picked_cae_n = _norm_cae_slice(picked_cae[picked_idx], triplet.cae_min, triplet.cae_max)

    other_per_slot = np.full(n_slots, -1, dtype=np.int64)
    if o_all.size > 0:
        other_xy_n = _norm_slice(other_xy[o_all], triplet.xy_min, triplet.xy_max)
        other_cae_n = _norm_cae_slice(other_cae[o_all], triplet.cae_min, triplet.cae_max)
        local_other = ot_match_picked_slots_to_other(
            picked_norm, picked_cae_n, other_xy_n, other_cae_n, lambda_xy=lambda_xy
        )
        for i in range(n_slots):
            if local_other[i] >= 0:
                other_per_slot[i] = o_all[local_other[i]]

    return n_infer, picked_side, picked_idx, other_per_slot


def build_patch_inference_batch(
    triplet: TripletSlices,
    patch_id: int,
    device: torch.device,
    *,
    n_infer: int,
    picked_is_bottom: bool,
    picked_idx: np.ndarray,
    other_per_slot: np.ndarray,
) -> PatchInferenceBatch:
    if picked_is_bottom:
        picked_xy, other_xy = triplet.b_xy, triplet.t_xy
        picked_cae, other_cae = triplet.bottom.cae, triplet.top.cae
    else:
        picked_xy, other_xy = triplet.t_xy, triplet.b_xy
        picked_cae, other_cae = triplet.top.cae, triplet.bottom.cae

    picked_norm_slots = _norm_slice(picked_xy[picked_idx], triplet.xy_min, triplet.xy_max)
    picked_cae_slots = _norm_cae_slice(picked_cae[picked_idx], triplet.cae_min, triplet.cae_max)

    other_norm = other_cae_norm = None
    other_per_slot = np.asarray(other_per_slot, dtype=np.int64)
    if other_per_slot.size > 0:
        mask = other_per_slot >= 0
        if np.any(mask):
            o_idx = other_per_slot[mask]
            other_norm = _norm_slice(other_xy[o_idx], triplet.xy_min, triplet.xy_max)
            other_cae_norm = _norm_cae_slice(other_cae[o_idx], triplet.cae_min, triplet.cae_max)

    ctx_tokens = _ctx_tokens_picked_other(
        picked_norm=picked_norm_slots,
        picked_cae_norm=picked_cae_slots,
        other_norm=other_norm,
        other_cae_norm=other_cae_norm,
        device=device,
    )

    n_slots = int(picked_idx.shape[0])
    n_infer = int(n_infer)
    slot_ix = (np.arange(n_infer, dtype=np.int64) % max(n_slots, 1)).astype(np.int64)
    xy_init_norm = picked_norm_slots[slot_ix].astype(np.float32)
    cae_init_norm = picked_cae_slots[slot_ix].astype(np.float32)

    return PatchInferenceBatch(
        ctx_tokens=ctx_tokens,
        xy_init_norm=xy_init_norm,
        cae_init_norm=cae_init_norm,
        patch_id=int(patch_id),
        n_work=n_infer,
        picked_is_bottom=bool(picked_is_bottom),
    )


def count_patch_inference_cells(triplet: TripletSlices, *, grid_n: int) -> tuple[int, int]:
    """Return (expected_total_n_mid, n_patches) over the grid."""
    total = 0
    n_patches = 0
    for patch_id in range(int(grid_n) * int(grid_n)):
        n_b, n_t = _patch_bottom_top_counts(triplet, patch_id)
        n_mid = target_n_mid_per_patch(n_b, n_t)
        if n_mid <= 0:
            continue
        picked_side = _patch_picked_side(n_b, n_t)
        n_picked = n_b if picked_side == "bottom" else n_t
        if n_picked <= 0:
            continue
        total += int(n_mid)
        n_patches += 1
    return total, n_patches


def iter_patch_inference_batches(
    triplet: TripletSlices,
    device: torch.device,
    *,
    grid_n: int,
    lambda_xy: float = DEFAULT_OT_LAMBDA_XY,
) -> Iterator[PatchInferenceBatch]:
    for patch_id in range(int(grid_n) * int(grid_n)):
        aligned = compute_patch_inference_alignment(triplet, patch_id, lambda_xy=lambda_xy)
        if aligned is None:
            continue
        n_infer, picked_side, picked_idx, other_per_slot = aligned
        yield build_patch_inference_batch(
            triplet,
            patch_id,
            device,
            n_infer=n_infer,
            picked_is_bottom=(picked_side == "bottom"),
            picked_idx=picked_idx,
            other_per_slot=other_per_slot,
        )


def residuals_to_physical_xy(
    mid_res_norm: np.ndarray,
    xy_init_norm: np.ndarray,
    xy_min: np.ndarray,
    xy_max: np.ndarray,
) -> np.ndarray:
    mid_xy_norm = np.asarray(xy_init_norm, dtype=np.float32) + np.asarray(mid_res_norm, dtype=np.float32)
    span = (np.asarray(xy_max, dtype=np.float32) - np.asarray(xy_min, dtype=np.float32)).reshape(1, 2)
    lo = np.asarray(xy_min, dtype=np.float32).reshape(1, 2)
    return (mid_xy_norm * span + lo).astype(np.float32)


def residuals_to_physical_cae(
    mid_cae_res_norm: np.ndarray,
    cae_init_norm: np.ndarray,
    cae_min: np.ndarray,
    cae_max: np.ndarray,
) -> np.ndarray:
    """cae_mid = denorm(norm(cae_picked) + sampled_cae_residual)."""
    mid_cae_norm = np.asarray(cae_init_norm, dtype=np.float32) + np.asarray(mid_cae_res_norm, dtype=np.float32)
    return denorm_cae_min_max(mid_cae_norm, cae_min, cae_max)


def physical_xy_to_ccf(mid_xy: np.ndarray, z_mid: float) -> np.ndarray:
    return xy_to_ccf_zyx(z_mid, mid_xy)


def infer_z_mid(b_xyz: np.ndarray, t_xyz: np.ndarray) -> float:
    zb = float(np.asarray(b_xyz[:, 0], dtype=np.float32).mean()) if b_xyz.shape[1] >= 3 else 0.0
    zt = float(np.asarray(t_xyz[:, 0], dtype=np.float32).mean()) if t_xyz.shape[1] >= 3 else 0.0
    return 0.5 * (zb + zt)
