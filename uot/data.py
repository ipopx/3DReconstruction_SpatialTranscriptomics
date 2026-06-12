"""H5AD I/O, triplet paths, and OT alignment cache columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

OT_PATCH_ID_COL = "OT_patch_id"
OT_SLOT_COL = "OT_slot"
OT_PICKED_PARTNER_COL = "OT_picked_partner"
OT_OTHER_PARTNER_COL = "OT_other_partner"


@dataclass
class SliceArrays:
    xyz: np.ndarray
    cae: np.ndarray
    feature_id: np.ndarray


def parse_triplets_file(path: str) -> List[Tuple[str, str, str]]:
    lines = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Each line needs 3 paths, got {line!r}")
            lines.append((parts[0], parts[1], parts[2]))
    return lines


def feature_ids_from_adata(adata, path: str = "") -> np.ndarray:
    if "feature_id" in adata.obs.columns:
        return np.asarray(adata.obs["feature_id"].astype(str))
    if getattr(adata.obs.index, "name", None) == "feature_id":
        return np.asarray(adata.obs.index.astype(str))
    names = np.asarray(adata.obs_names.astype(str))
    if names.shape[0] != adata.n_obs:
        raise ValueError(f"{path}: obs_names length mismatch")
    return names


def read_slice_from_h5ad(path: str) -> SliceArrays:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        xyz = np.asarray(adata.obsm["X_CCF"], dtype=np.float32)
        cae = np.asarray(adata.obsm["X_LLOKI_CAE"], dtype=np.float32)
        fid = feature_ids_from_adata(adata, path=path)
        if fid.shape[0] != xyz.shape[0]:
            raise ValueError(f"{path}: feature_id length {fid.shape[0]} != n_obs {xyz.shape[0]}")
        return SliceArrays(xyz=xyz.copy(), cae=cae.copy(), feature_id=fid.copy())
    finally:
        try:
            adata.file.close()
        except Exception:
            pass


def _close_backed_h5ad(adata) -> None:
    try:
        if getattr(adata, "isbacked", False) and getattr(adata, "file", None) is not None:
            adata.file.close()
    except Exception:
        pass


def middle_has_ot_alignment_cache(path: str) -> bool:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        cols = set(adata.obs.columns)
        return OT_SLOT_COL in cols and OT_PICKED_PARTNER_COL in cols
    finally:
        _close_backed_h5ad(adata)


def read_middle_ot_alignment_cache(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the four OT obs columns from a middle h5ad without loading X / full AnnData."""
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        cols = (OT_PATCH_ID_COL, OT_SLOT_COL, OT_PICKED_PARTNER_COL, OT_OTHER_PARTNER_COL)
        obs_cols = set(adata.obs.columns)
        missing = [c for c in cols if c not in obs_cols]
        if missing:
            raise KeyError(f"{path}: missing OT cache columns {missing}")

        n = adata.n_obs
        patch_id = np.asarray(adata.obs[OT_PATCH_ID_COL].values, dtype=np.int32)
        patch_slot = np.asarray(adata.obs[OT_SLOT_COL].values, dtype=np.int32)
        picked_partner = np.asarray(adata.obs[OT_PICKED_PARTNER_COL].values, dtype=str)
        other_partner = np.asarray(adata.obs[OT_OTHER_PARTNER_COL].values, dtype=str)
        if len(patch_id) != n:
            raise ValueError(f"{path}: cached patch_id length {len(patch_id)} != n_obs {n}")
        return patch_id, patch_slot, picked_partner, other_partner
    finally:
        _close_backed_h5ad(adata)


def write_middle_ot_alignment_cache(
    path: str,
    *,
    patch_id: np.ndarray,
    patch_slot: np.ndarray,
    ot_picked_partner: np.ndarray,
    ot_other_partner: np.ndarray,
) -> None:
    """Update OT alignment obs columns on middle h5ad in backed mode (no full-matrix load)."""
    import anndata as ad
    import pandas as pd

    adata = ad.read_h5ad(path, backed="r+")
    try:
        n = adata.n_obs
        if len(patch_id) != n:
            raise ValueError(f"patch_id length {len(patch_id)} != n_obs {n}")

        adata.obs[OT_PATCH_ID_COL] = patch_id.astype(np.int32)
        adata.obs[OT_SLOT_COL] = patch_slot.astype(np.int32)
        adata.obs[OT_PICKED_PARTNER_COL] = pd.array(
            np.asarray(ot_picked_partner, dtype=str), dtype="string"
        )
        adata.obs[OT_OTHER_PARTNER_COL] = pd.array(
            np.asarray(ot_other_partner, dtype=str), dtype="string"
        )
        if getattr(adata, "file", None) is not None:
            adata.file.flush()
    finally:
        _close_backed_h5ad(adata)
