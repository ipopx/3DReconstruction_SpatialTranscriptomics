"""Convert generic spatial AnnData slices to/from the UOT genespace representation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uot.data import feature_ids_from_adata
from uot.geometry import ccf_zyx_to_xy
from uot.interpolation_genespace import SliceGeneArrays

if TYPE_CHECKING:
    import anndata as ad


def xyz_from_adata(adata: "ad.AnnData", *, z_key: str = "z") -> np.ndarray:
    """
    Build pseudo ``obsm['X_CCF']`` rows as (z, y, x) from STARmap-style inputs.

    Accepts ``obsm['X_CCF']``, ``obsm['spatial']``, or ``obs['x']`` / ``obs['y']``.
    """
    if "X_CCF" in adata.obsm:
        xyz = np.asarray(adata.obsm["X_CCF"], dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError(f"obsm['X_CCF'] must be (n, 3+), got {xyz.shape}")
        return np.ascontiguousarray(xyz[:, :3], dtype=np.float32)

    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if spatial.ndim != 2 or spatial.shape[1] < 2:
            raise ValueError(f"obsm['spatial'] must be (n, 2+), got {spatial.shape}")
        x = spatial[:, 0]
        y = spatial[:, 1]
        if spatial.shape[1] >= 3:
            z = spatial[:, 2]
        elif z_key in adata.obs:
            z = np.asarray(adata.obs[z_key].values, dtype=np.float32)
        else:
            z = np.zeros(adata.n_obs, dtype=np.float32)
    elif "x" in adata.obs and "y" in adata.obs:
        x = np.asarray(adata.obs["x"].values, dtype=np.float32)
        y = np.asarray(adata.obs["y"].values, dtype=np.float32)
        if z_key in adata.obs:
            z = np.asarray(adata.obs[z_key].values, dtype=np.float32)
        else:
            z = np.zeros(adata.n_obs, dtype=np.float32)
    else:
        raise ValueError(
            "No spatial coordinates found (expected obsm['X_CCF'], obsm['spatial'], or obs['x']/obs['y'])"
        )

    return np.stack([z, y, x], axis=1).astype(np.float32)


def _dense_X(adata: "ad.AnnData") -> np.ndarray:
    from scipy import sparse

    x = adata.X
    if x is None:
        raise ValueError("adata.X is None")
    if sparse.issparse(x):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def slice_gene_arrays_from_adata(adata: "ad.AnnData", *, z_key: str = "z") -> SliceGeneArrays:
    xyz = xyz_from_adata(adata, z_key=z_key)
    genes = _dense_X(adata)
    fid = np.asarray(feature_ids_from_adata(adata), dtype=str)
    var_names = np.asarray(adata.var_names.astype(str))
    if genes.shape[0] != xyz.shape[0]:
        raise ValueError(f"n_obs mismatch: xyz={xyz.shape[0]} genes={genes.shape[0]}")
    if genes.shape[1] != var_names.shape[0]:
        raise ValueError(f"X cols {genes.shape[1]} != n_vars {var_names.shape[0]}")
    return SliceGeneArrays(
        xyz=xyz.copy(),
        genes=genes.copy(),
        feature_id=fid.copy(),
        var_names=var_names.copy(),
    )


def ann_data_from_interpolation(
    xyz: np.ndarray,
    genes: np.ndarray,
    var_names: np.ndarray,
    feature_ids: np.ndarray,
    *,
    cell_type_key: str | None = None,
    cell_types: np.ndarray | None = None,
) -> "ad.AnnData":
    """Build an AnnData object compatible with the existing STARmap metrics pipeline."""
    import anndata as ad
    import pandas as pd

    xy = ccf_zyx_to_xy(xyz)
    out = ad.AnnData(
        X=np.asarray(genes, dtype=np.float32),
        obs=pd.DataFrame({"feature_id": np.asarray(feature_ids, dtype=str)}),
        var=pd.DataFrame(index=np.asarray(var_names, dtype=str)),
    )
    out.obsm["spatial"] = np.asarray(xy, dtype=np.float32)
    out.obsm["X_CCF"] = np.asarray(xyz, dtype=np.float32)
    out.obs["x"] = xy[:, 0]
    out.obs["y"] = xy[:, 1]
    if cell_type_key and cell_types is not None:
        out.obs[cell_type_key] = pd.Series(np.asarray(cell_types, dtype=str), index=out.obs_names)
    return out


def ensure_spatial_obsm(adata: "ad.AnnData") -> "ad.AnnData":
    """Add obsm['spatial'] and obs x/y when only X_CCF is present."""
    if "spatial" in adata.obsm:
        return adata
    if "X_CCF" in adata.obsm:
        xy = ccf_zyx_to_xy(np.asarray(adata.obsm["X_CCF"], dtype=np.float32))
        adata.obsm["spatial"] = xy
        adata.obs["x"] = xy[:, 0]
        adata.obs["y"] = xy[:, 1]
    return adata
