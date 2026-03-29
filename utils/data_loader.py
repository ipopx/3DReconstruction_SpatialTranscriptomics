from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import anndata as ad
import numpy as np
import json


def split_data_into_slices(
    adata: ad.AnnData,
    *,
    slice_thickness: int | None = None,
    num_slices: int | None = None,
    z_key: str = "z",
    slice_key: str = "slice_id",
) -> Mapping[int, ad.AnnData]:
    """
    Split data into based on slice_thickness (or num_slices if former not available) along the discrete z-axis and annotate slices.
    """

    if z_key not in adata.obs:
        raise KeyError(f"'{z_key}' not found in adata.obs")
    unique_z = np.unique(adata.obs[z_key].values)
    if slice_thickness is not None and slice_thickness > 0:
        base_slice_size = slice_thickness
        num_slices = int(math.ceil(len(unique_z) / base_slice_size))
        slice_values_list = [
            unique_z[i : i + base_slice_size]
            for i in range(0, len(unique_z), base_slice_size)
        ]
    else:
        if num_slices is None:
            raise ValueError(
                "Provide either slice_thickness (>0) or num_slices (>0). "
                "If both are provided, slice_thickness takes precedence."
            )
        if num_slices <= 0:
            raise ValueError(f"num_slices must be positive, got {num_slices}")
        if len(unique_z) < num_slices:
            raise ValueError(
                f"num_slices ({num_slices}) is greater than number of unique z values ({len(unique_z)})"
            )
        slice_values_list = list(np.array_split(unique_z, num_slices))

    z_to_slice_id: Dict[object, int] = {}
    for i, slice_values in enumerate(slice_values_list, start=1):
        for z in slice_values:
            z_to_slice_id[z] = i

    slice_ids = adata.obs[z_key].map(z_to_slice_id)
    if slice_ids.isna().any():
        missing = adata.obs.loc[slice_ids.isna(), z_key].unique()
        raise ValueError(f"Some '{z_key}' values could not be assigned to slices: {missing}")
    adata.obs[slice_key] = slice_ids.astype(int)

    # Build and return per-slice views
    slices: Dict[int, ad.AnnData] = {}
    for i in range(1, num_slices + 1):
        slices[i] = adata[adata.obs[slice_key] == i].copy()
    return slices


def load_starmap_dataset(
    path: Union[str, Path],
    *,
    slice_thickness: int | None = None,
    num_slices: int | None = None,
    dropout_z_list: List[int] = [94],
    cell_class_key: str = "leiden",
    z_key: str = "z",
) -> Tuple[ad.AnnData, Mapping[int, ad.AnnData]]:
    """
    Load and preprocess a STARmap .h5ad dataset, and split it into slices.

    Steps:
    - Read .h5ad file.
    - Remove large/unused attributes in ``uns``, ``obsm``, and ``obsp``.
    - Create a ``cell_class`` column from ``leiden``.
    - Drop z-layers listed in ``dropout_z_list``.
    - Split along z-axis into ``num_slices`` and return a mapping of slices.

    Returns
    -------
    ad.AnnData
        The filtered full AnnData object with ``slice_id`` annotations.
    Mapping[int, ad.AnnData]
        A mapping from slice index (e.g. ``1``) to the corresponding AnnData.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"h5ad file not found: {path}")

    adata_raw = ad.read_h5ad(path.as_posix())
    adata = adata_raw.copy()

    # Drop heavy / unused containers if present
    if hasattr(adata, "uns"):
        adata.uns = {}
    if "X_umap" in adata.obsm:
        del adata.obsm["X_umap"]
    if hasattr(adata, "obsp"):
        adata.obsp = {}
        
    if cell_class_key != "cell_class":
        adata.obs["cell_class"] = adata.obs[cell_class_key].astype(str)

    # Filter out noisy z-layers
    if z_key not in adata.obs:
        raise KeyError(f"{z_key} not found in adata.obs; required for slicing")

    if dropout_z_list:
        adata = adata[~adata.obs[z_key].isin(dropout_z_list)].copy()

    # Split data into slices
    if slice_thickness is not None:
        slices = split_data_into_slices(adata, slice_thickness=slice_thickness, z_key=z_key)
    else:
        slices = split_data_into_slices(adata, num_slices=num_slices, z_key=z_key)

    return adata, slices

def load_merfish_dataset(
    path: Union[str, Path],
    cell_class_key: str = "cell_class",
) -> Tuple[ad.AnnData, Mapping[int, ad.AnnData]]:   
    """
    Load and preprocess a MERFISH .h5ad dataset, and split it into slices.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"h5ad file not found: {path}")
    adata_raw = ad.read_h5ad(path.as_posix())
    adata = adata_raw.copy()
    if cell_class_key != "cell_class":
        adata.obs["cell_class"] = adata.obs[cell_class_key].astype(str)

    # Build a mapping from integer slice_id -> AnnData slice
    adatas_id_list = list[Any](np.unique(adata.obs["slice_id"]))
    slices: Dict[int, ad.AnnData] = {}
    for i, section_id in enumerate(adatas_id_list):
        key = i + 1
        slices[key] = adata[adata.obs["slice_id"] == section_id].copy()

    return adata, slices

def get_test_slices(
    slices: Mapping[int, ad.AnnData],
    slices_to_test: Iterable[Iterable[int]],
) -> List[Tuple[ad.AnnData, ad.AnnData, ad.AnnData]]:
    """
    Given a mapping of slices (e.g. from ``split_data_into_slices``) and an
    explicit list of slice-id triples, return a list of (left, middle, right)
    AnnData triplets.

    Example
    -------
    ``slices_to_test = [[1, 2, 3], [2, 6, 10]]`` will return:
    ``[(slices[1], slices[2], slices[3]), (slices[2], slices[6], slices[10])]``.
    """

    if not slices:
        raise ValueError("slices mapping is empty")

    triplets: List[Tuple[ad.AnnData, ad.AnnData, ad.AnnData]] = []
    for t in slices_to_test:
        ids = list(t)
        if len(ids) != 3:
            raise ValueError(f"Each entry in slices_to_test must have length 3, got {ids}")
        left_id, middle_id, right_id = ids
        missing = [i for i in (left_id, middle_id, right_id) if i not in slices]
        if missing:
            raise KeyError(f"Requested slice ids not found in slices mapping: {missing}")
        triplets.append((slices[left_id], slices[middle_id], slices[right_id]))

    return triplets

def save_slice(
    adata_slice: ad.AnnData,
    name: str,
    *,
    output_dir: Union[str, Path] = "output",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (name if name.endswith(".h5ad") else f"{name}.h5ad")
    adata_slice.write_h5ad(out_path.as_posix())
    return out_path


def load_slice(
    name: str,
    *,
    output_dir: Union[str, Path] = "output",
) -> ad.AnnData:
    out_dir = Path(output_dir)
    out_path = out_dir / (name if name.endswith(".h5ad") else f"{name}.h5ad")
    return ad.read_h5ad(out_path.as_posix())

def save_metrics(
    metrics: dict,
    name: str,
    *,
    output_dir: Union[str, Path] = "output",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (name if name.endswith(".json") else f"{name}.json")
    with open(out_path.as_posix(), "w") as f:
        json.dump(metrics, f)
    return out_path