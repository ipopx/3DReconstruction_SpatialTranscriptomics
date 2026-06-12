"""High-level UOT generation for STARmap-style slice triplets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, Union

import anndata as ad

from uot.adapter import ann_data_from_interpolation, ensure_spatial_obsm, slice_gene_arrays_from_adata
from uot.infer_cell_types_baryinterpol_knn import infer_cell_types_for_middle
from uot.interpolation_genespace import build_interpolated_middle_slice


def slice_list_str(sl: Sequence[int]) -> str:
    return "[" + ",".join(map(str, sl)) + "]"


def triplet_for_middle_slice(
    slices_to_test: Iterable[Iterable[int]],
    middle_id: int,
) -> Tuple[int, int, int]:
    for triplet in slices_to_test:
        ids = [int(x) for x in triplet]
        if len(ids) != 3:
            raise ValueError(f"Each entry in slices_to_test must have length 3, got {ids}")
        if ids[1] == int(middle_id):
            return ids[0], ids[1], ids[2]
    raise KeyError(f"No triplet in slices_to_test has middle slice id {middle_id}")


def interpolation_alpha(
    left_slice: ad.AnnData,
    middle_slice: ad.AnnData,
    right_slice: ad.AnnData,
    *,
    alpha_key: str = "slice_id",
) -> float:
    mid_val = float(middle_slice.obs[alpha_key].values[0])
    right_val = float(right_slice.obs[alpha_key].values[0])
    left_val = float(left_slice.obs[alpha_key].values[0])
    return mid_val / (right_val + left_val)


def sim_middle_filename(prefix: str, triplet: Sequence[int]) -> str:
    return f"{prefix}_sim_middle_slice_{slice_list_str(triplet)}.h5ad"


def generate_uot(
    bottom: ad.AnnData,
    top: ad.AnnData,
    middle: ad.AnnData,
    *,
    t_interp: float,
    lambda_xy: float = 100.0,
    ot_reg: float = 0.005,
    ot_reg_m: float = 0.1,
    mass_eps: float = 1e-9,
    cell_type_key: str = "leiden",
    knn_k: int = 7,
    z_key: str = "z",
    output_path: Union[str, Path, None] = None,
    overwrite: bool = False,
) -> ad.AnnData:
    """
    Run slice-level UOT barycentric interpolation and KNN cell-type transfer.

    Loads from ``output_path`` when it exists and ``overwrite`` is False.
    """
    if output_path is not None:
        out_path = Path(output_path)
        if out_path.is_file() and not overwrite:
            loaded = ad.read_h5ad(out_path)
            return ensure_spatial_obsm(loaded)

    bottom_arr = slice_gene_arrays_from_adata(bottom, z_key=z_key)
    top_arr = slice_gene_arrays_from_adata(top, z_key=z_key)

    mid_xyz, mid_genes, mid_fids, _stats = build_interpolated_middle_slice(
        bottom_arr,
        top_arr,
        t_interp=float(t_interp),
        lambda_xy=float(lambda_xy),
        ot_reg=float(ot_reg),
        ot_reg_m=float(ot_reg_m),
        mass_eps=float(mass_eps),
    )

    labels = infer_cell_types_for_middle(
        bottom=bottom,
        top=top,
        query_genes=mid_genes,
        var_names=bottom_arr.var_names,
        label_col=cell_type_key,
        k=knn_k,
    )

    result = ann_data_from_interpolation(
        mid_xyz,
        mid_genes,
        bottom_arr.var_names,
        mid_fids,
        cell_type_key=cell_type_key,
        cell_types=labels,
    )

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_h5ad(out_path.as_posix())

    return result


def load_or_generate_uot_slice(
    *,
    slices: Mapping[int, ad.AnnData],
    slices_to_test: Iterable[Iterable[int]],
    middle_id: int,
    output_dir: Union[str, Path],
    save_name_prefix: str,
    uot_cfg: Mapping[str, object],
    data_cfg: Mapping[str, object],
) -> ad.AnnData:
    left_id, mid_id, right_id = triplet_for_middle_slice(slices_to_test, middle_id)
    triplet = (left_id, mid_id, right_id)
    filename = sim_middle_filename(save_name_prefix, triplet)
    out_path = Path(output_dir) / "uot" / filename

    left = slices[left_id]
    middle = slices[mid_id]
    right = slices[right_id]
    alpha = interpolation_alpha(
        left,
        middle,
        right,
        alpha_key=str(data_cfg.get("alpha_key", "slice_id")),
    )

    return generate_uot(
        bottom=left,
        top=right,
        middle=middle,
        t_interp=alpha,
        lambda_xy=float(uot_cfg.get("lambda_xy", 100.0)),
        ot_reg=float(uot_cfg.get("ot_reg", 0.005)),
        ot_reg_m=float(uot_cfg.get("ot_reg_m", 0.1)),
        mass_eps=float(uot_cfg.get("mass_eps", 1e-9)),
        cell_type_key=str(data_cfg.get("cell_class_key", "leiden")),
        knn_k=int(uot_cfg.get("knn_k", 7)),
        z_key=str(data_cfg.get("z_key", "z")),
        output_path=out_path,
        overwrite=bool(uot_cfg.get("overwrite", False)),
    )
