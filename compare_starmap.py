import json
from pathlib import Path
from typing import Union

import anndata as ad

from uot.generate_uot import load_or_generate_uot_slice, slice_list_str, triplet_for_middle_slice
from utils.comparative_metrics import compute_comparative_metrics
from utils.data_loader import load_starmap_dataset


def _load_json_config(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_slice(
    model_name: str,
    rel_path: str,
    *,
    output_dir: Path,
    slices,
    slices_to_test,
    middle_id: int,
    data_cfg: dict,
    uot_cfg: dict,
) -> ad.AnnData:
    full_path = output_dir / model_name / rel_path
    if full_path.is_file():
        return ad.read_h5ad(full_path)
    if model_name == "uot":
        return load_or_generate_uot_slice(
            slices=slices,
            slices_to_test=slices_to_test,
            middle_id=middle_id,
            output_dir=output_dir,
            save_name_prefix=data_cfg.get("save_name_prefix", "starmap"),
            uot_cfg=uot_cfg,
            data_cfg=data_cfg,
        )
    raise FileNotFoundError(
        f"Missing simulated slice for model {model_name!r}: {full_path}. "
        f"Run evaluation_starmap.py {model_name} first."
    )


def main():
    config = _load_json_config(Path("utils") / "config.json")
    data_cfg = config.get("data").get("starmap", config)
    spatialz_cfg = config.get("spatialz", config)
    metrics_cfg = config.get("metrics", config)
    compare_cfg = config.get("compare", config)
    uot_cfg = config.get("uot", {})

    adata, slices = load_starmap_dataset(
        data_cfg["path_to_starmap"],
        slice_thickness=data_cfg.get("slice_thickness"),
        num_slices=data_cfg.get("num_slices"),
        dropout_z_list=data_cfg.get("dropout_z_list", []),
        z_key=data_cfg.get("z_key", "z"),
        cell_class_key=data_cfg.get("cell_class_key", "leiden"),
    )

    model1_data_paths = compare_cfg.get("model1_data_paths")
    model2_data_paths = compare_cfg.get("model2_data_paths")
    model_names = compare_cfg.get("model_names")
    slices_to_test = data_cfg.get("slices_to_test", [])
    output_dir = Path(spatialz_cfg.get("output_dir", "output"))

    for i, slice_id in enumerate(compare_cfg.get("slices_to_compare")):
        print(f"Comparing slice {slice_id}")
        true_middle_slice = slices[slice_id]
        triplet = triplet_for_middle_slice(slices_to_test, slice_id)
        default_name = f"{data_cfg.get('save_name_prefix')}_sim_middle_slice_{slice_list_str(triplet)}.h5ad"
        path1 = model1_data_paths[i] if model1_data_paths else default_name
        path2 = model2_data_paths[i] if model2_data_paths else default_name

        model1_middle_slice = _load_model_slice(
            model_names[0],
            path1,
            output_dir=output_dir,
            slices=slices,
            slices_to_test=slices_to_test,
            middle_id=slice_id,
            data_cfg=data_cfg,
            uot_cfg=uot_cfg,
        )
        model2_middle_slice = _load_model_slice(
            model_names[1],
            path2,
            output_dir=output_dir,
            slices=slices,
            slices_to_test=slices_to_test,
            middle_id=slice_id,
            data_cfg=data_cfg,
            uot_cfg=uot_cfg,
        )

        plot_save_path = (
            Path(compare_cfg.get("output_dir", "output")) / "compare"
            / f"starmap_{model_names[0]}_vs_{model_names[1]}_slice_{slice_id}.png"
        )
        comparative_metrics = compute_comparative_metrics(
            true_adata=true_middle_slice,
            sim_adata_1=model1_middle_slice,
            sim_adata_2=model2_middle_slice,
            include_ari=metrics_cfg.get("include_ari"),
            include_spatial_autocorrelation=metrics_cfg.get("include_spatial_autocorrelation"),
            include_ssim=metrics_cfg.get("include_ssim"),
            include_ssim_gene_expression=metrics_cfg.get("include_ssim_gene_expression"),
            include_soft_metrics=metrics_cfg.get("include_soft_metrics"),
            ari_label_key=data_cfg.get("ari_label_key"),
            autocorrelation_n_neighbors=metrics_cfg.get("autocorrelation_n_neighbors"),
            ssim_grid_size=metrics_cfg.get("ssim_grid_size"),
            soft_radius=metrics_cfg.get("soft_radius"),
            show=metrics_cfg.get("show"),
            model_names=[model_names[0], model_names[1]],
            plot_save_path=plot_save_path,
            dataset_name="starmap",
        )

        print(comparative_metrics)


if __name__ == "__main__":
    main()
