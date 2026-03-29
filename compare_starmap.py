import json
from pathlib import Path
from typing import Union

from utils.data_loader import load_starmap_dataset
from utils.comparative_metrics import compute_comparative_metrics

import anndata as ad

def _load_json_config(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    config = _load_json_config(Path("utils") / "config.json")
    data_cfg = config.get("data").get("starmap", config)
    spatialz_cfg = config.get("spatialz", config)
    metrics_cfg = config.get("metrics", config)
    compare_cfg = config.get("compare", config)

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
    
    for i, slice_id in enumerate(compare_cfg.get("slices_to_compare")):
        print(f"Comparing slice {slice_id}")
        true_middle_slice = slices[slice_id]
        device = spatialz_cfg.get("device", "auto")
        
        output_dir = Path(spatialz_cfg.get("output_dir", "output")) 
        model1_middle_slice = ad.read_h5ad(output_dir / model_names[0] / model1_data_paths[i])
        model2_middle_slice = ad.read_h5ad(output_dir / model_names[1] / model2_data_paths[i])
        
        plot_save_path = (
            Path(compare_cfg.get("output_dir", "output")) / "compare"
            / f"starmap_{model_names[0]}_vs_{model_names[1]}_slice_{slice_id}.png"
        )
        comparative_metrics = compute_comparative_metrics(
            true_adata=true_middle_slice,
            sim_adata_1=model1_middle_slice,
            sim_adata_2=model2_middle_slice,
            include_ari = metrics_cfg.get("include_ari"),
            include_spatial_autocorrelation = metrics_cfg.get("include_spatial_autocorrelation"),
            include_ssim = metrics_cfg.get("include_ssim"),
            include_ssim_gene_expression = metrics_cfg.get("include_ssim_gene_expression"),
            include_soft_metrics = metrics_cfg.get("include_soft_metrics"),
            ari_label_key = data_cfg.get("ari_label_key"),
            autocorrelation_n_neighbors = metrics_cfg.get("autocorrelation_n_neighbors"),
            ssim_grid_size = metrics_cfg.get("ssim_grid_size"),
            soft_radius = metrics_cfg.get("soft_radius"),
            show = metrics_cfg.get("show"),
            model_names=[model_names[0], model_names[1]],
            plot_save_path=plot_save_path,
            dataset_name="starmap",
        )
        
        print(comparative_metrics)
        
if __name__ == "__main__":
    main()