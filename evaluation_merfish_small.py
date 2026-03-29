import sys
model_to_run = sys.argv[1]  # spatialz or scvi

import json
from pathlib import Path
from typing import Union

from utils.data_loader import get_test_slices, load_merfish_dataset, save_slice, save_metrics
from utils.metrics import summarize_and_plot_metrics

if model_to_run == "spatialz":
    from SpatialZ import *
elif model_to_run == "scvi":
    from my_method_scvi import generate_scvi
elif model_to_run == "pca":
    from my_method_pca import generate_pca
else:
    raise ValueError(f"Invalid model: {model_to_run}")

def _load_json_config(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _slice_list_str(sl: list) -> str:
    """Canonical string for slice list in filenames, e.g. [1,2,3] (no spaces)."""
    return "[" + ",".join(map(str, sl)) + "]"


def main():
    config = _load_json_config(Path("utils") / "config.json")
    data_cfg = config.get("data").get("merfish", config)
    spatialz_cfg = config.get("spatialz", config)
    metrics_cfg = config.get("metrics", config)

    adata, slices = load_merfish_dataset(
        data_cfg["path_to_merfish"],
        cell_class_key=data_cfg.get("cell_class_key", "cell_class"),
    )

    if data_cfg.get("evaluate_on_real_data", False):
        slices_to_test = data_cfg.get("slices_to_test")
        test_slices = get_test_slices(slices, slices_to_test)
        device = spatialz_cfg.get("device", "auto")
    
        print(f"Running SpatialZ for {len(test_slices)} test slices")
        for i, test_slice in enumerate(test_slices):  
            left_slice, middle_slice, right_slice = test_slice
            
            # Get parameters for SpatialZ
            alpha = middle_slice.obs['z'].values[0]/ (right_slice.obs['z'].values[0] +  left_slice.obs['z'].values[0])
            if spatialz_cfg.get("auto_n_cell", spatialz_cfg.get("spatialz_auto_n_cell", spatialz_cfg.get("sptialz_auto_n_cell", False))):
                n_cell_middle_slice = None
            else:
                n_cell_middle_slice = len(middle_slice.obs_names)

            k_ct = spatialz_cfg.get("k_ct", spatialz_cfg.get("spatialz_k_ct", 5))
            k_gex = spatialz_cfg.get("k_gex", spatialz_cfg.get("spatialz_k_gex", 5))
            nb_iter_max = spatialz_cfg.get("nb_iter_max", spatialz_cfg.get("spatialz_nb_iter_max", 100))
            output_dir = Path(spatialz_cfg.get("output_dir", "output")) / model_to_run
            
            if model_to_run == "spatialz":
                sim_middle_slice = Generate_spatialz(
                    adata1=left_slice,
                    adata2=right_slice,
                    adata1_id="above",
                    adata2_id="below",
                    alpha=alpha,
                    device=device,
                    n_cell=n_cell_middle_slice,
                    k_neighbors=k_ct,
                    n_mag=1.0,
                    lr=1e5,
                    nb_iter_max=nb_iter_max,
                    seed=42,
                    num_projections=80,
                    cell_type_key=data_cfg.get("cell_class_key", "leiden"),
                    syn_mode="default",
                    k_sam=k_gex,
                    micro_env_key="mender",
                    Beta=100,
                    # transfer domain / other annotations onto simulated slice
                    add_obs_list= data_cfg.get("added_obs_list", None),
                    verbose=spatialz_cfg.get("verbose"),
                )
            
            elif model_to_run == "scvi":
                sim_middle_slice = generate_scvi(
                    adata1=left_slice,
                    adata2=right_slice,
                    adata1_id="above",
                    adata2_id="below",
                    alpha=alpha,
                    device=device,
                    n_cell=n_cell_middle_slice,
                    k_ct=k_ct,
                    n_mag=1.0,
                    lr=1e5,
                    nb_iter_max=nb_iter_max,
                    seed=42,
                    num_projections=80,
                    cell_type_key=data_cfg.get("cell_class_key", "leiden"),
                    k_gex=k_gex,
                    add_obs_list= data_cfg.get("added_obs_list", None),
                    verbose=spatialz_cfg.get("verbose"),
                )
            elif model_to_run == "pca":
                sim_middle_slice = generate_pca(
                    adata1=left_slice,
                    adata2=right_slice,
                    adata1_id="above",
                    adata2_id="below",
                    alpha=alpha,
                    device=device,
                    n_cell=n_cell_middle_slice,
                    k_ct=k_ct,
                    n_mag=1.0,
                    lr=1e5,
                    nb_iter_max=nb_iter_max,
                    seed=42,
                    num_projections=80,
                    cell_type_key=data_cfg.get("cell_class_key", "leiden"),
                    k_gex=k_gex,
                    add_obs_list= data_cfg.get("added_obs_list", None),
                    verbose=spatialz_cfg.get("verbose"),
                )
            
            _sl = _slice_list_str(slices_to_test[i])
            save_name = f"{data_cfg.get('save_name_prefix')}_sim_middle_slice_{_sl}.h5ad"
            save_slice(sim_middle_slice, output_dir=output_dir, name=save_name)
            
            plot_save_dir = Path(spatialz_cfg.get("output_dir", "output")) / model_to_run / "metrics_plots"
            plot_save_name = f"{data_cfg.get('save_name_prefix')}_metrics_plot_{_sl}.png"
            plot_name = f"{data_cfg.get('save_name_prefix')} - Metrics Plot - Slices {slices_to_test[i]}"
            # Evaluate metrics on simulated middle slice
            metrics = summarize_and_plot_metrics(
                middle_slice,
                sim_middle_slice,
                ssim_grid_size=metrics_cfg.get("ssim_grid_size"),
                ari_label_key=data_cfg.get("ari_label_key"),
                autocorrelation_n_neighbors=metrics_cfg.get("autocorrelation_n_neighbors"),
                show=metrics_cfg.get("show"),
                include_ari=metrics_cfg.get("include_ari"),
                include_spatial_autocorrelation=metrics_cfg.get("include_spatial_autocorrelation"),
                include_ssim=metrics_cfg.get("include_ssim"),
                include_ssim_gene_expression=metrics_cfg.get("include_ssim_gene_expression"),
                include_soft_metrics=metrics_cfg.get("include_soft_metrics"),
                soft_radius=metrics_cfg.get("soft_radius"),
                plot_save_dir = plot_save_dir,
                plot_save_name= plot_save_name,
                plot_name = plot_name,
                model_name = model_to_run
            )

            save_name = f"{data_cfg.get('save_name_prefix')}_metrics_{_sl}.json"
            save_metrics(metrics, output_dir=output_dir, name=save_name)

if __name__ == "__main__":
    main()