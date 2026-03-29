import json
from pathlib import Path
from typing import Any, Dict

import anndata as ad

from utils.data_loader import save_metrics
from utils.metrics import summarize_and_plot_metrics


def _load_json_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    # Resolve config relative to this file so it works from any CWD
    config_path = Path(__file__).parent / "inference_config.json"
    config = _load_json_config(config_path)
 
    true_adata_path = Path(config["true_adata_path"])
    sim_adata_path = Path(config["sim_adata_path"])
    metrics_cfg: Dict[str, Any] = config.get("metrics_parameters", {})

    # Load AnnData objects
    if not true_adata_path.is_file():
        raise FileNotFoundError(f"true_adata_path not found: {true_adata_path}")
    if not sim_adata_path.is_file():
        raise FileNotFoundError(f"sim_adata_path not found: {sim_adata_path}")

    true_adata = ad.read_h5ad(true_adata_path.as_posix())
    sim_adata = ad.read_h5ad(sim_adata_path.as_posix())

    # Decide where to save things inside the inference folder
    inference_dir = Path("inference")
    metrics_dir = inference_dir / "metrics"
    plots_dir = inference_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Filter metrics config to only accepted summarize_and_plot_metrics kwargs
    allowed_keys = {
        "coord_keys",
        "include_ari",
        "include_spatial_autocorrelation",
        "include_ssim",
        "include_ssim_gene_expression",
        "include_soft_metrics",
        "ari_label_key",  # will be overridden below if needed
        "autocorrelation_n_neighbors",
        "ssim_grid_size",
        "soft_radius",
        "show"
    }
    metrics_kwargs: Dict[str, Any] = {
        k: v for k, v in metrics_cfg.items() if k in allowed_keys and v is not None
    }
    metrics_kwargs["plot_save_dir"] = plots_dir.as_posix()
    metrics_kwargs["plot_name"] = config["dataset"] + " - Metrics Plot - Slice {slice_id}"
    metrics_kwargs["model_name"] = config["model_name"]
    metrics_kwargs["plot_save_name"] = config["dataset"] + "_" + config["model_name"] + "_slice_{slice_id}.png"

    # Compute metrics and generate plots
    metrics = summarize_and_plot_metrics(
        true_adata,
        sim_adata,
        **metrics_kwargs,
    )
    
    metrics_save_name = config["dataset"] + "_" + config["model_name"] + "_slice_{slice_id}.json"
    save_metrics(metrics, name=metrics_save_name, output_dir=metrics_dir)


if __name__ == "__main__":
    main()
