import glob
import os
import anndata as ad
import numpy as np
import scanpy as sc
import torch
import copy
from lloki.fp.lloki_fp import propagate
from lloki.fp.metrics import get_clustering_scores
from lloki.utils import set_seed
import scgpt as scg

def run_lloki_fp(args):
    """
    Run Lloki propagation on spatial transcriptomics data files, perform batching and
    embedding, and save processed data.
    """
    # Limit CPU threads for PyTorch
    obsm_spatial_key = getattr(args, "obsm_spatial_key", "spatial")
    torch.set_num_threads(3)

    # Allow running FP + embedding on different devices (useful on macOS/MPS).
    # Defaults keep existing behavior.
    fp_device = getattr(args, "fp_device", getattr(args, "device", "cpu"))
    embed_device = getattr(args, "embed_device", getattr(args, "device", "cpu"))

    # Validate/normalize embed_device for MPS
    if embed_device == "mps" and not torch.backends.mps.is_available():
        print("WARNING: MPS is not available. Using CPU for embedding instead.")
        embed_device = "cpu"

    # Load reference data for propagation
    adata_scrna = sc.read_h5ad(args.reference_data_path)  # Update path

    # Process each .h5ad file in the data directory
    for h5ad_file in glob.glob(os.path.join(args.data_dir, "*.h5ad")):
        print(f"Processing file: {h5ad_file}")
        adata = ad.read_h5ad(h5ad_file)
        set_seed(args.seed)  # Ensure reproducibility
        batches = [adata]  # Default to single batch

        # Apply spatially aware batching if large data
        if adata.shape[0] >= 30000:
            sorted_indices = np.argsort(adata.obsm[obsm_spatial_key][:, 0])
            num_splits = 4 if "xenium" in h5ad_file else 2  # More splits for 'xenium' files
            batches = [adata[s] for s in np.array_split(sorted_indices, num_splits)]
        
        # Propagate each batch and combine results
        args_fp = copy.copy(args)
        args_fp.device = fp_device
        adata = ad.concat(
            [propagate(batch, adata_scrna, args_fp) for batch in batches],
            join="outer",
            index_unique=None,
        )
        adata.obsm[obsm_spatial_key] = np.vstack([b.obsm[obsm_spatial_key] for b in batches])
        adata = adata[:, adata.var_names.isin(adata_scrna.var_names)]  # Filter genes present in reference data

        # Copy data for embedding and prepare necessary attributes
        adata_copy = adata.copy()
        denoised = adata.obsm["denoised"]
        adata_copy.X = denoised.toarray() if hasattr(denoised, "toarray") else denoised
        adata_copy.var["gene_names"] = [v.upper() for v in adata_copy.var_names]

        # Perform embedding with scGPT
        # scgpt uses `os.sched_getaffinity` to pick DataLoader workers, but that
        # function is Linux-only and doesn't exist on macOS.
        if not hasattr(os, "sched_getaffinity"):
            os.sched_getaffinity = lambda _pid=0: set(range(os.cpu_count() or 1))  # type: ignore[attr-defined]

        # On macOS, scgpt's DataLoader can crash when it uses multiprocessing
        # workers because its inner Dataset class is not pickleable under spawn.
        # Patch scgpt's module-level DataLoader to force num_workers=0.
        from scgpt.tasks import cell_emb as _cell_emb

        _OrigDataLoader = _cell_emb.DataLoader

        def _PatchedDataLoader(*dl_args, **dl_kwargs):
            dl_kwargs["num_workers"] = 0
            dl_kwargs["pin_memory"] = False
            return _OrigDataLoader(*dl_args, **dl_kwargs)

        _cell_emb.DataLoader = _PatchedDataLoader
        try:
            processed_data = scg.tasks.embed_data(
                adata_copy,
                args.model_dir,
                batch_size=64,
                gene_col="gene_names",
                device=embed_device,
            )
        finally:
            _cell_emb.DataLoader = _OrigDataLoader

        # Save processed AnnData with embedding to the output directory
        basename = os.path.splitext(os.path.basename(h5ad_file))[0]
        output_path = os.path.join(args.output_dir, f"{basename}_processed.h5ad")
        processed_data.write_h5ad(output_path)
        # print(f"Saved: {output_path}\nScore: {get_clustering_scores(processed_data, 'X_scGPT', filename=basename)}")
        print(f"Saved: {output_path}")
