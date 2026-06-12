import argparse

from lloki.fp.run_lloki_fp import run_lloki_fp

import warnings
warnings.filterwarnings('ignore')

import os
import zipfile
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

default_values = {
    'data_dir': os.path.join(BASE_DIR, "data", "large_merfish_3"),
    'output_dir': os.path.join(BASE_DIR, "output_1may_cpu"),
    'model_dir': os.path.join(BASE_DIR, "external", "scgpt"),
    'reference_data_path': os.path.join(BASE_DIR, "data", "reference_data", "scref_full.h5ad"),
    'k': 40,  # used for knn
    'iter': 40,   # used for sparsity propagation
    'alpha': 0.05,        # was 0.5  — PDF: α = 0.05
    'fp_hard_alpha': 0.0,              # hard (masked) propagation mixing weight (usually 0.0)
    'fp_soft_iter': 1,                 # soft diffusion iterations (paper often uses 1)
    'fp_soft_use_imputed_input': False, # diffuse the imputed matrix (recommended / consistent with graph)
    'fp_avg_dist_samples': 100,        # samples used to estimate spatial kernel scale
    'fp_gcn_norm': True,               # apply GCN normalization to adjacency
    'fp_sym': True,                    # symmetrize adjacency
    'fp_early_stopping': True,         # early stopping inside propagation iterations
    'fp_tol': 1e-6,                    # tolerance for early stopping
    'fp_patience': 15,                 # consecutive stable steps required to stop
    'seed': 0,
    'device': "cpu",
    'obsm_spatial_key': 'X_spatial_coords'
}
args = argparse.Namespace(**default_values)
default_values.update({
    "npl_num_neighbors": 30,
    "mnn_num_neighbors": 40,
    "triplet_warmup": 10,
    "batch_dim": 10,
    "num_batches": 5,
    "lambda_neighborhood": 500,   # was 250  — PDF: λbc = 500
    "lambda_triplet": 2,          # was 0.2  — PDF: λtrip = 2
    "lr": 5e-4,
    "epochs": 20,
    "batch_size": 16000,          # was 4000 — PDF: 16,000 cells
    "checkpoint_dir": os.path.join(default_values["output_dir"], "checkpoints"),
})
args = argparse.Namespace(**default_values)

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.checkpoint_dir, exist_ok=True)

run_lloki_fp(args)