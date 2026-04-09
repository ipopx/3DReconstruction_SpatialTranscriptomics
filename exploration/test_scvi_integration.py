import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
import ot 
from anndata import AnnData
from tqdm import tqdm
import time
import scvi
from scvi.model import SCVI
import json
import scipy.sparse as sp
import scanpy as sc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
try:
    import umap
except Exception:
    umap = None


def prepare_adata_for_scvi(adata, dataset, symbol_to_ensembl):
    adata_ensbl = adata.copy()
    ensmbl_adata = [symbol_to_ensembl[symbol] for symbol in adata.var_names]
    adata_ensbl.var_names = ensmbl_adata
    adata_ensbl.obs['donor_id'] = 'new_donor'
    adata_ensbl.obs['assay'] = dataset.upper()
    return adata_ensbl

def expand_to_genes(adata: AnnData, ref_genes) -> AnnData:
    ref_genes = pd.Index(ref_genes)
    q_genes = pd.Index(adata.var_names)

    pos = ref_genes.get_indexer(q_genes)
    keep = pos >= 0

    n, p = adata.n_obs, len(ref_genes)
    if sp.issparse(adata.X):
        X_full = sp.csr_matrix((n, p), dtype=adata.X.dtype)
        Xq = adata.X[:, keep]
        cols = pos[keep]
        X_full[:, cols] = Xq
    else:
        X_full = np.zeros((n, p), dtype=adata.X.dtype)
        X_full[:, pos[keep]] = adata.X[:, keep]

    out = AnnData(X=X_full, obs=adata.obs.copy(), var=pd.DataFrame(index=ref_genes))
    return out

def query_scvi(adata, scvi_model):
    scvi.model.SCVI.prepare_query_anndata(adata, scvi_model)
    scvi_query = scvi.model.SCVI.load_query_data(
        adata,
        scvi_model,
    )
    scvi_query.train(max_epochs=15, plan_kwargs=dict(weight_decay=0.0))
    return scvi_query

if __name__ == "__main__":
    adata = sc.read_h5ad("./data/merfish/merfish_all_int.h5ad")
    dataset = "merfish"
    print(adata.shape)
    
    # Load pre-trained scVI model and prepare query adata
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        # Set weights_only=False if not explicitly specified
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    
    # Load symbol to Ensembl mapping ONCE
    with open(f"./data/{dataset}/{dataset}_symbol_to_ensembl.json", "r") as f:
        symbol_to_ensembl = json.load(f)
    
    with open(f"./data/large_merfish/ref_genes.txt", "r") as f:
        ref_genes = f.read().splitlines()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    adata_ensbl = prepare_adata_for_scvi(adata, dataset, symbol_to_ensembl)
    adata_ensbl = expand_to_genes(adata_ensbl, ref_genes)
    
    # Integrate new batch
    adata_query = adata_ensbl.copy()
    load_adata = sc.read_h5ad("./data/large_merfish/animal4_int.h5ad")
    scvi_model = scvi.model.SCVI.load("./models/scvi_subset_model_4March_16latent_2layers_128hidden_8epochs", adata=load_adata)
    scvi_model.to_device(device)  
    scvi_query = query_scvi(adata_query, scvi_model)

    # ---- Quick latent-space integration check (first 3k cells each) ----
    n_plot = 3000
    ref_n = min(n_plot, load_adata.n_obs)
    qry_n = min(n_plot, adata_query.n_obs)

    # Latents
    z_ref = scvi_model.get_latent_representation(adata=load_adata)[:ref_n]
    z_qry = scvi_query.get_latent_representation()[:qry_n]

    Z = np.concatenate([z_ref, z_qry], axis=0)
    labels = np.array(["reference"] * ref_n + ["query"] * qry_n)

    if umap is not None:
        Z2 = umap.UMAP(n_components=2, random_state=0, n_neighbors=15, min_dist=0.3).fit_transform(Z)
    else:
        Z2 = PCA(n_components=2, random_state=0).fit_transform(Z)

    plt.figure(figsize=(7, 6))
    for name, color in [("reference", "#1f77b4"), ("query", "#ff7f0e")]:
        m = labels == name
        plt.scatter(Z2[m, 0], Z2[m, 1], s=3, alpha=0.6, c=color, label=name)
    proj_name = "UMAP" if umap is not None else "PCA"
    plt.title(f"scVI latent space ({proj_name}) — reference vs query (first 3k cells)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.show()

    def _plot_umap_colored_by_obs(key_ref: str, key_qry: str) -> None:
        ref_vals = (
            load_adata.obs[key_ref].astype(str).to_numpy()[:ref_n]
            if key_ref in load_adata.obs.columns
            else np.array(["NA"] * ref_n)
        )
        qry_vals = (
            adata_query.obs[key_qry].astype(str).to_numpy()[:qry_n]
            if key_qry in adata_query.obs.columns
            else np.array(["NA"] * qry_n)
        )
        vals = np.concatenate([ref_vals, qry_vals], axis=0)

        # Encode categories to integers for coloring
        cats, codes = np.unique(vals, return_inverse=True)
        cmap = plt.get_cmap("tab20", max(1, min(len(cats), 20)))

        plt.figure(figsize=(7, 6))
        sc = plt.scatter(Z2[:, 0], Z2[:, 1], s=3, alpha=0.7, c=codes, cmap=cmap)
        plt.title(f"{proj_name} latent — colored by `{key_qry}`")
        plt.xlabel(f"{proj_name}1")
        plt.ylabel(f"{proj_name}2")

        # Legend can be huge; show only if reasonably small
        if len(cats) <= 20:
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, color=cmap(i), label=c)
                for i, c in enumerate(cats)
            ]
            plt.legend(handles=handles, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.0)
        else:
            plt.colorbar(sc, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    _plot_umap_colored_by_obs("cell_type", "cell_class")
    
    
    
    
    
    