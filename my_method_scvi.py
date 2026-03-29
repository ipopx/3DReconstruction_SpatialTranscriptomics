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


def train_local_scvi(
    adata1,
    adata2,
    use_gpu=None,
    n_latent=10,
    max_epochs=30,
):
    """
    Train a small scVI model on the concatenation of adata1 and adata2.
    """
    def _round_X_to_int_if_needed(adata, name: str) -> None:
        X = adata.X
        # Handle sparse matrices without importing heavy deps at module import time
        is_sparse = hasattr(X, "data") and hasattr(X, "tocoo")

        if is_sparse:
            data = X.data
            # If all entries are already (near) integers, do nothing
            if data.size and not np.allclose(data, np.rint(data)):
                X = X.copy()
                X.data = np.rint(X.data)
                adata.X = X
        else:
            arr = np.asarray(X)
            if arr.size and not np.allclose(arr, np.rint(arr)):
                adata.X = np.rint(arr)

    _round_X_to_int_if_needed(adata1, "adata1")
    _round_X_to_int_if_needed(adata2, "adata2")

    combined = adata1.concatenate(adata2, batch_key="slice", batch_categories=["left", "right"])
    SCVI.setup_anndata(combined)
    model = SCVI(combined, n_latent=n_latent)
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    model.train(max_epochs=max_epochs, use_gpu=use_gpu)
    return model


def generate_spatial_coord_wasserstein(
    adata1,
    adata2,
    alpha=0.5,
    device="cpu",
    n_cell=None,
    n_mag=1.0,
    lr=1e5,
    nb_iter_max=100,
    seed=42,
    num_projections=80,
    verbose=True,
    time_logger=None,
):
    if verbose:
        print("Begin to generate spatial coordinates......")

    start_time = time.time()
    coor1_torch = torch.tensor(adata1.obsm["spatial"], dtype=torch.float).to(device=device)
    coor2_torch = torch.tensor(adata2.obsm["spatial"], dtype=torch.float).to(device=device)

    local_n_cell = n_cell
    if local_n_cell is None:
        local_n_cell = int((alpha * adata1.n_obs + (1 - alpha) * adata2.n_obs) * n_mag)

    n_cell1 = int(local_n_cell * alpha)
    n_cell2 = local_n_cell - n_cell1

    sampled_indices1 = np.linspace(0, coor1_torch.shape[0] - 1, n_cell1, dtype=int)
    sampled_indices2 = np.linspace(0, coor2_torch.shape[0] - 1, n_cell2, dtype=int)

    Coor_init1 = coor1_torch[sampled_indices1].cpu().numpy()
    Coor_init2 = coor2_torch[sampled_indices2].cpu().numpy()
    Coor_init = np.concatenate([Coor_init1, Coor_init2], axis=0)

    Coor_torch = torch.tensor(Coor_init, dtype=torch.float32, device=device).requires_grad_(True)
    if time_logger is not None:
        time_logger("coordinate initialization", start_time)
    elif verbose:
        print(f"coordinate initialization time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    gen = torch.Generator(device=device).manual_seed(seed)
    for i in range(nb_iter_max):
        loss = (
            alpha
            * ot.sliced_wasserstein_distance(
                Coor_torch, coor1_torch, n_projections=num_projections, seed=gen
            )
            + (1 - alpha)
            * ot.sliced_wasserstein_distance(
                Coor_torch, coor2_torch, n_projections=num_projections, seed=gen
            )
        )
        loss.backward()

        with torch.no_grad():
            Coor_torch -= Coor_torch.grad * lr
            Coor_torch.grad.zero_()
        if verbose and i % 1000 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    if time_logger is not None:
        time_logger("Ot optimization", start_time)
    elif verbose:
        print(f"Ot optimization time: {time.time() - start_time:.2f} seconds")

    Coor_final = Coor_torch.detach().cpu().numpy()
    var_data = pd.DataFrame(index=adata1.var_names)
    adata3 = AnnData(
        X=np.zeros((Coor_final.shape[0], adata1.n_vars), dtype=np.float32),
        var=var_data,
        dtype=np.float32,
    )
    adata3.obsm["spatial"] = Coor_final

    return Coor_final, adata3


def generate_cell_type_knn(
    adata1,
    adata2,
    adata3,
    Coor_final,
    k_ct=1,
    cell_type_key="cell_type",
    adata1_id="above",
    adata2_id="below",
    add_obs_list=None,
    verbose=True,
    time_logger=None,
):
    if verbose:
        print("Begin to determine cell types......")

    start_time = time.time()
    nn_adata1 = NearestNeighbors(n_neighbors=k_ct).fit(adata1.obsm["spatial"])
    nn_adata2 = NearestNeighbors(n_neighbors=k_ct).fit(adata2.obsm["spatial"])
    distances_1, indices_1 = nn_adata1.kneighbors(Coor_final)
    distances_2, indices_2 = nn_adata2.kneighbors(Coor_final)

    epsilon = 0.1

    sim_celltype = []
    closest_indices = []
    for i in range(Coor_final.shape[0]):
        types_1 = adata1.obs.iloc[indices_1[i]][cell_type_key].values
        types_2 = adata2.obs.iloc[indices_2[i]][cell_type_key].values

        weights_1 = 1 / (distances_1[i] + epsilon)
        weights_2 = 1 / (distances_2[i] + epsilon)
        all_types = np.concatenate([types_1, types_2])
        all_weights = np.concatenate([weights_1, weights_2])

        type_weights = pd.Series(all_weights, index=all_types).groupby(level=0).sum()
        dominant_type = type_weights.idxmax() if not type_weights.empty else None

        if dominant_type:
            min_dist_1 = (
                np.min(distances_1[i][types_1 == dominant_type])
                if dominant_type in types_1
                else np.inf
            )
            min_dist_2 = (
                np.min(distances_2[i][types_2 == dominant_type])
                if dominant_type in types_2
                else np.inf
            )
            if min_dist_1 <= min_dist_2:
                closest_index = adata1.obs_names[
                    indices_1[i][np.where(types_1 == dominant_type)[0][0]]
                ]
            else:
                closest_index = adata2.obs_names[
                    indices_2[i][np.where(types_2 == dominant_type)[0][0]]
                ]
        else:
            closest_dist_1 = np.argmin(distances_1[i])
            closest_dist_2 = np.argmin(distances_2[i])
            if distances_1[i][closest_dist_1] < distances_2[i][closest_dist_2]:
                dominant_type = types_1[closest_dist_1]
                closest_index = adata1.obs_names[indices_1[i][closest_dist_1]]
            else:
                dominant_type = types_2[closest_dist_2]
                closest_index = adata2.obs_names[indices_2[i][closest_dist_2]]

        sim_celltype.append(dominant_type)
        closest_indices.append(closest_index)

    adata3.obs[cell_type_key] = sim_celltype

    if time_logger is not None:
        time_logger("Cell type determination", start_time)
    elif verbose:
        print(f"Cell type determination time: {time.time() - start_time:.2f} seconds")

    if add_obs_list is not None:
        start_time = time.time()
        if verbose:
            print("Begin to transfer the attribute......")
        for obs_key in add_obs_list:
            adata3.obs[obs_key] = [
                adata1.obs[obs_key][index] if adata1_id in index else adata2.obs[obs_key][index]
                for index in closest_indices
            ]

        if time_logger is not None:
            time_logger("Transfer the attribute", start_time)
        elif verbose:
            print(f"Transfer the attribute time: {time.time() - start_time:.2f} seconds")

    return sim_celltype, closest_indices, distances_1, distances_2, indices_1, indices_2, adata3


def generate_gex_scvi(
    adata1,
    adata2,
    adata3,
    Coor_final,
    sim_celltype,
    distances_1,
    distances_2,
    indices_1,
    indices_2,
    k_gex=3,
    cell_type_key="cell_type",
    verbose=True,
    time_logger=None,
    scvi_model=None,
):
    start_time = time.time()
    if verbose:
        print("Begin to synthesize gene expression with scVI......")

    if scvi_model is None:
        raise ValueError("scvi_model must be provided to generate_gene_expression_scFM.")

    n_cells = Coor_final.shape[0]
    n_genes = adata3.n_vars
    X_new = np.zeros((n_cells, n_genes), dtype=np.float32)

    # Number of cells in adata1, used as offset into the concatenated scVI reference
    n_adata1 = adata1.n_obs

    # Access the AnnData scVI was trained on (concatenation of adata1/adata2 in train_local_scvi)
    combined_adata = getattr(scvi_model, "adata", None)
    if combined_adata is None and hasattr(scvi_model, "adata_manager"):
        combined_adata = scvi_model.adata_manager.adata
    if combined_adata is None:
        raise RuntimeError("Could not access scvi_model training AnnData (expected scvi_model.adata).")

    # Precompute a log-library proxy from training data (used for decoding averaged latent)
    Xc = combined_adata.X
    if hasattr(Xc, "sum"):
        lib = np.asarray(Xc.sum(axis=1)).reshape(-1)
    else:
        lib = np.sum(np.asarray(Xc), axis=1)
    lib_log = np.log1p(lib).astype(np.float32, copy=False)

    # Determine device for module forward
    module_device = next(scvi_model.module.parameters()).device

    for i in range(n_cells):
        # Take k nearest neighbours across BOTH adjacent slices, filtered to the same cell type
        ctype = sim_celltype[i]
        idxs1 = np.asarray(indices_1[i])
        idxs2 = np.asarray(indices_2[i])
        dists1 = np.asarray(distances_1[i])
        dists2 = np.asarray(distances_2[i])

        neighbor_sources: list[tuple[str, int]] = []
        neighbor_dists: list[float] = []

        # Only keep neighbours of the same predicted cell type
        types_1 = adata1.obs.iloc[idxs1][cell_type_key].values
        types_2 = adata2.obs.iloc[idxs2][cell_type_key].values

        for j, t in enumerate(types_1):
            if t == ctype:
                neighbor_sources.append(("adata1", int(idxs1[j])))
                neighbor_dists.append(float(dists1[j]))
        for j, t in enumerate(types_2):
            if t == ctype:
                neighbor_sources.append(("adata2", int(idxs2[j])))
                neighbor_dists.append(float(dists2[j]))

        # Fallback: if no same-type neighbours found, use the nearest regardless of type
        if len(neighbor_sources) == 0:
            for j in range(len(idxs1)):
                neighbor_sources.append(("adata1", int(idxs1[j])))
                neighbor_dists.append(float(dists1[j]))
            for j in range(len(idxs2)):
                neighbor_sources.append(("adata2", int(idxs2[j])))
                neighbor_dists.append(float(dists2[j]))

        order = np.argsort(neighbor_dists)[:k_gex]

        # Map neighbor indices back to the concatenated AnnData that scVI was trained on
        combined_indices: list[int] = []
        for o in order:
            src, idx = neighbor_sources[int(o)]
            combined_indices.append(idx if src == "adata1" else (n_adata1 + idx))

        # Encode neighbours -> average latent z -> decode averaged z
        # scvi-tools versions differ; some don't support `return_numpy`.
        z_nb = scvi_model.get_latent_representation(indices=combined_indices)
        if torch.is_tensor(z_nb):
            z_nb = z_nb.detach().cpu().numpy()
        z_mean = np.mean(np.asarray(z_nb, dtype=np.float32), axis=0, keepdims=True)  # (1, n_latent)

        # Use mean log-library from neighbours as a decoding context
        lib_mean = float(np.mean(lib_log[np.asarray(combined_indices, dtype=int)]))

        z_t = torch.tensor(z_mean, device=module_device)
        library_t = torch.tensor([[lib_mean]], device=module_device)
        batch_index_t = torch.zeros((1, 1), dtype=torch.long, device=module_device)

        with torch.no_grad():
            gen_out = scvi_model.module.generative(
                z=z_t,
                library=library_t,
                batch_index=batch_index_t,
            )
            # scvi-tools versions differ in generative() outputs:
            # - newer: returns "px_rate"
            # - older: returns "px" distribution with .rate / .mean
            px_rate = gen_out.get("px_rate", None)
            if px_rate is None:
                px = gen_out.get("px", None)
                if px is None:
                    raise RuntimeError(
                        "scVI generative() did not return 'px_rate' or 'px'. "
                        f"Available keys: {list(gen_out.keys())}"
                    )
                if hasattr(px, "rate"):
                    px_rate = px.rate
                elif hasattr(px, "mean"):
                    px_rate = px.mean
                else:
                    raise RuntimeError(
                        "scVI generative() returned 'px' but it has no .rate or .mean attribute."
                    )

            X_new[i, :] = (
                px_rate.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            )

    adata3.X = X_new

    if time_logger is not None:
        time_logger("Gene expression synthesis with scVI", start_time)
    elif verbose:
        print(
            f"Gene expression synthesis with scVI time: {time.time() - start_time:.2f} seconds"
        )

    return adata3


def generate_scvi(adata1, adata2, adata1_id='above', adata2_id='below',
                      alpha=0.5, device='auto', n_cell=None, k_ct=1,
                      n_mag=1.0, lr=1e5, nb_iter_max=100, seed=42,
                      num_projections=80, cell_type_key='cell_type', k_gex=3, add_obs_list= None, verbose=True):
    """
    The generate_scvi function is designed to integrate spatial coordinates and gene expression from two AnnData objects and generate a new AnnData object. 
    """
    def print_time(message, start):
        if verbose:
            print(f"{message} time: {time.time() - start:.2f} seconds")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)

    # Check required fields in AnnData
    if 'spatial' not in adata1.obsm or 'spatial' not in adata2.obsm:
        raise ValueError("Both adata1 and adata2 must have 'spatial' coordinates in obsm.")
    if cell_type_key not in adata1.obs or cell_type_key not in adata2.obs:
        raise ValueError(f"Both adata1 and adata2 must have '{cell_type_key}' information in obs.")

    # Adjusting indices with unique identifiers
    adata1.obs_names = [f"{name}_{adata1_id}" for name in adata1.obs_names]
    adata2.obs_names = [f"{name}_{adata2_id}" for name in adata2.obs_names]

    # Train a small local scVI model on adata1 and adata2 to encode/decode gene expression
    scvi_use_gpu = device.type == "cuda"
    scvi_model = train_local_scvi(adata1, adata2, use_gpu=scvi_use_gpu)

    Coor_final, adata3 = generate_spatial_coord_wasserstein(
        adata1=adata1,
        adata2=adata2,
        alpha=alpha,
        device=device,
        n_cell=n_cell,
        n_mag=n_mag,
        lr=lr,
        nb_iter_max=nb_iter_max,
        seed=seed,
        num_projections=num_projections,
        verbose=verbose,
        time_logger=print_time,
    )

    (
        sim_celltype,
        closest_indices,
        distances_1,
        distances_2,
        indices_1,
        indices_2,
        adata3,
    ) = generate_cell_type_knn(
        adata1=adata1,
        adata2=adata2,
        adata3=adata3,
        Coor_final=Coor_final,
        k_ct=k_ct,
        cell_type_key=cell_type_key,
        adata1_id=adata1_id,
        adata2_id=adata2_id,
        add_obs_list=add_obs_list,
        verbose=verbose,
        time_logger=print_time,
    )

    adata3 = generate_gex_scvi(
        adata1=adata1,
        adata2=adata2,
        adata3=adata3,
        Coor_final=Coor_final,
        sim_celltype=sim_celltype,
        distances_1=distances_1,
        distances_2=distances_2,
        indices_1=indices_1,
        indices_2=indices_2,
        k_gex=k_gex,
        cell_type_key=cell_type_key,
        verbose=verbose,
        time_logger=print_time,
        scvi_model=scvi_model,
    )

    return adata3

def generate_multiple_scvi(adata1, adata2, num_sim, adata1_id='above', adata2_id='below',
                               device='auto', n_cell=None, n_mag=1.0, lr=1e5, nb_iter_max=3000, seed=42, num_projections=80,
                               cell_type_key='cell_type',syn_mode= 'default', k_gex=3, micro_env_key = 'mender', Beta = 100, add_obs_list=None, verbose=True,
                               include_raw=True):
    """
    The generate_multiple_scvi function extends the capabilities of the Generate_spatialz by generating multiple integrated AnnData objects. 
    """
    sim_adatas = []
    num_sim = num_sim + 1

    # Optionally include raw adata1 at the beginning
    if include_raw:
        adata1.obs['slice_id'] = f"{adata1_id}"
        adata1.obs['data_type'] = 'real'
        sim_adatas.append(adata1.copy())

    #for i in range(1, num_sim):  # Start from 1 to exclude alpha=1 and end at num_sim to exclude alpha=0
    for i in tqdm(range(1, num_sim), desc="Generating simulations"): 
        alpha = 1 - i / num_sim
        #print(alpha)
        sim_adata = generate_scvi(adata1, adata2, adata1_id=adata1_id, adata2_id=adata2_id,
                                      alpha=alpha, device=device, n_cell=n_cell, n_mag=n_mag, lr=lr,
                                      nb_iter_max=nb_iter_max, seed=seed, num_projections=num_projections,
                                      cell_type_key=cell_type_key, syn_mode= syn_mode, k_gex=k_gex, micro_env_key = micro_env_key, Beta = Beta, add_obs_list=add_obs_list,verbose=True
                                      )
        # Create slice_id
        #slice_id = f"{adata1_id}-{adata2_id}-{alpha:.2f}"
        slice_id = f"{adata1_id}-{adata2_id}-{i}"
        sim_adata.obs['slice_id'] = slice_id
        sim_adata.obs['data_type'] = 'synthetic'
        sim_adatas.append(sim_adata)
        if verbose:
            print(f"Completed {slice_id} generated!")

    # Optionally include raw adata2 at the end
    if include_raw:
        adata2.obs['slice_id'] = f"{adata2_id}"
        adata2.obs['data_type'] = 'real'
        sim_adatas.append(adata2.copy())

    # Concatenate all generated AnnData objects
    concatenated_adata = AnnData.concatenate(*sim_adatas, batch_key='slice_id', batch_categories=[s.obs['slice_id'][0] for s in sim_adatas])
    return concatenated_adata

