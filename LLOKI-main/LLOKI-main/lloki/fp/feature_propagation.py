import torch


class FeaturePropagation(torch.nn.Module):
    def __init__(
        self,
        num_iterations,
        adata,
        mask,
        alpha=0.0,
        max_imputation_per_cell=None,
        early_stopping=True,
        tol=1e-6,
        patience=15,
        device="cuda",
    ):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations
        self.mask = mask
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.tol = tol
        self.patience = patience
        self.adata = adata
        self.max_imputation_per_cell = max_imputation_per_cell
        self.device = device

    def forward(self, x, edge_index, edge_weight):
        device = self.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        original_x = x.clone()
        n_nodes, n_genes = x.shape
        res = (1 - self.alpha) * x
        adj = (
            torch.sparse.FloatTensor(edge_index, edge_weight, size=(n_nodes, n_nodes))
            .to(device)
            .float()
        )
        if self.max_imputation_per_cell is not None:
            existing_nonzero_mask = x != 0
            imputation_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
            max_imputation_per_cell = self.max_imputation_per_cell.to(device)
        stable_steps = 0

        for i in range(self.num_iterations):
            previous_x = x.clone()
            x = torch.sparse.mm(adj, x)
            if self.mask:
                if self.max_imputation_per_cell is not None:
                    x[existing_nonzero_mask] = original_x[existing_nonzero_mask]
                    newly_imputed = (x != 0) & ~existing_nonzero_mask & ~imputation_mask
                    imputation_mask |= newly_imputed
                    total_imputed_per_cell = imputation_mask.sum(dim=1)
                    over_imputed_cells = total_imputed_per_cell > max_imputation_per_cell
                    if over_imputed_cells.any():
                        for cell_idx in over_imputed_cells.nonzero(as_tuple=True)[0]:
                            allowed = int(max_imputation_per_cell[cell_idx].item())
                            imputed_gene_indices = imputation_mask[cell_idx].nonzero(
                                as_tuple=True
                            )[0]
                            if imputed_gene_indices.numel() <= allowed:
                                continue
                            if allowed <= 0:
                                x[cell_idx, imputed_gene_indices] = 0
                                imputation_mask[cell_idx, imputed_gene_indices] = False
                                continue

                            # Keep the strongest imputed values (deterministic).
                            imputed_vals = x[cell_idx, imputed_gene_indices]
                            keep = torch.topk(
                                imputed_vals, k=allowed, largest=True
                            ).indices
                            drop_mask = torch.ones(
                                imputed_gene_indices.numel(),
                                device=device,
                                dtype=torch.bool,
                            )
                            drop_mask[keep] = False
                            drop_gene_indices = imputed_gene_indices[drop_mask]
                            x[cell_idx, drop_gene_indices] = 0
                            imputation_mask[cell_idx, drop_gene_indices] = False

                    # Stop when all cells have reached their target sparsity
                    if (imputation_mask.sum(dim=1) >= max_imputation_per_cell).all():
                        break

                else:
                    nonzero_idx = torch.nonzero(original_x)
                    nonzero_i, nonzero_j = nonzero_idx.t()
                    x[nonzero_i, nonzero_j] = original_x[nonzero_i, nonzero_j]
            else:
                x.mul_(self.alpha).add_(res)

            # Frobenius norm early stopping with patience
            if self.early_stopping:
                if torch.norm(x - previous_x) < self.tol:
                    stable_steps += 1
                    if stable_steps >= self.patience:
                        break
                else:
                    stable_steps = 0

        return x
