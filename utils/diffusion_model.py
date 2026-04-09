"""
Diffusion model for spatial coordinates, conditioned on slice depth (z).

This module trains a DDPM-style denoiser that:
  f(x_t, z, t) -> x0_hat
where x_t is noisy (x,y) in *normalized* coordinates, z is the *true* slice depth
as a condition, and the output is the predicted clean (x,y).

Sampling supports optional KDE guidance using neighbor-slice coordinates in world (x,y).
Guidance uses ∇ log p_KDE(x,y) to bias the reverse process toward a neighbor slice.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import anndata as ad  # type: ignore
except Exception:  # pragma: no cover
    ad = None


@dataclass
class TrainConfig:
    # Model
    hidden_sizes: tuple[int, ...] = (512, 512)
    t_emb_dim: int = 32
    batchnorm: bool = False
    dropout: float = 0.0

    # Diffusion
    n_timesteps: int = 100
    beta_start: float = 1e-5
    beta_end: float = 2e-3

    # Training
    batch_size: int = 4096
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs: int = 100
    grad_clip: float | None = None
    ema_decay: float = 0.999


class TimestepEmbedding(nn.Module):
    """Sin/cos timestep embedding."""

    def __init__(self, dim: int = 32):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("t_emb_dim must be even")
        self.register_buffer(
            "freqs", torch.exp(torch.linspace(0, math.log(10000.0), steps=dim // 2))
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.float().unsqueeze(1) * self.freqs.view(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class EMA:
    """Exponential moving average of parameters (standard for diffusion)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters()}
        for v in self.shadow.values():
            v.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, p in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for k, p in model.named_parameters():
            p.copy_(self.shadow[k])


class DenoiserXYGivenZ(nn.Module):
    """
    Conditional denoiser:
      (x_t, z, t) -> x0_hat

    - x_t: (B, 2) noisy normalized xy
    - z:   (B, 1) normalized z (condition only)
    - t:   (B,) timestep indices
    - out: (B, 2) predicted clean normalized xy
    """

    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, ...] = (512, 512),
        t_emb_dim: int = 32,
        batchnorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.t_emb = TimestepEmbedding(dim=t_emb_dim)
        in_dim = 2 + 1 + t_emb_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([x_t, z, self.t_emb(t)], dim=-1)
        return self.net(feats)


# ---------------- KDE "potential" model (for guidance) ----------------
class KDEMixture(nn.Module):
    """
    Lightweight KDE that can also return a gradient signal for guidance.

    Given query points `points` (shape [P, d]), returns:
    - `fvals`: KDE density estimate at each point (shape [P])
    - `log_grad`: gradient of log density wrt the ORIGINAL (unscaled) coordinates (shape [P, d])

    This is designed to plug into `DDPMTrainer.sample_with_guidance(...)` as `potential_model`.
    """

    def __init__(self, spatial_data, bandwidth: float = 1.0, z_factor: float = 1.0):
        super().__init__()
        if not isinstance(spatial_data, torch.Tensor):
            spatial_data = torch.tensor(spatial_data).float()
        if not torch.isfinite(spatial_data).all():
            raise ValueError("KDEMixture spatial_data contains NaN/Inf.")
        self.N, self.d = spatial_data.shape
        if float(bandwidth) <= 0:
            raise ValueError("KDEMixture bandwidth must be > 0.")

        self.register_buffer("spatial_data", spatial_data)
        self.register_buffer("bandwidth", torch.tensor(float(bandwidth), device=spatial_data.device))
        self.register_buffer("z_factor", torch.tensor(float(z_factor), device=spatial_data.device))
        self.register_buffer("weights", torch.ones(self.N, device=spatial_data.device) / self.N)

    def forward(
        self,
        points: torch.Tensor,
        sample_frac: float = 1.0,
        eps: float = 1e-12,
        p_bs: int = 4096,  # points batch size
        s_bs: int = 8192,  # subset batch size
        normalize_weights: bool = True,
        rescale_grad_to_original_coords: bool = True,
    ):
        """
        Compute KDE density and ∇ log density for query points.

        Returns:
          - fvals: KDE density at each point, shape (P,)
          - log_grad: gradient of log density wrt ORIGINAL coords, shape (P, d)
        """
        device, out_dtype = points.device, points.dtype
        work_dtype = torch.float64

        centers, weights = self._select_centers_and_weights(
            device=device,
            dtype=work_dtype,
            sample_frac=sample_frac,
            eps=eps,
            normalize_weights=normalize_weights,
        )

        points_scaled, centers_scaled = self._scale_for_distance(points, centers, dtype=work_dtype)

        inv_sigma2 = self._inv_sigma2(device=device, dtype=work_dtype)
        n_points = int(points_scaled.shape[0])

        density = torch.zeros(n_points, device=device, dtype=work_dtype)
        grad_f = torch.zeros(n_points, self.d, device=device, dtype=work_dtype)

        self._accumulate_in_chunks(
            points_scaled=points_scaled,
            centers_scaled=centers_scaled,
            weights=weights,
            inv_sigma2=inv_sigma2,
            out_density=density,
            out_grad_f=grad_f,
            p_bs=p_bs,
            s_bs=s_bs,
        )

        grad_log_density = grad_f / (density[:, None] + eps)

        # Undo z-scaling on gradients so they are wrt ORIGINAL coordinates.
        if self.d >= 3 and rescale_grad_to_original_coords:
            grad_f[:, 2] *= self.z_factor
            grad_log_density[:, 2] *= self.z_factor

        return density.to(out_dtype), grad_log_density.to(out_dtype)

    def _inv_sigma2(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma2 = (self.bandwidth**2).to(device=device, dtype=dtype)
        return 1.0 / sigma2

    def _select_centers_and_weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        sample_frac: float,
        eps: float,
        normalize_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select KDE centers (optionally subsampled) and their weights."""
        if sample_frac < 1.0:
            k = max(1, int(self.N * sample_frac))
            idx = torch.randperm(self.N, device=self.spatial_data.device)[:k]
            centers = self.spatial_data[idx].to(device=device, dtype=dtype)
            weights = self.weights[idx].to(device=device, dtype=dtype)
        else:
            centers = self.spatial_data.to(device=device, dtype=dtype)
            weights = self.weights.to(device=device, dtype=dtype)

        if normalize_weights:
            weights = weights / (weights.sum() + eps)
        return centers, weights

    def _scale_for_distance(
        self, points: torch.Tensor, centers: torch.Tensor, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply anisotropic scaling (z-axis) before distance computation.
        Returns scaled copies in `dtype`.
        """
        p = points.to(dtype=dtype).clone()
        c = centers.to(dtype=dtype).clone()
        if self.d >= 3:
            p[:, 2] *= self.z_factor
            c[:, 2] *= self.z_factor
        return p, c

    def _accumulate_in_chunks(
        self,
        *,
        points_scaled: torch.Tensor,   # (P, d)
        centers_scaled: torch.Tensor,  # (K, d)
        weights: torch.Tensor,         # (K,)
        inv_sigma2: torch.Tensor,      # scalar
        out_density: torch.Tensor,     # (P,)
        out_grad_f: torch.Tensor,      # (P, d)
        p_bs: int,
        s_bs: int,
    ) -> None:
        """
        Chunked accumulation of density and grad f(x).

        KDE:
          f(x) = Σ w_i exp(-||x-c_i||^2 / (2σ^2))
        Gradient:
          ∇f(x) = Σ w_i [-(x-c_i)/σ^2] exp(-||x-c_i||^2 / (2σ^2))
        """
        n_points = int(points_scaled.shape[0])
        n_centers = int(centers_scaled.shape[0])

        for p0 in range(0, n_points, p_bs):
            p1 = min(p0 + p_bs, n_points)
            p_chunk = points_scaled[p0:p1]  # (bp, d)

            dens_chunk = torch.zeros(p1 - p0, device=points_scaled.device, dtype=points_scaled.dtype)
            grad_chunk = torch.zeros(p1 - p0, self.d, device=points_scaled.device, dtype=points_scaled.dtype)

            for c0 in range(0, n_centers, s_bs):
                c1 = min(c0 + s_bs, n_centers)
                c_chunk = centers_scaled[c0:c1]  # (bc, d)
                w_chunk = weights[c0:c1]         # (bc,)

                diffs = p_chunk[:, None, :] - c_chunk[None, :, :]          # (bp, bc, d)
                sq_dist = (diffs**2).sum(dim=-1)                           # (bp, bc)

                exponent = -sq_dist * (0.5 * inv_sigma2)
                exponent = torch.clamp(exponent, min=-80.0)                # exp(-80) ~ 1e-35
                kernels = torch.exp(exponent)                              # (bp, bc)

                dens_chunk = dens_chunk + (kernels * w_chunk[None, :]).sum(dim=1)

                grad_chunk = grad_chunk + (
                    -(diffs * inv_sigma2) * kernels[..., None] * w_chunk[None, :, None]
                ).sum(dim=1)

            out_density[p0:p1] = dens_chunk
            out_grad_f[p0:p1] = grad_chunk

    def sample(self, num_samples: int) -> torch.Tensor:
        indices = torch.randint(
            0, self.N, (int(num_samples),), device=self.spatial_data.device
        )
        noise = torch.randn(int(num_samples), self.d, device=self.spatial_data.device) * (
            self.bandwidth
        )
        if self.d >= 3:
            noise[:, 2] /= self.z_factor
        return self.spatial_data[indices] + noise


class KDEModelForGuidance(nn.Module):
    """
    Convenience wrapper used as `potential_model` for guided diffusion sampling.

    You can build it from:
    - `xyz`: an (N, d) array/tensor of reference points, OR
    - `slices`: a list of AnnData, where `obsm[spatial_key]` stores (N, d) points.
    """

    def __init__(
        self,
        slices=None,
        xyz=None,
        spatial_key: str = "aligned_spatial",
        bandwidth: float = 0.001,
        z_factor: float = 1.0,
    ):
        super().__init__()
        self.bandwidth = float(bandwidth)
        self.z_factor = float(z_factor)

        if xyz is not None:
            self.xyz = xyz
        elif slices is not None:
            if ad is None:
                raise ImportError(
                    "anndata is not available; pass xyz=... directly instead of slices=..."
                )
            self.xyz = ad.concat(slices).obsm[spatial_key]
        else:
            raise ValueError("Provide either xyz=... or slices=...")

        self.model: KDEMixture | None = None

    def fit(self):
        # If caller provided 3D reference points but the diffusion model is 2D (x,y),
        # you can still guide in x/y by passing only the first two columns here.
        xyz = self.xyz
        if isinstance(xyz, np.ndarray) and xyz.ndim == 2 and xyz.shape[1] >= 3:
            xyz = xyz[:, :2]
        elif isinstance(xyz, torch.Tensor) and xyz.ndim == 2 and xyz.shape[1] >= 3:
            xyz = xyz[:, :2]
        # Drop non-finite reference points (otherwise KDE returns NaNs).
        if isinstance(xyz, np.ndarray):
            mask = np.isfinite(xyz).all(axis=1)
            xyz = xyz[mask]
            if xyz.shape[0] == 0:
                raise ValueError("KDEModelForGuidance has no finite reference points after filtering.")
        elif isinstance(xyz, torch.Tensor):
            mask = torch.isfinite(xyz).all(dim=1)
            xyz = xyz[mask]
            if xyz.numel() == 0:
                raise ValueError("KDEModelForGuidance has no finite reference points after filtering.")

        self.model = KDEMixture(xyz, bandwidth=self.bandwidth, z_factor=self.z_factor)
        return self

    def forward(self, x: torch.Tensor, sample_frac: float = 1.0):
        if self.model is None:
            raise ValueError("KDEModelForGuidance.fit() must be called before forward().")
        return self.model(x, sample_frac=sample_frac)


class DDPMTrainer:
    """
    DDPM trainer for (x,y), conditioned on true z.

    Training data `coords_np` is expected to be (N, 3) or (N, >=3) with columns [x, y, z, ...].
    Only x,y are noised/denoised; z is provided as a condition.
    """

    def __init__(self, coords_np: np.ndarray, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        coords = np.asarray(coords_np, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 3:
            raise ValueError(f"coords_np must have shape (N, >=3); got {coords.shape}")
        coords_xyz = coords[:, :3]
        if not np.all(np.isfinite(coords_xyz)):
            raise ValueError("coords_np contains NaN/Inf in x/y/z. Filter rows before training.")

        xy = coords_xyz[:, :2]
        z = coords_xyz[:, 2:3]

        # Normalize xy (train+inference)
        self.xy_mean = xy.mean(axis=0, keepdims=True)
        self.xy_std = xy.std(axis=0, keepdims=True) + 1e-8
        xy_n = (xy - self.xy_mean) / self.xy_std

        # Normalize z for conditioning (keeps scale stable)
        self.z_mean = z.mean(axis=0, keepdims=True)
        self.z_std = z.std(axis=0, keepdims=True) + 1e-8
        z_n = (z - self.z_mean) / self.z_std

        self.data_xy = torch.tensor(xy_n, dtype=torch.float32)
        self.data_z = torch.tensor(z_n, dtype=torch.float32)
        self.loader = DataLoader(
            TensorDataset(self.data_xy, self.data_z),
            batch_size=int(cfg.batch_size),
            shuffle=True,
            drop_last=False,
        )

        # Linear beta schedule
        T = int(cfg.n_timesteps)
        betas = torch.linspace(float(cfg.beta_start), float(cfg.beta_end), T, dtype=torch.float64).clamp(1e-8, 0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(self.device, dtype=torch.float32)
        self.alphas = alphas.to(self.device, dtype=torch.float32)
        self.alpha_bars = alpha_bars.to(self.device, dtype=torch.float32)

        self.model = DenoiserXYGivenZ(
            hidden_sizes=cfg.hidden_sizes,
            t_emb_dim=cfg.t_emb_dim,
            batchnorm=cfg.batchnorm,
            dropout=cfg.dropout,
        ).to(self.device)
        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

    def train(self) -> None:
        self.model.train()
        n_timesteps = int(self.betas.shape[0])
        grad_clip = self.cfg.grad_clip

        for epoch in range(1, int(self.cfg.epochs) + 1):
            total_loss, n_batches = 0.0, 0
            for xy0, zc in self.loader:
                xy0 = xy0.to(self.device)
                zc = zc.to(self.device)
                B = int(xy0.shape[0])

                t = torch.randint(0, n_timesteps, (B,), device=self.device)
                eps = torch.randn_like(xy0)
                alpha_bar_t = self.alpha_bars[t].view(-1, 1)
                xy_t = torch.sqrt(alpha_bar_t) * xy0 + torch.sqrt(1 - alpha_bar_t) * eps

                # Predict noise ε (standard DDPM objective, more stable than x0-pred)
                eps_hat = self.model(xy_t, t, zc)
                loss = ((eps_hat - eps) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Loss became NaN/Inf. Check coordinate scaling and beta schedule.")

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and float(grad_clip) > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                self.opt.step()
                self.ema.update(self.model)

                total_loss += float(loss.item())
                n_batches += 1

            print(f"Epoch {epoch:03d} | loss {total_loss / max(1, n_batches):.6f}")

    def _make_eval_model(self, use_ema: bool) -> DenoiserXYGivenZ:
        model = DenoiserXYGivenZ(
            hidden_sizes=self.cfg.hidden_sizes,
            t_emb_dim=self.cfg.t_emb_dim,
            batchnorm=self.cfg.batchnorm,
            dropout=self.cfg.dropout,
        ).to(self.device)
        model.load_state_dict(self.model.state_dict(), strict=True)
        if use_ema:
            self.ema.copy_to(model)
        model.eval()
        return model

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        z_world: float | np.ndarray,
        *,
        use_ema: bool = True,
        x_clip: float = 20.0,
    ) -> np.ndarray:
        """Sample (x,y) conditioned on a fixed z (world units)."""
        model = self._make_eval_model(use_ema=use_ema)
        n_timesteps = int(self.betas.shape[0])

        # Normalize z condition
        z = np.asarray(z_world, dtype=np.float32).reshape(-1, 1)
        if z.shape[0] == 1:
            z = np.repeat(z, n_samples, axis=0)
        if z.shape[0] != n_samples:
            raise ValueError(f"z_world must be scalar or shape (n_samples,); got {z.shape}")
        z_n = (z - self.z_mean) / self.z_std
        z_t = torch.tensor(z_n, device=self.device, dtype=torch.float32)

        xy = torch.randn(n_samples, 2, device=self.device)
        for t in reversed(range(n_timesteps)):
            if x_clip and x_clip > 0:
                xy = torch.clamp(xy, -float(x_clip), float(x_clip))

            beta_t = self.betas[t]
            alpha_t = torch.clamp(self.alphas[t], min=1e-5)
            alpha_bar_t = torch.clamp(self.alpha_bars[t], min=1e-5, max=1.0 - 1e-6)
            sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

            t_vec = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            eps_hat = model(xy, t_vec, z_t)
            eps_hat = torch.nan_to_num(eps_hat, nan=0.0, posinf=0.0, neginf=0.0)

            z_step = torch.randn_like(xy) if t > 0 else 0.0
            xy = sqrt_recip_alpha * (xy - (1 - alpha_t) / sqrt_one_minus_alpha_bar * eps_hat) + torch.sqrt(beta_t) * z_step
            xy = torch.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)

        xy_world = xy.detach().cpu().numpy() * self.xy_std + self.xy_mean
        return xy_world

    @torch.no_grad()
    def sample_with_guidance(
        self,
        n_samples: int,
        z_world: float | np.ndarray,
        *,
        neighbor_coords: np.ndarray | torch.Tensor,
        kde_bandwidth: float = 1.0,
        guidance_scale: float = 1.0,
        guidance_clip: float = 5.0,
        sample_frac: float = 1.0,
        use_ema: bool = True,
        x_clip: float = 20.0,
        debug: bool = False,
        debug_every: int = 10,
    ) -> np.ndarray:
        """
        Sample with KDE guidance toward a neighboring slice.

        Guidance uses ∇ log p_KDE(x,y) computed in world coords (x,y),
        then converted to normalized coords and applied as:
          ε_guided = ε_hat - guidance_scale * grad_norm
        """
        model = self._make_eval_model(use_ema=use_ema)
        n_timesteps = int(self.betas.shape[0])

        # Normalize z condition
        z = np.asarray(z_world, dtype=np.float32).reshape(-1, 1)
        if z.shape[0] == 1:
            z = np.repeat(z, n_samples, axis=0)
        if z.shape[0] != n_samples:
            raise ValueError(f"z_world must be scalar or shape (n_samples,); got {z.shape}")
        z_n = (z - self.z_mean) / self.z_std
        z_t = torch.tensor(z_n, device=self.device, dtype=torch.float32)

        # Build KDE potential in world (x,y)
        nbr = neighbor_coords
        if isinstance(nbr, torch.Tensor):
            nbr_xy = nbr.detach().cpu().numpy()
        else:
            nbr_xy = np.asarray(nbr)
        if nbr_xy.ndim != 2 or nbr_xy.shape[1] < 2:
            raise ValueError(f"neighbor_coords must be (N,>=2); got {nbr_xy.shape}")
        nbr_xy = nbr_xy[:, :2]
        pot = KDEMixture(nbr_xy, bandwidth=float(kde_bandwidth), z_factor=1.0).to(self.device)

        xy = torch.randn(n_samples, 2, device=self.device)
        xy_mean_t = torch.tensor(self.xy_mean, device=self.device, dtype=torch.float32)
        xy_std_t = torch.tensor(self.xy_std, device=self.device, dtype=torch.float32)

        for t in reversed(range(n_timesteps)):
            if x_clip and x_clip > 0:
                xy = torch.clamp(xy, -float(x_clip), float(x_clip))

            beta_t = self.betas[t]
            alpha_t = torch.clamp(self.alphas[t], min=1e-5)
            alpha_bar_t = torch.clamp(self.alpha_bars[t], min=1e-5, max=1.0 - 1e-6)
            sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

            t_vec = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            eps_hat = model(xy, t_vec, z_t)
            eps_hat = torch.nan_to_num(eps_hat, nan=0.0, posinf=0.0, neginf=0.0)

            # Guidance query point: x0 estimate in world coords derived from eps_hat
            xy0_hat = (xy - sqrt_one_minus_alpha_bar * eps_hat) / torch.sqrt(alpha_bar_t)
            xy0_world = xy0_hat * xy_std_t + xy_mean_t
            _, grad_world = pot(xy0_world, sample_frac=float(sample_frac))
            grad_world = torch.nan_to_num(grad_world, nan=0.0, posinf=0.0, neginf=0.0)

            # Convert grad from world to normalized coords
            grad_norm = grad_world / xy_std_t
            if guidance_clip and guidance_clip > 0:
                grad_norm = torch.clamp(grad_norm, -float(guidance_clip), float(guidance_clip))

            if debug and (t % max(1, int(debug_every)) == 0 or t == n_timesteps - 1 or t == 0):
                with torch.no_grad():
                    g_absmax = float(grad_norm.abs().max().item()) if grad_norm.numel() else 0.0
                    g_med = float(grad_norm.abs().median().item()) if grad_norm.numel() else 0.0
                    print(f"[guidance] t={t:04d} |grad_norm| median={g_med:.3g} max={g_absmax:.3g}")

            eps_guided = eps_hat - float(guidance_scale) * grad_norm
            eps_guided = torch.nan_to_num(eps_guided, nan=0.0, posinf=0.0, neginf=0.0)

            z_step = torch.randn_like(xy) if t > 0 else 0.0
            xy = sqrt_recip_alpha * (xy - (1 - alpha_t) / sqrt_one_minus_alpha_bar * eps_guided) + torch.sqrt(beta_t) * z_step
            xy = torch.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)

        return xy.detach().cpu().numpy() * self.xy_std + self.xy_mean
