from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP projection."""

    def __init__(self, dim: int):
        super().__init__()
        # Width of the conditioning vector injected into every DiT block.
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed integer diffusion timesteps with shape ``B``."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = rearrange(t.float(), "b -> b 1") * rearrange(freqs, "d -> 1 d")
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class DiTBlock(nn.Module):
    """Transformer block with adaptive LayerNorm conditioning."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # Affine-free norms receive shift/scale from the conditioning vector.
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP residual branches modulated by ``cond``."""
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=1)
        h = self.norm1(x) * (1 + rearrange(scale1, "b d -> b 1 d")) + rearrange(shift1, "b d -> b 1 d")
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + rearrange(gate1, "b d -> b 1 d") * h
        h = self.norm2(x) * (1 + rearrange(scale2, "b d -> b 1 d")) + rearrange(shift2, "b d -> b 1 d")
        x = x + rearrange(gate2, "b d -> b 1 d") * self.mlp(h)
        return x


class DiT(nn.Module):
    """Latent diffusion transformer that predicts Gaussian noise on VAE latents."""

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 7,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        num_classes: int = 10,
        class_dropout: float = 0.1,
    ):
        super().__init__()
        # Channel count of the latent feature map produced by ConvVAE.
        self.latent_channels = latent_channels
        # Spatial side length of the latent map; MNIST uses 7x7 after downsampling.
        self.latent_size = latent_size
        # Number of digit classes; an extra null label enables classifier-free guidance.
        self.num_classes = num_classes
        self.null_label = num_classes
        # Probability of dropping labels during training.
        self.class_dropout = class_dropout
        self.in_proj = nn.Linear(latent_channels, dim)
        # Learned absolute position embedding for flattened latent patches.
        self.pos = nn.Parameter(torch.zeros(1, latent_size * latent_size, dim))
        self.time = SinusoidalTimeEmbedding(dim)
        self.label_emb = nn.Embedding(num_classes + 1, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.out_proj = nn.Linear(dim, latent_channels)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        force_drop_labels: bool = False,
    ) -> torch.Tensor:
        """Predict the noise tensor for noised latent maps at timesteps ``t``."""
        b, c, h, w = x.shape
        if h != self.latent_size or w != self.latent_size:
            raise ValueError(f"expected latent size {self.latent_size}x{self.latent_size}, got {h}x{w}")

        if self.training or force_drop_labels:
            drop = torch.rand(labels.shape, device=labels.device) < self.class_dropout
            if force_drop_labels:
                drop = torch.ones_like(drop, dtype=torch.bool)
            labels = torch.where(drop, torch.full_like(labels, self.null_label), labels)

        tokens = rearrange(x, "b c h w -> b (h w) c")
        tokens = self.in_proj(tokens) + self.pos
        cond = self.time(t) + self.label_emb(labels)
        for block in self.blocks:
            tokens = block(tokens, cond)
        shift, scale = self.out_ada(cond).chunk(2, dim=1)
        tokens = self.out_norm(tokens) * (1 + rearrange(scale, "b d -> b 1 d")) + rearrange(shift, "b d -> b 1 d")
        tokens = self.out_proj(tokens)
        return rearrange(tokens, "b (h w) c -> b c h w", h=h, w=w)


class Diffusion:
    """DDPM forward noising and ancestral reverse sampling utilities."""

    def __init__(self, timesteps: int = 1000, device: str = "cpu"):
        # Number of discrete noising steps in the DDPM chain.
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        # Per-step noise variances and derived alpha schedules.
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``x_t`` from ``q(x_t | x_0)`` and return the noise target."""
        if x0.ndim != 4:
            raise ValueError(f"expected BCHW latent tensor, got shape {tuple(x0.shape)}")
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = rearrange(self.sqrt_alpha_bars[t], "b -> b 1 1 1")
        sqrt_one_minus_alpha_bar = rearrange(self.sqrt_one_minus_alpha_bars[t], "b -> b 1 1 1")
        return (
            sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise,
            noise,
        )

    @torch.no_grad()
    def sample(
        self,
        model: DiT,
        shape: tuple[int, int, int, int],
        labels: torch.Tensor,
        cfg_scale: float = 2.0,
    ) -> torch.Tensor:
        """Generate latent samples by reversing the DDPM chain with optional CFG."""
        x = torch.randn(shape, device=labels.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=labels.device, dtype=torch.long)
            eps = model(x, t, labels)
            if cfg_scale != 1.0:
                uncond = model(x, t, labels, force_drop_labels=True)
                eps = uncond + cfg_scale * (eps - uncond)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            x = (x - beta / torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha)
            if i > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x
