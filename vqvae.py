from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def vqvae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    quantized_raw: torch.Tensor,
    z_e: torch.Tensor,
    commitment_cost: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total VQ-VAE loss plus reconstruction and codebook terms."""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
    codebook = F.mse_loss(quantized_raw, z_e.detach())
    commitment = F.mse_loss(quantized_raw.detach(), z_e)
    return recon + codebook + commitment_cost * commitment, recon, codebook + commitment


class VectorQuantizer(nn.Module):
    """Nearest-neighbor vector quantizer for 2D latent feature maps."""

    def __init__(self, num_codes: int = 128, code_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()
        # Number of discrete entries in the learned codebook.
        self.num_codes = num_codes
        # Channel width of each latent/codebook vector.
        self.code_dim = code_dim
        # Weight for encoder commitment loss, kept here for checkpoint/config clarity.
        self.commitment_cost = commitment_cost
        # Codebook mapping integer token ids to continuous latent vectors.
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder latents and return straight-through latents, ids, and raw codebook values."""
        b, c, h, w = z_e.shape
        flat = rearrange(z_e, "b c h w -> (b h w) c")
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + rearrange(self.embedding.weight.pow(2).sum(dim=1), "codes -> 1 codes")
        )
        indices = distances.argmin(dim=1)
        quantized_raw = rearrange(self.embedding(indices), "(b h w) c -> b c h w", b=b, h=h, w=w)
        quantized = z_e + (quantized_raw - z_e).detach()
        return quantized, rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w), quantized_raw

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert a grid of token ids with shape ``B H W`` into latent maps ``B C H W``."""
        z_q = self.embedding(indices)
        return rearrange(z_q, "b h w c -> b c h w")


class VQVAE(nn.Module):
    """Convolutional VQ-VAE that maps MNIST images to a 7x7 grid of discrete codes."""

    def __init__(
        self,
        num_codes: int = 128,
        code_dim: int = 64,
        hidden_dim: int = 128,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        # Size of the discrete codebook used by the quantizer and MaskGIT.
        self.num_codes = num_codes
        # Channel width of the encoder output before vector quantization.
        self.code_dim = code_dim
        # Spatial side length after two stride-2 downsampling layers for 28x28 MNIST.
        self.latent_size = 7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, code_dim, 1),
        )
        # Bottleneck that replaces continuous encoder vectors with codebook entries.
        self.quantizer = VectorQuantizer(num_codes, code_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.Conv2d(code_dim, hidden_dim, 3, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, 1, 3, 1, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images into quantized latents, token ids, and pre-quantized latents."""
        z_e = self.encoder(x)
        z_q, indices, _ = self.quantizer(z_e)
        return z_q, indices, z_e

    def encode_for_loss(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images and keep raw codebook vectors needed by the VQ loss."""
        z_e = self.encoder(x)
        z_q, indices, z_q_raw = self.quantizer(z_e)
        return z_q, indices, z_e, z_q_raw

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent maps into image logits."""
        return self.decoder(z_q)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode a grid of codebook ids directly into image logits."""
        return self.decode(self.quantizer.embed(indices))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction logits and quantization tensors for loss computation."""
        z_q, indices, z_e, z_q_raw = self.encode_for_loss(x)
        return self.decode(z_q), z_q_raw, z_e, indices