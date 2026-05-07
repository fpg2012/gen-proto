from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def vae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return beta-VAE loss plus reconstruction BCE and KL terms."""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


class ConvVAE(nn.Module):
    """Small convolutional VAE that compresses MNIST images into 7x7 latent maps."""

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        # Number of channels in the continuous latent map consumed by DiT.
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(128, latent_channels, 1)
        self.to_logvar = nn.Conv2d(128, latent_channels, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images into latent Gaussian mean and clipped log variance."""
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-20, 10)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latents with the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent maps into image logits."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, sample, and decode a batch of images."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar