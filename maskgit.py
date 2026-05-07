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


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal scalar embedding followed by an MLP projection."""

    def __init__(self, dim: int):
        super().__init__()
        # Output embedding width used by the transformer conditioning path.
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of scalar values with shape ``B``."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device, dtype=torch.float32) / half
        )
        args = rearrange(x.float(), "b -> b 1") * rearrange(freqs, "d -> 1 d")
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class MaskGIT(nn.Module):
    """Bidirectional transformer that predicts masked VQ tokens in parallel."""

    def __init__(
        self,
        num_codes: int = 128,
        seq_len: int = 49,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        class_dropout: float = 0.1,
    ):
        super().__init__()
        # Real codebook ids are [0, num_codes); num_codes itself is reserved for [MASK].
        self.num_codes = num_codes
        self.mask_token = num_codes
        # Number of VQ tokens per image, 7x7 for this MNIST setup.
        self.seq_len = seq_len
        # Number of digit labels; an extra null label supports classifier-free guidance.
        self.num_classes = num_classes
        self.null_label = num_classes
        # Probability of replacing labels by the null label during conditional training.
        self.class_dropout = class_dropout

        self.token_emb = nn.Embedding(num_codes + 1, dim)
        # Learned absolute position embedding over the flattened 7x7 token grid.
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.mask_ratio_emb = SinusoidalEmbedding(dim)
        self.label_emb = nn.Embedding(num_classes + 1, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_codes)
        nn.init.normal_(self.pos, std=0.02)

    def _condition(
        self,
        batch_size: int,
        mask_ratio: torch.Tensor,
        labels: torch.Tensor | None,
        force_uncond: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Build additive conditioning from mask ratio and optional class labels."""
        cond = self.mask_ratio_emb(mask_ratio)
        if labels is None:
            labels = torch.full((batch_size,), self.null_label, device=device, dtype=torch.long)
        elif self.training or force_uncond:
            drop = torch.rand(labels.shape, device=device) < self.class_dropout
            if force_uncond:
                drop = torch.ones_like(drop, dtype=torch.bool)
            labels = torch.where(drop, torch.full_like(labels, self.null_label), labels)
        cond = cond + self.label_emb(labels)
        return cond

    def forward(
        self,
        tokens: torch.Tensor,
        mask_ratio: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """Predict logits for every token position in a possibly masked sequence."""
        b, n = tokens.shape
        if n != self.seq_len:
            raise ValueError(f"expected sequence length {self.seq_len}, got {n}")
        if mask_ratio is None:
            mask_ratio = (tokens == self.mask_token).float().mean(dim=1)
        cond = self._condition(b, mask_ratio, labels, force_uncond, tokens.device)
        h = self.token_emb(tokens) + self.pos + rearrange(cond, "b d -> b 1 d")
        h = self.blocks(h)
        return self.to_logits(self.norm(h))

    @staticmethod
    def random_mask(tokens: torch.Tensor, mask_token: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask at least one token per sequence and return masked tokens, mask, and ratio."""
        b, n = tokens.shape
        ratio = torch.rand(b, device=tokens.device)
        num_mask = (ratio * n).long().clamp(min=1, max=n)
        order = torch.rand(b, n, device=tokens.device).argsort(dim=1)
        rank = order.argsort(dim=1)
        mask = rank < rearrange(num_mask, "b -> b 1")
        masked = torch.where(mask, torch.full_like(tokens, mask_token), tokens)
        return masked, mask, num_mask.float() / n

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        labels: torch.Tensor | None = None,
        steps: int = 12,
        cfg_scale: float = 1.0,
        temperature: float = 1.0,
        topk: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Iteratively sample VQ token sequences with optional class conditioning and CFG."""
        if device is None:
            device = next(self.parameters()).device
        tokens = torch.full((batch_size, self.seq_len), self.mask_token, device=device, dtype=torch.long)
        if labels is not None:
            labels = labels.to(device)

        for step in range(steps):
            mask = tokens == self.mask_token
            mask_ratio = mask.float().mean(dim=1)
            logits = self(tokens, mask_ratio, labels)
            if labels is not None and cfg_scale != 1.0:
                uncond = self(tokens, mask_ratio, labels, force_uncond=True)
                logits = uncond + cfg_scale * (logits - uncond)
            logits = logits / max(temperature, 1e-6)
            if topk is not None and topk > 0:
                values, _ = logits.topk(min(topk, logits.shape[-1]), dim=-1)
                logits = logits.masked_fill(logits < values[..., -1:].contiguous(), -torch.inf)
            probs = logits.softmax(dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()
            conf = rearrange(probs.gather(-1, rearrange(sampled, "b n -> b n 1")), "b n 1 -> b n")
            conf = conf.masked_fill(~mask, torch.inf)
            tokens = torch.where(mask, sampled, tokens)

            if step == steps - 1:
                break
            keep_ratio = math.cos(0.5 * math.pi * (step + 1) / steps)
            next_mask_count = (self.seq_len * keep_ratio * repeat(torch.ones((), device=device), "-> b", b=batch_size)).long()
            next_mask_count = torch.minimum(next_mask_count, mask.sum(dim=1) - 1).clamp(min=0)
            order = conf.argsort(dim=1)
            rank = order.argsort(dim=1)
            remask = rank < rearrange(next_mask_count, "b -> b 1")
            tokens = torch.where(remask, torch.full_like(tokens, self.mask_token), tokens)

        return tokens
