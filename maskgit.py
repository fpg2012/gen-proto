from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


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
