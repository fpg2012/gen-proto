from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from einops import rearrange, repeat
from PIL import Image
from torchvision import utils

from maskgit import MaskGIT
from train_maskgit_mnist import build_maskgit, build_vqvae
from vqvae import VQVAE


def default_device() -> str:
    """Choose the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a CHW tensor in [0, 1] to a PIL image."""
    x = x.detach().cpu().clamp(0, 1)
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape {tuple(x.shape)}")
    if x.shape[0] == 1:
        x = repeat(x, "1 h w -> c h w", c=3)
    x = rearrange(x, "c h w -> h w c").numpy()
    return Image.fromarray((x * 255).astype("uint8"))


def save_grid(x: torch.Tensor, path: Path, nrow: int) -> torch.Tensor:
    """Save a torchvision grid and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = utils.make_grid(x, nrow=nrow, padding=2)
    utils.save_image(grid, str(path))
    return grid


def token_grid_to_image(tokens: torch.Tensor, num_codes: int, nrow: int, side: int) -> torch.Tensor:
    """Render token ids as a normalized grayscale grid."""
    token_grid = rearrange(tokens, "b (h w) -> b 1 h w", h=side, w=side)
    token_grid = token_grid.float() / float(max(num_codes, 1))
    return utils.make_grid(token_grid, nrow=nrow, padding=2, normalize=False)


def decode_tokens(vqvae: VQVAE, tokens: torch.Tensor) -> torch.Tensor:
    """Decode a batch of token sequences into image probabilities."""
    token_grid = tokens.clone()
    token_grid = torch.where(token_grid == vqvae.num_codes, torch.zeros_like(token_grid), token_grid)
    token_grid = rearrange(token_grid, "b (h w) -> b h w", h=vqvae.latent_size, w=vqvae.latent_size)
    return torch.sigmoid(vqvae.decode_indices(token_grid))


@torch.no_grad()
def sample_trajectory(
    args: argparse.Namespace,
    device: torch.device,
    vqvae: VQVAE,
    model: MaskGIT,
    labels: torch.Tensor | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Collect decoded frames and token grids for every MaskGIT step."""
    vqvae.eval()
    model.eval()

    batch_size = args.sample_count if labels is None else labels.numel()
    tokens = torch.full((batch_size, model.seq_len), model.mask_token, device=device, dtype=torch.long)
    decoded_frames: list[torch.Tensor] = []
    token_frames: list[torch.Tensor] = []

    def record_frame(current_tokens: torch.Tensor) -> None:
        decoded = decode_tokens(vqvae, current_tokens).cpu()
        decoded_frames.append(decoded)
        token_frames.append(current_tokens.cpu())

    record_frame(tokens)

    for step in range(args.sample_steps):
        mask = tokens == model.mask_token
        mask_ratio = mask.float().mean(dim=1)
        logits = model(tokens, mask_ratio, labels)
        if labels is not None and args.cfg_scale != 1.0:
            uncond = model(tokens, mask_ratio, labels, force_uncond=True)
            logits = uncond + args.cfg_scale * (logits - uncond)

        logits = logits / max(args.temperature, 1e-6)
        if args.topk is not None and args.topk > 0:
            values, _ = logits.topk(min(args.topk, logits.shape[-1]), dim=-1)
            logits = logits.masked_fill(logits < values[..., -1:].contiguous(), -torch.inf)

        probs = logits.softmax(dim=-1)
        sampled = torch.distributions.Categorical(probs=probs).sample()
        conf = rearrange(probs.gather(-1, rearrange(sampled, "b n -> b n 1")), "b n 1 -> b n")
        conf = conf.masked_fill(~mask, torch.inf)
        tokens = torch.where(mask, sampled, tokens)
        record_frame(tokens)

        if step == args.sample_steps - 1:
            break

        keep_ratio = math.cos(0.5 * math.pi * (step + 1) / args.sample_steps)
        next_mask_count = (
            model.seq_len * keep_ratio * repeat(torch.ones((), device=device), "-> b", b=batch_size)
        ).long()
        next_mask_count = torch.minimum(next_mask_count, mask.sum(dim=1) - 1).clamp(min=0)
        order = conf.argsort(dim=1)
        rank = order.argsort(dim=1)
        remask = rank < rearrange(next_mask_count, "b -> b 1")
        tokens = torch.where(remask, torch.full_like(tokens, model.mask_token), tokens)

    return decoded_frames, token_frames


def save_trajectory(
    args: argparse.Namespace,
    latent_size: int,
    decoded_frames: list[torch.Tensor],
    token_frames: list[torch.Tensor],
) -> None:
    """Save per-step PNGs, GIFs, and contact sheets."""
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    decoded_pils: list[Image.Image] = []
    token_pils: list[Image.Image] = []
    decoded_grids: list[torch.Tensor] = []
    token_grids: list[torch.Tensor] = []

    for step, (decoded, tokens) in enumerate(zip(decoded_frames, token_frames)):
        decoded_grid = save_grid(decoded, args.viz_dir / f"decoded_step_{step:03d}.png", nrow=args.nrow)
        token_grid = save_grid(
            token_grid_to_image(tokens, args.num_codes, nrow=args.nrow, side=latent_size),
            args.viz_dir / f"tokens_step_{step:03d}.png",
            nrow=args.nrow,
        )
        decoded_grids.append(decoded_grid)
        token_grids.append(token_grid)
        decoded_pils.append(tensor_to_pil(decoded_grid))
        token_pils.append(tensor_to_pil(token_grid))

    if decoded_pils:
        decoded_pils[0].save(
            args.viz_dir / "decoded_sampling.gif",
            save_all=True,
            append_images=decoded_pils[1:],
            duration=int(1000 / args.fps),
            loop=0,
        )
    if token_pils:
        token_pils[0].save(
            args.viz_dir / "token_sampling.gif",
            save_all=True,
            append_images=token_pils[1:],
            duration=int(1000 / args.fps),
            loop=0,
        )

    if decoded_grids:
        decoded_strip = torch.stack(decoded_grids, dim=0)
        save_grid(decoded_strip, args.viz_dir / "decoded_contact_sheet.png", nrow=1)
    if token_grids:
        token_strip = torch.stack(token_grids, dim=0)
        save_grid(token_strip, args.viz_dir / "token_contact_sheet.png", nrow=1)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for MaskGIT visualization."""
    parser = argparse.ArgumentParser(description="Visualize multi-step MaskGIT generation on MNIST.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mnist_vqvae_maskgit"))
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--vqvae-ckpt", type=Path, default=None)
    parser.add_argument("--maskgit-ckpt", type=Path, default=None)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-codes", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--vq-hidden-dim", type=int, default=128)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--maskgit-dim", type=int, default=192)
    parser.add_argument("--maskgit-depth", type=int, default=6)
    parser.add_argument("--maskgit-heads", type=int, default=6)
    parser.add_argument("--class-dropout", type=float, default=0.1)

    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--samples-per-digit", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--label", type=int, default=-1)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def build_labels(args: argparse.Namespace, device: torch.device) -> torch.Tensor | None:
    """Build optional conditional labels for visualization."""
    if not args.conditional:
        return None
    if args.label >= 0:
        labels = torch.full((args.sample_count,), args.label, device=device, dtype=torch.long)
        return labels
    labels = repeat(
        torch.arange(10, device=device),
        "digit -> (digit sample)",
        sample=args.samples_per_digit,
    )
    return labels


def main() -> None:
    """Load checkpoints, run iterative sampling, and save the visualization."""
    args = parse_args()
    if args.topk <= 0:
        args.topk = None
    if args.viz_dir is None:
        args.viz_dir = args.run_dir / "maskgit_viz"
    if args.vqvae_ckpt is None:
        args.vqvae_ckpt = args.run_dir / "vqvae.pt"
    if args.maskgit_ckpt is None:
        args.maskgit_ckpt = args.run_dir / ("maskgit_cond.pt" if args.conditional else "maskgit_uncond.pt")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"using device: {device}")
    print(f"loading VQ-VAE: {args.vqvae_ckpt}")
    print(f"loading MaskGIT: {args.maskgit_ckpt}")

    vqvae = build_vqvae(args).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_ckpt, map_location=device))

    model = build_maskgit(args).to(device)
    model.load_state_dict(torch.load(args.maskgit_ckpt, map_location=device))

    labels = build_labels(args, device)
    decoded_frames, token_frames = sample_trajectory(args, device, vqvae, model, labels)
    save_trajectory(args, vqvae.latent_size, decoded_frames, token_frames)
    print(f"saved MaskGIT visualization to {args.viz_dir}")


if __name__ == "__main__":
    main()
