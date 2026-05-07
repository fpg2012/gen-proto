from __future__ import annotations

import argparse
from pathlib import Path

import torch
from einops import rearrange, repeat
from PIL import Image
from torchvision import utils

from diffusion import ConvVAE, DiT, Diffusion


def default_device() -> str:
    """Choose the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a CHW tensor in [0, 1] to a PIL RGB image."""
    x = x.detach().cpu().clamp(0, 1)
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape {tuple(x.shape)}")
    if x.shape[0] == 1:
        x = repeat(x, "1 h w -> c h w", c=3)
    x = (rearrange(x, "c h w -> h w c").numpy() * 255).astype("uint8")
    return Image.fromarray(x)


def make_grid(x: torch.Tensor, nrow: int) -> torch.Tensor:
    """Convert a batch of images into a clamped torchvision grid."""
    return utils.make_grid(x.clamp(0, 1), nrow=nrow, padding=2)


def save_grid(x: torch.Tensor, path: Path, nrow: int) -> torch.Tensor:
    """Save a clamped image grid and return the grid tensor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(x, nrow=nrow)
    utils.save_image(grid, str(path))
    return grid


def build_save_steps(timesteps: int, frames: int) -> set[int]:
    """Choose evenly spaced timesteps to save from a reverse diffusion chain."""
    if frames <= 1:
        return {0}
    steps = torch.linspace(timesteps - 1, 0, frames).round().long().tolist()
    return set(int(step) for step in steps)


@torch.no_grad()
def sample_trajectory(
    args: argparse.Namespace,
    device: torch.device,
    vae: ConvVAE,
    dit: DiT,
    diffusion: Diffusion,
) -> list[tuple[int, torch.Tensor]]:
    """Run reverse diffusion while collecting decoded frames at selected timesteps."""
    labels = repeat(torch.arange(10, device=device), "digit -> (digit sample)", sample=args.samples_per_digit)
    shape = (labels.numel(), args.latent_channels, args.latent_size, args.latent_size)
    x = torch.randn(shape, device=device)
    save_steps = build_save_steps(args.timesteps, args.frames)
    trajectory: list[tuple[int, torch.Tensor]] = []

    vae.eval()
    dit.eval()
    for i in reversed(range(args.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = dit(x, t, labels)
        if args.cfg_scale != 1.0:
            uncond = dit(x, t, labels, force_drop_labels=True)
            eps = uncond + args.cfg_scale * (eps - uncond)

        beta = diffusion.betas[i]
        alpha = diffusion.alphas[i]
        alpha_bar = diffusion.alpha_bars[i]
        x = (x - beta / torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha)
        if i > 0:
            x = x + torch.sqrt(beta) * torch.randn_like(x)

        if i in save_steps:
            decoded = torch.sigmoid(vae.decode(x / args.latent_scale))
            trajectory.append((i, decoded.cpu()))

    trajectory.sort(key=lambda item: item[0], reverse=True)
    return trajectory


def save_trajectory(args: argparse.Namespace, trajectory: list[tuple[int, torch.Tensor]]) -> None:
    """Save individual frames, an animated GIF, and a contact sheet."""
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    frame_grids: list[torch.Tensor] = []

    for frame_idx, (timestep, images) in enumerate(trajectory):
        grid = save_grid(
            images,
            args.viz_dir / f"frame_{frame_idx:03d}_t_{timestep:04d}.png",
            nrow=args.samples_per_digit,
        )
        frame_grids.append(grid)
        frames.append(tensor_to_pil(grid))

    if frames:
        frames[0].save(
            args.viz_dir / "sampling.gif",
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / args.fps),
            loop=0,
        )
        strip = torch.stack(frame_grids, dim=0)
        save_grid(strip, args.viz_dir / "sampling_contact_sheet.png", nrow=args.contact_nrow)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for sampling visualization."""
    parser = argparse.ArgumentParser(description="Visualize a trained MNIST latent DiT sampling trajectory.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mnist_vae_dit"))
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-size", type=int, default=7)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument("--dit-dim", type=int, default=192)
    parser.add_argument("--dit-depth", type=int, default=6)
    parser.add_argument("--dit-heads", type=int, default=6)
    parser.add_argument("--class-dropout", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=2.0)

    parser.add_argument("--samples-per-digit", type=int, default=4)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--contact-nrow", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Load checkpoints, sample a trajectory, and write visualization files."""
    args = parse_args()
    if args.viz_dir is None:
        args.viz_dir = args.run_dir / "sampling_viz"

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    vae_ckpt = args.run_dir / "vae.pt"
    dit_ckpt = args.run_dir / "dit.pt"
    print(f"using device: {device}")
    print(f"loading VAE: {vae_ckpt}")
    print(f"loading DiT: {dit_ckpt}")

    vae = ConvVAE(args.latent_channels).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    dit = DiT(
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        dim=args.dit_dim,
        depth=args.dit_depth,
        heads=args.dit_heads,
        class_dropout=args.class_dropout,
    ).to(device)
    dit.load_state_dict(torch.load(dit_ckpt, map_location=device))

    diffusion = Diffusion(args.timesteps, str(device))
    trajectory = sample_trajectory(args, device, vae, dit, diffusion)
    save_trajectory(args, trajectory)
    print(f"saved sampling visualization to {args.viz_dir}")


if __name__ == "__main__":
    main()
