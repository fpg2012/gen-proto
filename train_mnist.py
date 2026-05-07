from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

from diffusion import DiT, Diffusion
from vae import ConvVAE, vae_loss


def default_device() -> str:
    """Choose the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_loader(data_dir: Path, batch_size: int, workers: int, train: bool = True) -> DataLoader:
    """Build an MNIST dataloader with the project default transform."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(str(data_dir), train=train, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def image_grid(x: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Convert a batch of images into a clamped torchvision grid."""
    return utils.make_grid(x.clamp(0, 1), nrow=nrow)


def save_image_grid(x: torch.Tensor, path: Path, nrow: int = 8) -> torch.Tensor:
    """Save a clamped image grid and return the grid tensor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = image_grid(x, nrow=nrow)
    utils.save_image(grid, str(path))
    return grid


def train_vae(
    args: argparse.Namespace,
    device: torch.device,
    writer: SummaryWriter | None = None,
) -> ConvVAE:
    """Train the convolutional VAE stage and save reconstruction previews."""
    loader = get_loader(args.data_dir, args.batch_size, args.workers)
    val_loader = get_loader(args.data_dir, 64, args.workers, train=False)
    fixed_x = next(iter(val_loader))[0].to(device)
    vae = ConvVAE(args.latent_channels).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.vae_epochs + 1):
        vae.train()
        for x, _ in loader:
            x = x.to(device)
            logits, mu, logvar = vae(x)
            loss, recon, kl = vae_loss(logits, x, mu, logvar, args.kl_beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if writer is not None:
                writer.add_scalar("vae/loss", loss.item(), step)
                writer.add_scalar("vae/recon_bce", recon.item(), step)
                writer.add_scalar("vae/kl", kl.item(), step)
                writer.add_scalar("vae/lr", opt.param_groups[0]["lr"], step)
            if step % args.log_every == 0:
                print(
                    f"vae epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"recon={recon.item():.4f} kl={kl.item():.4f}"
                )

        vae.eval()
        with torch.no_grad():
            logits, mu, logvar = vae(fixed_x)
            recon = torch.sigmoid(logits)
            comparison = torch.cat([fixed_x[:32], recon[:32]], dim=0)
            recon_grid = save_image_grid(recon.cpu(), args.out_dir / f"vae_recon_epoch_{epoch}.png")
            if writer is not None:
                writer.add_image("vae/input", image_grid(fixed_x[:32].cpu()), epoch)
                writer.add_image("vae/reconstruction", recon_grid, epoch)
                writer.add_image("vae/input_vs_recon", image_grid(comparison.cpu(), nrow=8), epoch)
                writer.add_histogram("vae/mu", mu.detach().cpu(), epoch)
                writer.add_histogram("vae/logvar", logvar.detach().cpu(), epoch)
        torch.save(vae.state_dict(), args.out_dir / "vae.pt")

    return vae


def load_or_train_vae(
    args: argparse.Namespace,
    device: torch.device,
    writer: SummaryWriter | None = None,
) -> ConvVAE:
    """Load a saved VAE unless forced to retrain it."""
    vae = ConvVAE(args.latent_channels).to(device)
    ckpt = args.out_dir / "vae.pt"
    if ckpt.exists() and not args.force_vae:
        vae.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded VAE from {ckpt}")
        return vae
    return train_vae(args, device, writer)


def train_dit(
    args: argparse.Namespace,
    device: torch.device,
    vae: ConvVAE,
    writer: SummaryWriter | None = None,
) -> DiT:
    """Train DiT to predict VAE latent noise at randomly sampled timesteps."""
    loader = get_loader(args.data_dir, args.batch_size, args.workers)
    dit = DiT(
        latent_channels=args.latent_channels,
        latent_size=7,
        dim=args.dit_dim,
        depth=args.dit_depth,
        heads=args.dit_heads,
        class_dropout=args.class_dropout,
    ).to(device)
    diffusion = Diffusion(args.timesteps, str(device))
    opt = torch.optim.AdamW(dit.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vae.eval()

    step = 0
    for epoch in range(1, args.dit_epochs + 1):
        dit.train()
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z = ConvVAE.reparameterize(mu, logvar) * args.latent_scale
            t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)
            noisy, noise = diffusion.q_sample(z, t)
            pred = dit(noisy, t, labels)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            opt.step()
            step += 1
            if writer is not None:
                writer.add_scalar("dit/loss", loss.item(), step)
                writer.add_scalar("dit/lr", opt.param_groups[0]["lr"], step)
            if step % args.log_every == 0:
                print(f"dit epoch={epoch} step={step} loss={loss.item():.4f}")

        torch.save(dit.state_dict(), args.out_dir / "dit.pt")
        samples = make_samples(args, device, vae, dit, diffusion, epoch)
        if writer is not None:
            writer.add_image(
                "dit/samples_by_label",
                image_grid(samples.cpu(), nrow=args.samples_per_digit),
                epoch,
            )
            writer.add_histogram("dit/noisy_latent", noisy.detach().cpu(), epoch)
            writer.add_histogram("dit/pred_noise", pred.detach().cpu(), epoch)
            writer.add_histogram("dit/target_noise", noise.detach().cpu(), epoch)

    return dit


@torch.no_grad()
def make_samples(
    args: argparse.Namespace,
    device: torch.device,
    vae: ConvVAE,
    dit: DiT,
    diffusion: Diffusion,
    epoch: int,
) -> torch.Tensor:
    """Sample labeled MNIST digits from the trained latent DiT and decode them."""
    vae.eval()
    dit.eval()
    labels = repeat(torch.arange(10, device=device), "digit -> (digit sample)", sample=args.samples_per_digit)
    z = diffusion.sample(
        dit,
        (labels.numel(), args.latent_channels, 7, 7),
        labels,
        cfg_scale=args.cfg_scale,
    )
    x = torch.sigmoid(vae.decode(z / args.latent_scale))
    save_image_grid(x.cpu(), args.out_dir / f"dit_samples_epoch_{epoch}.png", nrow=args.samples_per_digit)
    return x


def parse_args() -> argparse.Namespace:
    """Parse command-line options for VAE and DiT training."""
    parser = argparse.ArgumentParser(description="Train a small VAE and DiT on MNIST.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist_vae_dit"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vae-epochs", type=int, default=10)
    parser.add_argument("--kl-beta", type=float, default=1e-3)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument("--force-vae", action="store_true")

    parser.add_argument("--dit-epochs", type=int, default=20)
    parser.add_argument("--dit-dim", type=int, default=192)
    parser.add_argument("--dit-depth", type=int, default=6)
    parser.add_argument("--dit-heads", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--class-dropout", type=float, default=0.1)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--samples-per-digit", type=int, default=8)

    parser.add_argument(
        "--stage",
        choices=["vae", "dit", "all", "sample"],
        default="all",
        help="vae trains only the VAE; dit loads/trains VAE then trains DiT; sample uses saved checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested training or sampling stage."""
    args = parse_args()
    if args.log_dir is None:
        args.log_dir = args.out_dir / "tensorboard"
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"using device: {device}")
    writer = SummaryWriter(args.log_dir)
    writer.add_text("config/args", "\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    print(f"tensorboard log dir: {args.log_dir}")

    try:
        if args.stage == "vae":
            train_vae(args, device, writer)
            return

        vae = load_or_train_vae(args, device, writer)
        if args.stage in {"dit", "all"}:
            train_dit(args, device, vae, writer)
            return

        dit = DiT(
            latent_channels=args.latent_channels,
            dim=args.dit_dim,
            depth=args.dit_depth,
            heads=args.dit_heads,
            class_dropout=args.class_dropout,
        ).to(device)
        ckpt = args.out_dir / "dit.pt"
        dit.load_state_dict(torch.load(ckpt, map_location=device))
        samples = make_samples(args, device, vae, dit, Diffusion(args.timesteps, str(device)), epoch=0)
        writer.add_image("dit/samples_by_label", image_grid(samples.cpu(), nrow=args.samples_per_digit), 0)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
