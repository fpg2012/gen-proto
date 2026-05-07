from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

from maskgit import MaskGIT
from vqvae import VQVAE, vqvae_loss


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


def build_vqvae(args: argparse.Namespace) -> VQVAE:
    """Construct the VQ-VAE from command-line arguments."""
    return VQVAE(
        num_codes=args.num_codes,
        code_dim=args.code_dim,
        hidden_dim=args.vq_hidden_dim,
        commitment_cost=args.commitment_cost,
    )


def build_maskgit(args: argparse.Namespace) -> MaskGIT:
    """Construct the MaskGIT transformer from command-line arguments."""
    return MaskGIT(
        num_codes=args.num_codes,
        seq_len=49,
        dim=args.maskgit_dim,
        depth=args.maskgit_depth,
        heads=args.maskgit_heads,
        class_dropout=args.class_dropout,
    )


def train_vqvae(
    args: argparse.Namespace,
    device: torch.device,
    writer: SummaryWriter | None = None,
) -> VQVAE:
    """Train the VQ-VAE stage and save reconstruction previews."""
    loader = get_loader(args.data_dir, args.batch_size, args.workers)
    val_loader = get_loader(args.data_dir, 64, args.workers, train=False)
    fixed_x = next(iter(val_loader))[0].to(device)
    model = build_vqvae(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.vqvae_epochs + 1):
        model.train()
        for x, _ in loader:
            x = x.to(device)
            logits, z_q_raw, z_e, indices = model(x)
            loss, recon, vq = vqvae_loss(logits, x, z_q_raw, z_e, args.commitment_cost)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if writer is not None:
                writer.add_scalar("vqvae/loss", loss.item(), step)
                writer.add_scalar("vqvae/recon_bce", recon.item(), step)
                writer.add_scalar("vqvae/vq", vq.item(), step)
                writer.add_scalar("vqvae/code_usage", indices.unique().numel(), step)
            if step % args.log_every == 0:
                print(
                    f"vqvae epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"recon={recon.item():.4f} vq={vq.item():.4f}"
                )

        model.eval()
        with torch.no_grad():
            logits, _, _, indices = model(fixed_x)
            recon = torch.sigmoid(logits)
            recon_grid = save_image_grid(recon.cpu(), args.out_dir / f"vqvae_recon_epoch_{epoch}.png")
            if writer is not None:
                writer.add_image("vqvae/input", image_grid(fixed_x[:32].cpu()), epoch)
                writer.add_image("vqvae/reconstruction", recon_grid, epoch)
                writer.add_histogram("vqvae/codes", indices.float().cpu(), epoch)
        torch.save(model.state_dict(), args.out_dir / "vqvae.pt")
    return model


def load_or_train_vqvae(
    args: argparse.Namespace,
    device: torch.device,
    writer: SummaryWriter | None = None,
) -> VQVAE:
    """Load a saved VQ-VAE unless forced to retrain it."""
    model = build_vqvae(args).to(device)
    ckpt = args.out_dir / "vqvae.pt"
    if ckpt.exists() and not args.force_vqvae:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded VQ-VAE from {ckpt}")
        return model
    return train_vqvae(args, device, writer)


def flatten_codes(indices: torch.Tensor) -> torch.Tensor:
    """Flatten a ``B H W`` token grid into a MaskGIT sequence ``B (H W)``."""
    return rearrange(indices, "b h w -> b (h w)")


def train_maskgit(
    args: argparse.Namespace,
    device: torch.device,
    vqvae: VQVAE,
    conditional: bool,
    writer: SummaryWriter | None = None,
) -> MaskGIT:
    """Train unconditional or class-conditional MaskGIT over frozen VQ tokens."""
    loader = get_loader(args.data_dir, args.batch_size, args.workers)
    model = build_maskgit(args).to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if conditional:
        ckpt = args.out_dir / "maskgit_uncond.pt"
        if ckpt.exists() and not args.force_maskgit:
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"continued conditional MaskGIT from {ckpt}")
    else:
        ckpt = args.out_dir / "maskgit_uncond.pt"
        if ckpt.exists() and not args.force_maskgit:
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"loaded unconditional MaskGIT from {ckpt}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    vqvae.eval()
    ckpt_name = "maskgit_cond.pt" if conditional else "maskgit_uncond.pt"
    tag = "maskgit_cond" if conditional else "maskgit_uncond"
    epochs = args.cond_epochs if conditional else args.uncond_epochs

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                _, indices, _ = vqvae.encode(x)
                tokens = flatten_codes(indices)
            masked, mask, mask_ratio = MaskGIT.random_mask(tokens, model.mask_token)
            logits = model(masked, mask_ratio, labels if conditional else None)
            loss = F.cross_entropy(logits[mask], tokens[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if writer is not None:
                writer.add_scalar(f"{tag}/loss", loss.item(), step)
                writer.add_scalar(f"{tag}/mask_ratio", mask.float().mean().item(), step)
            if step % args.log_every == 0:
                print(f"{tag} epoch={epoch} step={step} loss={loss.item():.4f}")

        torch.save(model.state_dict(), args.out_dir / ckpt_name)
        samples = make_samples(args, device, vqvae, model, conditional, epoch)
        if writer is not None:
            nrow = args.samples_per_digit if conditional else 8
            writer.add_image(f"{tag}/samples", image_grid(samples.cpu(), nrow=nrow), epoch)
    return model


@torch.no_grad()
def make_samples(
    args: argparse.Namespace,
    device: torch.device,
    vqvae: VQVAE,
    model: MaskGIT,
    conditional: bool,
    epoch: int,
) -> torch.Tensor:
    """Sample token grids from MaskGIT, decode them with VQ-VAE, and save previews."""
    vqvae.eval()
    model.eval()
    if conditional:
        labels = repeat(torch.arange(10, device=device), "digit -> (digit sample)", sample=args.samples_per_digit)
        batch_size = labels.numel()
        tokens = model.sample(
            batch_size,
            labels=labels,
            steps=args.sample_steps,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            topk=args.topk,
            device=device,
        )
        no_cfg_tokens = model.sample(
            batch_size,
            labels=labels,
            steps=args.sample_steps,
            cfg_scale=1.0,
            temperature=args.temperature,
            topk=args.topk,
            device=device,
        )
        token_grid = rearrange(tokens, "b (h w) -> b h w", h=7, w=7)
        no_cfg_token_grid = rearrange(no_cfg_tokens, "b (h w) -> b h w", h=7, w=7)
        x = torch.sigmoid(vqvae.decode_indices(token_grid))
        x_no_cfg = torch.sigmoid(vqvae.decode_indices(no_cfg_token_grid))
        save_image_grid(
            x_no_cfg.cpu(),
            args.out_dir / f"maskgit_cond_samples_no_cfg_epoch_{epoch}.png",
            nrow=args.samples_per_digit,
        )
        save_image_grid(
            x.cpu(),
            args.out_dir / f"maskgit_cond_samples_cfg_{args.cfg_scale:g}_epoch_{epoch}.png",
            nrow=args.samples_per_digit,
        )
        return torch.cat([x_no_cfg, x], dim=0)

    batch_size = args.sample_count
    tokens = model.sample(
        batch_size,
        labels=None,
        steps=args.sample_steps,
        cfg_scale=1.0,
        temperature=args.temperature,
        topk=args.topk,
        device=device,
    )
    token_grid = rearrange(tokens, "b (h w) -> b h w", h=7, w=7)
    x = torch.sigmoid(vqvae.decode_indices(token_grid))
    save_image_grid(x.cpu(), args.out_dir / f"maskgit_uncond_samples_epoch_{epoch}.png", nrow=8)
    return x


def load_maskgit(args: argparse.Namespace, device: torch.device, conditional: bool) -> MaskGIT:
    """Load a saved unconditional or conditional MaskGIT checkpoint."""
    model = build_maskgit(args).to(device)
    ckpt = args.out_dir / ("maskgit_cond.pt" if conditional else "maskgit_uncond.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"loaded MaskGIT from {ckpt}")
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line options for VQ-VAE and MaskGIT training."""
    parser = argparse.ArgumentParser(description="Train VQ-VAE and MaskGIT on MNIST.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mnist_vqvae_maskgit"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vqvae-epochs", type=int, default=10)
    parser.add_argument("--num-codes", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--vq-hidden-dim", type=int, default=128)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--force-vqvae", action="store_true")

    parser.add_argument("--uncond-epochs", type=int, default=20)
    parser.add_argument("--cond-epochs", type=int, default=10)
    parser.add_argument("--maskgit-dim", type=int, default=192)
    parser.add_argument("--maskgit-depth", type=int, default=6)
    parser.add_argument("--maskgit-heads", type=int, default=6)
    parser.add_argument("--class-dropout", type=float, default=0.1)
    parser.add_argument("--force-maskgit", action="store_true")

    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--samples-per-digit", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=2.0)

    parser.add_argument(
        "--stage",
        choices=["vqvae", "uncond", "cond", "all", "sample-uncond", "sample-cond"],
        default="all",
        help="all trains VQ-VAE, unconditional MaskGIT, then continues with conditional MaskGIT.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested VQ-VAE, MaskGIT, or sampling stage."""
    args = parse_args()
    if args.log_dir is None:
        args.log_dir = args.out_dir / "tensorboard"
    if args.topk <= 0:
        args.topk = None
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"using device: {device}")
    writer = SummaryWriter(args.log_dir)
    writer.add_text("config/args", "\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    print(f"tensorboard log dir: {args.log_dir}")

    try:
        if args.stage == "vqvae":
            train_vqvae(args, device, writer)
            return

        vqvae = load_or_train_vqvae(args, device, writer)
        if args.stage in {"uncond", "all"}:
            train_maskgit(args, device, vqvae, conditional=False, writer=writer)
            if args.stage == "uncond":
                return

        if args.stage in {"cond", "all"}:
            train_maskgit(args, device, vqvae, conditional=True, writer=writer)
            return

        conditional = args.stage == "sample-cond"
        model = load_maskgit(args, device, conditional=conditional)
        samples = make_samples(args, device, vqvae, model, conditional=conditional, epoch=0)
        nrow = args.samples_per_digit if conditional else 8
        writer.add_image(f"sample/{args.stage}", image_grid(samples.cpu(), nrow=nrow), 0)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
