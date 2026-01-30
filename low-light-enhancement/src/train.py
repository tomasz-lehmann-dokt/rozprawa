"""
Training script for Swin-UNet low-light image enhancement.

Implements training with composite perceptual loss, cosine annealing schedule,
and validation on LOL/LOL-v2 benchmarks.
"""

import argparse
from pathlib import Path
from collections import OrderedDict
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SwinTransformerSys
from dataset import LowLightDataset
from loss import CombinedLoss
from metrics import compute_metrics


class AverageMeter:
    """Running average tracker."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def train_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run single training epoch."""
    model.train()
    loss_meter = AverageMeter()

    with tqdm(loader, desc=f"Train [{epoch}]") as pbar:
        for low, high, _ in pbar:
            low = low.to(device)
            high = high.to(device)

            optimizer.zero_grad()
            output = model(low)
            loss = criterion(output, high)

            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), low.size(0))
            pbar.set_postfix(OrderedDict(loss=f"{loss_meter.avg:.4f}"))

    return loss_meter.avg


@torch.no_grad()
def validate(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    desc: str = "Valid",
) -> tuple:
    """Run validation and compute metrics."""
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with tqdm(loader, desc=desc) as pbar:
        for low, high, _ in pbar:
            low = low.to(device)
            high = high.to(device)

            output = model(low)
            psnr, ssim = compute_metrics(output, high)

            psnr_meter.update(psnr)
            ssim_meter.update(ssim)

            pbar.set_postfix(
                OrderedDict(
                    psnr=f"{psnr_meter.avg:.2f}",
                    ssim=f"{ssim_meter.avg:.4f}",
                )
            )

    return psnr_meter.avg, ssim_meter.avg


def main(args: argparse.Namespace) -> None:
    """Main training loop."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = SwinTransformerSys(
        img_size=args.crop_size,
        patch_size=4,
        in_chans=3,
        num_classes=3,
        embed_dim=args.embed_dim,
        depths=args.depths,
        window_size=args.window_size,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = CombinedLoss(
        device=device,
        alpha_lpips=args.alpha_lpips,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Datasets
    train_dataset = LowLightDataset(
        split="train",
        crop_size=(args.crop_size, args.crop_size),
        datasets_dir=args.datasets_dir,
        use_lol=True,
        use_lol_v2=True,
        use_sid=args.use_sid,
        sid_selection=args.sid_selection,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    # Validation loaders for each LOL variant
    val_loaders = {}
    for subset in ["lol_v1", "lol_v2_real", "lol_v2_synth"]:
        if subset == "lol_v1":
            ds = LowLightDataset(
                split="valid",
                crop_size=(args.crop_size, args.crop_size),
                datasets_dir=args.datasets_dir,
                use_lol=True,
                use_lol_v2=False,
                use_sid=False,
            )
        elif subset == "lol_v2_real":
            ds = LowLightDataset(
                split="valid",
                crop_size=(args.crop_size, args.crop_size),
                datasets_dir=args.datasets_dir,
                use_lol=False,
                use_lol_v2=True,
                use_sid=False,
                lol_v2_subsets=["Real_captured"],
            )
        else:
            ds = LowLightDataset(
                split="valid",
                crop_size=(args.crop_size, args.crop_size),
                datasets_dir=args.datasets_dir,
                use_lol=False,
                use_lol_v2=True,
                use_sid=False,
                lol_v2_subsets=["Synthetic"],
            )
        val_loaders[subset] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

    print(f"Train samples: {len(train_dataset)}")
    for name, loader in val_loaders.items():
        print(f"Val {name}: {len(loader.dataset)}")

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate on each subset
        val_results = {}
        total_loss = 0
        for name, loader in val_loaders.items():
            psnr, ssim = validate(loader, model, device, desc=f"Val {name}")
            val_results[f"{name}_psnr"] = psnr
            val_results[f"{name}_ssim"] = ssim
            total_loss += 1 - ssim  # Use SSIM as proxy for loss

        avg_loss = total_loss / len(val_loaders)

        print(f"\nEpoch {epoch}:")
        for name in val_loaders:
            print(
                f"  {name}: PSNR={val_results[f'{name}_psnr']:.2f}, SSIM={val_results[f'{name}_ssim']:.4f}"
            )

        history.append({"epoch": epoch, "train_loss": train_loss, **val_results})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_dir / "swin_unet_best.pth",
            )
            print(f"Saved best model")

        scheduler.step()
        torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(checkpoint_dir / "training_log.csv", index=False)
    print(f"\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Swin-UNet for low-light enhancement"
    )
    parser.add_argument(
        "--datasets_dir", type=str, required=True, help="Datasets root directory"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Model architecture
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs=4, default=[2, 4, 8, 2])
    parser.add_argument("--depths_decoder", type=int, nargs=4, default=[2, 8, 4, 2])
    parser.add_argument("--num_heads", type=int, nargs=4, default=[4, 8, 16, 32])
    parser.add_argument("--window_size", type=int, default=7)

    # Loss weights
    parser.add_argument("--alpha_lpips", type=float, default=0.1)

    # SID configuration
    parser.add_argument("--use_sid", action="store_true")
    parser.add_argument(
        "--sid_selection",
        type=str,
        default="darkest",
        choices=["darkest", "three_darkest", "random"],
    )

    main(parser.parse_args())
