"""
Training script for Attention U-Net moire removal.

Implements training with combined MSE+SSIM loss and rotating fixed-budget
cross-sampling (RFBCS) for efficient use of imbalanced datasets.
"""

import argparse
from pathlib import Path
from collections import OrderedDict
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AttentionUNet
from dataset import DemoireDataset
from metrics import compute_metrics, SSIM


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


class CombinedLoss(nn.Module):
    """MSE + SSIM composite loss for image restoration."""

    def __init__(self, mse_weight: float = 0.1, ssim_weight: float = 1.0) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIM()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        ssim_loss = torch.clamp(1 - self.ssim(pred, target), 0, 1)
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run single training epoch."""
    model.train()
    loss_meter = AverageMeter()

    with tqdm(loader, desc=f"Train [{epoch}]") as pbar:
        for moire, clean, _ in pbar:
            moire = moire.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(moire)
            loss = criterion(output, clean)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), moire.size(0))
            pbar.set_postfix(OrderedDict(loss=f"{loss_meter.avg:.4f}"))

    return loss_meter.avg


@torch.no_grad()
def validate(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple:
    """Run validation and compute metrics."""
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with tqdm(loader, desc=f"Valid [{epoch}]") as pbar:
        for moire, clean, _ in pbar:
            moire = moire.to(device)
            clean = clean.to(device)

            output = model(moire)
            loss = criterion(output, clean)

            psnr, ssim = compute_metrics(output, clean)

            loss_meter.update(loss.item(), moire.size(0))
            psnr_meter.update(psnr)
            ssim_meter.update(ssim)

            pbar.set_postfix(OrderedDict(
                loss=f"{loss_meter.avg:.4f}",
                psnr=f"{psnr_meter.avg:.2f}",
                ssim=f"{ssim_meter.avg:.4f}",
            ))

    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


def main(args: argparse.Namespace) -> None:
    """Main training loop with RFBCS."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = AttentionUNet().to(device)
    criterion = CombinedLoss(mse_weight=args.mse_weight, ssim_weight=args.ssim_weight)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Create validation loaders (fixed)
    uhdm_val = DemoireDataset(
        split="uhdm_test",
        image_size=args.image_size,
        uhdm_dir=args.uhdm_dir,
    )
    uhdm_loader = DataLoader(
        uhdm_val, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    tip_val = DemoireDataset(
        split="tip_test",
        image_size=args.image_size,
        tip_dir=args.tip_dir,
    )
    tip_loader = DataLoader(
        tip_val, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    print(f"UHDM val: {len(uhdm_val)}, TIP val: {len(tip_val)}")

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # Refresh training dataset each epoch (RFBCS)
        train_dataset = DemoireDataset(
            split="train",
            image_size=args.image_size,
            uhdm_dir=args.uhdm_dir,
            tip_dir=args.tip_dir,
            tip_budget=args.tip_budget,
            seed=random.randint(0, 10000),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        print(f"\nEpoch {epoch}: {len(train_dataset)} training samples")

        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        val_loss, val_psnr, val_ssim = validate(
            uhdm_loader, model, criterion, device, epoch
        )
        tip_loss, tip_psnr, tip_ssim = validate(
            tip_loader, model, criterion, device, epoch
        )

        print(f"UHDM - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        print(f"TIP  - Loss: {tip_loss:.4f}, PSNR: {tip_psnr:.2f}, SSIM: {tip_ssim:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "uhdm_loss": val_loss,
            "uhdm_psnr": val_psnr,
            "uhdm_ssim": val_ssim,
            "tip_loss": tip_loss,
            "tip_psnr": tip_psnr,
            "tip_ssim": tip_ssim,
        })

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_dir / "attention_unet_best.pth",
            )
            print(f"Saved best model (loss={val_loss:.4f})")

        scheduler.step()

    pd.DataFrame(history).to_csv(checkpoint_dir / "training_log.csv", index=False)
    print(f"\nTraining complete. Best UHDM loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention U-Net for demoireing")
    parser.add_argument("--uhdm_dir", type=str, required=True, help="UHDM dataset root")
    parser.add_argument("--tip_dir", type=str, required=True, help="TIP2018 dataset root")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--mse_weight", type=float, default=0.1)
    parser.add_argument("--ssim_weight", type=float, default=1.0)
    parser.add_argument("--tip_budget", type=int, default=15500)
    parser.add_argument("--gpu", type=int, default=0)

    main(parser.parse_args())

