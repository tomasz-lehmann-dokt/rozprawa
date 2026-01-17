"""
Training script for BIQA quality predictor.

Implements training with mixed precision, cosine annealing, and validation.

Architectures:
- ConvNeXt-MLP: ConvNeXt backbone + MLP parameter encoder
- dual-Xception: Xception backbone + MLP parameter encoder
"""

import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ConvNeXtMLP
from model_xception import DualXception
from dataset import KonIQDataset
from metrics import compute_correlation_metrics


class AverageMeter:
    """Computes and stores running average."""

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
        self.avg = self.sum / self.count


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    """Run single training epoch."""
    model.train()
    loss_meter = AverageMeter()

    with tqdm(loader, desc=f"Train [{epoch}]") as pbar:
        for images, params, targets in pbar:
            images = images.to(device)
            params = params.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images, params)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), images.size(0))
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
    all_preds = []
    all_targets = []

    with tqdm(loader, desc=f"Valid [{epoch}]") as pbar:
        for images, params, targets in pbar:
            images = images.to(device)
            params = params.to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model(images, params)
                loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            pbar.set_postfix(OrderedDict(loss=f"{loss_meter.avg:.4f}"))

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    srocc, plcc, rmse = compute_correlation_metrics(preds, targets)

    return srocc, plcc, rmse


def main(args: argparse.Namespace) -> None:
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    param_stats_path = Path("param_stats.npz")
    if param_stats_path.exists():
        data = np.load(param_stats_path)
        param_stats = {"mean": data["mean"], "std": data["std"]}
        print("Loaded parameter statistics from file.")
    else:
        print("Computing parameter statistics...")
        temp_dataset = KonIQDataset(
            csv_path=args.csv_path,
            images_dir=args.images_dir,
            split="train",
            image_size=args.image_size,
        )
        param_stats = temp_dataset.param_stats
        np.savez(param_stats_path, **param_stats)
        print("Saved parameter statistics.")

    train_dataset = KonIQDataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        split="train",
        image_size=args.image_size,
        augment=True,
        param_stats=param_stats,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_dataset = KonIQDataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        split="valid",
        image_size=args.image_size,
        param_stats=param_stats,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    if args.model == "convnext":
        model = ConvNeXtMLP().to(device)
        model_name = "convnext_mlp"
    elif args.model == "xception":
        model = DualXception().to(device)
        model_name = "dual_xception"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Model: {model_name}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    best_rmse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, scaler, device, epoch
        )
        srocc, plcc, rmse = validate(val_loader, model, criterion, device, epoch)

        print(f"\nEpoch {epoch}: SROCC={srocc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_srocc": srocc,
            "val_plcc": plcc,
            "val_rmse": rmse,
        })

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_dir / f"{model_name}_best.pth",
            )
            print(f"Saved best model (RMSE={rmse:.4f})")

        scheduler.step()

    pd.DataFrame(history).to_csv(checkpoint_dir / "training_log.csv", index=False)
    print(f"Training complete. Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BIQA quality predictor")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to KonIQ CSV")
    parser.add_argument("--images_dir", type=str, required=True, help="Images directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--model", type=str, default="convnext", choices=["convnext", "xception"],
                        help="Model architecture: convnext (ConvNeXt-MLP) or xception (dual-Xception)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 384])

    args = parser.parse_args()
    args.image_size = tuple(args.image_size)
    main(args)

