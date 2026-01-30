"""
Training script for motion deblurring with MOS-based perceptual loss.

Attention U-Net with combined loss: MSE + SSIM + MOS (dual-Xception).
"""

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import HIDEDataset
from features import extract_features, normalize_features
from loss import CombinedLoss
from metrics import compute_metrics
from model import AttentionUNet
from mos_model import DualXception


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_param_stats(path: str) -> Dict[str, np.ndarray]:
    """Load feature normalization statistics."""
    data = np.load(path)
    return {"mean": data["mean"], "std": data["std"]}


def create_feature_fn(param_stats: Dict[str, np.ndarray]):
    """Create feature extraction function with normalization."""

    def fn(image: np.ndarray) -> np.ndarray:
        features = extract_features(image)
        return normalize_features(features, param_stats["mean"], param_stats["std"])

    return fn


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedLoss,
    mos_model: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for blurred, sharp in pbar:
        blurred = blurred.to(device)
        sharp = sharp.to(device)

        optimizer.zero_grad()
        output = model(blurred)
        loss, mse, ssim_l, mos_l = loss_fn(blurred, sharp, output, mos_model)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * blurred.size(0)
        pbar.set_postfix(loss=loss.item(), mse=mse.item())

    return total_loss / len(loader.dataset)


def validate(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: CombinedLoss,
    mos_model: nn.Module,
    device: torch.device,
) -> tuple:
    """Run validation and compute metrics."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for blurred, sharp in tqdm(loader, desc="Validate"):
            blurred = blurred.to(device)
            sharp = sharp.to(device)

            output = model(blurred)
            loss, _, _, _ = loss_fn(blurred, sharp, output, mos_model)

            psnr, ssim = compute_metrics(output, sharp)

            batch_size = blurred.size(0)
            total_loss += loss.item() * batch_size
            total_psnr += psnr * batch_size
            total_ssim += ssim * batch_size
            count += batch_size

    return total_loss / count, total_psnr / count, total_ssim / count


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dataset = HIDEDataset(
        args.data_dir, "train", (args.image_size, args.image_size)
    )
    val_dataset = HIDEDataset(args.data_dir, "val", (args.image_size, args.image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Models
    model = AttentionUNet(dim=args.dim).to(device)

    mos_model = DualXception(pretrained=False).to(device)
    if args.mos_checkpoint:
        ckpt = torch.load(args.mos_checkpoint, map_location=device)
        mos_model.load_state_dict(ckpt["state_dict"])
    mos_model.eval()

    # Loss
    param_stats = load_param_stats(args.param_stats)
    feature_fn = create_feature_fn(param_stats)

    image_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    loss_fn = CombinedLoss(
        w_mse=args.w_mse,
        w_ssim=args.w_ssim,
        w_mos=args.w_mos,
        mos_variant=args.mos_variant,
        mos_input_size=(512, 384),
        image_transform=image_transform,
        feature_fn=feature_fn,
    )

    # Optimizer
    optimizer = RMSprop(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            train_loader, model, optimizer, loss_fn, mos_model, device, epoch
        )
        val_loss, val_psnr, val_ssim = validate(
            val_loader, model, loss_fn, mos_model, device
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict()},
                output_dir / "best_model.pth",
            )
            print(f"  -> New best model saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"Early stopping after {args.patience} epochs without improvement"
                )
                break

        scheduler.step()

    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deblurring model")
    parser.add_argument("--data_dir", type=str, required=True, help="HIDE dataset path")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--mos_checkpoint", type=str, help="Path to MOS model checkpoint"
    )
    parser.add_argument(
        "--param_stats", type=str, required=True, help="Path to param_stats.npz"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--w_mse", type=float, default=50.0)
    parser.add_argument("--w_ssim", type=float, default=0.00699)
    parser.add_argument("--w_mos", type=float, default=0.01792)
    parser.add_argument(
        "--mos_variant", type=str, default="ratio", choices=["ratio", "difference"]
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
