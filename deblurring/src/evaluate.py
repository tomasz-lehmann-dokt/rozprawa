"""
Evaluation script for motion deblurring model.
"""

import argparse
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import HIDEDataset
from features import extract_features, normalize_features
from loss import CombinedLoss
from metrics import compute_metrics
from model import AttentionUNet
from mos_model import DualXception


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


def evaluate(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model on test set."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for blurred, sharp in tqdm(loader, desc="Evaluating"):
            blurred = blurred.to(device)
            sharp = sharp.to(device)

            output = model(blurred)
            psnr, ssim = compute_metrics(output, sharp)

            batch_size = blurred.size(0)
            total_psnr += psnr * batch_size
            total_ssim += ssim * batch_size
            count += batch_size

    return total_psnr / count, total_ssim / count


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    test_dataset = HIDEDataset(
        args.data_dir, "test", (args.image_size, args.image_size)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model
    model = AttentionUNet(dim=args.dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Evaluate
    psnr, ssim = evaluate(test_loader, model, device)

    print(f"\nTest Results:")
    print(f"  PSNR: {psnr:.3f} dB")
    print(f"  SSIM: {ssim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deblurring model")
    parser.add_argument("--data_dir", type=str, required=True, help="HIDE dataset path")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    main(args)
