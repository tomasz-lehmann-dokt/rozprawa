"""
Evaluation script for BIQA quality predictor.

Evaluates on test set with correlation metrics and per-range RMSE.

Architectures:
- ConvNeXt-MLP: ConvNeXt backbone + MLP parameter encoder
- dual-Xception: Xception backbone + MLP parameter encoder
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ConvNeXtMLP
from model_xception import DualXception
from dataset import KonIQDataset
from metrics import compute_correlation_metrics, compute_range_metrics


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate model on dataset.

    Returns:
        Tuple of (predictions, ground_truth) arrays.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with tqdm(loader, desc="Evaluating") as pbar:
        for images, params, targets in pbar:
            images = images.to(device)
            params = params.to(device)

            with autocast():
                outputs = model(images, params)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def main(args: argparse.Namespace) -> None:
    """Main evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    param_stats_path = Path("param_stats.npz")
    if not param_stats_path.exists():
        raise FileNotFoundError("param_stats.npz not found. Run training first.")

    data = np.load(param_stats_path)
    param_stats = {"mean": data["mean"], "std": data["std"]}

    test_dataset = KonIQDataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        split="test",
        image_size=tuple(args.image_size),
        param_stats=param_stats,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    if args.model == "convnext":
        model = ConvNeXtMLP().to(device)
    elif args.model == "xception":
        model = DualXception().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    predictions, ground_truth = evaluate(test_loader, model, device)

    srocc, plcc, rmse = compute_correlation_metrics(predictions, ground_truth)
    range_metrics = compute_range_metrics(predictions, ground_truth)

    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    print(f"SROCC: {srocc:.4f}")
    print(f"PLCC:  {plcc:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print("\nPer-range RMSE:")
    for (low, high), val in range_metrics.items():
        print(f"  MOS [{low:3d}, {high:3d}): {val:.4f}")

    results = {
        "srocc": srocc,
        "plcc": plcc,
        "rmse": rmse,
        **{f"rmse_{l}_{h}": v for (l, h), v in range_metrics.items()},
    }
    pd.DataFrame([results]).to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BIQA quality predictor")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to KonIQ CSV")
    parser.add_argument("--images_dir", type=str, required=True, help="Images directory")
    parser.add_argument("--model", type=str, default="convnext", choices=["convnext", "xception"],
                        help="Model architecture: convnext (ConvNeXt-MLP) or xception (dual-Xception)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 384])
    parser.add_argument("--output", type=str, default="test_results.csv")

    main(parser.parse_args())

