"""
Evaluation script for Swin-UNet low-light enhancement.

Evaluates trained models on LOL, LOL-v2 test sets computing PSNR and SSIM.
"""

import argparse
from pathlib import Path
from collections import OrderedDict

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SwinTransformerSys
from dataset import LowLightDataset
from metrics import compute_metrics


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    desc: str = "Eval",
) -> tuple:
    """
    Evaluate model on dataset.

    Returns:
        Tuple of (mean_psnr, mean_ssim).
    """
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    with tqdm(loader, desc=desc) as pbar:
        for low, high, _ in pbar:
            low = low.to(device)
            high = high.to(device)

            output = model(low)
            psnr, ssim = compute_metrics(output, high)

            psnr_sum += psnr
            ssim_sum += ssim
            count += 1

            pbar.set_postfix(OrderedDict(
                psnr=f"{psnr_sum / count:.2f}",
                ssim=f"{ssim_sum / count:.4f}",
            ))

    return psnr_sum / count, ssim_sum / count


def main(args: argparse.Namespace) -> None:
    """Main evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    # Load model
    model = SwinTransformerSys(
        img_size=args.crop_size,
        patch_size=4,
        in_chans=3,
        num_classes=3,
        embed_dim=args.embed_dim,
        depths=args.depths,
        window_size=args.window_size,
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    results = []

    # Evaluate on each test set
    test_configs = [
        ("LOL-v1", {"use_lol": True, "use_lol_v2": False, "use_sid": False}),
        ("LOL-v2-Real", {"use_lol": False, "use_lol_v2": True, "use_sid": False, "lol_v2_subsets": ["Real_captured"]}),
        ("LOL-v2-Synth", {"use_lol": False, "use_lol_v2": True, "use_sid": False, "lol_v2_subsets": ["Synthetic"]}),
    ]

    for name, config in test_configs:
        dataset = LowLightDataset(
            split="test",
            crop_size=(args.crop_size, args.crop_size),
            datasets_dir=args.datasets_dir,
            **config,
        )

        if len(dataset) == 0:
            print(f"\n{name}: No samples found, skipping")
            continue

        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        print(f"\n{name} test samples: {len(dataset)}")
        psnr, ssim = evaluate(loader, model, device, desc=name)

        print(f"{name} Results: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        results.append({"dataset": name, "psnr": psnr, "ssim": ssim})

    if results:
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Swin-UNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--datasets_dir", type=str, required=True, help="Datasets root")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--output", type=str, default="eval_results.csv")

    # Model architecture (must match checkpoint)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs=4, default=[2, 4, 8, 2])
    parser.add_argument("--depths_decoder", type=int, nargs=4, default=[2, 8, 4, 2])
    parser.add_argument("--num_heads", type=int, nargs=4, default=[4, 8, 16, 32])
    parser.add_argument("--window_size", type=int, default=7)

    main(parser.parse_args())


