"""
Evaluation script for Attention U-Net moire removal.

Evaluates trained models on UHDM and TIP2018 test sets,
computing PSNR and SSIM metrics.
"""

import argparse
from pathlib import Path
from collections import OrderedDict

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AttentionUNet
from dataset import DemoireDataset
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
        for moire, clean, _ in pbar:
            moire = moire.to(device)
            clean = clean.to(device)

            output = model(moire)
            psnr, ssim = compute_metrics(output, clean)

            psnr_sum += psnr
            ssim_sum += ssim
            count += 1

            pbar.set_postfix(
                OrderedDict(
                    psnr=f"{psnr_sum / count:.2f}",
                    ssim=f"{ssim_sum / count:.4f}",
                )
            )

    return psnr_sum / count, ssim_sum / count


def main(args: argparse.Namespace) -> None:
    """Main evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    model = AttentionUNet().to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    results = []

    # Evaluate on UHDM
    if args.uhdm_dir:
        uhdm_dataset = DemoireDataset(
            split="uhdm_test",
            image_size=args.image_size,
            uhdm_dir=args.uhdm_dir,
        )
        uhdm_loader = DataLoader(
            uhdm_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        print(f"\nUHDM test samples: {len(uhdm_dataset)}")
        psnr, ssim = evaluate(uhdm_loader, model, device, desc="UHDM")

        print(f"\nUHDM Results: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        results.append({"dataset": "UHDM", "psnr": psnr, "ssim": ssim})

    # Evaluate on TIP2018
    if args.tip_dir:
        tip_dataset = DemoireDataset(
            split="tip_test",
            image_size=args.image_size,
            tip_dir=args.tip_dir,
        )
        tip_loader = DataLoader(
            tip_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        print(f"\nTIP2018 test samples: {len(tip_dataset)}")
        psnr, ssim = evaluate(tip_loader, model, device, desc="TIP2018")

        print(f"\nTIP2018 Results: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        results.append({"dataset": "TIP2018", "psnr": psnr, "ssim": ssim})

    if results:
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Attention U-Net")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    parser.add_argument("--uhdm_dir", type=str, default="", help="UHDM dataset root")
    parser.add_argument("--tip_dir", type=str, default="", help="TIP2018 dataset root")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output", type=str, default="eval_results.csv")

    main(parser.parse_args())
