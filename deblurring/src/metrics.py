"""
Image quality metrics: PSNR and SSIM.
"""

import torch
import torchmetrics.functional as tmf
from typing import Tuple


def compute_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        output: Predicted image (B, C, H, W), range [0, 1].
        target: Reference image (B, C, H, W), range [0, 1].

    Returns:
        Mean PSNR in dB.
    """
    data_range = (target.max() - target.min()).item()
    return tmf.peak_signal_noise_ratio(output, target, data_range=data_range).item()


def compute_ssim(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index.

    Args:
        output: Predicted image (B, C, H, W), range [0, 1].
        target: Reference image (B, C, H, W), range [0, 1].

    Returns:
        Mean SSIM.
    """
    data_range = (target.max() - target.min()).item()
    return tmf.structural_similarity_index_measure(
        output, target, data_range=data_range
    ).item()


def compute_metrics(output: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
    Compute both PSNR and SSIM.

    Returns:
        Tuple of (PSNR, SSIM).
    """
    return compute_psnr(output, target), compute_ssim(output, target)
