"""
Quality metrics for image restoration.

Provides SSIM and PSNR implementations for evaluating moire removal results.
"""

from math import exp
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _create_window(window_size: int, channels: int) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM computation."""
    window_1d = _gaussian_window(window_size).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    val_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index.

    Args:
        pred: Predicted image (B, C, H, W).
        target: Ground truth image (B, C, H, W).
        window_size: Size of Gaussian window.
        val_range: Dynamic range of pixel values.

    Returns:
        Mean SSIM value.
    """
    channels = pred.size(1)
    window = _create_window(window_size, channels).to(pred.device).type(pred.dtype)

    mu1 = F.conv2d(pred, window, padding=0, groups=channels)
    mu2 = F.conv2d(target, window, padding=0, groups=channels)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred**2, window, padding=0, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target**2, window, padding=0, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=0, groups=channels) - mu12

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return ssim_map.mean()


class SSIM(nn.Module):
    """SSIM metric as a module."""

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ssim(pred, target, self.window_size)


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio metric."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2)
        return -10 * torch.log10(mse)


def compute_metrics(output: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
    Compute PSNR and SSIM for output-target pair.

    Args:
        output: Model output, will be clamped to [0, 1].
        target: Ground truth, will be clamped to [0, 1].

    Returns:
        Tuple of (psnr, ssim) values.
    """
    output = torch.clamp(output, 0, 1)
    target = torch.clamp(target, 0, 1)

    psnr_metric = PSNR()
    ssim_metric = SSIM()

    return psnr_metric(output, target).item(), ssim_metric(output, target).item()
