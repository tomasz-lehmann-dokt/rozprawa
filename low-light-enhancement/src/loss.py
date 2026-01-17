"""
Loss functions for low-light image enhancement.

Provides composite loss combining perceptual, structural, and pixel-level components
for training Swin-UNet to produce visually pleasing enhanced images.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lpips


def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 loss (Huber loss)."""
    return F.smooth_l1_loss(pred, target)


def color_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Color consistency loss based on channel-wise mean difference."""
    diff = torch.mean(target, dim=[1, 2, 3]) - torch.mean(pred, dim=[1, 2, 3]) + 1e-6
    return torch.mean(torch.abs(diff))


def psnr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverted PSNR loss (to be minimized)."""
    mse = torch.clamp(F.mse_loss(pred, target), min=eps)
    psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse))
    return 40.0 - psnr


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sobel gradient consistency loss for edge preservation."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device)

    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=3)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=3)
    target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=3)
    target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=3)

    loss_x = F.l1_loss(pred_grad_x, target_grad_x)
    loss_y = F.l1_loss(pred_grad_y, target_grad_y)

    return loss_x + loss_y


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16]
        self.model = vgg.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat = self.model(pred)
        target_feat = self.model(target)
        return F.l1_loss(pred_feat, target_feat)


class MultiscaleSSIMLoss(nn.Module):
    """Multi-scale SSIM loss using pytorch-msssim."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from pytorch_msssim import msssim
            self.msssim = msssim
        except ImportError:
            self.msssim = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.msssim is None:
            return torch.tensor(0.0, device=pred.device)
        ms = self.msssim(pred, target, val_range=1.0, size_average=True)
        if torch.isnan(ms):
            return torch.tensor(0.0, device=pred.device)
        return 1.0 - ms.detach()


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss using pretrained AlexNet."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.model = lpips.LPIPS(net="alex").to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS expects [-1, 1] range
        pred_norm = pred * 2 - 1
        target_norm = target * 2 - 1
        return self.model(pred_norm, target_norm).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for low-light enhancement.
    
    Aggregates smooth L1, LPIPS, MS-SSIM, PSNR, color, and gradient losses.
    """

    def __init__(
        self,
        device: torch.device,
        alpha_smooth: float = 1.0,
        alpha_lpips: float = 0.1,
        alpha_msssim: float = 0.5,
        alpha_psnr: float = 0.0083,
        alpha_color: float = 0.25,
        alpha_grad: float = 0.1,
    ) -> None:
        super().__init__()
        self.lpips_loss = LPIPSLoss(device)
        self.msssim_loss = MultiscaleSSIMLoss()

        self.alpha_smooth = alpha_smooth
        self.alpha_lpips = alpha_lpips
        self.alpha_msssim = alpha_msssim
        self.alpha_psnr = alpha_psnr
        self.alpha_color = alpha_color
        self.alpha_grad = alpha_grad

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Optional[torch.Tensor]:
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

        if torch.isnan(pred).any() or torch.isnan(target).any():
            return None

        l_smooth = smooth_l1_loss(pred, target)
        l_lpips = self.lpips_loss(pred, target)
        l_msssim = self.msssim_loss(pred, target)
        l_psnr = psnr_loss(pred, target)
        l_color = color_loss(pred, target)
        l_grad = gradient_loss(pred, target)

        total = (
            self.alpha_smooth * l_smooth
            + self.alpha_lpips * l_lpips
            + self.alpha_msssim * l_msssim
            + self.alpha_psnr * l_psnr
            + self.alpha_color * l_color
            + self.alpha_grad * l_grad
        )

        return total


