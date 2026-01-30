"""
Combined loss function with MSE, SSIM, and MOS components.

Implements perceptual loss based on predicted MOS from dual-Xception model.
Two MOS loss variants:
- Sigmoid ratio: L_MOS = 1 - sigmoid(-MOS(out)/MOS(in))
- Normalized difference: L_MOS = 1 - sigmoid(-d/max|d|), d = MOS(in) - MOS(out)
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn


class CombinedLoss(nn.Module):
    """
    Combined loss: λ_MSE * MSE + λ_SSIM * (1-SSIM) + λ_MOS * L_MOS.

    Args:
        w_mse: Weight for MSE component.
        w_ssim: Weight for SSIM component.
        w_mos: Weight for MOS component.
        mos_variant: 'ratio' or 'difference'.
        mos_input_size: Size for MOS model input.
        image_transform: Normalization transform for MOS model.
        feature_fn: Function to extract image features for MOS.
    """

    def __init__(
        self,
        w_mse: float = 50.0,
        w_ssim: float = 0.01,
        w_mos: float = 0.01,
        mos_variant: str = "ratio",
        mos_input_size: Tuple[int, int] = (512, 384),
        image_transform: Optional[nn.Module] = None,
        feature_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.w_mse = w_mse
        self.w_ssim = w_ssim
        self.w_mos = w_mos
        self.mos_variant = mos_variant
        self.mos_input_size = mos_input_size
        self.image_transform = image_transform
        self.feature_fn = feature_fn
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        blurred: torch.Tensor,
        sharp: torch.Tensor,
        output: torch.Tensor,
        mos_model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            blurred: Input blurred image (B, 3, H, W).
            sharp: Target sharp image (B, 3, H, W).
            output: Deblurred output (B, 3, H, W).
            mos_model: Pretrained MOS estimator (optional).

        Returns:
            Tuple of (total_loss, mse_loss, ssim_loss, mos_loss).
        """
        device = output.device

        # MSE component
        mse_val = self.mse_loss(output, sharp)
        total = self.w_mse * mse_val

        # SSIM component
        ssim_loss = torch.tensor(0.0, device=device)
        if self.w_ssim > 0:
            ssim_val = ssim_fn(output, sharp, data_range=1.0)
            ssim_loss = 1.0 - ssim_val
            total = total + self.w_ssim * ssim_loss

        # MOS component
        mos_loss = torch.tensor(0.0, device=device)
        if self.w_mos > 0 and mos_model is not None:
            mos_loss = self._compute_mos_loss(blurred, output, mos_model, device)
            total = total + self.w_mos * mos_loss

        return total, mse_val, ssim_loss, mos_loss

    def _compute_mos_loss(
        self,
        blurred: torch.Tensor,
        output: torch.Tensor,
        mos_model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute MOS-based perceptual loss."""
        batch_size = output.shape[0]

        # Resize for MOS model
        out_resized = F.interpolate(
            output, size=self.mos_input_size, mode="bilinear", align_corners=False
        )
        blur_resized = F.interpolate(
            blurred, size=self.mos_input_size, mode="bilinear", align_corners=False
        )

        # Apply normalization
        if self.image_transform is not None:
            out_tensor = self.image_transform(out_resized)
            blur_tensor = self.image_transform(blur_resized)
        else:
            out_tensor = out_resized
            blur_tensor = blur_resized

        # Extract features
        out_params_list = []
        blur_params_list = []
        for i in range(batch_size):
            out_np = out_resized[i].detach().cpu().permute(1, 2, 0).numpy()
            blur_np = blur_resized[i].detach().cpu().permute(1, 2, 0).numpy()
            out_params_list.append(self.feature_fn(out_np))
            blur_params_list.append(self.feature_fn(blur_np))

        out_params = torch.tensor(
            np.stack(out_params_list), dtype=torch.float32, device=device
        )
        blur_params = torch.tensor(
            np.stack(blur_params_list), dtype=torch.float32, device=device
        )

        # Predict MOS scores
        with torch.no_grad():
            out_mos = mos_model(out_tensor, out_params)
            blur_mos = mos_model(blur_tensor, blur_params)

        # Compute loss variant
        eps = 1e-6
        if self.mos_variant == "ratio":
            # Eq. (6.1): 1 - sigmoid(-MOS(out)/MOS(in))
            ratio = out_mos / (blur_mos + eps)
            loss = 1.0 - 1.0 / (1.0 + torch.exp(-ratio))
        else:
            # Eq. (6.2): normalized difference
            diff = blur_mos - out_mos
            normalized = diff / (diff.abs().max() + eps)
            loss = 1.0 - 1.0 / (1.0 + torch.exp(-normalized))

        return loss.mean()
