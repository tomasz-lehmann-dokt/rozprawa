"""
Dual-Xception architecture for blind image quality assessment.

Dual-stream approach as described in the dissertation:
1. Image pathway: Xception backbone extracts visual features from RGB input
2. Parameter pathway: MLP processes global image attributes (brightness, contrast,
   sharpness, colorfulness, entropy, edge density, saturation, exposure ratios)

Both streams are fused via concatenation for MOS prediction.
"""

import torch
import torch.nn as nn
import timm


class DualXception(nn.Module):
    """
    Dual-Xception for no-reference image quality assessment.

    Combines Xception visual features with MLP-encoded global image parameters
    to predict MOS in [0, 100] range.
    """

    def __init__(
        self,
        num_params: int = 9,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        """
        Initialize the dual-stream architecture.

        Args:
            num_params: Number of input image parameters.
            hidden_dim: Hidden dimension for MLP layers.
            dropout: Dropout rate for regularization.
            pretrained: Whether to use ImageNet pretrained weights.
        """
        super().__init__()

        self.feature_extractor = timm.create_model(
            "xception", pretrained=pretrained, num_classes=0
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.visual_dim = self.feature_extractor.num_features

        self.param_encoder = nn.Sequential(
            nn.Linear(num_params, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.regressor = nn.Sequential(
            nn.Linear(self.visual_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, image: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass combining visual and parametric features.

        Args:
            image: Input image tensor of shape (B, 3, H, W).
            params: Normalized image parameters of shape (B, num_params).

        Returns:
            Predicted MOS score scaled to [0, 100].
        """
        visual_features = self.feature_extractor(image)
        param_features = self.param_encoder(params)
        combined = torch.cat([visual_features, param_features], dim=1)
        output = self.regressor(combined).squeeze(1) * 100
        return output

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias for backward compatibility
XceptionMLP = DualXception

