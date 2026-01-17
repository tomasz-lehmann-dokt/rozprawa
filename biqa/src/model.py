"""
ConvNeXt-MLP model for blind image quality assessment.

Combines ConvNeXt backbone for visual features with MLP for global
image parameters. Dual-stream architecture for MOS prediction.
"""

import torch
import torch.nn as nn
import timm


class ConvNeXtMLP(nn.Module):
    """
    ConvNeXt-MLP for no-reference image quality assessment.

    Visual features from ConvNeXt + global parameters from MLP
    are fused to predict MOS in [0, 100] range.
    """

    def __init__(
        self,
        backbone: str = "convnext_base",
        num_params: int = 9,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Visual pathway - ConvNeXt backbone
        self.img_feature_extractor = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )

        for param in self.img_feature_extractor.parameters():
            param.requires_grad = True

        self.dim_feats = self.img_feature_extractor.num_features

        # Parameter pathway - MLP
        self.param_mlp = nn.Sequential(
            nn.Linear(num_params, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Fusion and regression
        self.fc = nn.Sequential(
            nn.Linear(self.dim_feats + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor (B, 3, H, W).
            params: Normalized parameters (B, 9).

        Returns:
            MOS prediction [0, 100].
        """
        img_feats = self.img_feature_extractor(image)
        param_feats = self.param_mlp(params)
        combined = torch.cat([img_feats, param_feats], dim=1)
        return self.fc(combined).squeeze(1) * 100
