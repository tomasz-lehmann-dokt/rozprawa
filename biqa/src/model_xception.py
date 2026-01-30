"""
Dual-Xception architecture for blind image quality assessment.

Dual-stream approach as described in the dissertation:
1. Image pathway: Xception backbone extracts visual features from RGB input
2. Parameter pathway: MLP processes global image attributes (brightness, contrast,
   sharpness, colorfulness, entropy, edge density, saturation, exposure ratios)

Both streams are fused via concatenation for MOS prediction.

Two variants available:
- DualXception: Xception (timm) + MLP for scalar parameters
- DualXceptionV2: Two Xception backbones (pretrainedmodels), parameters as spatial maps
"""

import ssl
import torch
import torch.nn as nn
import timm

ssl._create_default_https_context = ssl._create_unverified_context


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

    def forward(self, image: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
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


class Identity(nn.Module):
    """Identity layer for replacing classification heads."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DualXceptionV2(nn.Module):
    """
    Dual-Xception V2 with two Xception backbones.

    Architecture using pretrainedmodels library:
    1. Image pathway: Xception backbone for RGB input (3 channels)
    2. Parameter pathway: Xception backbone for spatial parameter maps (4 channels)

    Parameters are converted to 4-channel spatial maps and processed by
    a separate Xception network with modified input convolution.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """
        Initialize dual-backbone architecture.

        Args:
            pretrained: Whether to use ImageNet pretrained weights.
        """
        super().__init__()

        try:
            import pretrainedmodels
        except ImportError:
            raise ImportError(
                "pretrainedmodels required for DualXceptionV2. "
                "Install via: pip install pretrainedmodels"
            )

        pretrained_str = "imagenet" if pretrained else None

        self.img_feature_extractor = pretrainedmodels.__dict__["xception"](
            pretrained=pretrained_str
        )
        self.params_feature_extractor = pretrainedmodels.__dict__["xception"](
            pretrained=pretrained_str
        )

        for param in self.img_feature_extractor.parameters():
            param.requires_grad = True
        for param in self.params_feature_extractor.parameters():
            param.requires_grad = True

        self.dim_feats = self.img_feature_extractor.last_linear.in_features

        self.params_feature_extractor.last_linear = Identity()
        self.img_feature_extractor.last_linear = Identity()

        # Modify params branch input to accept 4 channels
        self.params_feature_extractor.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
        )
        self.params_feature_extractor.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Linear(2 * self.dim_feats, 1)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, image: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dual Xception backbones.

        Args:
            image: Input image tensor of shape (B, 3, H, W).
            params: Normalized image parameters of shape (B, num_params).
                    Converted to spatial maps of shape (B, 4, H, W).

        Returns:
            Predicted MOS score scaled to [0, 100].
        """
        B, _, H, W = image.shape

        # Convert scalar params to 4-channel spatial map
        # Select first 4 params (or repeat if fewer) and broadcast spatially
        if params.shape[1] >= 4:
            param_channels = params[:, :4]
        else:
            # Repeat params to fill 4 channels
            repeats = (4 + params.shape[1] - 1) // params.shape[1]
            param_channels = params.repeat(1, repeats)[:, :4]

        param_map = param_channels.view(B, 4, 1, 1).expand(B, 4, H, W)

        img_feats = self.img_feature_extractor(image)
        param_feats = self.params_feature_extractor(param_map)

        combined = torch.cat([img_feats, param_feats], dim=1)
        output = self.regressor(combined).squeeze(1) * 100

        return output

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias for backward compatibility
XceptionMLP = DualXception
