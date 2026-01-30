"""
Attention U-Net for moire pattern reduction.

Encoder-decoder architecture with self-attention gates on skip connections
and residual convolution blocks. Global residual learning (x + output).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block with residual (1x1) connection."""

    def __init__(self, in_channel: int, out_channel: int, strides: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=strides, padding=1
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channel, out_channel, kernel_size=3, stride=strides, padding=1
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=strides, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.conv11(x)


class UpConv(nn.Module):
    """Upsampling block with convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionBlock(nn.Module):
    """
    Attention gate for skip connections.

    Computes spatial attention based on gating signal from decoder
    and features from encoder, enabling selective feature propagation.
    """

    def __init__(self, F_g: int, F_l: int, n_coefficients: int) -> None:
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(
                F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(n_coefficients),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(n_coefficients),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, gate: torch.Tensor, skip_connection: torch.Tensor
    ) -> torch.Tensor:
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net for image restoration.

    U-Net with attention gates on skip connections, residual convolution blocks,
    and global residual learning (out = x + conv(features)).
    """

    def __init__(self, dim: int = 64) -> None:
        super().__init__()

        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Unused but kept for checkpoint compatibility
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)
        self.pool2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)
        self.pool3 = nn.Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)
        self.pool4 = nn.Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = ConvBlock(dim, dim * 2, strides=1)
        self.ConvBlock3 = ConvBlock(dim * 2, dim * 4, strides=1)
        self.ConvBlock4 = ConvBlock(dim * 4, dim * 8, strides=1)
        self.ConvBlock5 = ConvBlock(dim * 8, dim * 16, strides=1)

        self.upv6 = UpConv(1024, 512)
        self.ConvBlock6 = ConvBlock(dim * 16, dim * 8, strides=1)

        self.upv7 = UpConv(512, 256)
        self.ConvBlock7 = ConvBlock(dim * 8, dim * 4, strides=1)

        self.upv8 = UpConv(256, 128)
        self.ConvBlock8 = ConvBlock(dim * 4, dim * 2, strides=1)

        self.upv9 = UpConv(128, 64)
        self.ConvBlock9 = ConvBlock(dim * 2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

        self.Att1 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.Att2 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.Att4 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        conv1 = self.ConvBlock1(x)
        pool1 = self.MaxPool(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.MaxPool(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.MaxPool(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.MaxPool(conv4)

        conv5 = self.ConvBlock5(pool4)

        # Decoder with attention
        up6 = self.upv6(conv5)
        conv4 = self.Att1(gate=up6, skip_connection=conv4)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        conv3 = self.Att2(gate=up7, skip_connection=conv3)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        conv2 = self.Att3(gate=up8, skip_connection=conv2)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        conv1 = self.Att4(gate=up9, skip_connection=conv1)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        # Global residual
        return x + self.conv10(conv9)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
