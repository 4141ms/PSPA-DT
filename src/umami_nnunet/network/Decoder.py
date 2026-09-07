"""Original 3-D pyramid decoder for PSPA-DT.

This implementation was written for this public repository and does not copy
the EMCAD implementation. It uses standard PyTorch operations: trilinear
upsampling, channel projection, residual depthwise convolutions, and learned
skip gates.
"""

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ResidualPyramidBlock3D(nn.Module):
    """Fuse local context at three dilation rates with a residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.paths = nn.ModuleList(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            )
            for dilation in (1, 2, 3)
        )
        self.mix = nn.Sequential(
            nn.Conv3d(3 * channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
        )
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.mix(torch.cat([path(x) for path in self.paths], dim=1))
        return self.activation(self.norm(x + update))


class SkipGate3D(nn.Module):
    """Condition a skip connection on the current decoder representation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return skip * self.channel_gate(decoder + skip) * self.spatial_gate(
            torch.cat([decoder, skip], dim=1)
        )


class DecoderStage3D(nn.Module):
    """Upsample, gate a projected skip, and refine the fused features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.decoder_projection = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.skip_projection = nn.Conv3d(out_channels, out_channels, 1, bias=False)
        self.gate = SkipGate3D(out_channels)
        self.fusion = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            ResidualPyramidBlock3D(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.decoder_projection(x)
        skip = self.skip_projection(skip)
        skip = self.gate(x, skip)
        return self.fusion(torch.cat([x, skip], dim=1))


class PSPADecoder3D(nn.Module):
    """Four-level decoder used by the fixed UmamiRefine architecture.

    The returned list is ordered from the bottleneck to full resolution so it
    remains compatible with the four deep-supervision heads.
    """

    def __init__(self, channels: Sequence[int] = (256, 128, 64, 32)) -> None:
        super().__init__()
        self.bottleneck = ResidualPyramidBlock3D(channels[0])
        self.stages = nn.ModuleList(
            DecoderStage3D(source, target)
            for source, target in zip(channels[:-1], channels[1:])
        )

    def forward(
        self, x: torch.Tensor, skips: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        if len(skips) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} skip tensors, received {len(skips)}"
            )
        x = self.bottleneck(x)
        outputs = [x]
        for stage, skip in zip(self.stages, skips):
            x = stage(x, skip)
            outputs.append(x)
        return outputs


class DynamicPSPADecoder3D(PSPADecoder3D):
    """Plans-aware decoder accepting arbitrary encoder channel widths."""

    def __init__(self, channels: Sequence[int], decoder_strides=None) -> None:
        # Spatial sizes are taken directly from the skip tensors, which also
        # supports anisotropic nnU-Net plans without hard-coded scale factors.
        super().__init__(channels=channels)
