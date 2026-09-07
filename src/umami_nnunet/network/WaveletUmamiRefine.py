"""Wavelet-domain Umami with a full-resolution refinement head."""

from typing import Sequence

import torch
from torch import nn

from umami_nnunet.network.WaveletUmami import (
    SharedHighFrequencyGate3D,
    WaveletUmami,
    haar_dwt3d,
    haar_idwt3d,
)
from umami_nnunet.network.model.VSSM import VSSM3D


class DeltaWaveletSSM3D(nn.Module):
    """Wavelet SSM that injects only the learned wavelet-domain update."""

    def __init__(self, channels: int, residual_scale: float = 1e-3):
        super().__init__()
        self.low_frequency_ssm = VSSM3D(channels)
        self.high_frequency_gate = SharedHighFrequencyGate3D(channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        subbands, original_shape = haar_dwt3d(x)
        low = self.low_frequency_ssm(subbands[0])
        highs = self.high_frequency_gate(subbands[1:])
        reconstructed = haar_idwt3d([low, *highs], original_shape)
        return x + self.residual_scale * (reconstructed - x)


class WaveletUmamiRefine(WaveletUmami):
    """Plans-aware WaveletUmami with refinement at the finest output scale."""

    def __init__(self, in_c: int, num_classes: int, arch_params: dict,
                 deep_supervision: bool = True, wavelet_stages: int = 2,
                 residual_scale: float = 1e-3):
        super().__init__(
            in_c=in_c,
            num_classes=num_classes,
            arch_params=arch_params,
            deep_supervision=deep_supervision,
            wavelet_stages=wavelet_stages,
        )

        features: Sequence[int] = arch_params["features_per_stage"]
        wavelet_stages = max(1, min(int(wavelet_stages), len(features)))
        first_wavelet_stage = len(features) - wavelet_stages
        for stage_index in range(first_wavelet_stage, len(features)):
            # Preserve the following LoEN block created by WaveletUmami while
            # replacing its wavelet operation with delta-only residual fusion.
            self.feature_blocks[stage_index][0] = DeltaWaveletSSM3D(
                features[stage_index], residual_scale=residual_scale
            )

        finest_channels = features[0]
        self.out_heads[0] = nn.Sequential(
            nn.Conv3d(finest_channels, finest_channels, kernel_size=3,
                      padding=1, bias=True),
            nn.InstanceNorm3d(finest_channels, affine=True,
                              track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(finest_channels, num_classes, kernel_size=1, bias=True),
        )
