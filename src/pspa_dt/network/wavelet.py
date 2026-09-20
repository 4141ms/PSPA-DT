"""Plans-aware PSPA-DT with state-space scanning in the 3-D Haar domain."""

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from pspa_dt.network.backbones import PSPADTDynamicBackbone, replace_bn_with_in
from pspa_dt.network.model.LoE import LoENm3D
from pspa_dt.network.model.VSSM import VSSM3D


_SUBBAND_BITS: Tuple[Tuple[int, int, int], ...] = tuple(
    (d, h, w) for d in range(2) for h in range(2) for w in range(2)
)


def haar_dwt3d(x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int, int]]:
    """Orthogonal one-level 3D Haar transform in LLL..HHH order."""
    original_shape = tuple(x.shape[2:])
    pad_d, pad_h, pad_w = (size % 2 for size in original_shape)
    if pad_d or pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")

    samples = [x[:, :, d::2, h::2, w::2] for d, h, w in _SUBBAND_BITS]
    scale = 1.0 / math.sqrt(8.0)
    subbands = []
    for fd, fh, fw in _SUBBAND_BITS:
        band = sum(
            (-1.0 if (fd * d + fh * h + fw * w) % 2 else 1.0) * sample
            for sample, (d, h, w) in zip(samples, _SUBBAND_BITS)
        )
        subbands.append(band * scale)
    return subbands, original_shape


def haar_idwt3d(subbands: Sequence[torch.Tensor],
                original_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Inverse of :func:`haar_dwt3d`, including removal of odd-size padding."""
    if len(subbands) != 8:
        raise ValueError(f"3D Haar IDWT expects 8 subbands, got {len(subbands)}")

    b, c, d, h, w = subbands[0].shape
    output = subbands[0].new_empty((b, c, d * 2, h * 2, w * 2))
    scale = 1.0 / math.sqrt(8.0)
    for sd, sh, sw in _SUBBAND_BITS:
        sample = sum(
            (-1.0 if (fd * sd + fh * sh + fw * sw) % 2 else 1.0) * band
            for band, (fd, fh, fw) in zip(subbands, _SUBBAND_BITS)
        )
        output[:, :, sd::2, sh::2, sw::2] = sample * scale

    od, oh, ow = original_shape
    return output[:, :, :od, :oh, :ow]


class SharedHighFrequencyGate3D(nn.Module):
    """A parameter-shared lightweight gate for all seven high-frequency bands."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.activation = nn.GELU()
        self.channel_mix = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        # Start as an identity high-frequency path.
        nn.init.zeros_(self.channel_mix.weight)
        nn.init.zeros_(self.channel_mix.bias)

    def forward(self, bands: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(bands) != 7:
            raise ValueError(f"Expected 7 high-frequency bands, got {len(bands)}")
        shape = bands[0].shape
        merged = torch.cat(list(bands), dim=0)
        delta = torch.tanh(self.channel_mix(self.activation(self.depthwise(merged))))
        gated = merged * (1.0 + delta)
        return list(gated.split(shape[0], dim=0))


class WaveletSSM3D(nn.Module):
    """SSM on LLL, shared lightweight processing on high-frequency subbands."""

    def __init__(self, channels: int):
        super().__init__()
        self.low_frequency_ssm = VSSM3D(channels)
        self.high_frequency_gate = SharedHighFrequencyGate3D(channels)
        # ReZero makes inserting this branch safe at initialization.
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        subbands, original_shape = haar_dwt3d(x)
        low = self.low_frequency_ssm(subbands[0])
        highs = self.high_frequency_gate(subbands[1:])
        reconstructed = haar_idwt3d([low, *highs], original_shape)
        return x + self.residual_scale * reconstructed


class PSPADTWavelet(nn.Module):
    """Dynamic PSPA-DT backbone using wavelet SSMs in its deepest stages."""

    def __init__(self, in_c: int, num_classes: int, arch_params: dict,
                 deep_supervision: bool = True, wavelet_stages: int = 2):
        super().__init__()
        backbone = PSPADTDynamicBackbone(in_c=in_c, num_classes=num_classes, arch_params=arch_params)
        self.encoder_stages = backbone.encoder_stages
        self.decoder = backbone.decoder
        self.out_heads = backbone.out_heads
        self.deep_supervision = deep_supervision

        features = list(arch_params["features_per_stage"])
        wavelet_stages = max(1, min(int(wavelet_stages), len(features)))
        first_wavelet_stage = len(features) - wavelet_stages
        self.feature_blocks = nn.ModuleList()
        for stage_index, channels in enumerate(features):
            if stage_index >= first_wavelet_stage:
                self.feature_blocks.append(nn.Sequential(
                    WaveletSSM3D(channels),
                    LoENm3D(channels),
                ))
            else:
                self.feature_blocks.append(nn.Identity())

        replace_bn_with_in(self.decoder)

    def forward(self, x: torch.Tensor, mode=None):
        skips = []
        feat = x
        for encoder, feature_block in zip(self.encoder_stages, self.feature_blocks):
            feat = feature_block(encoder(feat))
            skips.append(feat)

        decoder_features = self.decoder(feat, skips[-2::-1])[::-1]
        logits = [head(feature) for head, feature in
                  zip(self.out_heads, decoder_features)]
        logits = logits[:-1]
        return logits if self.deep_supervision else logits[0]
