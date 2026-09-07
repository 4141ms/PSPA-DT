"""Backward-compatible imports for the packaged Umami trainers.

New code should import these classes from :mod:`umami_nnunet.training`.
"""

from umami_nnunet.training import (
    nnUNetTrainer_umami_refine_wavelet_control,
    nnUNetTrainer_umami_refine_wavelet_control_flux,
)
from umami_nnunet.training.trainers import (
    WaveletUmamiRefineControl,
    WaveletUmamiRefineControlFlux,
)

__all__ = [
    "WaveletUmamiRefineControl",
    "WaveletUmamiRefineControlFlux",
    "nnUNetTrainer_umami_refine_wavelet_control",
    "nnUNetTrainer_umami_refine_wavelet_control_flux",
]
