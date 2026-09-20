"""Public implementation of the PSPA-DT training method.

This module collects the method-specific parts of the segmentation and flux
regression trainers in one place. It is intended to be used with nnU-Net v2
and the PSPA-DT modules included in this project.
"""

from contextlib import nullcontext
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from pspa_dt.network.backbones import PSPADTRefineBackbone
from pspa_dt.network.wavelet_refine import DeltaWaveletSSM3D
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from pspa_dt.training.flux_base import (
    nnUNetTrainerFluxRegression,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class PSPADT(PSPADTRefineBackbone):
    """PSPA-DT network for 3-D anatomical and photon-flux prediction."""

    def __init__(
        self,
        in_c: int = 3,
        num_classes: int = 10,
        residual_scale: float = 1e-3,
    ) -> None:
        super().__init__(in_c=in_c, num_classes=num_classes)
        self.mblk2[0] = DeltaWaveletSSM3D(64, residual_scale=residual_scale)
        self.mblk3[0] = DeltaWaveletSSM3D(128, residual_scale=residual_scale)
        self.mblk4[0] = DeltaWaveletSSM3D(256, residual_scale=residual_scale)


class nnUNetTrainer_PSPADT(nnUNetTrainer):
    """Segmentation trainer with deep supervision and topology loss."""

    def _set_batch_size_and_oversample(self):
        self.batch_size = 1

    @staticmethod
    def _resize_logits_to_target(output, target):
        def resize_one(logits, labels):
            spatial_shape = labels.shape[2:] if labels.ndim >= 3 else labels.shape[1:]
            if logits.shape[2:] == spatial_shape:
                return logits
            return F.interpolate(
                logits, size=spatial_shape, mode="trilinear", align_corners=False
            )

        if isinstance(output, (tuple, list)):
            if isinstance(target, (tuple, list)):
                return type(output)(resize_one(o, t) for o, t in zip(output, target))
            return type(output)(resize_one(o, target) for o in output)
        labels = target[0] if isinstance(target, (tuple, list)) else target
        return resize_one(output, labels)

    def _build_loss(self):
        base_loss = super()._build_loss()
        outer = self

        class ConnectivityLoss(nn.Module):
            def forward(self, output, target):
                return base_loss(output, target) + 0.2 * outer._cldice_loss(
                    output, target
                )

        return ConnectivityLoss()

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
        *args,
        **kwargs,
    ) -> nn.Module:
        return PSPADT(
            in_c=num_input_channels,
            num_classes=num_output_channels,
            residual_scale=1e-3,
        )

    def _get_deep_supervision_scales(self):
        scales = super()._get_deep_supervision_scales()
        if scales is None:
            return None
        if len(scales) < 4:
            raise RuntimeError(
                f"PSPA-DT requires 4 deep-supervision scales, got {len(scales)}"
            )
        return scales[:4]

    def _bf16_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA device with BF16 support is required")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def _move_target(target, device):
        if isinstance(target, list):
            return [item.to(device, non_blocking=True) for item in target]
        return target.to(device, non_blocking=True)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._move_target(batch["target"], self.device)
        if not torch.isfinite(data).all():
            raise FloatingPointError("training data contains NaN or Inf")

        self.optimizer.zero_grad(set_to_none=True)
        with self._bf16_context():
            output = self._resize_logits_to_target(self.network(data), target)
            loss = self.loss(output, target)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("training loss contains NaN or Inf")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), 12.0, error_if_nonfinite=True
        )
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu())}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._move_target(batch["target"], self.device)
        with self._bf16_context():
            output = self.network(data, mode="train")
            output = self._resize_logits_to_target(output, target)
            loss = self.loss(output, target)

        if self.enable_deep_supervision:
            output, target = output[0], target[0]
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_onehot = torch.zeros_like(output, dtype=torch.float32)
            predicted_onehot.scatter_(1, output_seg, 1)

        if self.label_manager.has_ignore_label:
            if self.label_manager.has_regions:
                mask = (
                    ~target[:, -1:]
                    if target.dtype == torch.bool
                    else 1 - target[:, -1:]
                )
                target = target[:, :-1]
            else:
                mask = (target != self.label_manager.ignore_label).float()
                target = target.clone()
                target[target == self.label_manager.ignore_label] = 0
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_onehot, target, axes=axes, mask=mask
        )
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard, fp_hard, fn_hard = tp_hard[1:], fp_hard[1:], fn_hard[1:]
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    @staticmethod
    def _soft_erode_3d(x: torch.Tensor) -> torch.Tensor:
        eroded_d = -F.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0))
        eroded_h = -F.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))
        eroded_w = -F.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1))
        return torch.minimum(torch.minimum(eroded_d, eroded_h), eroded_w)

    @staticmethod
    def _soft_dilate_3d(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)

    def _soft_open_3d(self, x: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate_3d(self._soft_erode_3d(x))

    def _soft_skeletonize_3d(
        self, x: torch.Tensor, iters: int = 8
    ) -> torch.Tensor:
        skeleton = F.relu(x - self._soft_open_3d(x))
        for _ in range(iters):
            x = self._soft_erode_3d(x)
            delta = F.relu(x - self._soft_open_3d(x))
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton

    def _cldice_single(
        self, logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        if self.label_manager.has_regions:
            pred, gt = torch.sigmoid(logits).float(), target.float()
            if self.label_manager.has_ignore_label:
                mask = 1.0 - gt[:, -1:]
                gt = gt[:, :-1]
                pred, gt = pred * mask, gt * mask
        else:
            pred = torch.softmax(logits.float(), dim=1)
            target_long = target.long()
            if target_long.ndim == logits.ndim - 1:
                target_long = target_long.unsqueeze(1)
            if self.label_manager.has_ignore_label:
                mask = target_long != self.label_manager.ignore_label
                safe_target = target_long.clone()
                safe_target[~mask] = 0
            else:
                mask = torch.ones_like(target_long, dtype=torch.bool)
                safe_target = target_long
            gt = torch.zeros_like(pred)
            gt.scatter_(1, safe_target, 1.0)
            pred, gt = pred[:, 1:] * mask.float(), gt[:, 1:] * mask.float()

        skel_pred = checkpoint(self._soft_skeletonize_3d, pred, use_reentrant=False)
        with torch.no_grad():
            skel_gt = self._soft_skeletonize_3d(gt)
        spatial_axes = tuple(range(2, pred.ndim))
        pred_mass = skel_pred.sum(dim=spatial_axes)
        gt_mass = skel_gt.sum(dim=spatial_axes)
        precision = (skel_pred * gt).sum(dim=spatial_axes) / (pred_mass + eps)
        sensitivity = (skel_gt * pred).sum(dim=spatial_axes) / (gt_mass + eps)
        cldice = 2 * precision * sensitivity / (precision + sensitivity + eps)
        valid = gt_mass > eps
        return 1.0 - cldice[valid].mean() if valid.any() else pred.sum() * 0.0

    def _cldice_loss(self, logits, target, eps: float = 1e-6):
        if isinstance(logits, (list, tuple)):
            if not isinstance(target, (list, tuple)) or len(target) < len(logits):
                raise RuntimeError("Deep-supervision clDice scale mismatch")
            topology_scale = 1 if len(logits) > 1 else 0
            return self._cldice_single(
                logits[topology_scale], target[topology_scale], eps
            )
        return self._cldice_single(logits, target, eps)


class PSPADTFlux(nn.Module):
    """Adapt the segmentation backbone to single-output flux regression."""

    def __init__(self, in_channels: int, residual_scale: float = 1e-3) -> None:
        super().__init__()
        self.model = PSPADT(
            in_c=in_channels, num_classes=1, residual_scale=residual_scale
        )
        self.model.out_head2 = nn.Identity()
        self.model.out_head3 = nn.Identity()
        self.model.out_head4 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x, mode="eval")
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(
                output, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        return output


class nnUNetTrainer_PSPADTFlux(
    nnUNetTrainerFluxRegression
):
    """Train the method for normalized voxelwise flux regression."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.enable_deep_supervision = False

    def _set_batch_size_and_oversample(self):
        self.batch_size = 1

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
        *args,
        **kwargs,
    ) -> nn.Module:
        return PSPADTFlux(num_input_channels)

    def _bf16_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA device with BF16 support is required")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def _assert_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True).float()
        self._assert_finite("training data", data)
        self._assert_finite("training target", target)
        self.optimizer.zero_grad(set_to_none=True)
        with self._bf16_context():
            logits = self.network(data)
            output = self.output_for_loss(logits)
            loss = self.loss(output, target)
        self._assert_finite("training logits", logits)
        self._assert_finite("training output", output)
        self._assert_finite("training loss", loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), 12.0, error_if_nonfinite=True
        )
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu())}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True).float()
        self._assert_finite("validation data", data)
        self._assert_finite("validation target", target)
        with self._bf16_context():
            logits = self.network(data)
            output = self.output_for_loss(logits)
            loss = self.loss(output, target)
        self._assert_finite("validation logits", logits)
        self._assert_finite("validation output", output)
        self._assert_finite("validation loss", loss)
        error = output.float() - target
        return {
            "loss": float(loss.detach().cpu()),
            "mae": float(error.abs().mean().detach().cpu()),
            "mse": float(error.square().mean().detach().cpu()),
        }


# Legacy aliases allow checkpoints produced before the paper-aligned rename to
# be loaded. New experiments and documentation must use the PSPA-DT names.
WaveletUmamiRefineControl = PSPADT
WaveletUmamiRefineControlFlux = PSPADTFlux
nnUNetTrainer_umami_refine_wavelet_control = nnUNetTrainer_PSPADT
nnUNetTrainer_umami_refine_wavelet_control_flux = nnUNetTrainer_PSPADTFlux
