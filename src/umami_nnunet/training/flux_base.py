from time import time
from typing import List, Tuple, Union

import numpy as np
import torch
import multiprocessing
import warnings
from time import time, sleep
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from torch import autocast, nn
from torch import distributed as dist

from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from umami_nnunet.inference.metrics import compute_metrics_on_folder_regression
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian


class nnUNetDataLoaderFluxRegression(nnUNetDataLoader):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        target_all = np.zeros(self.seg_shape, dtype=np.float32)

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, target, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            target_all[j] = crop_and_pad_nd(target, bbox, 0)

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            target_all = target_all[:, :, 0]

        return {
            'data': torch.from_numpy(data_all).float(),
            'target': torch.from_numpy(target_all).float(),
            'keys': selected_keys
        }


class nnUNetTrainerFluxRegression(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # nnU-Net 2.6.2 does not expose ``unpack_dataset`` in this constructor.
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self.enable_deep_supervision = False
        self.initial_lr = 1e-3
        self.num_epochs = 100
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 20

        # self.num_epochs = 5
        # self.num_iterations_per_epoch = 5
        # self.num_val_iterations_per_epoch = 2

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   *args, **kwargs) -> nn.Module:
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            1,
            allow_init=True,
            deep_supervision=False)

    def _build_loss(self):
        return nn.SmoothL1Loss(beta=0.05)

    def output_for_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Map network logits to the space used by the regression loss."""
        return torch.sigmoid(logits)

    def output_for_inference(self, logits: torch.Tensor) -> torch.Tensor:
        """Map network logits to normalized flux values in [0, 1]."""
        return torch.sigmoid(logits)

    def set_deep_supervision_enabled(self, enabled: bool):
        return

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderFluxRegression(
            dataset_tr, self.batch_size, patch_size, patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=None,
            probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoaderFluxRegression(
            dataset_val, self.batch_size, patch_size, patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=None,
            probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True).float()

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            output = self.output_for_loss(output)
            loss = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': float(loss.detach().cpu())}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True).float()

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            output = self.output_for_loss(output)
            loss = self.loss(output, target)

        error = output - target
        mae = torch.mean(torch.abs(error))
        mse = torch.mean(error ** 2)
        return {
            'loss': float(loss.detach().cpu()),
            'mae': float(mae.detach().cpu()),
            'mse': float(mse.detach().cpu())
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs = collate_outputs(val_outputs)
        val_loss = float(np.mean(outputs['loss']))
        val_mae = float(np.mean(outputs['mae']))
        val_mse = float(np.mean(outputs['mse']))

        self.logger.log('val_losses', val_loss, self.current_epoch)
        self.logger.log('mean_fg_dice', -val_mae, self.current_epoch)
        self.logger.log('dice_per_class_or_region', [val_mae, val_mse], self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        metrics = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        self.print_to_log_file('val_mae', np.round(metrics[0], decimals=6), 'val_mse', np.round(metrics[1], decimals=6))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"New best EMA validation MAE: {np.round(-self._best_ema, decimals=6)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)

                # we do [:] to convert blosc2 to numpy
                data = data[:]

                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)
                
                # data.shape is (C, X, Y, Z) but predictor expects (N, C, X, Y, Z)
                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = self.output_for_inference(prediction)
                prediction = prediction.cpu().float()

                # Export synchronously. Sending a bound trainer method to a spawn Pool can pickle
                # self.network and trigger CUDA IPC errors such as pidfd_getfd: Operation not permitted.
                self.export_regression_result(
                    prediction,
                    properties,
                    output_filename_truncated,
                    self.dataset_json,
                )
            _ = [r.get() for r in results]


        if self.local_rank == 0:
            metrics = compute_metrics_on_folder_regression(
                                                join(self.preprocessed_dataset_folder_base, 'gt_regressions'),
                                                validation_output_folder,
                                                join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)

            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation: ", (metrics['foreground_mean']["MAE"], metrics['foreground_mean']["RMSE"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def export_regression_result(self, prediction: torch.Tensor, properties: dict, 
                             output_file_truncated: str, dataset_json: dict):
        # 1. 确保是 numpy 或 tensor (推荐在 CPU 上操作，因为后续还原通常涉及 numpy 容器)
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().float().numpy()

        # 2. 反归一化处理 (保持为 numpy)
        if 'flux_normalization' in properties:
            f_min = properties['flux_normalization']['min']
            f_max = properties['flux_normalization']['max']
            prediction = prediction * (f_max - f_min) + f_min

        # 3. 【核心修复】强制对齐维度为 (C, D, H, W)
        # 不管 prediction 原本是 (D, H, W) 还是 (1, 1, D, H, W)
        # 我们都把它统一处理
        if prediction.ndim == 3:
            # (D, H, W) -> (1, D, H, W)
            prediction = prediction[None]
        elif prediction.ndim == 4:
            # 已经是 (C, D, H, W)，检查第一个维度是否为通道
            pass
        elif prediction.ndim == 5:
            # (1, 1, D, H, W) -> (1, D, H, W)
            prediction = prediction[0]

        # 4. 执行重采样还原
        # 注意：nnU-Net 的这个函数可以接受 numpy 也可以接受 tensor
        # 如果它是 data.numpy() 报错，传 numpy 进去最稳妥
        target_shape = properties['shape_after_cropping_and_before_resampling']
        current_spacing = self.configuration_manager.spacing
        original_spacing = properties['spacing']
        
        # 确保传入的是 4 维 Numpy 数组
        resampled = self.configuration_manager.resampling_fn_probabilities(
            prediction, target_shape, current_spacing, original_spacing
        )
        # resampled 返回的通常是 (C, D, H, W) 的 numpy 数组

        # 5. 还原裁剪并填充
        final_result = np.full(properties['shape_before_cropping'], f_min, dtype=np.float32)
        bbox = properties['bbox_used_for_cropping']
    
        # 填充时取第一个通道 [0]
        final_result[bbox[0][0]:bbox[0][1], 
                    bbox[1][0]:bbox[1][1], 
                    bbox[2][0]:bbox[2][1]] = resampled[0]
        


        # 6. 撤销转置
        if hasattr(self.plans_manager, 'transpose_backward'):
            final_result = final_result.transpose(self.plans_manager.transpose_backward)

        # 7. 使用 SimpleITK 保存物理意义上的 Float32 图像
        import SimpleITK as sitk
        full_output_file = output_file_truncated + dataset_json['file_ending']
        
        # Preserve the historical floor snapping for existing trainers. Models
        # that regress continuous low flux can disable it explicitly.
        if getattr(self, 'snap_low_flux_to_floor', True):
            final_result[(final_result > -12) & (final_result < -11)] = -12
        
        itk_image = sitk.GetImageFromArray(final_result)
        
        # 恢复该数据特有的空间元数据
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])
        
        # 保存并开启压缩
        sitk.WriteImage(itk_image, full_output_file, True)
