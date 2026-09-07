import argparse
import os
from types import SimpleNamespace
from typing import List, Union

import numpy as np
import torch
import time
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import isdir, join, load_json, maybe_mkdir_p, save_json

from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.file_path_utilities import get_output_folder
from umami_nnunet.training.trainers import (
    nnUNetTrainer_umami_refine_wavelet_control_flux,
)


def export_flux_prediction(prediction: Union[np.ndarray, torch.Tensor], properties: dict,
                           configuration_manager, plans_manager, output_file: str,
                           flux_normalization: dict) -> None:
    """Restore a normalized continuous prediction to the raw image geometry."""
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu()

    spacing_transposed = [properties['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if (
        len(configuration_manager.spacing) == len(properties['shape_after_cropping_and_before_resampling'])
    ) else [spacing_transposed[0], *configuration_manager.spacing]

    prediction = configuration_manager.resampling_fn_probabilities(
        prediction,
        properties['shape_after_cropping_and_before_resampling'],
        current_spacing,
        spacing_transposed,
    )
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim != 4 or prediction.shape[0] != 1:
        raise RuntimeError(
            f'Flux regression expects one output channel with shape (1, D, H, W), got {prediction.shape}'
        )

    flux_min = float(flux_normalization['min'])
    flux_max = float(flux_normalization['max'])
    prediction = prediction[0] * (flux_max - flux_min) + flux_min

    restored = np.full(properties['shape_before_cropping'], flux_min, dtype=np.float32)
    bbox = properties['bbox_used_for_cropping']
    bbox_slices = tuple(slice(int(start), int(end)) for start, end in bbox)
    restored[bbox_slices] = prediction
    restored = restored.transpose(plans_manager.transpose_backward)

    # write_seg casts to an integer label type, so continuous regression output
    # must be written explicitly as Float32.
    if 'sitk_stuff' not in properties:
        raise RuntimeError('Flux export currently requires SimpleITK image geometry in data_properties.')
    import SimpleITK as sitk

    itk_image = sitk.GetImageFromArray(restored.astype(np.float32, copy=False))
    itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
    itk_image.SetOrigin(properties['sitk_stuff']['origin'])
    itk_image.SetDirection(properties['sitk_stuff']['direction'])
    sitk.WriteImage(itk_image, output_file, True)


class FluxRegressionPredictor(nnUNetPredictor):
    """nnU-Net predictor that preserves a one-channel output as continuous flux."""


    @torch.inference_mode()
    def predict_whole_volume_flux(
        self,
        data: torch.Tensor,
        warmup: int = 1,
        repeat: int = 10,
    ):
        """
        Whole-volume flux prediction.

        Timing ONLY includes:
            forward_flux()

        Timing DOES NOT include:
            preprocessing
            prepare_flux_inputs()
            NIfTI export
            resampling
            saving

        This is intended for comparison with MCX fluence computation time.
        """

        self.network = self.network.to(self.device)
        self.network.eval()

        # ------------------------------------------------------
        # For runtime benchmarking, strongly recommend one fold
        # ------------------------------------------------------
        if len(self.list_of_parameters) != 1:
            print(
                f"WARNING: {len(self.list_of_parameters)} folds are loaded. "
                "For runtime comparison, use '-f 0'."
            )

        # ------------------------------------------------------
        # data from nnU-Net preprocessor:
        # [C, D, H, W]
        # ------------------------------------------------------
        data = data.to(
            self.device,
            dtype=torch.float32,
            non_blocking=True
        )

        print("\n========================================")
        print("Whole-volume inference")
        print("========================================")
        print("Preprocessed shape:", tuple(data.shape))

        original_shape = data.shape[1:]

        # ------------------------------------------------------
        # Determine divisibility required by the network
        #
        # pool_op_kernel_sizes corresponds to encoder strides.
        # Example:
        # [[1,1,1],
        #  [2,2,2],
        #  [2,2,2],
        #  [2,2,2]]
        #
        # -> divisibility = [8,8,8]
        # ------------------------------------------------------
        strides = np.asarray(
            self.configuration_manager.pool_op_kernel_sizes,
            dtype=np.int64
        )

        divisibility = np.prod(
            strides,
            axis=0
        ).astype(int)

        print(
            "Required divisibility:",
            divisibility.tolist()
        )

        d, h, w = original_shape

        pad_d = (
            int(divisibility[0])
            - d % int(divisibility[0])
        ) % int(divisibility[0])

        pad_h = (
            int(divisibility[1])
            - h % int(divisibility[1])
        ) % int(divisibility[1])

        pad_w = (
            int(divisibility[2])
            - w % int(divisibility[2])
        ) % int(divisibility[2])

        # F.pad order:
        # W-left, W-right,
        # H-left, H-right,
        # D-left, D-right
        data = F.pad(
            data,
            (
                0, pad_w,
                0, pad_h,
                0, pad_d
            ),
            mode="constant",
            value=0
        )

        print(
            "Padded shape:",
            tuple(data.shape)
        )

        # add batch dimension
        # [C,D,H,W] -> [1,C,D,H,W]
        x = data.unsqueeze(0)

        predictions = []
        all_times = []

        # ======================================================
        # Fold loop
        # ======================================================

        for fold_idx, params in enumerate(
            self.list_of_parameters
        ):

            print(
                f"\nFold {fold_idx + 1}/"
                f"{len(self.list_of_parameters)}"
            )

            # --------------------------------------------------
            # Load checkpoint
            # --------------------------------------------------
            if hasattr(self.network, "_orig_mod"):
                self.network._orig_mod.load_state_dict(
                    params
                )
                net = self.network._orig_mod
            else:
                self.network.load_state_dict(
                    params
                )
                net = self.network

            net.eval()

            # ==================================================
            # Step 1:
            # Prepare anatomical features and conditions
            #
            # NOT included in timing
            # ==================================================

            with torch.autocast(
                device_type=self.device.type,
                enabled=(self.device.type == "cuda")
            ):

                bottleneck, skip, conditions = \
                    net.prepare_flux_inputs(x)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # ==================================================
            # Step 2:
            # Warm-up
            #
            # NOT included in final statistics
            # ==================================================

            print(
                f"Warm-up: {warmup} runs"
            )

            for _ in range(warmup):

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=(self.device.type == "cuda")
                ):

                    _ = net.forward_flux(
                        bottleneck,
                        skip,
                        conditions
                    )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # ==================================================
            # Step 3:
            # Benchmark whole-volume flux regression
            # ==================================================

            print(
                f"Benchmark: {repeat} runs"
            )

            fold_times = []

            prediction = None

            for run_idx in range(repeat):

                if self.device.type == "cuda":

                    start_event = torch.cuda.Event(
                        enable_timing=True
                    )

                    end_event = torch.cuda.Event(
                        enable_timing=True
                    )

                    start_event.record()

                    with torch.autocast(
                        device_type="cuda",
                        enabled=True
                    ):

                        prediction = net.forward_flux(
                            bottleneck,
                            skip,
                            conditions
                        )

                    end_event.record()

                    torch.cuda.synchronize()

                    elapsed_ms = \
                        start_event.elapsed_time(
                            end_event
                        )

                    elapsed_s = \
                        elapsed_ms / 1000.0

                else:

                    start_time = \
                        time.perf_counter()

                    prediction = net.forward_flux(
                        bottleneck,
                        skip,
                        conditions
                    )

                    elapsed_s = \
                        time.perf_counter() - start_time

                    elapsed_ms = \
                        elapsed_s * 1000.0

                fold_times.append(
                    elapsed_s
                )

                print(
                    f"Run {run_idx + 1:02d}: "
                    f"{elapsed_s:.6f} s "
                    f"({elapsed_ms:.3f} ms)"
                )

            all_times.extend(
                fold_times
            )

            # --------------------------------------------------
            # Remove batch dimension:
            #
            # [1,1,D,H,W] -> [1,D,H,W]
            # --------------------------------------------------
            prediction = prediction[0]

            # --------------------------------------------------
            # Remove padding
            # --------------------------------------------------
            prediction = prediction[
                :,
                :d,
                :h,
                :w
            ]

            predictions.append(
                prediction.float().cpu()
            )

            fold_times = np.asarray(
                fold_times
            )

            print(
                f"\nFold mean: "
                f"{fold_times.mean():.6f} s"
            )

            print(
                f"Fold std:  "
                f"{fold_times.std():.6f} s"
            )

        # ======================================================
        # Fold ensemble
        # ======================================================

        if len(predictions) == 1:

            prediction = predictions[0]

        else:

            prediction = torch.stack(
                predictions,
                dim=0
            ).mean(dim=0)

        # ======================================================
        # Runtime summary
        # ======================================================

        all_times = np.asarray(
            all_times
        )

        print("\n========================================")
        print("Whole-volume Flux Regression Runtime")
        print("========================================")

        print(
            f"Mean: "
            f"{all_times.mean():.6f} s/case"
        )

        print(
            f"Std:  "
            f"{all_times.std():.6f} s"
        )

        print(
            f"Min:  "
            f"{all_times.min():.6f} s"
        )

        print(
            f"Max:  "
            f"{all_times.max():.6f} s"
        )

        print(
            f"Mean: "
            f"{all_times.mean() * 1000:.3f} ms/case"
        )

        print("========================================\n")

        return prediction

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds, checkpoint_name: str = 'checkpoint_final.pth'):
        dataset_name = os.path.basename(os.path.dirname(model_training_output_dir))
        env_name = 'nnUNet_flux_inference_dataset_name'
        loading_env_name = 'nnUNet_flux_loading_trained_checkpoint'
        previous_dataset_name = os.environ.get(env_name)
        previous_loading_value = os.environ.get(loading_env_name)
        os.environ[env_name] = dataset_name
        os.environ[loading_env_name] = '1'
        # nnU-Net normally discovers trainers only inside its own package.
        # Temporarily provide this extension's trainer without modifying the
        # installed nnU-Net distribution.
        import nnunetv2.inference.predict_from_raw_data as prediction_module
        original_finder = prediction_module.recursive_find_python_class

        def find_trainer(folder, class_name, current_module):
            if class_name == "nnUNetTrainer_umami_refine_wavelet_control_flux":
                return nnUNetTrainer_umami_refine_wavelet_control_flux
            return original_finder(folder, class_name, current_module)

        prediction_module.recursive_find_python_class = find_trainer
        try:
            super().initialize_from_trained_model_folder(
                model_training_output_dir, use_folds, checkpoint_name
            )
        finally:
            prediction_module.recursive_find_python_class = original_finder
            if previous_dataset_name is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous_dataset_name
            if previous_loading_value is None:
                os.environ.pop(loading_env_name, None)
            else:
                os.environ[loading_env_name] = previous_loading_value
        # The base sliding-window implementation allocates its accumulator from
        # label_manager.num_segmentation_heads. Dataset 511 has 17 segmentation
        # classes, but every flux-regression trainer returns one continuous
        # channel. This predictor never performs label conversion, so a minimal
        # one-head label-manager view is the correct allocation contract here.
        self.segmentation_label_manager = self.label_manager
        self.label_manager = SimpleNamespace(num_segmentation_heads=1)

    def _set_network_output(self, output: str) -> None:
        network = self.network._orig_mod if hasattr(self.network, '_orig_mod') else self.network
        if not hasattr(network, 'inference_output'):
            if output == 'flux':
                # The public WaveletUmamiRefineControlFlux model is already a
                # dedicated one-channel regressor and needs no output switch.
                return
            raise RuntimeError(
                f'{type(network).__name__} does not support optional segmentation output. '
                'Use nnUNetTrainer_umami_flux_v6 or disable --save_seg.'
            )
        network.inference_output = output

    def predict_flux_from_files(self, input_folder: str, output_folder: str,
                                flux_normalization: dict, overwrite: bool = True,
                                num_parts: int = 1, part_id: int = 0,
                                seg_output_folder: Union[str, None] = None) -> List[str]:
        maybe_mkdir_p(output_folder)
        if seg_output_folder is not None:
            maybe_mkdir_p(seg_output_folder)
        input_lists, output_names, previous_stage = self._manage_input_and_output_lists(
            input_folder, output_folder, None,
            overwrite or seg_output_folder is not None, part_id, num_parts, False
        )
        if any(i is not None for i in previous_stage):
            raise RuntimeError('Cascaded configurations are not supported by the flux regression predictor.')

        save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
        save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        save_json(flux_normalization, join(output_folder, 'flux_normalization.json'), sort_keys=False)
        if seg_output_folder is not None:
            save_json(self.dataset_json, join(seg_output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(seg_output_folder, 'plans.json'), sort_keys=False)

        if not overwrite and seg_output_folder is not None:
            keep = []
            for idx, output_truncated in enumerate(output_names):
                case_name = os.path.basename(output_truncated)
                flux_file = output_truncated + self.dataset_json['file_ending']
                seg_file = join(seg_output_folder, case_name) + self.dataset_json['file_ending']
                if not (os.path.isfile(flux_file) and os.path.isfile(seg_file)):
                    keep.append(idx)
            input_lists = [input_lists[i] for i in keep]
            output_names = [output_names[i] for i in keep]

        # FluxRegressionPreprocessor.run_case is training-only because it needs a
        # ground-truth flux file. Raw inference uses the standard image pipeline.
        preprocessor = DefaultPreprocessor(verbose=self.verbose_preprocessing)
        written = []
        for image_files, output_truncated in zip(input_lists, output_names):
            output_file = output_truncated + self.dataset_json['file_ending']
            print(f'\nPredicting {os.path.basename(output_truncated)}:')
            data, _, properties = preprocessor.run_case(
                image_files,
                None,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json,
            )

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
            # predict_logits_from_preprocessed_data returns an inference tensor.
            # Use an out-of-place operation because inference tensors cannot be
            # modified in-place after leaving torch.inference_mode().
            self.label_manager = SimpleNamespace(num_segmentation_heads=1)
            self._set_network_output('flux')
            logits = self.predict_logits_from_preprocessed_data(data)
            prediction = torch.sigmoid(logits)
            # prediction = self.predict_whole_volume_flux(
            #     data,
            #     warmup=1,
            #     repeat=10
            # ).clamp(0.0, 1.0)
            export_flux_prediction(
                prediction,
                properties,
                self.configuration_manager,
                self.plans_manager,
                output_file,
                flux_normalization,
            )
            written.append(output_file)
            print(f'Wrote {output_file}')

            if seg_output_folder is not None:
                case_name = os.path.basename(output_truncated)
                seg_output_truncated = join(seg_output_folder, case_name)
                print(f'Predicting segmentation for {case_name}:')
                self.label_manager = self.segmentation_label_manager
                self._set_network_output('segmentation')
                seg_logits = self.predict_logits_from_preprocessed_data(data)
                export_prediction_from_logits(
                    seg_logits,
                    properties,
                    self.configuration_manager,
                    self.plans_manager,
                    self.dataset_json,
                    seg_output_truncated,
                    save_probabilities=False,
                )
                print(f'Wrote {seg_output_truncated + self.dataset_json["file_ending"]}')

        self.label_manager = SimpleNamespace(num_segmentation_heads=1)
        self._set_network_output('flux')
        return written


def predict_flux_entry_point() -> None:
    parser = argparse.ArgumentParser(
        description='Predict a continuous flux volume with a flux-regression nnU-Net model.'
    )
    parser.add_argument('-i', required=True, help='Input folder containing nnU-Net channel-suffixed images.')
    parser.add_argument('-o', required=True, help='Output folder for Float32 flux volumes.')
    parser.add_argument('-d', required=True, help='Dataset name or ID.')
    parser.add_argument('-p', default='nnUNetPlans_flux_regression', help='Plans identifier.')
    parser.add_argument('-tr', default='nnUNetTrainer_umami_refine_wavelet_control_flux',
                        help='Trainer class used for training.')
    parser.add_argument('-c', required=True, help='nnU-Net configuration.')
    parser.add_argument('-f', nargs='+', default=(0, 1, 2, 3, 4), help='Fold(s) used for prediction.')
    parser.add_argument('-chk', default='checkpoint_final.pth', help='Checkpoint filename.')
    parser.add_argument('-step_size', type=float, default=0.5, help='Sliding-window step size.')
    parser.add_argument('--disable_tta', action='store_true', help='Disable mirroring test-time augmentation.')
    parser.add_argument('--save_seg', action='store_true',
                        help='Only for compatible joint models; unavailable for the public flux-only model.')
    parser.add_argument('--seg_output_folder', default=None,
                        help='Segmentation output folder. Default: <output folder>/seg.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Do not overwrite existing predictions.')
    parser.add_argument('-num_parts', type=int, default=1)
    parser.add_argument('-part_id', type=int, default=0)
    parser.add_argument('-device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--disable_progress_bar', action='store_true')
    args = parser.parse_args()

    if args.part_id >= args.num_parts:
        raise ValueError('part_id must be smaller than num_parts')
    folds = [i if i == 'all' else int(i) for i in args.f]
    dataset_name = maybe_convert_to_dataset_name(args.d)
    normalization_file = join(nnUNet_preprocessed, dataset_name, 'flux_normalization.json')
    if not os.path.isfile(normalization_file):
        raise FileNotFoundError(
            f'Missing {normalization_file}. Run flux preprocessing first or restore this file from training.'
        )
    flux_normalization = load_json(normalization_file)

    if args.device == 'cpu':
        torch.set_num_threads(os.cpu_count() or 1)
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    device = torch.device(args.device)

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)
    if not isdir(model_folder):
        raise FileNotFoundError(f'Model folder does not exist: {model_folder}')

    predictor = FluxRegressionPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=not args.disable_progress_bar,
    )
    predictor.initialize_from_trained_model_folder(model_folder, folds, checkpoint_name=args.chk)
    seg_output_folder = (args.seg_output_folder or join(args.o, 'seg')) if args.save_seg else None
    predictor.predict_flux_from_files(
        args.i,
        args.o,
        flux_normalization,
        overwrite=not args.continue_prediction,
        num_parts=args.num_parts,
        part_id=args.part_id,
        seg_output_folder=seg_output_folder,
    )


if __name__ == '__main__':
    predict_flux_entry_point()
