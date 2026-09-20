import multiprocessing
import os
import shutil
from time import sleep
from typing import List, Union

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import isdir, isfile, join, load_json, maybe_mkdir_p, save_json
from tqdm import tqdm

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


class FluxRegressionPreprocessor(DefaultPreprocessor):
    """
    Preprocesses regular nnU-Net images plus a continuous-valued flux target.

    The dataset.json must keep the anatomical segmentation in "label" and add a
    per-case "flux" entry. The segmentation is used for planning, normalization
    masks and foreground sampling. The saved target is the resampled flux image.
    """
    flux_min = -12.0
    flux_normalization = 'dataset_minmax'

    @staticmethod
    def _resolve_path(raw_dataset_folder: str, filename: str) -> str:
        expanded = os.path.expandvars(filename)
        return os.path.abspath(join(raw_dataset_folder, expanded)) if not os.path.isabs(expanded) else expanded

    @classmethod
    def normalize_flux(cls, flux: np.ndarray, flux_max: float) -> np.ndarray:
        if flux_max <= cls.flux_min:
            raise RuntimeError(f'Invalid flux_max={flux_max}. It must be greater than flux_min={cls.flux_min}.')
        flux = np.clip(flux, cls.flux_min, flux_max)
        return (flux - cls.flux_min) / (flux_max - cls.flux_min)

    @classmethod
    def denormalize_flux(cls, flux: np.ndarray, flux_max: float) -> np.ndarray:
        return flux * (flux_max - cls.flux_min) + cls.flux_min

    @classmethod
    def flux_normalization_properties(cls, flux_max: float) -> dict:
        return {
            'name': cls.flux_normalization,
            'min': cls.flux_min,
            'max': float(flux_max),
            'normalized_min': 0.0,
            'normalized_max': 1.0,
            'denormalize': 'flux = normalized_flux * (max - min) + min'
        }

    @staticmethod
    def determine_flux_max(flux_files: List[str], plans_manager: PlansManager) -> float:
        rw = plans_manager.image_reader_writer_class()
        flux_max = -np.inf
        for flux_file in flux_files:
            flux, _ = rw.read_images([flux_file])
            if not np.all(np.isfinite(flux)):
                raise RuntimeError(f'Flux target contains NaN or Inf values: {flux_file}')
            flux_max = max(flux_max, float(np.max(flux)))
        return flux_max

    def run_case_npy(self, data: np.ndarray, seg: np.ndarray, flux: np.ndarray, properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str], flux_max: float):
        data = data.astype(np.float32)
        flux = flux.astype(np.float32)
        seg = np.copy(seg)

        assert data.shape[1:] == seg.shape[1:], \
            "Shape mismatch between image and segmentation. Please fix your dataset."
        assert data.shape[1:] == flux.shape[1:], \
            "Shape mismatch between image and flux target. Please fix your dataset."
        if not np.all(np.isfinite(flux)):
            raise RuntimeError("Flux target contains NaN or Inf values. Please fix your dataset.")

        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        flux = flux.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping

        data, seg, bbox = crop_to_nonzero(data, seg)
        flux = flux[(slice(None),) + bounding_box_to_slice(bbox)]
        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        flux = configuration_manager.resampling_fn_data(flux, new_shape, original_spacing, target_spacing)
        properties['flux_normalization'] = self.flux_normalization_properties(flux_max)
        properties['flux_before_normalization_min'] = float(np.min(flux))
        properties['flux_before_normalization_max'] = float(np.max(flux))
        flux = self.normalize_flux(flux, flux_max)
        properties['flux_after_normalization_min'] = float(np.min(flux))
        properties['flux_after_normalization_max'] = float(np.max(flux))
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        label_manager = plans_manager.get_label_manager(dataset_json)
        collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
            else label_manager.foreground_labels
        if label_manager.has_ignore_label:
            collect_for_this.append([-1] + label_manager.all_labels)
        properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                          verbose=self.verbose)

        return data.astype(np.float32, copy=False), flux.astype(np.float32, copy=False), properties

    def run_case(self, image_files: List[str], seg_file: str, flux_file: str,
                 plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str], flux_max: float):
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()
        data, data_properties = rw.read_images(image_files)
        seg, _ = rw.read_seg(seg_file)
        flux, flux_properties = rw.read_images([flux_file])

        if not np.allclose(data_properties['spacing'], flux_properties['spacing']):
            raise RuntimeError(f"Spacing mismatch between image and flux target: {image_files} vs {flux_file}")

        if self.verbose:
            print(flux_file)
        data, flux, data_properties = self.run_case_npy(data, seg, flux, data_properties, plans_manager,
                                                        configuration_manager, dataset_json, flux_max)
        return data, flux, data_properties

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str, flux_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str], flux_max: float):
        data, flux, properties = self.run_case(image_files, seg_file, flux_file, plans_manager,
                                               configuration_manager, dataset_json, flux_max)
        block_size_data, chunk_size_data = nnUNetDatasetBlosc2.comp_blosc2_params(
            data.shape, tuple(configuration_manager.patch_size), data.itemsize)
        block_size_flux, chunk_size_flux = nnUNetDatasetBlosc2.comp_blosc2_params(
            flux.shape, tuple(configuration_manager.patch_size), flux.itemsize)

        nnUNetDatasetBlosc2.save_case(data, flux, properties, output_filename_truncated,
                                      chunks=chunk_size_data, blocks=block_size_data,
                                      chunks_seg=chunk_size_flux, blocks_seg=block_size_flux)

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)
        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)
        raw_dataset_folder = join(nnUNet_raw, dataset_name)
        for k in dataset:
            if 'flux' not in dataset[k]:
                raise RuntimeError(f'Case {k} is missing a "flux" entry in dataset.json')
            dataset[k]['flux'] = self._resolve_path(raw_dataset_folder, dataset[k]['flux'])
        flux_max = self.determine_flux_max([dataset[k]['flux'] for k in dataset], plans_manager)
        flux_normalization = self.flux_normalization_properties(flux_max)
        save_json(flux_normalization, join(nnUNet_preprocessed, dataset_name, 'flux_normalization.json'),
                  sort_keys=False)
        if self.verbose:
            print(f'Using flux normalization range [{self.flux_min}, {flux_max}]')

        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            workers = [j for j in p._pool]
            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                           dataset[k]['flux'], plans_manager, configuration_manager,
                                           dataset_json, flux_max),)))

            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('One of your background preprocessing processes died. This can happen if '
                                           'there is an error in a case or if the system runs out of RAM.')
                    done = [i for i in remaining if r[i].ready()]
                    for index in done:
                        r[index].get()
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)
