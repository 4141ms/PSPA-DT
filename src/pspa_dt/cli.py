"""Command-line entry points that do not patch the installed nnU-Net package."""

import argparse
import os
import shutil
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprint_dataset,
    plan_experiment_dataset,
    preprocess_dataset,
)
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

from .preprocessing import FluxRegressionPreprocessor
from .training import (
    nnUNetTrainer_PSPADT,
    nnUNetTrainer_PSPADTFlux,
)


def preprocess_main() -> None:
    parser = argparse.ArgumentParser(description="Plan and preprocess a PSPA-DT dataset")
    parser.add_argument("dataset_id", type=int)
    parser.add_argument("--mode", choices=("seg", "flux"), required=True)
    parser.add_argument("-np", "--num-processes", type=int, default=8)
    parser.add_argument("-c", "--configuration", default="3d_fullres")
    args = parser.parse_args()
    extract_fingerprint_dataset(args.dataset_id, num_processes=args.num_processes)
    plans_name = "nnUNetPlans_flux_regression" if args.mode == "flux" else "nnUNetPlans"
    plan_experiment_dataset(
        args.dataset_id,
        preprocess_class_name="DefaultPreprocessor",
        overwrite_plans_name=plans_name,
    )
    if args.mode == "seg":
        preprocess_dataset(args.dataset_id, plans_name, (args.configuration,), (args.num_processes,))
        return

    preprocessor = FluxRegressionPreprocessor(verbose=True)
    preprocessor.run(args.dataset_id, args.configuration, plans_name, args.num_processes)
    dataset_name = maybe_convert_to_dataset_name(args.dataset_id)
    raw_folder = join(nnUNet_raw, dataset_name)
    preprocessed_folder = join(nnUNet_preprocessed, dataset_name)
    dataset_json = load_json(join(raw_folder, "dataset.json"))
    dataset = get_filenames_of_train_images_and_targets(raw_folder, dataset_json)
    regression_destination = join(preprocessed_folder, "gt_regressions")
    segmentation_destination = join(preprocessed_folder, "gt_segmentations")
    maybe_mkdir_p(regression_destination)
    maybe_mkdir_p(segmentation_destination)
    for case_id, files in dataset.items():
        flux = dataset_json["dataset"][case_id]["flux"]
        flux = os.path.expandvars(flux)
        if not os.path.isabs(flux):
            flux = str(Path(raw_folder) / flux)
        ending = dataset_json["file_ending"]
        shutil.copy2(flux, join(regression_destination, case_id + ending))
        shutil.copy2(files["label"], join(segmentation_destination, case_id + ending))


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train the PSPA-DT nnU-Net extension")
    parser.add_argument("dataset")
    parser.add_argument("--mode", choices=("seg", "flux"), required=True)
    parser.add_argument("--fold", default="0")
    parser.add_argument("-c", "--configuration", default="3d_fullres")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    plans_name = "nnUNetPlans_flux_regression" if args.mode == "flux" else "nnUNetPlans"
    base = join(nnUNet_preprocessed, dataset_name)
    plans = load_json(join(base, plans_name + ".json"))
    dataset_json = load_json(join(base, "dataset.json"))
    trainer_class = (
        nnUNetTrainer_PSPADTFlux
        if args.mode == "flux"
        else nnUNetTrainer_PSPADT
    )
    fold = args.fold if args.fold == "all" else int(args.fold)
    trainer = trainer_class(
        plans=plans,
        configuration=args.configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=torch.device(args.device),
    )
    if args.continue_training:
        candidates = ["checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth"]
        checkpoint = next(
            (join(trainer.output_folder, name) for name in candidates if os.path.isfile(join(trainer.output_folder, name))),
            None,
        )
        if checkpoint:
            trainer.load_checkpoint(checkpoint)
    trainer.run_training()
    trainer.perform_actual_validation(False)
