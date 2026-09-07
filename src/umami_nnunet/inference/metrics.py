"""Region-wise metrics for continuous flux predictions."""

import multiprocessing
from typing import List, Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, subfiles
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
from nnunetv2.evaluation.evaluate_predictions import save_summary_json
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from scipy.stats import pearsonr


def compute_metrics_regression(
    reference_file: str,
    prediction_file: str,
    segmentation_file: str,
    image_reader_writer: BaseReaderWriter,
    labels_or_regions,
    ignore_label: int = None,
) -> dict:
    reference, _ = image_reader_writer.read_seg(reference_file)
    prediction, _ = image_reader_writer.read_seg(prediction_file)
    segmentation, _ = image_reader_writer.read_seg(segmentation_file)
    reference = reference.astype(np.float32)
    prediction = prediction.astype(np.float32)
    result = {
        "reference_file": reference_file,
        "prediction_file": prediction_file,
        "segmentation_file": segmentation_file,
        "metrics": {},
    }
    for region in labels_or_regions:
        mask = region_or_label_to_mask(segmentation, region)
        if ignore_label is not None:
            mask &= segmentation != ignore_label
        truth, estimate = reference[mask], prediction[mask]
        if truth.size == 0:
            values = {k: np.nan for k in ("MAE", "MSE", "RMSE", "Pearson")}
            values["n_pixels"] = 0
        else:
            error = truth - estimate
            mse = np.mean(error**2)
            correlation = (
                pearsonr(truth, estimate).statistic
                if truth.size > 1 and np.std(truth) > 0 and np.std(estimate) > 0
                else np.nan
            )
            values = {
                "MAE": np.mean(np.abs(error)),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "Pearson": correlation,
                "n_pixels": int(truth.size),
            }
        result["metrics"][region] = values
    return result


def compute_metrics_on_folder_regression(
    folder_ref: str,
    folder_pred: str,
    folder_seg: str,
    output_file: str,
    image_reader_writer: BaseReaderWriter,
    file_ending: str,
    regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
    ignore_label: int = None,
    num_processes: int = 8,
    chill: bool = True,
) -> dict:
    prediction_names = subfiles(folder_pred, suffix=file_ending, join=False)
    if not chill:
        missing = [name for name in prediction_names if not isfile(join(folder_ref, name))]
        if missing:
            raise FileNotFoundError(f"Missing references: {missing}")
    jobs = [
        (
            join(folder_ref, name),
            join(folder_pred, name),
            join(folder_seg, name),
            image_reader_writer,
            regions_or_labels,
            ignore_label,
        )
        for name in prediction_names
    ]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        per_case = pool.starmap(compute_metrics_regression, jobs)
    means = {
        region: {
            metric: np.nanmean([case["metrics"][region][metric] for case in per_case])
            for metric in ("MAE", "MSE", "RMSE", "Pearson", "n_pixels")
        }
        for region in regions_or_labels
    }
    foreground_mean = {
        metric: np.nanmean([values[metric] for region, values in means.items() if region not in (0, "0")])
        for metric in ("MAE", "MSE", "RMSE", "Pearson", "n_pixels")
    }
    result = {"metric_per_case": per_case, "mean": means, "foreground_mean": foreground_mean}
    recursive_fix_for_json_export(result)
    if output_file:
        save_summary_json(result, output_file)
    return result
