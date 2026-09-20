# PSPA-DT

PSPA-DT is an end-to-end deep learning framework for generating 3-D photon
flux distributions. It takes multimodal medical images as input and combines
pyramid spatial projection aggregation, wavelet-domain feature modeling, and
state-space models to directly predict photon flux within biological tissues.
Built on nnU-Net, the project provides a complete pipeline for preprocessing,
training, inference, flux denormalization, and evaluation, offering a fast
alternative to computationally expensive conventional optical simulations.

This repository packages PSPA-DT as an extension of the released
`nnunetv2==2.6.2`. It contains the method's network blocks, segmentation and
flux-regression trainers, continuous-target preprocessing, evaluation metrics,
dataset conversion, and command-line entry points.

## 1. Installation

Python 3.10 and a CUDA GPU with BF16 support are recommended. Install the
PyTorch build matching the local CUDA toolkit first, then install this package:

```bash
conda create -n pspa-dt python=3.10 -y
conda activate pspa-dt
# Example only; select the correct command at https://pytorch.org/get-started/
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation -e .
```

The `--no-build-isolation` option helps `mamba-ssm` locate the already installed
PyTorch/CUDA environment. Exact project dependencies are declared in both
`pyproject.toml` and `requirements.txt`.

## 2. Dataset layout

Each case contains three registered input modalities and an anatomical label:

```text
DATA/
  case001/
    t1.nii.gz
    t2.nii.gz
    mra.nii.gz
    seg.nii.gz
  case002/
    ...
```

Flux regression additionally expects:

```text
FLUX/
  case001_logFlux.nii.gz
  case002_logFlux.nii.gz
```

All volumes belonging to a case must have the same voxel grid and physical
metadata. The converter uses symbolic links by default. The included default
label map is the ten-class IXI tissue definition used in the experiments;
change `DEFAULT_LABELS` in `prepare_dataset.py`, or use `--labels-json` when
running the converter directly, for a different dataset.

The dataset itself is not redistributed. Readers must obtain it under its
original access conditions and cite its source.

## 3. One-command reproduction

Segmentation preprocessing and fold-0 training:

```bash
./run.sh seg \
  --data-root /absolute/path/to/DATA \
  --dataset-id 41 \
  --gpus 0
```

Flux-regression preprocessing and training:

```bash
./run.sh flux \
  --data-root /absolute/path/to/DATA \
  --flux-root /absolute/path/to/FLUX \
  --dataset-id 411 \
  --gpus 0
```

The script installs the local package if necessary, creates
`workspace/{raw,preprocessed,results}`, produces `dataset.json`, performs
nnU-Net planning and preprocessing, and starts training. It never modifies the
installed `nnunetv2` package. See all options with:

```bash
./run.sh --help
```

Resume an interrupted experiment without repeating preparation:

```bash
./run.sh flux --dataset-id 411 --skip-prepare --skip-preprocess \
  --continue-training --gpus 0
```

## 4. Individual commands

After setting the three standard nnU-Net environment variables, each stage can
also be run separately:

```bash
export nnUNet_raw=/path/to/workspace/raw
export nnUNet_preprocessed=/path/to/workspace/preprocessed
export nnUNet_results=/path/to/workspace/results

python prepare_dataset.py --mode flux --dataset-id 411 \
  --data-root /path/to/DATA --flux-root /path/to/FLUX \
  --raw-root "$nnUNet_raw"

pspa-dt-preprocess 411 --mode flux -np 8
pspa-dt-train 411 --mode flux --fold 0 --device cuda
```

For segmentation, replace `flux` with `seg` and omit `--flux-root`.

Flux inference on channel-suffixed nnU-Net test images:

```bash
pspa-dt-predict-flux -d 411 -c 3d_fullres -f 0 \
  -i /path/to/imagesTs -o /path/to/predictions --disable_tta
```

The predictor restores the original image geometry and reverses the fitted
flux normalization before writing Float32 NIfTI volumes.

## 5. Method-specific preprocessing

The flux preprocessor:

1. reads anatomical images, segmentation and continuous flux target;
2. checks finite values and matching spatial shape/spacing;
3. applies the same transpose, crop and resampling to all volumes;
4. uses image interpolation—not label interpolation—for continuous flux;
5. clips flux at the experimental lower bound of `-12` and the training-set
   maximum, then min-max normalizes it to `[0, 1]`;
6. records normalization metadata for physical-scale reconstruction.

The fitted values are written to
`preprocessed/DatasetXXX_*/flux_normalization.json`; this file is part of an
experiment's reproducibility record and should be retained with checkpoints.

## 6. Repository structure

```text
src/pspa_dt/
  network/          complete PSPA-DT network implementation
  training/         segmentation and flux trainers plus flux data loader
  preprocessing/    continuous flux preprocessor
  inference/        continuous regression evaluation metrics
  cli.py             preprocessing and training orchestration
prepare_dataset.py   raw-data conversion
run.sh               one-command entry point
```

## 7. Reproducibility checklist

For a paper release, archive the following alongside this repository:

- the exact dataset split (`splits_final.json`);
- generated plans and `dataset_fingerprint.json`;
- `flux_normalization.json` for flux experiments;
- trained checkpoints and validation summaries;
- CUDA, driver, GPU model and package versions (`pip freeze`);
- dataset download/access instructions and citation;
- random seeds and any deviation from the commands above.

The current public CLI intentionally uses one process/GPU for training. This
avoids depending on nnU-Net's internal trainer-name discovery in distributed
workers. Multi-GPU training can be added later through a package-native DDP
launcher without altering nnU-Net itself.

## License and attribution

See `LICENSE` and `THIRD_PARTY_NOTICE.md`. An EMCAD-derived decoder used in an
earlier internal version was removed because its official license restricts
redistribution. The public package instead contains the independently written
`PSPADecoder3D`. This architectural replacement means public-release results
must be reported separately from results obtained with the earlier decoder.

Official EMCAD source and terms:

- Paper: <https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html>
- Repository: <https://github.com/SLDGroup/EMCAD>
- License: <https://github.com/SLDGroup/EMCAD/blob/main/LICENSE>
