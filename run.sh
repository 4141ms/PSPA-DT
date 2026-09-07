#!/usr/bin/env bash
set -euo pipefail

usage() {
  sed -n '2,39p' "$0" | sed -n 's/^# //p'
}

# Install, preprocess, and train the standalone Umami extension.
#
# Usage:
#   ./run.sh seg  --data-root DATA --dataset-id 41 [options]
#   ./run.sh flux --data-root DATA --flux-root FLUX --dataset-id 411 [options]
#
# DATA/<case> must contain t1.nii.gz, t2.nii.gz, mra.nii.gz and seg.nii.gz.
# FLUX must contain <case>_logFlux.nii.gz (flux mode only).
#
# Options:
#   --workspace DIR       nnU-Net raw/preprocessed/results root (default: ./workspace)
#   --python EXECUTABLE   Python interpreter (default: python)
#   --fold N              fold to train (default: 0)
#   --gpus LIST           CUDA_VISIBLE_DEVICES (default: 0; training uses one GPU)
#   --processes N         preprocessing/augmentation processes (default: 8)
#   --skip-prepare        reuse an existing raw dataset
#   --skip-preprocess     reuse existing plans and preprocessed data
#   --continue-training   resume the latest available checkpoint
#   --copy                copy raw inputs instead of symlinking them
#   --force               replace generated input links
#   --no-install          fail instead of installing this package when absent
#   --help                show this help

if [[ $# -lt 1 ]] || [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
mode="$1"; shift
if [[ "$mode" != "seg" && "$mode" != "flux" ]]; then
  echo "ERROR: first argument must be seg or flux" >&2; exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$script_dir/workspace"; python_bin=python
data_root=""; flux_root=""; dataset_id=""; fold=0; gpus=0; processes=8
skip_prepare=0; skip_preprocess=0; install=1; continue_flag=""; copy_flag=""; force_flag=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) data_root="$2"; shift 2 ;;
    --flux-root) flux_root="$2"; shift 2 ;;
    --dataset-id) dataset_id="$2"; shift 2 ;;
    --workspace) workspace="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --fold) fold="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --processes) processes="$2"; shift 2 ;;
    --skip-prepare) skip_prepare=1; shift ;;
    --skip-preprocess) skip_preprocess=1; shift ;;
    --continue-training) continue_flag="--continue-training"; shift ;;
    --copy) copy_flag="--copy"; shift ;;
    --force) force_flag="--force"; shift ;;
    --no-install) install=0; shift ;;
    --help) usage; exit 0 ;;
    *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
  esac
done

[[ -n "$dataset_id" ]] || { echo "ERROR: --dataset-id is required" >&2; exit 2; }
if [[ "$skip_prepare" -eq 0 ]]; then
  [[ -n "$data_root" ]] || { echo "ERROR: --data-root is required" >&2; exit 2; }
  [[ "$mode" == "seg" || -n "$flux_root" ]] || {
    echo "ERROR: --flux-root is required in flux mode" >&2; exit 2;
  }
fi

if [[ "$install" -eq 1 ]]; then
  "$python_bin" -m pip install --no-build-isolation -e "$script_dir"
elif ! "$python_bin" -c 'import umami_nnunet' >/dev/null 2>&1; then
    echo "ERROR: package not installed; run: $python_bin -m pip install -e $script_dir" >&2
    exit 1
fi

export nnUNet_raw="$workspace/raw"
export nnUNet_preprocessed="$workspace/preprocessed"
export nnUNet_results="$workspace/results"
export nnUNet_n_proc_DA="$processes"
export CUDA_VISIBLE_DEVICES="$gpus"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

if [[ "$skip_prepare" -eq 0 ]]; then
  prepare=("$python_bin" "$script_dir/prepare_dataset.py" --mode "$mode"
    --dataset-id "$dataset_id" --data-root "$data_root" --raw-root "$nnUNet_raw")
  [[ -n "$flux_root" ]] && prepare+=(--flux-root "$flux_root")
  [[ -n "$copy_flag" ]] && prepare+=("$copy_flag")
  [[ -n "$force_flag" ]] && prepare+=("$force_flag")
  "${prepare[@]}"
fi

if [[ "$skip_preprocess" -eq 0 ]]; then
  "$python_bin" -m umami_nnunet.cli_preprocess "$dataset_id" \
    --mode "$mode" -np "$processes"
fi

train=("$python_bin" -m umami_nnunet.cli_train "$dataset_id"
  --mode "$mode" --fold "$fold")
[[ -n "$continue_flag" ]] && train+=("$continue_flag")
"${train[@]}"
