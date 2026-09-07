#!/usr/bin/env python3
"""Create an nnU-Net v2 dataset from the public Umami data layout."""

import argparse
import json
import os
from pathlib import Path


DEFAULT_LABELS = {
    "background": 0,
    "WM": 1,
    "GM": 2,
    "CSF": 3,
    "Muscle": 4,
    "Scalp": 5,
    "Eye balls": 6,
    "Compact bone": 7,
    "Spongy bone": 8,
    "Vessel": 9,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare segmentation or flux-regression data for Umami."
    )
    parser.add_argument("--mode", choices=("seg", "flux"), required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--flux-root", type=Path)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="Umami")
    parser.add_argument("--image-files", nargs="+", default=["t1.nii.gz", "t2.nii.gz", "mra.nii.gz"])
    parser.add_argument("--channel-names", nargs="+", default=["T1", "T2", "MRA"])
    parser.add_argument("--label-file", default="seg.nii.gz")
    parser.add_argument("--flux-suffix", default="_logFlux.nii.gz")
    parser.add_argument("--labels-json", type=Path)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink (uses more disk space).")
    parser.add_argument("--force", action="store_true", help="Replace links/files previously created by this script.")
    return parser.parse_args()


def place(source: Path, destination: Path, copy: bool, force: bool) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if not force:
            if destination.resolve() == source:
                return
            raise FileExistsError(f"{destination} already exists; pass --force to replace it")
        destination.unlink()
    if copy:
        import shutil
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def main() -> None:
    args = arguments()
    if len(args.image_files) != len(args.channel_names):
        raise ValueError("--image-files and --channel-names must have the same length")
    if args.mode == "flux" and args.flux_root is None:
        raise ValueError("--flux-root is required in flux mode")
    if not 1 <= args.dataset_id <= 999:
        raise ValueError("--dataset-id must be between 1 and 999")

    cases = sorted(path for path in args.data_root.resolve().iterdir() if path.is_dir())
    if not cases:
        raise RuntimeError(f"No case directories found under {args.data_root}")

    suffix = "_flux" if args.mode == "flux" else ""
    folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}{suffix}"
    dataset_dir = args.raw_root.resolve() / folder_name
    images_dir, labels_dir = dataset_dir / "imagesTr", dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "flux":
        (dataset_dir / "fluxTr").mkdir(exist_ok=True)

    dataset_entries = {}
    for case_dir in cases:
        case_id = case_dir.name
        images = []
        for channel, filename in enumerate(args.image_files):
            destination = images_dir / f"{case_id}_{channel:04d}.nii.gz"
            place(case_dir / filename, destination, args.copy, args.force)
            images.append(f"./imagesTr/{destination.name}")
        label_destination = labels_dir / f"{case_id}.nii.gz"
        place(case_dir / args.label_file, label_destination, args.copy, args.force)
        entry = {"images": images, "label": f"./labelsTr/{label_destination.name}"}
        if args.mode == "flux":
            flux_source = args.flux_root.resolve() / f"{case_id}{args.flux_suffix}"
            flux_destination = dataset_dir / "fluxTr" / flux_source.name
            place(flux_source, flux_destination, args.copy, args.force)
            entry["flux"] = f"./fluxTr/{flux_destination.name}"
        dataset_entries[case_id] = entry

    labels = DEFAULT_LABELS
    if args.labels_json:
        labels = json.loads(args.labels_json.read_text(encoding="utf-8"))
    dataset_json = {
        "channel_names": {str(i): name for i, name in enumerate(args.channel_names)},
        "labels": labels,
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": folder_name,
        "licence": "See the license of the original dataset.",
        "description": "Dataset prepared by the public Umami pipeline.",
        "dataset": dataset_entries,
    }
    output = dataset_dir / "dataset.json"
    output.write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(cases)} cases in {dataset_dir}")


if __name__ == "__main__":
    main()
