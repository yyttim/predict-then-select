"""Prepare every eligible BSCCMNIST cell with a fixed, outcome-independent FOV split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import zarr

from cellselect.data import (
    CHANNELS,
    CLASS_NAMES,
    MARKER_COLUMNS,
    assign_group_split,
    make_cohort,
    sha256_file,
    training_statistics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-seed", type=int, default=20261021)
    args = parser.parse_args()
    source = args.source or args.data_root / "raw" / "BSCCMNIST"
    output = args.output or args.data_root / "prepared" / "bsccmnist_cd16_v1"
    if (output / "manifest.json").exists():
        raise RuntimeError(
            "Prepared data already locked. Reuse it; do not silently overwrite a split."
        )
    source_files = ["BSCCM_index.csv", "BSCCM_surface_markers.csv", "BSCCM_global_metadata.json"]
    if not all((source / name).is_file() for name in source_files):
        raise FileNotFoundError(
            f"Source metadata missing in {source}; complete download/extraction first"
        )
    global_metadata = json.loads((source / source_files[2]).read_text())
    if "MNIST" not in global_metadata["name"] or "tiny" in global_metadata["name"].lower():
        raise ValueError("Require the complete BSCCMNIST variant, not tiny")
    index = pd.read_csv(source / source_files[0], index_col="global_index", low_memory=False)
    markers = pd.read_csv(source / source_files[1], index_col="global_index", low_memory=False)
    metadata, qc = make_cohort(index, markers)
    source_zarr = zarr.open(str(source / "BSCCM_images.zarr"), mode="r")
    indices = [global_metadata["led_array"]["channel_names"].index(name) for name in CHANNELS]
    output.mkdir(parents=True, exist_ok=True)
    temporary_image_path = output / "images.incomplete.npy"
    images = np.lib.format.open_memmap(
        temporary_image_path, mode="w+", dtype=np.float32, shape=(len(metadata), 4, 28, 28)
    )
    hashes = defaultdict(list)
    last_log = time.monotonic()
    dtypes = set()
    minimum, maximum = float("inf"), -float("inf")
    # Exact duplicate inputs across FOVs are joined before splitting. This is an
    # input-only leakage precaution; marker values do not influence partitioning.
    for row, (global_index, entry) in enumerate(metadata.iterrows()):
        path = f"{entry['data_path'].strip('/')}/led_array/cell_{global_index}"
        array = source_zarr[path]
        # Zarr otherwise silently returns its fill value for missing chunks.
        # Extraction/download problems must never turn into apparently valid cells.
        if len(array.shape) != 3:
            raise ValueError(f"Expected channel-first source array for cell {global_index}")
        for channel in indices:
            ranges = [range(channel // array.chunks[0], channel // array.chunks[0] + 1)]
            ranges += [
                range((array.shape[d] + array.chunks[d] - 1) // array.chunks[d]) for d in (1, 2)
            ]
            for chunk in product(*ranges):
                key = array._chunk_key(chunk)  # pinned Zarr 2 API; honors dimension separator
                if key not in array.chunk_store:
                    raise ValueError(f"Missing raw image chunk for cell {global_index}: {key}")
        image = np.stack([np.asarray(array[channel]) for channel in indices])
        if image.shape != (4, 28, 28) or not np.isfinite(image).all():
            raise ValueError(f"Invalid image shape/values for cell {global_index}")
        dtypes.add(str(image.dtype))
        minimum, maximum = min(minimum, float(image.min())), max(maximum, float(image.max()))
        images[row] = image.astype(np.float32)
        hashes[hashlib.sha256(images[row].tobytes()).hexdigest()].append(row)
        if time.monotonic() - last_log > 20:
            print(f"Prepared {row + 1:,}/{len(metadata):,} cells", flush=True)
            last_log = time.monotonic()
    images.flush()
    groups = metadata["fov_id"].to_numpy()
    parents = {group: group for group in np.unique(groups)}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    duplicate_sets = [rows for rows in hashes.values() if len(rows) > 1]
    cross_field_duplicate_sets = 0
    for rows in duplicate_sets:
        if len(np.unique(groups[rows])) > 1:
            cross_field_duplicate_sets += 1
            for row in rows[1:]:
                a, b = sorted((find(groups[rows[0]]), find(groups[row])))
                parents[b] = a
    metadata["split_group"] = [find(group) for group in groups]
    metadata["split"] = assign_group_split(
        metadata, seed=args.split_seed, group_column="split_group"
    )
    qc.update(
        {
            "exact_duplicate_image_sets": len(duplicate_sets),
            "cross_field_duplicate_image_sets": cross_field_duplicate_sets,
            "source_image_dtypes": sorted(dtypes),
            "image_range": [minimum, maximum],
            "split_groups": int(metadata["split_group"].nunique()),
        }
    )
    for split, frame in metadata.groupby("split"):
        if set(frame["class_id"].unique()) != {0, 1, 2}:
            raise ValueError(
                f"Partition {split} lacks a class; discuss revised protocol rather than silently search seeds"
            )
    stats = training_statistics(images, metadata)
    del images
    temporary_image_path.replace(output / "images.npy")
    metadata.reset_index().to_csv(output / "metadata.csv", index=False)
    split_summary = {
        split: {
            "cells": len(frame),
            "fovs": int(frame["fov_id"].nunique()),
            "groups": int(frame["split_group"].nunique()),
            "class_counts": {CLASS_NAMES[i]: int((frame["class_id"] == i).sum()) for i in range(3)},
        }
        for split, frame in metadata.groupby("split")
    }
    manifest = {
        "schema_version": 1,
        "dataset": "BSCCMNIST",
        "doi": "10.5061/dryad.sxksn038s",
        "dataset_license": "CC0-1.0",
        "sources": [
            "https://arxiv.org/abs/2402.06191",
            "https://github.com/Waller-Lab/BSCCM",
            "https://datadryad.org/dataset/doi:10.5061/dryad.sxksn038s",
        ],
        "source": str(source.resolve()),
        "source_sha256": {name: sha256_file(source / name) for name in source_files},
        "input_channels": list(CHANNELS),
        "input_semantics": "Released BSCCMNIST four-direction LED-array intensity images; not reconstructed quantitative phase",
        "image_shape": [len(metadata), 4, 28, 28],
        "image_dtype": "float32",
        "class_names": list(CLASS_NAMES),
        "target_columns": MARKER_COLUMNS,
        "target_units": "Released four-spectrum-unmixed abundance estimates in arbitrary units; this pipeline adds no log transform or clipping",
        "cohort_qc": qc,
        "normalization": stats,
        "split": {
            "seed": args.split_seed,
            "fractions_of_groups": [0.70, 0.15, 0.15],
            "group_key": "physical slide data_path + stage XY; exact-input-duplicate FOVs merged",
            "outcomes_used_for_assignment": False,
            "summary": split_summary,
        },
        "limitations": [
            "Gated class labels reuse the fluorescence markers; they are not independent clinical diagnoses.",
            "The eligible labelled cohort is batch 0, all-antibody condition; held-out FOVs do not establish new-donor generalization.",
            "Official fluorescence processing/gating was performed before this ML split; training-only normalization does not undo that upstream processing.",
            "The authors' processing notebook floors each marker below its batch-level 0.5th percentile of nonzero abundance before release; targets are processed estimates, not untouched raw fluorescence.",
            "Exact-image duplicate checks cannot rule out near-duplicate or spatially overlapping acquisitions.",
        ],
        "prepared_sha256": {
            name: sha256_file(output / name) for name in ["images.npy", "metadata.csv"]
        },
    }
    provenance = args.data_root / "download_manifest.json"
    if provenance.exists():
        manifest["download_provenance"] = json.loads(provenance.read_text())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps({"output": str(output), "cohort_qc": qc, "split": split_summary}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
