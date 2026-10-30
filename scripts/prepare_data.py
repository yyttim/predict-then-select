"""Prepare both single-CD16 cohorts from existing BSCCMNIST archives.

Example:
  python scripts/prepare_data.py --source data/raw/BSCCMNIST --archives data/archives --output data/prepared/cd16_selection_v3

Missing source metadata is recovered from verified archives without replacing
existing files. Images are unpacked under <archives-parent>/raw_cd16. Batch 0
defines the main split and threshold; all batch-1 cells remain external.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import product
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import zarr

from cellselect.data import (
    CHANNELS,
    QUARTER_SEED,
    SPLIT_SEED,
    TARGET_COLUMN,
    assign_partitions,
    eligible_metadata,
    sha256_file,
    threshold_from_quarter,
)

ARCHIVES = [
    {
        "name": "BSCCMNIST.tar.gz_chunk00000.bin",
        "bytes": 4294967296,
        "sha256": "41d588f4aa12fd756ea9e5438a2feff8a539bfd9b18f2af36733c7d710ddaf23",
    },
    {
        "name": "BSCCMNIST.tar.gz_chunk00001.bin",
        "bytes": 1930936954,
        "sha256": "ce42e1bba0d77e8ae72c9ceb14dc30e197f8b46986b9d2c979c7ae7bdcd28618",
    },
]
SOURCE_NAMES = ("BSCCM_index.csv", "BSCCM_surface_markers.csv", "BSCCM_global_metadata.json")


class ChunkReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths, self.current = iter(paths), None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    return 0
            size = self.current.readinto(buffer)
            if size:
                return size
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def safe_destination(root, name):
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or any(":" in p for p in parts.parts):
        raise ValueError(f"Unsafe archive path: {name}")
    root = root.resolve()
    target = root.joinpath(*parts.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Archive member escapes the extraction directory")
    return target


def ensure_metadata(paths, source):
    """Recover missing metadata after archive verification, preserving existing files."""
    destinations = {name: safe_destination(source, name) for name in SOURCE_NAMES}
    if any(path.exists() and not path.is_file() for path in destinations.values()):
        raise ValueError("A required metadata path exists but is not a regular file")
    if all(path.is_file() for path in destinations.values()):
        return
    source.mkdir(parents=True, exist_ok=True)
    buffered, digests = {}, {}
    with ExitStack() as stack:
        with (
            io.BufferedReader(ChunkReader(paths), 8 * 1024 * 1024) as stream,
            tarfile.open(fileobj=stream, mode="r|gz") as archive,
        ):
            for member in archive:
                archive.members.clear()
                name = member.name.removeprefix("./")
                leaf = PurePosixPath(name).name
                if leaf not in destinations:
                    continue
                safe_destination(source, name)
                if name != f"BSCCMNIST/{leaf}" or not member.isfile():
                    raise ValueError(f"Unexpected metadata archive member: {name}")
                if leaf in buffered:
                    raise ValueError(f"Duplicate metadata archive member: {name}")
                temporary = stack.enter_context(tempfile.TemporaryFile(dir=source))
                digest = hashlib.sha256()
                with archive.extractfile(member) as contents:
                    for block in iter(lambda: contents.read(1024 * 1024), b""):
                        temporary.write(block)
                        digest.update(block)
                buffered[leaf], digests[leaf] = temporary, digest.hexdigest()
                if len(buffered) == len(SOURCE_NAMES):
                    break
        if set(buffered) != set(SOURCE_NAMES):
            raise ValueError("Verified archives do not contain all three source metadata files")
        for name, path in destinations.items():
            if path.exists() and sha256_file(path) != digests[name]:
                raise ValueError(f"Existing metadata differs from the verified archive: {name}")
        for name, path in destinations.items():
            if path.exists():
                continue
            buffered[name].seek(0)
            # Exclusive creation never overwrites a concurrently created file.
            with path.open("xb") as destination:
                try:
                    while block := buffered[name].read(1024 * 1024):
                        destination.write(block)
                except BaseException:
                    destination.close()
                    path.unlink()
                    raise


def extract_inputs(paths, raw_root, metadata, channels):
    arrays = {
        f"BSCCMNIST/BSCCM_images.zarr/{row.data_path.strip('/')}/led_array/cell_{i}"
        for i, row in metadata.iterrows()
    }
    groups = set()
    for name in arrays:
        parent = PurePosixPath(name).parent
        while len(parent.parts) >= 2:
            groups.add(str(parent / ".zgroup"))
            parent = parent.parent
    scanned = written = 0
    last_log = time.monotonic()
    with (
        io.BufferedReader(ChunkReader(paths), 8 * 1024 * 1024) as stream,
        tarfile.open(fileobj=stream, mode="r|gz") as archive,
    ):
        for member in archive:
            archive.members.clear()  # Avoid retaining millions of TarInfo objects.
            scanned += 1
            name = member.name.removeprefix("./")
            parent, _, leaf = name.rpartition("/")
            take = name in groups
            if parent in arrays:
                chunk = leaf.split(".")
                take |= leaf in {".zarray", ".zattrs"} or (
                    len(chunk) == 3
                    and all(x.isdigit() for x in chunk)
                    and int(chunk[0]) in channels
                )
            if take:
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"Unexpected archive member: {name}")
                target = safe_destination(raw_root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                written += 1
            if time.monotonic() - last_log > 30:
                print(
                    f"Archive: scanned {scanned:,} entries; retained {written:,} files",
                    flush=True,
                )
                last_log = time.monotonic()
    return {"members_scanned": scanned, "files_extracted": written, "cells_requested": len(arrays)}


def read_image(raw_root, row, channels):
    cell, entry = row
    path = (
        raw_root
        / "BSCCMNIST"
        / "BSCCM_images.zarr"
        / entry.data_path.strip("/")
        / "led_array"
        / f"cell_{cell}"
    )
    array = zarr.open_array(str(path), mode="r")
    if len(array.shape) != 3:
        raise ValueError(f"Invalid source shape for cell {cell}")
    for channel in channels:
        ranges = [range(channel // array.chunks[0], channel // array.chunks[0] + 1)]
        ranges += [range((array.shape[d] + array.chunks[d] - 1) // array.chunks[d]) for d in (1, 2)]
        for chunk in product(*ranges):
            if array._chunk_key(chunk) not in array.chunk_store:
                raise ValueError(f"Missing image chunk for cell {cell}")
    image = np.stack([np.asarray(array[c]) for c in channels])
    if image.shape != (4, 28, 28) or not np.isfinite(image).all():
        raise ValueError(f"Nonfinite or incorrectly shaped image for cell {cell}")
    return image.astype(np.float32), str(image.dtype)


def prepare(source, archives, output, archive_specs=None):
    source, archives, output = (Path(p).resolve() for p in (source, archives, output))
    raw_root = archives.parent / "raw_cd16"
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared cohort already locked; reuse it rather than overwriting it")
    if output == source or raw_root == source:
        raise ValueError("Preparation must use separate source and output paths")
    records = ARCHIVES if archive_specs is None else archive_specs
    paths = []
    for record in records:
        if PurePosixPath(record["name"]).name != record["name"]:
            raise ValueError("Archive names must be basenames")
        path = archives / record["name"]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Archive failed the declared integrity check: {path.name}")
        paths.append(path)
    ensure_metadata(paths, source)
    source_names = SOURCE_NAMES
    source_hashes = {name: sha256_file(source / name) for name in source_names}
    global_metadata = json.loads((source / source_names[2]).read_text())
    if global_metadata["name"] != "BSCCMNIST":
        raise ValueError("Require the full native 28x28 BSCCMNIST release")
    index = pd.read_csv(
        source / source_names[0],
        index_col="global_index",
        low_memory=False,
        float_precision="round_trip",
    )
    markers = pd.read_csv(
        source / source_names[1],
        usecols=["global_index", TARGET_COLUMN],
        index_col="global_index",
        float_precision="round_trip",
    )
    main_metadata, main_audit = eligible_metadata(index, markers, batch=0)
    external_metadata, external_audit = eligible_metadata(index, markers, batch=1)
    metadata = pd.concat([main_metadata, external_metadata]).sort_index()
    channels = [global_metadata["led_array"]["channel_names"].index(name) for name in CHANNELS]
    extraction = extract_inputs(paths, raw_root, metadata, channels)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "images.incomplete.npy"
    images = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(metadata), 4, 28, 28)
    )
    fingerprints, duplicates, dtypes = {}, [], set()
    minimum, maximum = float("inf"), -float("inf")
    last_log = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for row, (image, dtype) in enumerate(
            executor.map(lambda r: read_image(raw_root, r, channels), metadata.iterrows())
        ):
            images[row] = image
            dtypes.add(dtype)
            minimum = min(minimum, float(image.min()))
            maximum = max(maximum, float(image.max()))
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            cell = int(metadata.index[row])
            if digest in fingerprints:
                duplicates.append([fingerprints[digest], cell])
            else:
                fingerprints[digest] = cell
            if time.monotonic() - last_log > 20:
                print(f"Image QC: {row + 1:,}/{len(metadata):,} cells", flush=True)
                last_log = time.monotonic()
    images.flush()
    del images
    if duplicates:
        (output / "duplicate_image_audit.json").write_text(
            json.dumps({"pairs": duplicates}, indent=2)
        )
        raise ValueError(
            "Exact duplicate images found; inspect before altering cohort or field grouping"
        )
    main_metadata = assign_partitions(main_metadata)
    external_metadata["split"] = "external"
    external_metadata["train_quarter"] = False
    metadata = pd.concat([main_metadata, external_metadata]).sort_index()
    metadata.reset_index().to_csv(output / "metadata.csv", index=False, float_format="%.17g")
    persisted = pd.read_csv(output / "metadata.csv", float_precision="round_trip")
    threshold = threshold_from_quarter(persisted)
    temporary.replace(output / "images.npy")
    counts = {
        split: {"cells": len(frame), "fields": int(frame["fov_id"].nunique())}
        for split, frame in metadata.groupby("split")
    }
    quarter = metadata.loc[metadata["train_quarter"]]
    counts["train_quarter"] = {"cells": len(quarter), "fields": int(quarter["fov_id"].nunique())}
    manifest = {
        "schema_version": 2,
        "dataset": "BSCCMNIST",
        "doi": "10.5061/dryad.sxksn038s",
        "dataset_version": "362939",
        "cohort": {
            "main_batch": 0,
            "external_batch": 1,
            "antibodies": "CD16",
            "target_column": TARGET_COLUMN,
        },
        "input_channels": list(CHANNELS),
        "source_channel_indices": channels,
        "image_shape": [len(metadata), 4, 28, 28],
        "image_dtype": "float32",
        "image_qc": {
            "finite_images": len(metadata),
            "exact_duplicate_image_pairs": 0,
            "source_dtypes": sorted(dtypes),
            "source_range": [minimum, maximum],
            "resized_or_per_image_scaled": False,
        },
        "target_qc": {"main": main_audit, "external": external_audit},
        "threshold": threshold,
        "split": {
            "group_key": "physical slide data_path plus stage XY",
            "seed": SPLIT_SEED,
            "fractions_of_fields": [0.70, 0.15, 0.15],
            "quarter_seed": QUARTER_SEED,
            "quarter_fraction_of_training_fields": 0.25,
            "quarter_rounding": "ceil",
            "subsets_shared_across_methods_and_training_seeds": True,
            "counts": counts,
            "outcomes_used_to_assign_fields": False,
        },
        "source_sha256": source_hashes,
        "source_archives": records,
        "extraction": extraction,
        "prepared_sha256": {
            name: sha256_file(output / name) for name in ("images.npy", "metadata.csv")
        },
        "normalization_rule": "Fit image and target moments on the actual training-budget rows only.",
        "target_semantics": "Released two-spectrum CD16 plus autofluorescence unmixing estimate in arbitrary units; no additional transform or tail clipping.",
        "limitations": [
            "The release applies a batch-specific 0.5th-percentile lower floor to nonzero unmixed marker estimates upstream.",
            "The marker-high cutoff is an operational threshold estimated from the shared quarter-training labels, not clinical positivity.",
            "The main and external cohorts each come from one physical slide; donor independence is not established.",
            "The same source-derived threshold is carried to batch 1 without recalibration; assay and acquisition variation can affect its interpretation.",
        ],
        "numpy_version": np.__version__,
    }
    if any(sha256_file(source / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Original source metadata changed during preparation")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": counts,
                "threshold": threshold,
                "image_qc": manifest["image_qc"],
                "manifest_sha256": sha256_file(output / "manifest.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/BSCCMNIST"))
    parser.add_argument("--archives", type=Path, default=Path("data/archives"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/cd16_selection_v3"))
    args = parser.parse_args()
    prepare(args.source, args.archives, args.output)


if __name__ == "__main__":
    main()
