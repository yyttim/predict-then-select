"""Prepare batch-1 image and marker arrays from the existing BSCCMNIST archives.

Each slide is a separate candidate pool. The main cohort and its normalization
remain unchanged; external cell-type labels are unavailable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from itertools import product
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import zarr

from cellselect.data import CHANNELS, MARKER_COLUMNS, sha256_file
from cellselect.external_data import make_external_cohort
from download_data import ConcatenatedReader, safe_target


def extract_external(archives, destination, global_metadata, metadata):
    """Unpack only metadata, hierarchy, and four required channels for frozen IDs."""
    channels = {global_metadata["led_array"]["channel_names"].index(name) for name in CHANNELS}
    arrays = {
        f"BSCCMNIST/BSCCM_images.zarr/{row.data_path.strip('/')}/led_array/cell_{cell_id}"
        for cell_id, row in metadata.iterrows()
    }
    scanned = written = 0
    last_log = time.monotonic()
    with io.BufferedReader(ConcatenatedReader(archives), buffer_size=8 * 1024 * 1024) as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                archive.members.clear()
                scanned += 1
                path = PurePosixPath(member.name)
                if member.isfile() and path.parent.as_posix() in arrays:
                    leaf = path.name
                    requested = leaf in {".zarray", ".zattrs"}
                    numbers = leaf.split(".")
                    requested |= (
                        len(numbers) == 3
                        and all(x.isdigit() for x in numbers)
                        and int(numbers[0]) in channels
                    )
                else:
                    requested = path.name == ".zgroup"
                if requested:
                    if not member.isfile() or member.issym() or member.islnk():
                        raise ValueError(f"Unexpected member type in source archive: {member.name}")
                    target = safe_target(destination, member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                    written += 1
                if time.monotonic() - last_log > 30:
                    print(
                        f"External extraction: scanned {scanned:,} members, wrote {written:,} files",
                        flush=True,
                    )
                    last_log = time.monotonic()
    return {
        "scanned_members": scanned,
        "written_files": written,
        "all_requested_cell_ids": len(arrays),
    }


def read_input(store, row, channel_indices):
    global_index, entry = row
    key = f"{entry.data_path.strip('/')}/led_array/cell_{global_index}"
    array = store[key]
    if len(array.shape) != 3:
        raise ValueError(f"Malformed source array {global_index}")
    for channel in channel_indices:
        ranges = [range(channel // array.chunks[0], channel // array.chunks[0] + 1)]
        ranges += [range((array.shape[d] + array.chunks[d] - 1) // array.chunks[d]) for d in (1, 2)]
        for chunk in product(*ranges):
            if array._chunk_key(chunk) not in array.chunk_store:
                raise ValueError(f"Missing required external chunk for cell {global_index}")
    image = np.stack([np.asarray(array[c]) for c in channel_indices])
    if image.shape != (4, 28, 28) or not np.isfinite(image).all():
        raise ValueError(f"Invalid external input {global_index}")
    return image.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--main-data", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    data_root = args.data_root
    main_root = args.main_data or data_root / "prepared" / "bsccmnist_cd16_v1"
    output = data_root / "prepared" / "bsccmnist_batch1_all_v1"
    raw_root = data_root / "raw_external" / "batch1_all"
    source_root = data_root / "raw" / "BSCCMNIST"
    if (output / "manifest.json").exists():
        raise FileExistsError(
            "External dataset already frozen; reuse it rather than silently overwriting it"
        )
    main_manifest = json.loads((main_root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in main_manifest["source_sha256"].items():
        if sha256_file(source_root / name) != expected:
            raise ValueError(f"Source metadata changed since main preparation: {name}")
    for name, expected in main_manifest["prepared_sha256"].items():
        if sha256_file(main_root / name) != expected:
            raise ValueError(f"Main prepared artifact changed: {name}")
    index = pd.read_csv(source_root / "BSCCM_index.csv", index_col="global_index", low_memory=False)
    markers = pd.read_csv(
        source_root / "BSCCM_surface_markers.csv", index_col="global_index", low_memory=False
    )
    metadata, qc = make_external_cohort(index, markers)
    main_metadata = pd.read_csv(main_root / "metadata.csv")
    if np.intersect1d(metadata.index, main_metadata["global_index"]).size:
        raise ValueError("External global indices overlap the original cohort")
    global_metadata = json.loads((source_root / "BSCCM_global_metadata.json").read_text())
    if global_metadata["name"] != "BSCCMNIST":
        raise ValueError("Require the complete BSCCMNIST archive")
    download = json.loads((data_root / "download_manifest.json").read_text())
    records = sorted(download["files"], key=lambda entry: entry["path"])
    archive_paths = []
    for record in records:
        if PurePosixPath(record["path"]).name != record["path"]:
            raise ValueError("Archive record must be a basename")
        path = data_root / "archives" / record["path"]
        if path.stat().st_size != record["size"] or sha256_file(path) != record["digest"]:
            raise ValueError(f"Archive checksum mismatch: {path.name}")
        archive_paths.append(path)
    output.mkdir(parents=True, exist_ok=True)
    # This ID-only eligibility lock is saved before image preparation or inference.
    id_hash = hashlib.sha256(metadata.index.to_numpy(dtype=np.int64).tobytes()).hexdigest()
    lock = {
        "role": "external_acquisition_evaluation",
        "global_ids_sha256": id_hash,
        "selection_rule": "batch=1, antibodies=all, finite CD16; no outcome quantile or class gate",
        "qc": qc,
        "pool_rule": "one physical slide per candidate pool",
    }
    lock_path = output / "cohort_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("External eligibility changed from the saved lock")
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    raw_metadata_root = raw_root / "BSCCMNIST"
    raw_metadata_root.mkdir(parents=True, exist_ok=True)
    for name in main_manifest["source_sha256"]:
        shutil.copy2(source_root / name, raw_metadata_root / name)
    extraction = extract_external(archive_paths, raw_root, global_metadata, metadata)
    store = zarr.open(str(raw_metadata_root / "BSCCM_images.zarr"), mode="r")
    channels = [global_metadata["led_array"]["channel_names"].index(name) for name in CHANNELS]
    main_images = np.load(main_root / "images.npy", mmap_mode="r")
    main_hashes = {
        hashlib.sha256(np.asarray(image, dtype=np.float32).tobytes()).hexdigest()
        for image in main_images
    }
    seen = defaultdict(list)
    duplicates_main = []
    temporary = output / "images.incomplete.npy"
    images = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(metadata), 4, 28, 28)
    )
    last_log = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(lambda row: read_input(store, row, channels), metadata.iterrows())
        for row, image in enumerate(results):
            images[row] = image
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            seen[digest].append(int(metadata.index[row]))
            if digest in main_hashes:
                duplicates_main.append(int(metadata.index[row]))
            if time.monotonic() - last_log > 20:
                print(f"External arrays prepared: {row + 1:,}/{len(metadata):,}", flush=True)
                last_log = time.monotonic()
    images.flush()
    del images
    if duplicates_main:
        (output / "input_overlap_warning.json").write_text(
            json.dumps({"matching_external_ids": duplicates_main}, indent=2)
        )
        raise ValueError(
            "Exact external inputs overlap the main cohort; inspect before permitting evaluation"
        )
    duplicate_sets = [ids for ids in seen.values() if len(ids) > 1]
    qc["exact_input_duplicate_sets"] = duplicate_sets
    qc["exact_input_duplicates_with_main"] = 0
    qc["global_id_overlap_with_main"] = 0
    if duplicate_sets:
        # Do not silently alter population or bootstrap grouping after eligibility lock.
        (output / "external_duplicate_warning.json").write_text(
            json.dumps(duplicate_sets, indent=2)
        )
        raise ValueError(
            "Repeated external input arrays require an explicit audit before evaluation"
        )
    temporary.replace(output / "images.npy")
    metadata.reset_index().to_csv(output / "metadata.csv", index=False)
    manifest = {
        "schema_version": 1,
        "role": "external_acquisition_evaluation",
        "dataset": "BSCCMNIST",
        "doi": "10.5061/dryad.sxksn038s",
        "dataset_license": "CC0-1.0",
        "input_channels": list(CHANNELS),
        "image_shape": [len(metadata), 4, 28, 28],
        "image_dtype": "float32",
        "target_columns": MARKER_COLUMNS,
        "target_units": "Released processed unmixed abundance estimates, arbitrary units",
        "cohort_qc": qc,
        "pool_rule": "one physical slide per candidate pool",
        "normalization": {
            "fitted_on_external": False,
            "rule": "use each frozen source-trained checkpoint's normalization; no external fitting",
        },
        "main_manifest_sha256": sha256_file(main_root / "manifest.json"),
        "main_prepared_sha256": main_manifest["prepared_sha256"],
        "source_sha256": main_manifest["source_sha256"],
        "source_archives": records,
        "eligibility_lock_sha256": sha256_file(lock_path),
        "extraction": extraction,
        "source_files_unchanged_after_preparation": all(
            sha256_file(source_root / name) == expected
            for name, expected in main_manifest["source_sha256"].items()
        ),
        "prepared_sha256": {
            name: sha256_file(output / name) for name in ("images.npy", "metadata.csv")
        },
        "limitations": [
            "This is another cell-isolation batch and acquisition dates, not verified independent donors or an independent laboratory.",
            "No published reference cell-type labels exist here; classification scores, type quotas, and class-based gap decomposition are unavailable.",
            "All source-filtered batch-1/all cells are included, unlike the three-class-gated source cohort; population selection also changes.",
            "The authors report batch-1/all CD19 antibody at 35 percent of the other stains; assay and acquisition shift coexist.",
            "Fluorescence unmixing and lower-tail flooring were performed upstream, with batch-dependent spectra; absolute units are not assumed calibrated across batches/slides.",
            "Per-slide utility comparisons describe offline image-based ranking of processed measurements, not clinical benefit.",
        ],
        "sources": [
            "https://arxiv.org/abs/2402.06191",
            "https://github.com/Waller-Lab/BSCCM/blob/main/bsccm/bsccm.py",
            "https://github.com/Waller-Lab/BSCCM/blob/main/bsccm/fluorescence_processing/do_demixing.ipynb",
        ],
        "external_model_inference_performed": False,
    }
    if not manifest["source_files_unchanged_after_preparation"]:
        raise ValueError("Original metadata unexpectedly changed during preparation")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "cohort_qc": qc, "external_model_inference_performed": False},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
