"""Export a small real-data Colab demonstration from the main training split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cellselect.data import load_prepared, sha256_file


def export_demo(
    prepared,
    output,
    min_cells=1200,
    seed=20261021,
    overwrite=False,
    notebook="cell_selection_extended.ipynb",
):
    """Choose an outcome-independent prefix of complete training FOV groups.

    Duplicate-linked FOVs stay together. Missing classes cause an explicit
    failure; the exporter never searches alternative random seeds or labels.
    """
    prepared, output = Path(prepared), Path(output)
    if notebook not in {"cell_selection.ipynb", "cell_selection_extended.ipynb"}:
        raise ValueError("Unknown demonstration notebook")
    if not isinstance(min_cells, int) or isinstance(min_cells, bool) or min_cells < 1:
        raise ValueError("min_cells must be a positive integer")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Export already exists: {output}; use --overwrite to refresh it")
    data = load_prepared(prepared, verify_hashes=True)
    train = data.metadata.loc[data.metadata["split"] == "train"]
    if len(train) < min_cells:
        raise ValueError("Main training split is smaller than the requested demo")
    group_column = "split_group" if "split_group" in train else "fov_id"
    order = np.random.default_rng(seed).permutation(np.sort(train[group_column].unique()))
    sizes = train.groupby(group_column).size()
    selected, count = [], 0
    for group in order:
        selected.append(group)
        count += int(sizes.loc[group])
        if count >= min_cells and len(selected) >= 5:
            break
    if len(selected) < 5:
        raise ValueError("At least five complete training groups are needed for the demo")
    selected_rows = np.flatnonzero(
        (data.metadata["split"] == "train") & data.metadata[group_column].isin(selected)
    )
    sample = data.metadata.iloc[selected_rows]
    if set(sample.class_id.to_numpy()) != {0, 1, 2}:
        raise ValueError("The fixed sample does not contain three classes; increase --min-cells")
    split_seed = seed + 1
    groups = sample[group_column].to_numpy(dtype=str)
    split_order = np.random.default_rng(split_seed).permutation(np.unique(groups))
    demo_train_groups = split_order[: int(0.8 * len(split_order))]
    demo_train = np.isin(groups, demo_train_groups)
    if set(sample.loc[demo_train, "class_id"]) != {0, 1, 2}:
        raise ValueError("Fixed demo training groups lack a class; increase --min-cells")
    images = np.asarray(data.images[selected_rows], dtype=np.float32)
    targets = sample.CD16.to_numpy(dtype=np.float64)
    if not np.isfinite(images).all() or not np.isfinite(targets).all():
        raise ValueError("Nonfinite sample images or CD16 values")
    if targets[demo_train].std() <= 0:
        raise ValueError("Demo training marker must vary")
    # Reading selected_rows accesses only main-training pixels.
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        images=images,
        marker=targets,
        class_id=sample.class_id.to_numpy(dtype=np.int64),
        global_index=sample.global_index.to_numpy(dtype=np.int64),
        fov_id=sample.fov_id.to_numpy(dtype=str),
        split_group=groups,
    )
    entries = {"sample.npz": payload.getvalue()}
    for path in sorted((ROOT / "src" / "cellselect").glob("*.py")):
        entries[f"src/cellselect/{path.name}"] = path.read_bytes()
    for relative in ["vendor/LICENSE", "vendor/README.md", f"notebooks/{notebook}"]:
        entries[relative] = (ROOT / relative).read_bytes()
    metadata = {
        "purpose": "Small runnable course demonstration, not evidence for the main study",
        "notebook": f"notebooks/{notebook}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_partition": "train",
        "main_validation_or_test_cells_included": 0,
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "selection": "fixed random prefix of complete training groups; no label-based resampling",
        "group_column": group_column,
        "sample_seed": seed,
        "demo_split_seed": split_seed,
        "demo_split": "80/20 by complete duplicate-linked FOV groups",
        "requested_minimum_cells": min_cells,
        "n_cells": len(sample),
        "n_fields": int(sample.fov_id.nunique()),
        "n_groups": len(selected),
        "class_counts": {
            str(k): int(v) for k, v in sample.class_id.value_counts().sort_index().items()
        },
        "marker": "CD16",
        "marker_scale": "original unmixed abundance units",
        "channels": ["DPC_Left", "DPC_Right", "DPC_Top", "DPC_Bottom"],
        "dataset": {
            "citation": "Pinkard and Waller. Berkeley Single-Cell Computational Microscopy (BSCCM) dataset. Dryad.",
            "doi": "10.5061/dryad.sxksn038s",
            "url": "https://datadryad.org/dataset/doi:10.5061/dryad.sxksn038s",
            "license": "CC0-1.0",
            "license_source": "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.sxksn038s",
        },
        "reference_model": {
            "url": "https://github.com/saqibnaziir/Single-Cell-Phenotyping",
            "commit": "56a42e77bf6a7c5fcbbf8967dc209e2ee8f3d5df",
            "license": "MIT; retained in vendor/LICENSE",
            "paper_doi": "10.1007/978-3-032-31666-0_41",
        },
        "files_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in entries.items()
        },
    }
    entries["metadata.json"] = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name, content in entries.items():
                archive.writestr(name, content)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "n_cells": len(sample),
        "n_fields": metadata["n_fields"],
        "source_partition": "train",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Locked prepared-data directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--notebook",
        choices=["cell_selection.ipynb", "cell_selection_extended.ipynb"],
        default="cell_selection_extended.ipynb",
    )
    parser.add_argument("--min-cells", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / "results" / (
        "colab_extended_demo.zip" if "extended" in args.notebook else "colab_demo.zip"
    )
    print(
        json.dumps(
            export_demo(
                args.data, output, args.min_cells, args.seed, args.overwrite, args.notebook
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
