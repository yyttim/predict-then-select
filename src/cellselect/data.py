"""BSCCMNIST cohort construction, grouped splitting, and prepared-array access.

The four inputs are the LED-array intensity channels DPC_Left/Right/Top/Bottom,
not the separate reconstructed quantitative-phase ``dpc`` image.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("DPC_Left", "DPC_Right", "DPC_Top", "DPC_Bottom")
CLASS_NAMES = ("Lymphocyte", "Granulocyte", "Monocyte")
MARKER_COLUMNS = {"CD16": "CD16_full_model_unmixed", "CD45": "CD45_full_model_unmixed"}
# Class mapping from the authors' get_cell_type_classification_data API.
GATE_TO_CLASS = {0: 0, 1: 1, 2: 1, 3: 1, 7: 2, 8: 2, 9: 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate_boolean(series: pd.Series) -> np.ndarray:
    """Treat missing gates as unselected, never as Python's bool(nan)."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    values = series.fillna(False).astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "1.0", "0.0"}
    if not set(values.unique()).issubset(allowed):
        raise ValueError(f"Unexpected gate values in {series.name}")
    return values.isin(["true", "1", "1.0"]).to_numpy()


def make_cohort(index: pd.DataFrame, markers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep all uniquely three-class-labelled batch-0/all cells with finite CD16.

    Missing optional CD45 is retained rather than reducing the CD16 cohort. Raw
    marker estimates stay in the authors' original arbitrary abundance units.
    """
    if index.index.name != "global_index" or markers.index.name != "global_index":
        raise ValueError("Both metadata frames must use global_index as their index")
    if not index.index.is_unique or not markers.index.is_unique:
        raise ValueError("Duplicate global_index in source metadata")
    gate_names = [name for name in markers.columns if "selection_gated" in name]
    if len(gate_names) != 10:
        raise ValueError(
            f"Expected the official 10 ordered gates, found {len(gate_names)}; inspect data/API before proceeding"
        )
    missing = [name for name in MARKER_COLUMNS.values() if name not in markers]
    if missing:
        raise ValueError(f"Missing source marker columns: {missing}")
    class_flags = np.zeros((len(markers), 3), dtype=bool)
    for gate, class_id in GATE_TO_CLASS.items():
        class_flags[:, class_id] |= _gate_boolean(markers[gate_names[gate]])
    number_of_classes = class_flags.sum(axis=1)
    labels = pd.Series(
        np.where(number_of_classes == 1, class_flags.argmax(axis=1), -1), index=markers.index
    )
    candidates = index.loc[(index["batch"] == 0) & (index["antibodies"] == "all")].copy()
    candidates["class_id"] = labels.reindex(candidates.index).fillna(-1).astype(int)
    for name, column in MARKER_COLUMNS.items():
        candidates[name] = pd.to_numeric(markers[column].reindex(candidates.index), errors="coerce")
    classified = candidates["class_id"] >= 0
    finite = np.isfinite(candidates["CD16"])
    cohort = candidates.loc[classified & finite].sort_index().copy()
    if (cohort["CD16"] < 0).any():
        raise ValueError("Negative CD16 abundance in the official nonnegative unmixed target")
    if cohort.empty:
        raise ValueError("No eligible cells")
    # data_path identifies antibody condition, batch and physical slide.
    # Imaging date is not added: re-imaging a physical slide must not split it.
    cohort["slide_id"] = cohort["data_path"].astype(str).str.strip("/")
    required = ["fov_center_x", "fov_center_y"]
    if not np.isfinite(cohort[required].to_numpy(dtype=np.float64)).all():
        raise ValueError("Missing field-of-view coordinates; cannot create leakage-safe split")
    # Repeated focal planes at the same XY remain in one field, hence z is omitted.
    xy = cohort[required].apply(lambda s: s.map(lambda v: format(float(v), ".12g")))
    cohort["fov_id"] = cohort["slide_id"] + "|x=" + xy["fov_center_x"] + "|y=" + xy["fov_center_y"]
    qc = {
        "source_cells": len(index),
        "batch0_all_cells": len(candidates),
        "unique_three_class_cells": int(classified.sum()),
        "excluded_missing_or_nonfinite_CD16": int((classified & ~finite).sum()),
        "ambiguous_across_class_gates_global": int((number_of_classes > 1).sum()),
        "eligible_cells": len(cohort),
        "optional_CD45_missing": int((~np.isfinite(cohort["CD45"])).sum()),
        "class_counts": {CLASS_NAMES[i]: int((cohort["class_id"] == i).sum()) for i in range(3)},
        "slides": int(cohort["slide_id"].nunique()),
        "fovs": int(cohort["fov_id"].nunique()),
        "gate_columns_in_source_order": gate_names,
        "gate_mapping": GATE_TO_CLASS,
    }
    return cohort, qc


def assign_group_split(
    metadata: pd.DataFrame,
    seed: int = 20261021,
    fractions=(0.70, 0.15, 0.15),
    group_column="fov_id",
) -> pd.Series:
    """Shuffle unique groups once, without using outcomes or class composition."""
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError("Three positive split fractions must sum to one")
    groups = np.array(sorted(metadata[group_column].unique()), dtype=object)
    if len(groups) < 7:
        raise ValueError("At least seven independent groups are needed for the requested split")
    groups = np.random.default_rng(seed).permutation(groups)
    n_train = int(np.floor(len(groups) * fractions[0]))
    n_val = int(np.floor(len(groups) * fractions[1]))
    if min(n_train, n_val, len(groups) - n_train - n_val) < 1:
        raise ValueError("Empty split")
    assignment = dict(
        zip(
            groups,
            ["train"] * n_train + ["val"] * n_val + ["test"] * (len(groups) - n_train - n_val),
        )
    )
    return metadata[group_column].map(assignment).rename("split")


def training_statistics(images: np.ndarray, metadata: pd.DataFrame, chunk_size=1024) -> dict:
    """Compute channel pixel and target moments using training rows only."""
    selected = np.flatnonzero(metadata["split"].to_numpy() == "train")
    if not len(selected):
        raise ValueError("No training rows")
    sums = np.zeros(images.shape[1], np.float64)
    sumsquares = np.zeros_like(sums)
    count = 0
    for start in range(0, len(selected), chunk_size):
        block = np.asarray(images[selected[start : start + chunk_size]], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Non-finite input images")
        sums += block.sum(axis=(0, 2, 3))
        sumsquares += np.square(block).sum(axis=(0, 2, 3))
        count += len(block) * block.shape[2] * block.shape[3]
    mean = sums / count
    std = np.sqrt(np.maximum(sumsquares / count - mean**2, 0))
    if (std <= 0).any():
        raise ValueError("Constant training image channel")
    marker_stats = {}
    for marker in MARKER_COLUMNS:
        values = metadata.iloc[selected][marker].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) and values.std() > 0:
            marker_stats[marker] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "n": len(values),
            }
    return {
        "scope": "full_training_split_only",
        "ddof": 0,
        "image_mean": mean.tolist(),
        "image_std": std.tolist(),
        "markers": marker_stats,
    }


@dataclass
class PreparedData:
    images: np.ndarray
    metadata: pd.DataFrame
    manifest: dict

    def rows(self, split: str) -> np.ndarray:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split}")
        return np.flatnonzero(self.metadata["split"].to_numpy() == split)


def load_prepared(path: str | Path, verify_hashes=False) -> PreparedData:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(root / "metadata.csv")
    images = np.load(root / "images.npy", mmap_mode="r")
    if images.shape != (len(metadata), 4, 28, 28):
        raise ValueError("Image and metadata shapes do not match the four-channel prepared schema")
    if not metadata["global_index"].is_unique:
        raise ValueError("Duplicate cells in prepared dataset")
    if (metadata.groupby("split_group")["split"].nunique() != 1).any():
        raise ValueError("A split group crosses partitions")
    if verify_hashes:
        for filename, expected in manifest["prepared_sha256"].items():
            if sha256_file(root / filename) != expected:
                raise ValueError(f"Prepared file checksum mismatch: {filename}")
    return PreparedData(images, metadata, manifest)
