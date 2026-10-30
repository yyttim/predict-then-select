"""Prepared single-antibody CD16 data and training-only normalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("DPC_Left", "DPC_Right", "DPC_Top", "DPC_Bottom")
TARGET_COLUMN = "CD16_single_antibody_model_unmixed"
SPLIT_SEED = 20261030
QUARTER_SEED = 20261031


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_metadata(index, markers, batch=0):
    """Retain every single-CD16 cell with a finite target in the requested batch."""
    if batch not in {0, 1}:
        raise ValueError("Only the two released single-CD16 batches are supported")
    if index.index.name != "global_index" or markers.index.name != "global_index":
        raise ValueError("Source tables must be indexed by global_index")
    if not index.index.is_unique or not markers.index.is_unique:
        raise ValueError("Duplicate global_index in source tables")
    names = [
        "batch",
        "antibodies",
        "slide_replicate",
        "imaging_date",
        "data_path",
        "fov_center_x",
        "fov_center_y",
        "fov_center_z",
        "position_in_fov_x_pix",
        "position_in_fov_y_pix",
    ]
    required = {"batch", "antibodies", "data_path", "fov_center_x", "fov_center_y"}
    if not required.issubset(index.columns) or TARGET_COLUMN not in markers:
        raise ValueError("Required source metadata or single-antibody CD16 target is missing")
    selected = index.loc[
        (index["batch"] == batch) & (index["antibodies"] == "CD16"),
        [name for name in names if name in index],
    ].copy()
    selected["target"] = pd.to_numeric(
        markers[TARGET_COLUMN].reindex(selected.index), errors="coerce"
    )
    valid = np.isfinite(selected["target"])
    metadata = selected.loc[valid].sort_index().copy()
    if metadata.empty or (metadata["target"] < 0).any():
        raise ValueError("Empty cohort or negative values in the nonnegative released target")
    if (
        metadata["data_path"].isna().any()
        or not np.isfinite(metadata[["fov_center_x", "fov_center_y"]].to_numpy(float)).all()
    ):
        raise ValueError("Missing slide identity or nonfinite FOV coordinates")
    metadata["slide_id"] = metadata["data_path"].astype(str).str.strip("/")
    xy = metadata[["fov_center_x", "fov_center_y"]].apply(
        lambda values: values.map(lambda x: format(float(x), ".12g"))
    )
    # Keep repeat focal planes at the same physical XY in the same group.
    metadata["fov_id"] = (
        metadata["slide_id"] + "|x=" + xy["fov_center_x"] + "|y=" + xy["fov_center_y"]
    )
    audit = {
        "source_cells": len(index),
        "batch": batch,
        "single_cd16_cells": len(selected),
        "finite_target_cells": len(metadata),
        "excluded_nonfinite_target": int((~valid).sum()),
        "negative_targets": 0,
        "tail_clipping_or_trimming": False,
    }
    return metadata, audit


def assign_partitions(metadata):
    """Assign fields independently of targets, then fix the nested quarter subset."""
    if "batch" in metadata and not metadata["batch"].eq(0).all():
        raise ValueError("Only batch 0 may enter the training/validation/test split")
    metadata = metadata.copy()
    groups = np.array(sorted(metadata["fov_id"].unique()), dtype=object)
    if len(groups) < 7:
        raise ValueError("At least seven fields are needed for the declared split")
    order = np.random.default_rng(SPLIT_SEED).permutation(groups)
    n_train, n_val = int(np.floor(0.70 * len(order))), int(np.floor(0.15 * len(order)))
    mapping = dict(
        zip(
            order, ["train"] * n_train + ["val"] * n_val + ["test"] * (len(order) - n_train - n_val)
        )
    )
    metadata["split"] = metadata["fov_id"].map(mapping)
    train_fields = np.array(
        sorted(metadata.loc[metadata["split"] == "train", "fov_id"].unique()), dtype=object
    )
    quarter_fields = np.random.default_rng(QUARTER_SEED).permutation(train_fields)[
        : int(np.ceil(0.25 * len(train_fields)))
    ]
    metadata["train_quarter"] = (metadata["split"] == "train") & metadata["fov_id"].isin(
        quarter_fields
    )
    return metadata


def threshold_from_quarter(metadata):
    reference = metadata.loc[metadata["train_quarter"]]
    if reference.empty or not (reference["split"] == "train").all():
        raise ValueError("Threshold reference must be a nonempty training subset")
    values = reference["target"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite threshold-reference targets")
    ids = (
        reference.index.to_numpy(dtype=np.int64)
        if metadata.index.name == "global_index"
        else reference["global_index"].to_numpy(dtype=np.int64)
    )
    return {
        "value": float(np.quantile(values, 0.9, method="linear")),
        "quantile": 0.9,
        "operator": ">=",
        "source": "quarter_training",
        "method": "linear",
        "reference_count": len(reference),
        "reference_fields": int(reference["fov_id"].nunique()),
        "reference_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
    }


@dataclass
class PreparedData:
    images: np.ndarray
    metadata: pd.DataFrame
    manifest: dict
    root: Path | None = None
    manifest_sha256: str = ""

    @property
    def directory(self):
        return self.root

    def rows(self, split, fraction=1.0):
        if split not in {"train", "val", "test", "external"}:
            raise ValueError(f"Unknown partition: {split}")
        if fraction not in {0.25, 1.0} or (split != "train" and fraction != 1.0):
            raise ValueError(
                "Only training supports the quarter/full budgets; evaluation pools stay complete"
            )
        mask = self.metadata["split"].to_numpy() == split
        if fraction == 0.25:
            mask &= self.metadata["train_quarter"].to_numpy(dtype=bool)
        return np.flatnonzero(mask)


def load_data(path, verify_hashes=True):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("cohort", {}).get("target_column") != TARGET_COLUMN
    ):
        raise ValueError("Expected the single-antibody CD16 selection dataset")
    if verify_hashes:
        for name in ("images.npy", "metadata.csv"):
            if sha256_file(root / name) != manifest["prepared_sha256"][name]:
                raise ValueError(f"Prepared data checksum mismatch: {name}")
    metadata = pd.read_csv(root / "metadata.csv", float_precision="round_trip")
    images = np.load(root / "images.npy", mmap_mode="r", allow_pickle=False)
    if images.shape != (len(metadata), 4, 28, 28) or images.dtype != np.float32:
        raise ValueError("Prepared image shape or dtype mismatch")
    if not metadata["global_index"].is_unique or not np.isfinite(metadata["target"]).all():
        raise ValueError("Invalid prepared IDs or targets")
    if not pd.api.types.is_bool_dtype(metadata["train_quarter"]):
        raise ValueError("train_quarter must contain booleans")
    roles = set(metadata["split"].unique())
    if not {"train", "val", "test"}.issubset(roles) or not roles.issubset(
        {"train", "val", "test", "external"}
    ):
        raise ValueError("Require the main partitions and, optionally, an external pool")
    if "batch" in metadata:
        external = metadata["split"].eq("external")
        if (
            not metadata.loc[external, "batch"].eq(1).all()
            or not metadata.loc[~external, "batch"].eq(0).all()
        ):
            raise ValueError(
                "Batch 1 must remain external and batch 0 must remain in the main split"
            )
    if (metadata.groupby("fov_id")["split"].nunique() != 1).any():
        raise ValueError("A field crosses partitions")
    if (
        metadata.loc[metadata["train_quarter"], "split"].ne("train").any()
        or (metadata.groupby("fov_id")["train_quarter"].nunique() != 1).any()
    ):
        raise ValueError("Quarter membership must consist of complete training fields")
    if threshold_from_quarter(metadata) != manifest["threshold"]:
        raise ValueError("Threshold does not match the locked quarter-training reference")
    return PreparedData(images, metadata, manifest, root, sha256_file(root / "manifest.json"))


def normalization(data, indices):
    """Population moments from the specified training rows, with no label clipping."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Training indices must be a nonempty integer vector")
    if (
        indices.min() < 0
        or indices.max() >= len(data.metadata)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("Invalid or duplicated training indices")
    metadata = data.metadata.iloc[indices]
    if not metadata["split"].eq("train").all():
        raise ValueError("Normalization may use training rows only")
    sums, squares, count = np.zeros(4), np.zeros(4), 0
    for start in range(0, len(indices), 512):
        values = np.asarray(data.images[indices[start : start + 512]], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training images")
        sums += values.sum(axis=(0, 2, 3))
        squares += np.square(values).sum(axis=(0, 2, 3))
        count += len(values) * values.shape[2] * values.shape[3]
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    target = metadata["target"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or target.std() <= 0 or (std <= 0).any():
        raise ValueError("Training inputs and targets must have nonzero finite variance")
    return {
        "image_mean": mean.tolist(),
        "image_std": std.tolist(),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
    }
