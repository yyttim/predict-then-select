"""Batch-1 pools for acquisition-shift evaluation.

No normalization, calibration, model selection, or outcome-based gating is fitted
to these data. Each physical slide is a distinct candidate pool.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MARKER_COLUMNS, _gate_boolean, sha256_file


def make_external_cohort(index: pd.DataFrame, markers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select all batch-1/all-antibody cells with finite CD16 and no type gates."""
    if index.index.name != "global_index" or markers.index.name != "global_index":
        raise ValueError("Both metadata frames must be indexed by global_index")
    if not index.index.is_unique or not markers.index.is_unique:
        raise ValueError("Source global_index values must be unique")
    candidates = index.loc[(index["batch"] == 1) & (index["antibodies"] == "all")].copy()
    if candidates.empty:
        raise ValueError("No batch-1 all-antibody cells found")
    if not candidates.index.isin(markers.index).all():
        raise ValueError("Some external cells lack a marker-table row")
    selected_markers = markers.loc[candidates.index]
    gates = [name for name in markers if "selection_gated" in name]
    if any(_gate_boolean(selected_markers[name]).any() for name in gates):
        raise ValueError(
            "External source unexpectedly has published type gates; inspect changed metadata"
        )
    for name, column in MARKER_COLUMNS.items():
        if column not in markers:
            raise ValueError(f"Missing released marker {column}")
        candidates[name] = pd.to_numeric(selected_markers[column], errors="coerce")
    finite_primary = np.isfinite(candidates["CD16"])
    cohort = candidates.loc[finite_primary].sort_index().copy()
    if cohort.empty:
        raise ValueError("No finite external CD16 measurements")
    for name in MARKER_COLUMNS:
        if (cohort[name] < 0).any():
            raise ValueError(f"Negative {name} in released nonnegative abundance")
        cohort[f"{name}_available"] = np.isfinite(cohort[name])
    if not np.isfinite(cohort[["fov_center_x", "fov_center_y"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite external FOV coordinates")
    if cohort["data_path"].isna().any():
        raise ValueError("Missing external slide identity")
    cohort["slide_id"] = cohort["data_path"].astype(str).str.strip("/")
    xy = cohort[["fov_center_x", "fov_center_y"]].apply(
        lambda s: s.map(lambda v: format(float(v), ".12g"))
    )
    cohort["fov_id"] = cohort["slide_id"] + "|x=" + xy["fov_center_x"] + "|y=" + xy["fov_center_y"]
    cohort["pool_id"] = cohort["slide_id"]
    cohort["class_id"] = -1
    cohort["class_label_available"] = False
    cohort["split"] = "external"
    cohort["split_group"] = cohort["fov_id"]
    qc = {
        "source_cells": len(index),
        "batch1_all_candidates": len(candidates),
        "eligible_cells": len(cohort),
        "excluded_nonfinite_CD16": int((~finite_primary).sum()),
        "CD45_unavailable": int((~cohort["CD45_available"]).sum()),
        "published_type_labels_available": False,
        "slides": int(cohort["slide_id"].nunique()),
        "fovs": int(cohort["fov_id"].nunique()),
        "pools": {
            name: {
                "cells": len(frame),
                "fovs": int(frame["fov_id"].nunique()),
                "imaging_dates": sorted(frame["imaging_date"].astype(str).unique().tolist()),
            }
            for name, frame in cohort.groupby("pool_id")
        },
    }
    return cohort, qc


@dataclass
class ExternalData:
    images: np.ndarray
    metadata: pd.DataFrame
    manifest: dict

    def pool_rows(self, pool_id: str) -> np.ndarray:
        rows = np.flatnonzero(self.metadata["pool_id"].to_numpy() == pool_id)
        if not len(rows):
            raise ValueError(f"Unknown external pool: {pool_id}")
        return rows


def load_external(path: str | Path, verify_hashes=True) -> ExternalData:
    """Load raw archived intensities; use frozen source-model normalization later."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("role") != "external_acquisition_evaluation":
        raise ValueError("Not a frozen external acquisition dataset")
    if verify_hashes:
        for name, expected in manifest["prepared_sha256"].items():
            if sha256_file(root / name) != expected:
                raise ValueError(f"External prepared checksum mismatch: {name}")
    metadata = pd.read_csv(root / "metadata.csv")
    images = np.load(root / "images.npy", mmap_mode="r")
    if images.shape != (len(metadata), 4, 28, 28):
        raise ValueError("External image/metadata shape mismatch")
    if not metadata["global_index"].is_unique:
        raise ValueError("Duplicate external global indices")
    if not ((metadata["batch"] == 1) & (metadata["antibodies"] == "all")).all():
        raise ValueError("External dataset includes an unstated staining condition")
    if not ((metadata["split"] == "external") & (metadata["class_id"] == -1)).all():
        raise ValueError("External type labels must be explicitly unavailable")
    if not (metadata["pool_id"] == metadata["slide_id"]).all():
        raise ValueError("Each physical external slide must remain a separate candidate pool")
    return ExternalData(images, metadata, manifest)
