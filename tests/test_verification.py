"""Synthetic saved files exercise independent ranking and checkpoint bindings."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cellselect.metrics import prediction_metrics, selection_metrics

spec = importlib.util.spec_from_file_location(
    "verify_results", Path(__file__).resolve().parents[1] / "scripts/verify_results.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_saved_metrics_and_checkpoint_binding_are_independently_checked(tmp_path):
    output = tmp_path / "test"
    output.mkdir()
    ids, target = np.array([8, 2, 10, 1]), np.array([0.0, 5.0, 9.0, 20.0])
    predictions = {
        "mean": np.array([3.0, 3.0, 2.0, 1.0]),
        "hit_logit": np.array([-2.0, 1.0, 1.0, 0.0]),
    }
    config = {
        "method": "joint",
        "fraction": 1.0,
        "lambda_binary": 0.3,
        "seed": 17,
        "checkpoint_steps": [3],
    }
    study = {
        "runs": [config],
        "settings": {"seeds": [17], "selection_fractions": [0.5, 1.0], "primary_fraction": 0.5},
        "threshold": {"value": 5.0},
        "data_manifest_sha256": "a" * 64,
    }
    write(tmp_path / "study.json", study)
    checkpoint = {
        "model": {},
        "normalization": {},
        "config": {**config, "device": "cpu"},
        "step": 3,
        "threshold": 5.0,
        "manifest_sha256": "a" * 64,
    }
    torch.save(checkpoint, tmp_path / "weights.pt")
    state = {
        "seed": 17,
        "checkpoint": "weights.pt",
        "checkpoint_sha256": audit.sha(tmp_path / "weights.pt"),
    }
    choices = [
        {
            "train_fraction": 1.0,
            "method": "joint",
            "selected_for": purpose,
            "weight": 0.3,
            "step": 3,
            "score": score,
            "states": [state.copy()],
        }
        for purpose, score in [("mean", "mean"), ("purity", "hit_logit")]
    ]
    selection = {
        "study_sha256": audit.sha(tmp_path / "study.json"),
        "data_manifest_sha256": "a" * 64,
        "threshold": 5.0,
        "primary_fraction": 0.5,
        "selections": choices,
    }
    write(tmp_path / "selection.json", selection)
    np.savez_compressed(output / "prediction.npz", global_index=ids, **predictions)
    pd.DataFrame(
        {
            "global_index": ids,
            "target": target,
            "fov_id": ["a", "a", "b", "b"],
            "slide_id": "synthetic",
        }
    ).to_csv(output / "cells.csv", index=False)
    rows = []
    for choice in choices:
        for score, values in predictions.items():
            for fraction in [0.5, 1.0]:
                rows.append(
                    {
                        k: choice[k]
                        for k in ["train_fraction", "method", "selected_for", "weight", "step"]
                    }
                    | {
                        "seed": 17,
                        "score": score,
                        "selection_fraction": fraction,
                        "primary_score": score == choice["score"],
                    }
                    | selection_metrics(target, values, ids, 5.0, fraction)
                    | prediction_metrics(target, 5.0, **predictions)
                )
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    evaluation = {
        "status": "complete",
        "n_cells": 4,
        "n_fields": 2,
        "data_manifest_sha256": "a" * 64,
        "selection_sha256": audit.sha(tmp_path / "selection.json"),
        "cells_sha256": audit.sha(output / "cells.csv"),
        "metrics_sha256": audit.sha(output / "metrics.csv"),
        "prediction_files": [
            {
                "checkpoint": "weights.pt",
                "file": "prediction.npz",
                "sha256": audit.sha(output / "prediction.npz"),
            }
        ],
    }
    write(output / "evaluation.json", evaluation)
    checked = audit.verify(tmp_path)
    assert checked["metric_rows"] == 8 and checked["max_absolute_error"] < 1e-12
    assert audit.independent_metrics(target, predictions, "mean", ids, 5.0, 0.5)["purity"] == 0.5
    assert (
        audit.independent_metrics(target, predictions, "hit_logit", ids, 5.0, 0.5)["mean_marker"]
        == 7.0
    )
    # Rebind the file hash: the independent arithmetic must still catch a wrong row.
    altered = pd.read_csv(output / "metrics.csv")
    altered.loc[0, "purity"] += 0.1
    altered.to_csv(output / "metrics.csv", index=False)
    evaluation["metrics_sha256"] = audit.sha(output / "metrics.csv")
    write(output / "evaluation.json", evaluation)
    with pytest.raises(ValueError, match="Metric mismatch"):
        audit.verify(tmp_path)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    checkpoint["config"]["seed"] = 18
    torch.save(checkpoint, tmp_path / "weights.pt")
    for choice in selection["selections"]:
        choice["states"][0]["checkpoint_sha256"] = audit.sha(tmp_path / "weights.pt")
    write(tmp_path / "selection.json", selection)
    evaluation.update(
        selection_sha256=audit.sha(tmp_path / "selection.json"),
        metrics_sha256=audit.sha(output / "metrics.csv"),
    )
    write(output / "evaluation.json", evaluation)
    with pytest.raises(ValueError, match="Checkpoint configuration"):
        audit.verify(tmp_path)
