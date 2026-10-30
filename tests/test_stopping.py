import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellselect.study import file_hash, save_json


def test_stopping_averages_mse_not_rmse_and_preserves_primary(tmp_path):
    path = Path(__file__).parents[1] / "scripts/stopping_sensitivity.py"
    spec = importlib.util.spec_from_file_location("stopping_sensitivity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    save_json(tmp_path / "study.json", {"settings": {"seeds": [1, 2, 3]}})
    save_json(
        tmp_path / "selection.json",
        {
            "study_sha256": file_hash(tmp_path / "study.json"),
            "data_manifest_sha256": "a" * 64,
            "threshold": 1.0,
            "primary_fraction": 0.1,
            "selections": [
                {
                    "train_fraction": 1.0,
                    "selected_for": "mean",
                    "method": "mse",
                    "score": "mean",
                    "weight": 0.0,
                    "step": 1,
                }
            ],
        },
    )
    primary_hash = file_hash(tmp_path / "selection.json")
    rows = []
    for step, errors in ((1, [0.0, 0.0, 3.0]), (2, [1.2, 1.2, 1.2])):
        for seed, error in enumerate(errors, 1):
            checkpoint = tmp_path / "runs" / f"s{seed}_step{step}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(f"{seed}-{step}".encode())
            rows.append(
                {
                    "train_fraction": 1.0,
                    "selected_for": "mean",
                    "method": "mse",
                    "weight": 0.0,
                    "step": step,
                    "seed": seed,
                    "rmse": error,
                    "validation_utility": 99 - step,
                    "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
                    "checkpoint_sha256": file_hash(checkpoint),
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "validation_candidates.csv", index=False)
    selected = module.freeze_stopping(tmp_path)
    assert selected[0]["step"] == 2  # Mean RMSE would incorrectly choose step1.
    assert np.isclose(selected[0]["validation_loss"], 1.44)
    assert file_hash(tmp_path / "selection.json") == primary_hash
    assert selected[0]["states"][0]["checkpoint"].startswith("../runs/")
    with pytest.raises(FileExistsError):
        module.freeze_stopping(tmp_path)
