from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cellselect.evaluation import evaluate_pool, freeze_selection
from cellselect.study import file_hash, run_name, save_json


@pytest.fixture
def toy_study(tmp_path):
    meta = pd.DataFrame(
        {
            "global_index": [10, 11, 12, 13, 14],
            "target": [1.0, 2.0, 9.0, 10.0, -999.0],
            "split": ["val"] * 4 + ["test"],
        }
    )
    data = SimpleNamespace(
        metadata=meta,
        manifest_sha256="data-hash",
        manifest={"threshold": {"value": 8.0}},
        rows=lambda split: np.flatnonzero(meta.split.to_numpy() == split),
    )
    settings = {"seeds": [1, 2], "train_fractions": [0.25], "primary_fraction": 0.25}
    configs = []
    for method, weight in (("mse", 0.0), ("bce", 0.0), ("joint", 0.3), ("joint", 1.0)):
        for seed in settings["seeds"]:
            config = {
                "method": method,
                "lambda_binary": weight,
                "seed": seed,
                "fraction": 0.25,
                "max_steps": 2,
                "checkpoint_steps": [1, 2],
            }
            configs.append(config)
            folder = tmp_path / "runs" / run_name(config)
            save_json(folder / "config.json", {**config, "device": "cpu"})
            checkpoints = []
            for step in (1, 2):
                checkpoint = folder / f"step_{step:06d}.pt"
                checkpoint.write_bytes(f"model-{method}-{weight}-{seed}-{step}".encode())
                path = folder / f"val_{step:06d}.npz"
                arrays = {"global_index": meta.global_index.to_numpy()[:4]}
                if method != "bce":
                    arrays["mean"] = np.array([0, 0, 9, 10] if step == 1 else [0, 0, 10, 9])
                if method != "mse":
                    arrays["hit_logit"] = np.array([4, 3, 2, 1] if step == 1 else [0, 0, 10, 9])
                np.savez_compressed(path, **arrays)
                checkpoints.append(
                    {
                        "step": step,
                        "checkpoint_file": checkpoint.name,
                        "checkpoint_sha256": file_hash(checkpoint),
                        "predictions_file": path.name,
                        "predictions_sha256": file_hash(path),
                    }
                )
            save_json(
                folder / "complete.json",
                {"manifest_sha256": "data-hash", "steps": 2, "checkpoints": checkpoints},
            )
    save_json(
        tmp_path / "study.json",
        {"settings": settings, "runs": configs, "data_manifest_sha256": "data-hash"},
    )
    return data, tmp_path


def test_utility_specific_choices_and_joint_checkpoint_pairing(toy_study):
    data, root = toy_study
    result = freeze_selection(data, root)
    joint = {c["selected_for"]: c for c in result["selections"] if c["method"] == "joint"}
    assert joint["mean"]["step"] == 1
    assert joint["purity"]["step"] == 2
    assert joint["mean"]["score"] == "mean"
    assert joint["purity"]["score"] == "hit_logit"
    assert all(c["weight"] == 0.3 for c in joint.values())
    assert all([s["seed"] for s in c["states"]] == [1, 2] for c in result["selections"])
    # BCE has no regression output, including when selected for mean abundance.
    assert all(c["score"] == "hit_logit" for c in result["selections"] if c["method"] == "bce")


def test_changing_unread_test_targets_does_not_change_choices(toy_study):
    data, root = toy_study
    first = freeze_selection(data, root)
    (root / "selection.json").unlink()
    data.metadata.loc[data.metadata.split == "test", "target"] = 1e12
    second = freeze_selection(data, root)
    assert first == second


def test_changed_training_predictions_rejected(toy_study):
    data, root = toy_study
    prediction = next((root / "runs").glob("*/val_*.npz"))
    prediction.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed training artifact"):
        freeze_selection(data, root)


def test_missing_run_is_not_silently_skipped(toy_study):
    data, root = toy_study
    next((root / "runs").glob("*/complete.json")).unlink()
    with pytest.raises(FileNotFoundError):
        freeze_selection(data, root)


def test_test_evaluation_requires_frozen_choices(tmp_path):
    with pytest.raises(ValueError, match="Freeze validation"):
        evaluate_pool(None, tmp_path)
