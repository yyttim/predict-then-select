"""Artificial validation pools check shared resampling and frozen inputs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellselect.training import file_sha256

spec = importlib.util.spec_from_file_location(
    "validation_stability", Path(__file__).parents[1] / "scripts" / "validation_stability.py"
)
stability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stability)


def test_identical_models_choose_lower_weight_then_earlier_epoch_in_every_draw():
    y = np.array([1.0, 5.0, 2.0, 4.0])
    scores = np.broadcast_to(y, (4, 3, 4)).copy()
    result = stability.choice_stability(
        y,
        scores,
        [0, 0, 1, 1],
        [4, 2, 8, 1],
        [(1.0, 40), (0.0, 40), (1.0, 20), (0.0, 20)],
        0.5,
        repetitions=20,
    )
    assert result["weights"] == [0.0, 1.0]
    assert result["rmse_choice_counts"] == result["utility_choice_counts"] == [20, 0, 0, 0]
    assert result["rmse_lambda_counts"] == result["utility_lambda_counts"] == [20, 0]
    assert result["observed_selected_by_rmse"] == {"lambda_class": 0.0, "epoch": 20}
    assert result["selectors_agreement_frequency"] == 1.0


def test_mean_seed_rmse_is_not_pooled_error_or_ensemble_prediction():
    y = np.arange(4.0)
    # Per-seed RMSEs: lambda=0 gives (0, 4); lambda=1 gives (2.2, 2.2).
    # Mean RMSE chooses lambda=0; sqrt(pooled MSE) chooses lambda=1.
    scores = np.array([[y, y + 4], [y + 2.2, y + 2.2]])
    result = stability.choice_stability(
        y, scores, [0, 0, 1, 1], np.arange(4), [(0.0, 40), (1.0, 20)], 0.5, repetitions=20
    )
    np.testing.assert_allclose(result["observed_mean_rmse"], [2.0, 2.2])
    assert result["observed_selected_by_rmse"] == {"lambda_class": 0.0, "epoch": 40}
    assert result["rmse_choice_counts"] == [20, 0]


def test_common_field_draws_match_independent_calculation_and_row_permutation():
    y = np.array([0.0, 10.0, 3.0, 8.0, 1.0, 7.0])
    groups, ids = np.array([0, 1, 1, 2, 2, 2]), np.array([9, 4, 5, 1, 2, 3])
    scores = np.array(
        [[[2, 5, 3, 0, 1, 4], [3, 0, 5, 2, 4, 1]], [[0, 2, 1, 5, 4, 3], [2, 1, 0, 3, 5, 4]]],
        dtype=float,
    )
    repetitions, seed, fraction = 40, 7, 0.4
    pairs = [(0.0, 20), (0.0, 40)]
    result = stability.choice_stability(y, scores, groups, ids, pairs, fraction, repetitions, seed)
    counts, agreement = np.zeros((2, 2), dtype=int), 0
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        index = np.concatenate([np.flatnonzero(groups == g) for g in rng.integers(3, size=3)])
        k = int(np.ceil(fraction * len(index)))
        rmse, utility = [], []
        for family in scores:
            rmse.append(np.mean([np.sqrt(np.mean((s[index] - y[index]) ** 2)) for s in family]))
            utilities = []
            for s in family:
                selected = sorted(index.tolist(), key=lambda i: (-s[i], ids[i]))[:k]
                utilities.append(y[selected].mean())
            utility.append(np.mean(utilities))
        a, b = np.argmin(rmse), np.argmax(utility)
        counts[0, a] += 1
        counts[1, b] += 1
        agreement += a == b
    assert result["rmse_choice_counts"] == counts[0].tolist()
    assert result["utility_choice_counts"] == counts[1].tolist()
    assert result["selectors_agreement_frequency"] == agreement / repetitions
    assert result["selectors_lambda_agreement_frequency"] == 1.0
    assert result["rmse_lambda_counts"] == result["utility_lambda_counts"] == [repetitions]
    p = np.array([3, 0, 5, 1, 4, 2])
    permuted = stability.choice_stability(
        y[p], scores[:, :, p], groups[p], ids[p], pairs, fraction, repetitions, seed
    )
    for key in ("rmse_choice_counts", "utility_choice_counts", "selectors_agreement_frequency"):
        assert result[key] == permuted[key]


def test_loader_uses_only_validation_outcomes_and_rejects_changed_artifacts(tmp_path):
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    directory = runs / "regression"
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "global_index": [0, 1, 2, 3],
            "split": ["train", "val", "val", "test"],
            "fov_id": [0, 1, 2, 3],
            "CD16": ["unused", "2", "4", "unused"],
        }
    ).to_csv(data / "metadata.csv", index=False)
    (data / "manifest.json").write_text(
        json.dumps({"prepared_sha256": {"metadata.csv": file_sha256(data / "metadata.csv")}})
    )
    np.savez(directory / "validation_epoch_40.npz", global_index=[1, 2], score=[2.0, 4.0])
    run = {
        "name": "regression",
        "epoch": 40,
        "checkpoint_file": "epoch_40.pt",
        "predictions_file": "validation_epoch_40.npz",
        "config": {"mode": "joint", "lambda_class": 0.0, "seed": 16},
    }
    for filename, key in [
        ("epoch_40.pt", "checkpoint_sha256"),
        ("config.json", "config_sha256"),
        ("normalization.json", "normalization_sha256"),
    ]:
        (directory / filename).write_bytes(b"engineering fixture")
        run[key] = file_sha256(directory / filename)
    run["validation_predictions_sha256"] = file_sha256(directory / "validation_epoch_40.npz")
    protocol = {
        "schema_version": 2,
        "expected_experiment_verified": True,
        "test_used_to_choose_protocol": False,
        "checkpoint_epochs": [40],
        "manifest_sha256": file_sha256(data / "manifest.json"),
        "marker": "CD16",
        "weights": [0.0],
        "seeds": [16],
        "runs": [run],
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    _, arrays = stability.load_frozen_validation(data, runs, path)
    np.testing.assert_array_equal(arrays[0], [2.0, 4.0])
    np.testing.assert_array_equal(arrays[3], [1, 2])
    assert arrays[4] == [(0.0, 40)]
    (directory / "normalization.json").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        stability.load_frozen_validation(data, runs, path)
