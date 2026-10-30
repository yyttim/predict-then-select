import numpy as np
import pytest

from cellselect.study import (
    choose_policies,
    family,
    outcome_metrics,
    class_mean_scores,
    SCORING_PRECISION,
    study_prediction_metrics,
)


@pytest.mark.parametrize("constant_target", [False, True])
def test_exactly_constant_nonbinary_values_have_serializable_undefined_correlations(
    constant_target,
):
    import json
    import warnings

    constant = np.full(1000, 0.1)
    assert constant.std() > 0  # Floating reduction noise must not imply variation.
    varied = np.arange(1000, dtype=float)
    y, score = (constant, varied) if constant_target else (varied, constant)
    with warnings.catch_warnings(record=True) as messages:
        actual = study_prediction_metrics(y, score)
    assert not messages
    assert actual["pearson"] is None and actual["spearman"] is None
    assert actual["rmse"] == float(np.sqrt(np.mean((y - score) ** 2)))
    assert actual["mae"] == float(np.mean(np.abs(y - score)))
    json.dumps(actual, allow_nan=False)


def test_equal_class_means_cannot_create_a_numerical_ranking_signal():
    logits = np.random.default_rng(21).normal(size=(1000, 3)).astype(np.float32) * 5
    scores = class_mean_scores(logits, [123.45, 123.45, 123.45])
    assert np.array_equal(scores, np.full(1000, 123.45))


def test_float64_class_expectation_preserves_near_saturated_score_order():
    import torch

    # Increasing class-1 probability must lower the marker expectation.
    logits = np.array([[8.0, 0.0, 0.0], [8.0, 1e-6, 0.0]], dtype=np.float32)
    means = np.array([9.0, 1.0, 4.0])
    scores = class_mean_scores(logits, means)
    historical = torch.softmax(torch.from_numpy(logits), dim=1).numpy() @ means
    assert scores.dtype == np.float64
    assert scores[0] > scores[1]
    assert historical[0] < historical[1]  # Float32 roundoff reverses the order.
    weights = np.exp(logits.astype(float))
    np.testing.assert_allclose(scores, (weights @ means) / weights.sum(axis=1), rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        class_mean_scores(logits.astype(float) + 1000, means), scores, rtol=0, atol=2e-15
    )
    with pytest.raises(ValueError, match="Invalid"):
        class_mean_scores(logits, [1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="Invalid"):
        class_mean_scores(logits, [1.0, 3.0])


def validation_table():
    rows = []
    for seed in [16, 26, 36]:
        for name, error, utility in [
            ("mse_lambda_0", 2, 5),
            ("mse_lambda_1", 1, 4),
            ("mse_lambda_0.3", 3, 6),
            ("class_means", 5, 2),
            ("spo_plus", 6, 7),
            ("pg_central_h_0.1", 9, 8),
            ("pg_central_h_1", 4, 3),
        ]:
            for epoch in [60, 100]:
                rows.append(
                    dict(
                        family=name,
                        epoch=epoch,
                        seed=seed,
                        rmse=error,
                        utility=utility + (epoch == 100),
                    )
                )
    return rows


def test_distinct_prediction_and_decision_choices_with_strong_baselines():
    result = choose_policies(validation_table())
    assert result["predictive_selector"]["family"] == "mse_lambda_1"
    assert result["predictive_selector"]["epoch"] == 60
    assert result["utility_selector"]["family"] == "mse_lambda_0.3"
    assert result["utility_selector"]["epoch"] == 100
    assert result["regression"]["family"] == "mse_lambda_0"
    assert result["regression"]["epoch"] == 100
    assert result["pg_central"]["family"] == "pg_central_h_0.1"
    assert result["spo_plus"]["family"] == "spo_plus"


def test_incomplete_or_duplicate_paired_candidates_rejected():
    table = validation_table()
    with pytest.raises(ValueError, match="Unpaired"):
        choose_policies(table[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        choose_policies(table + [table[0]])


def test_missing_required_method_rejected():
    with pytest.raises(ValueError, match="Missing required comparator"):
        choose_policies([r for r in validation_table() if r["family"] != "class_means"])


def test_external_evaluation_needs_no_class_labels():
    y, s, ids = np.array([1.0, 7.0, 3.0, 8.0]), np.array([0.0, 3.0, 2.0, 1.0]), np.arange(4)
    result = outcome_metrics(y, s, ids, 0.5)
    assert result["utility"] == 5
    assert result["oracle_utility"] == 7.5
    assert result["oracle_gap"] == 2.5
    assert "within_type_gap" not in result
    assert "selected_type_counts" not in result
    assert family({"objective": "pg_central", "pg_h": 0.1}) == "pg_central_h_0.1"


def test_type_matched_random_control_uses_entire_observed_type():
    result = outcome_metrics(
        np.array([1.0, 9.0, 2.0, 8.0]),
        np.array([0.0, 3.0, 1.0, 2.0]),
        np.arange(4),
        0.5,
        np.array([0, 0, 1, 1]),
    )
    assert result["utility"] == 8.5
    assert result["type_matched_random_utility"] == 5
    assert result["within_type_enrichment"] == 3.5


def test_utility_exchange_diagnostic_counts_changes_without_inventing_sample_size():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from extended_study import exchange_diagnostics

    y = np.array([1.0, 9.0, 2.0, 8.0])
    a = np.array([[0.0, 3.0, 1.0, 2.0], [0.0, 3.0, 1.0, 2.0]])
    b = np.array([[4.0, 3.0, 2.0, 1.0], [0.0, 3.0, 1.0, 2.0]])
    result = exchange_diagnostics(y, a, b, np.arange(4), 0.5)
    first, second = result["paired_replicates"]
    assert first["swapped_cells_per_policy"] == 1
    assert first["mean_added_outcome"] == 8
    assert first["mean_removed_outcome"] == 1
    assert first["utility_difference"] == first["difference_from_exchanges"] == 3.5
    assert second["swapped_cells_per_policy"] == 0
    assert second["mean_added_outcome"] is None
    assert second["utility_difference"] == 0
    assert result["mean_selected_overlap_fraction"] == 0.75


def test_real_training_freeze_evaluate_and_artifact_guard(tmp_path):
    import json
    import torch
    from test_training import _write_fixture, _config
    from cellselect.training import PreparedData, run_training
    from cellselect.study import freeze_study, evaluate_study

    data_root = tmp_path / "data"
    _write_fixture(data_root)
    data = PreparedData(data_root)
    common = _config(
        epochs=1,
        checkpoint_epochs=[1],
        reference_epoch=1,
        marker="CD16",
        train_fraction=1.0,
        lambda_class=0.0,
        gradient_every=0,
    )
    overrides = [
        dict(name="mse", mode="joint", objective="mse"),
        dict(name="joint", mode="joint", objective="mse", lambda_class=1.0),
        dict(name="ce", mode="classifier", objective="mse"),
        dict(name="spo", mode="joint", objective="spo_plus"),
        dict(name="pg", mode="joint", objective="pg_central", pg_h=0.1),
    ]
    runs = []
    previous_threads = torch.get_num_threads()
    try:
        for item in overrides:
            config = {**common, **item}
            name = config.pop("name")
            run_training(data, config, tmp_path / "runs" / name)
            runs.append(item)
        spec = tmp_path / "spec.json"
        spec.write_text(
            json.dumps(dict(common=common, runs=runs, primary_fraction=0.25, fractions=[0.25, 0.5]))
        )
        protocol = tmp_path / "protocol.json"
        frozen = freeze_study(data_root, tmp_path / "runs", spec, protocol)
        assert len(frozen["records"]) == 5
        assert "already evaluated" in frozen["test_status"]
        assert frozen["scoring_precision"] == SCORING_PRECISION
        assert "study.py" in frozen["inference_source_sha256"]
        ce = next(record for record in frozen["records"] if record["family"] == "class_means")
        with np.load(tmp_path / "runs" / ce["name"] / ce["predictions_file"]) as saved:
            reconstructed = class_mean_scores(saved["logits"], data.class_means)
            y = data.metadata.iloc[data.indices["val"]].CD16.to_numpy()
            expected_rmse = np.sqrt(np.mean((y - reconstructed) ** 2))
        ce_row = next(
            row
            for row in frozen["groups"]["CD16_fraction_1"]["validation_table"]
            if row["family"] == "class_means"
        )
        assert ce_row["rmse"] == expected_rmse
        result = evaluate_study(
            data_root, tmp_path / "runs", protocol, tmp_path / "evaluation", device="cpu"
        )
        assert len(result["metrics"]) == 14
        assert all(np.isfinite(r["utility"]) for r in result["metrics"])
        with pytest.raises(FileExistsError):
            freeze_study(data_root, tmp_path / "runs", spec, protocol)
        # Integrity must be checked even if the manifest itself was not edited.
        with (data_root / "images.npy").open("ab") as stream:
            stream.write(b"tamper")
        with pytest.raises(ValueError, match="prepared data changed"):
            evaluate_study(
                data_root, tmp_path / "runs", protocol, tmp_path / "evaluation", device="cpu"
            )
    finally:
        torch.set_num_threads(previous_threads)
