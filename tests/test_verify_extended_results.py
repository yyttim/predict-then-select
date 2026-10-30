"""End-to-end verification tests using synthetic cohorts."""

import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from cellselect.study import freeze_study, evaluate_study
from cellselect.training import PreparedData, run_training, file_sha256, write_json
from test_training import _config, _write_fixture

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_script("verify_extended_results")
extended = load_script("extended_study")


def test_numpy_metrics_have_known_decomposition_and_external_scope():
    y, score, ids = (
        np.array([10.0, 2.0, 8.0, 1.0]),
        np.array([0.0, 4.0, 3.0, 2.0]),
        np.array([4, 3, 2, 1]),
    )
    classes = np.array([0, 0, 1, 1])
    actual = audit.numpy_metrics(y, score, ids, 0.5, {"slope": 2.0, "intercept": 1.0}, classes)
    assert actual["utility"] == 5 and actual["oracle_utility"] == 9
    assert actual["within_type_gap"] == 4 and actual["type_allocation_gap"] == 0
    assert actual["type_matched_random_utility"] == 5.25
    assert actual["within_type_enrichment"] == -0.25
    assert actual["rmse"] == pytest.approx(np.sqrt(130 / 4))
    external = audit.numpy_metrics(y, score, ids, 0.5, {"slope": 1.0, "intercept": 0.0})
    assert not audit.TYPE_FIELDS & external.keys()
    constant = audit.numpy_metrics(y, np.ones(4), ids, 0.5, {"slope": 1.0, "intercept": 0.0})
    assert constant["pearson"] is None and constant["spearman"] is None
    constant = audit.numpy_metrics(
        np.arange(1000.0),
        np.full(1000, 0.1),
        np.arange(1000),
        0.1,
        {"slope": 1.0, "intercept": 0.0},
    )
    assert constant["pearson"] is None and constant["spearman"] is None
    json.dumps(constant, allow_nan=False)


def test_independent_policy_choices_use_means_and_stable_family_epoch_ties():
    rows = []
    for family in ["mse_lambda_0", "mse_lambda_1", "class_means", "spo_plus", "pg_central_h_1"]:
        for epoch in (20, 40):
            for seed in (1, 2):
                rows.append(
                    {
                        "family": family,
                        "epoch": epoch,
                        "seed": seed,
                        "rmse": 0.5 if family == "mse_lambda_1" and epoch == 40 else 2.0,
                        "utility": 10.0,
                    }
                )
    choices = audit.independent_choices(rows)
    assert choices["predictive_selector"]["family"] == "mse_lambda_1"
    assert choices["predictive_selector"]["epoch"] == 40
    assert choices["utility_selector"]["family"] == "mse_lambda_0"
    assert choices["utility_selector"]["epoch"] == 20
    with pytest.raises(ValueError, match="paired"):
        audit.independent_choices(rows[:-1])


def test_exchange_identity_including_identical_selected_sets():
    y = np.array([1.0, 5.0, 9.0, 3.0])
    a = np.array([[1.0, 2.0, 4.0, 3.0], [1.0, 2.0, 4.0, 3.0]])
    b = np.array([[1.0, 2.0, 4.0, 3.0], [4.0, 3.0, 2.0, 1.0]])
    result = audit.independent_exchange(y, a, b, np.arange(4), 0.5)
    same, different = result["paired_replicates"]
    assert same["swapped_cells_per_policy"] == 0
    assert same["mean_added_outcome"] is None and same["mean_removed_outcome"] is None
    assert different["swapped_cells_per_policy"] == 2
    assert different["utility_difference"] == different["difference_from_exchanges"] == 3.0


@pytest.fixture(scope="module")
def completed_synthetic_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("extended_audit_synthetic")
    data_root, external_root, runs, results = [
        root / name for name in ("data", "external", "runs", "results")
    ]
    _write_fixture(data_root)
    external_root.mkdir()
    rng = np.random.default_rng(8)
    np.save(external_root / "images.npy", rng.normal(size=(6, 4, 28, 28)).astype(np.float32))
    frame = pd.DataFrame(
        {
            "global_index": np.arange(2000, 2006),
            "CD16": [4.0, 8.0, 5.0, 6.0, 1.0, 9.0],
            "class_id": -1,
            "class_label_available": False,
            "batch": 1,
            "antibodies": "all",
            "split": "external",
            "slide_id": "synthetic_slide",
            "pool_id": "synthetic_slide",
            "fov_id": ["fov_a"] * 3 + ["fov_b"] * 3,
        }
    )
    frame.to_csv(external_root / "metadata.csv", index=False)
    write_json(
        external_root / "manifest.json",
        {
            "role": "external_acquisition_evaluation",
            "purpose": "synthetic engineering fixture, not real biological data",
            "prepared_sha256": {
                name: file_sha256(external_root / name) for name in ["images.npy", "metadata.csv"]
            },
        },
    )
    common = _config(
        epochs=2,
        checkpoint_epochs=[1, 2],
        reference_epoch=1,
        gradient_every=0,
        primary_fraction=0.5,
        marker="CD16",
        train_fraction=1.0,
        subset_seed=7,
        objective="mse",
    )
    overrides = []
    methods = [
        ("mse0", {"lambda_class": 0.0}),
        ("mse1", {"lambda_class": 1.0}),
        ("class", {"mode": "classifier", "lambda_class": 0.0}),
        ("spo", {"objective": "spo_plus", "lambda_class": 0.0}),
        ("pg", {"objective": "pg_central", "pg_h": 1.0, "lambda_class": 0.0}),
    ]
    previous = torch.get_num_threads()
    data = PreparedData(data_root, subset_seed=7)
    for seed in (11, 12):
        for name, method in methods:
            override = {"name": f"{name}_seed{seed}", "seed": seed, **method}
            overrides.append(override)
            config = {**common, **override}
            run_name = config.pop("name")
            run_training(data, config, runs / run_name)
    declaration = root / "spec.json"
    write_json(
        declaration,
        {"common": common, "runs": overrides, "primary_fraction": 0.5, "fractions": [0.5, 1.0]},
    )
    protocol = root / "protocol.json"
    freeze_study(data_root, runs, declaration, protocol, external_root)
    evaluate_study(data_root, runs, protocol, results, external_root, device="cpu")
    extended.uncertainty(data_root, protocol, results, external_root, repetitions=16)
    torch.set_num_threads(previous)
    return data_root, runs, declaration, protocol, results, external_root


def test_complete_synthetic_study_passes_all_independent_checks(completed_synthetic_study):
    result = audit.verify_study(*completed_synthetic_study)
    assert result["status"] == "passed"
    assert result["declared_training_runs"] == 10 and result["candidate_states"] == 20
    assert result["reported_metric_rows_recomputed"] == 56
    assert result["paired_intervals_recomputed_full_pool"] == 20
    assert result["exchange_diagnostics_recomputed"] == 20
    assert result["external_subsampling"] is False
    assert result["pool_sizes"]["synthetic_slide"]["cells"] == 6
    assert result["pool_sizes"]["synthetic_slide"]["type_labels_used"] is False


def copied_results(study, tmp_path):
    values = list(study)
    copied = tmp_path / "results"
    shutil.copytree(values[4], copied)
    values[4] = copied
    return values


def rewrite_results_and_binding(path, results):
    write_json(path / "results.json", results)
    uncertainty = json.loads((path / "uncertainty.json").read_text())
    uncertainty["results_sha256"] = file_sha256(path / "results.json")
    write_json(path / "uncertainty.json", uncertainty)


def test_corrupted_metric_is_detected_beyond_file_hashes(completed_synthetic_study, tmp_path):
    args = copied_results(completed_synthetic_study, tmp_path)
    result = json.loads((args[4] / "results.json").read_text())
    result["metrics"][0]["utility"] += 1
    rewrite_results_and_binding(args[4], result)
    with pytest.raises(ValueError, match="numerical mismatch"):
        audit.verify_study(*args)


def test_external_class_diagnostics_are_rejected(completed_synthetic_study, tmp_path):
    args = copied_results(completed_synthetic_study, tmp_path)
    result = json.loads((args[4] / "results.json").read_text())
    next(row for row in result["metrics"] if row["pool"] == "synthetic_slide")["macro_f1"] = 1.0
    rewrite_results_and_binding(args[4], result)
    with pytest.raises(ValueError, match="unavailable class diagnostics"):
        audit.verify_study(*args)


def test_incomplete_grid_fails_before_any_external_outcome_read(
    completed_synthetic_study, tmp_path
):
    args = copied_results(completed_synthetic_study, tmp_path)
    result = json.loads((args[4] / "results.json").read_text())
    result["metrics"].pop()
    rewrite_results_and_binding(args[4], result)
    args[5] = tmp_path / "must_not_be_opened"
    with pytest.raises(ValueError, match="Incomplete/duplicate reported metric grid"):
        audit.verify_study(*args)
