"""Validation selection and evaluation tests using synthetic scores and images."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import cellselect.evaluation as evaluation
from cellselect.metrics import paired_seed_field_bootstrap, topk_indices
from cellselect.training import PreparedData, file_sha256, run_training, write_json
from test_training import _config, _write_fixture


def _spec():
    common = _config(epochs=1, primary_fraction=1 / 3, marker="CD16", train_fraction=1.0)
    common.pop("mode")
    common.pop("lambda_class")
    common.pop("seed")
    return {
        "common": common,
        "runs": [
            {"name": "regression", "seed": 17, "mode": "joint", "lambda_class": 0.0},
            {"name": "joint", "seed": 17, "mode": "joint", "lambda_class": 1.0},
            {"name": "classifier", "seed": 17, "mode": "classifier", "lambda_class": 0.0},
        ],
    }


def _fake_run(root, name, config, data, scores):
    directory = root / name
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    write_json(directory / "normalization.json", data.normalization())
    # A non-checkpoint file detects unintended model loading during freeze.
    (directory / "final.pt").write_bytes(b"synthetic engineering placeholder checkpoint")
    validation = data.metadata.iloc[data.indices["val"]]
    np.savez_compressed(
        directory / "validation_predictions.npz",
        global_index=validation.global_index.to_numpy(),
        score=np.asarray(scores),
        logits=np.zeros((len(validation), 3)),
    )
    train_ids = data.metadata.iloc[data.indices["train"]].global_index.to_numpy()
    write_json(
        directory / "provenance.json",
        {
            "manifest_sha256": data.manifest_hash,
            "train_ids_sha256": hashlib.sha256(train_ids.tobytes()).hexdigest(),
            "test_evaluated": False,
            "source_sha256": {
                p.name: file_sha256(p) for p in Path(evaluation.__file__).parent.glob("*.py")
            },
        },
    )
    write_json(
        directory / "complete.json",
        {
            "checkpoint_sha256": file_sha256(directory / "final.pt"),
            "config_sha256": file_sha256(directory / "config.json"),
            "normalization_sha256": file_sha256(directory / "normalization.json"),
            "validation_predictions_sha256": file_sha256(directory / "validation_predictions.npz"),
            "test_evaluated": False,
        },
    )


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def fabricated(tmp_path):
    directory = tmp_path / "cohort"
    _write_fixture(directory)
    data = PreparedData(directory)
    # Regression has lower RMSE; the joint model ranks the top outcomes correctly.
    y = np.array([2.0, 9.0, 8.0, 1.0, 4.0, 3.0])
    data.metadata.loc[data.indices["val"], "CD16"] = y
    scores = {
        "regression": np.array([2.0, 6.0, 8.0, 1.0, 7.0, 3.0]),
        "joint": y + 100,
        "classifier": np.ones(6),
    }
    root = tmp_path / "runs"
    spec = _spec()
    for item in spec["runs"]:
        config = {**spec["common"], **item}
        name = config.pop("name")
        _fake_run(root, name, config, data, scores[name])
    return data, root, spec, tmp_path / "protocol.json"


def _freeze(fabricated, **overrides):
    data, root, spec, output = fabricated
    arguments = dict(primary_fraction=1 / 3, fractions=(1 / 3, 0.5), expected_spec=spec)
    arguments.update(overrides)
    return evaluation.freeze_protocol(data, root, output, **arguments)


def _add_two_epoch_candidates(fabricated, scores=None):
    """Create two synthetic checkpoint candidates per run."""
    data, root, spec, _ = fabricated
    spec["common"].update(epochs=2, checkpoint_epochs=[1, 2], reference_epoch=1)
    ids = data.metadata.iloc[data.indices["val"]].global_index.to_numpy()
    for item in spec["runs"]:
        config = {**spec["common"], **item}
        name = config.pop("name")
        directory = root / name
        write_json(directory / "config.json", config)
        with np.load(directory / "validation_predictions.npz") as old:
            default_scores = old["score"].copy()
        complete = json.loads((directory / "complete.json").read_text())
        complete["candidates"] = []
        (directory / "snapshots").mkdir(exist_ok=True)
        for epoch in (1, 2):
            checkpoint_file = f"snapshots/epoch_{epoch:03d}.pt"
            predictions_file = f"snapshots/epoch_{epoch:03d}_validation.npz"
            (directory / checkpoint_file).write_bytes(
                f"synthetic candidate {name}, epoch {epoch}".encode()
            )
            values = default_scores if scores is None else scores[(name, epoch)]
            np.savez_compressed(
                directory / predictions_file,
                global_index=ids,
                score=np.asarray(values, dtype=float),
                logits=np.zeros((len(ids), 3)),
            )
            complete["candidates"].append(
                {
                    "epoch": epoch,
                    "checkpoint_file": checkpoint_file,
                    "checkpoint_sha256": file_sha256(directory / checkpoint_file),
                    "predictions_file": predictions_file,
                    "predictions_sha256": file_sha256(directory / predictions_file),
                    "validation": {},
                }
            )
        (directory / "validation_predictions.npz").write_bytes(
            (directory / complete["candidates"][-1]["predictions_file"]).read_bytes()
        )
        complete["validation_predictions_sha256"] = file_sha256(
            directory / "validation_predictions.npz"
        )
        complete["config_sha256"] = file_sha256(directory / "config.json")
        write_json(directory / "complete.json", complete)
    return fabricated


def test_positive_affine_fit_is_constrained_least_squares_and_preserves_ranks():
    x = np.array([3.0, -2.0, 4.0, 4.0, 0.0])
    fit = evaluation.fit_positive_affine(2.5 * x - 7, x)
    assert fit["slope"] == pytest.approx(2.5)
    assert fit["intercept"] == pytest.approx(-7)
    calibrated = fit["slope"] * x + fit["intercept"]
    ids = np.array([4, 3, 2, 1, 0])
    for k in range(1, len(x) + 1):
        np.testing.assert_array_equal(topk_indices(x, k, ids), topk_indices(calibrated, k, ids))
    inverse = evaluation.fit_positive_affine(-x, x, minimum_slope=0.01)
    assert inverse["slope"] == pytest.approx(0.01)
    constant = evaluation.fit_positive_affine([1, 4, 7], [2, 2, 2])
    assert constant["slope"] > 0
    assert constant["slope"] * 2 + constant["intercept"] == pytest.approx(4)


def test_selectors_depend_only_on_validation_without_loading_model_or_test(fabricated, monkeypatch):
    data, _, _, _ = fabricated
    monkeypatch.setattr(
        data, "partition", lambda *a, **k: pytest.fail("Freeze requested image partition")
    )
    monkeypatch.setattr(
        evaluation, "build_model", lambda *a, **k: pytest.fail("Freeze built a model")
    )
    # Test outcomes may be unavailable at this stage.
    data.metadata.loc[data.indices["test"], "CD16"] = np.nan
    protocol = _freeze(fabricated)
    assert protocol["lambda_by_rmse"] == 0.0
    assert protocol["lambda_by_utility"] == 1.0
    assert protocol["expected_experiment_verified"] is True
    assert protocol["test_used_to_choose_protocol"] is False
    joint = next(r for r in protocol["runs"] if r["name"] == "joint")
    assert joint["calibration"]["intercept"] == pytest.approx(-100)
    assert joint["calibration"]["fit_partition"] == "val"


def test_protocol_cannot_be_overwritten(fabricated):
    _freeze(fabricated)
    with pytest.raises(FileExistsError):
        _freeze(fabricated)


def test_equal_validation_metrics_choose_smaller_auxiliary_weight(fabricated):
    _, root, _, _ = fabricated
    target = root / "joint" / "validation_predictions.npz"
    target.write_bytes((root / "regression" / "validation_predictions.npz").read_bytes())
    complete_path = root / "joint" / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["validation_predictions_sha256"] = file_sha256(target)
    write_json(complete_path, complete)
    protocol = _freeze(fabricated)
    assert protocol["lambda_by_rmse"] == protocol["lambda_by_utility"] == 0.0


def test_candidate_ties_choose_lower_weight_then_earlier_epoch(fabricated):
    data, root, spec, _ = fabricated
    y = data.metadata.iloc[data.indices["val"]].CD16.to_numpy()
    _add_two_epoch_candidates(
        fabricated, {(item["name"], epoch): y for item in spec["runs"] for epoch in (1, 2)}
    )
    protocol = _freeze(fabricated)
    assert (
        protocol["selected_by_rmse"]
        == protocol["selected_by_utility"]
        == {"lambda_class": 0.0, "epoch": 1}
    )


def test_weight_epoch_selection_uses_paired_seed_means(fabricated):
    data, root, spec, _ = fabricated
    y = data.metadata.iloc[data.indices["val"]].CD16.to_numpy()
    extra = []
    for item in spec["runs"]:
        new = {**item, "name": item["name"] + "_seed18", "seed": 18}
        config = {**spec["common"], **new}
        name = config.pop("name")
        _fake_run(root, name, config, data, y)
        extra.append(new)
    spec["runs"].extend(extra)
    slightly_wrong = np.array([2.0, 6.0, 8.0, 1.0, 7.0, 3.0])
    scores = {}
    for item in spec["runs"]:
        name = item["name"]
        if item["mode"] == "classifier":
            scores[(name, 1)], scores[(name, 2)] = np.ones(6), y + 300
        elif item["lambda_class"] == 0:
            scores[(name, 1)] = y if item["seed"] == 17 else slightly_wrong
            scores[(name, 2)] = 100 - y
        else:
            scores[(name, 1)] = y + 10 if item["seed"] == 17 else 20 - y
            scores[(name, 2)] = y + 100
    _add_two_epoch_candidates(fabricated, scores)
    protocol = _freeze(fabricated)
    assert protocol["selected_by_rmse"] == {"lambda_class": 0.0, "epoch": 1}
    assert protocol["selected_by_utility"] == {"lambda_class": 1.0, "epoch": 2}
    assert protocol["regression_baseline"]["epoch"] == 1
    assert protocol["classifier_baseline"]["epoch"] == 2
    assert protocol["reference_epoch"] == 1
    assert len(protocol["validation_table"]) == 12


def test_missing_declared_candidate_cannot_freeze(fabricated):
    _add_two_epoch_candidates(fabricated)
    _, root, _, output = fabricated
    path = root / "joint" / "complete.json"
    complete = json.loads(path.read_text())
    complete["candidates"].pop()
    write_json(path, complete)
    with pytest.raises(ValueError, match="checkpoint grid"):
        _freeze(fabricated)
    assert not output.exists()


@pytest.mark.parametrize("artifact", ["epoch_002.pt", "epoch_002_validation.npz"])
def test_changed_candidate_refused_before_freeze(fabricated, artifact):
    _add_two_epoch_candidates(fabricated)
    _, root, _, _ = fabricated
    with (root / "joint" / "snapshots" / artifact).open("ab") as stream:
        stream.write(b"changed candidate")
    with pytest.raises(ValueError, match="Candidate artifact"):
        _freeze(fabricated)


def test_unselected_candidate_tampering_is_rejected_before_test(fabricated, monkeypatch, tmp_path):
    data, root, _, protocol_file = _add_two_epoch_candidates(fabricated)
    protocol = _freeze(fabricated)
    # All candidate scores tie across epochs; the earlier epoch is selected.
    assert protocol["selected_by_utility"]["epoch"] == 1
    with (root / "joint" / "snapshots" / "epoch_002.pt").open("ab") as stream:
        stream.write(b"unselected snapshot changed")
    monkeypatch.setattr(
        data, "partition", lambda *a, **k: pytest.fail("Invalid freeze opened test data")
    )
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        evaluation.evaluate_frozen(
            data, root, protocol_file, tmp_path / "test", n_bootstrap=8, device="cpu"
        )


def test_selection_averages_paired_seeds_instead_of_choosing_best_single_run(fabricated):
    data, root, spec, _ = fabricated
    y = data.metadata.iloc[data.indices["val"]].CD16.to_numpy()
    extra = []
    for item in spec["runs"]:
        new = {**item, "name": item["name"] + "_seed18", "seed": 18}
        config = {**spec["common"], **new}
        name = config.pop("name")
        # Regression's first seed wins RMSE but its mean over seeds loses.
        offset = 300 if new["mode"] == "joint" and new["lambda_class"] == 0 else 100
        _fake_run(root, name, config, data, y + offset)
        extra.append(new)
    spec["runs"].extend(extra)
    protocol = _freeze(fabricated)
    assert protocol["seeds"] == [17, 18]
    assert protocol["lambda_by_rmse"] == protocol["lambda_by_utility"] == 1.0


def test_partial_declared_grid_cannot_freeze(fabricated):
    _, root, _, output = fabricated
    (root / "joint" / "complete.json").unlink()
    with pytest.raises(ValueError, match="grid"):
        _freeze(fabricated)
    assert not output.exists()


def test_paired_seed_mismatch_cannot_freeze_even_without_expected_spec(fabricated):
    _, root, _, _ = fabricated
    path = root / "joint" / "config.json"
    config = json.loads(path.read_text())
    config["seed"] = 18
    write_json(path, config)
    complete = json.loads((root / "joint" / "complete.json").read_text())
    complete["config_sha256"] = file_sha256(path)
    write_json(root / "joint" / "complete.json", complete)
    with pytest.raises(ValueError):
        _freeze(fabricated, expected_spec=None)


def test_different_training_cell_subsets_cannot_be_compared(fabricated):
    _, root, _, _ = fabricated
    path = root / "joint" / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["train_ids_sha256"] = "0" * 64
    write_json(path, provenance)
    with pytest.raises(ValueError):
        _freeze(fabricated)


def test_stale_validation_predictions_are_rejected_before_freeze(fabricated):
    data, root, _, _ = fabricated
    path = root / "joint" / "validation_predictions.npz"
    np.savez_compressed(
        path,
        global_index=data.metadata.iloc[data.indices["val"]].global_index.to_numpy(),
        score=np.zeros(6),
        logits=np.zeros((6, 3)),
    )
    with pytest.raises(ValueError):
        _freeze(fabricated)


@pytest.mark.parametrize("fractions", [(0.0, 1 / 3), (1 / 3, 1.2), (0.1, 0.2)])
def test_fraction_protocol_is_valid_before_test_opening(fabricated, fractions):
    with pytest.raises(ValueError):
        _freeze(fabricated, fractions=fractions)


@pytest.mark.parametrize(
    "filename", ["final.pt", "config.json", "normalization.json", "validation_predictions.npz"]
)
def test_frozen_artifact_tampering_fails_before_test_access(
    fabricated, monkeypatch, tmp_path, filename
):
    data, root, _, protocol_file = fabricated
    _freeze(fabricated)
    with (root / "joint" / filename).open("ab") as stream:
        stream.write(b"modified after protocol freeze")
    monkeypatch.setattr(
        data, "partition", lambda *a, **k: pytest.fail("Tampered run opened test data")
    )
    output = tmp_path / "test_results"
    with pytest.raises(ValueError, match="changed"):
        evaluation.evaluate_frozen(data, root, protocol_file, output, n_bootstrap=8, device="cpu")
    assert not (output / "test_started.json").exists()


def test_quota_pair_and_composition_are_fixed_using_validation(fabricated):
    data, root, spec, _ = fabricated
    data.metadata.loc[data.indices["val"], "CD16"] = [10, 9, 5, 4, 1, 0]
    data.metadata.loc[data.indices["val"], "class_id"] = [0, 0, 1, 1, 2, 2]
    scores = {"regression": [5, 0, 0, 0, 6, 0], "joint": [0, 5, 0, 6, 0, 0]}
    for name, values in scores.items():
        path = root / name / "validation_predictions.npz"
        np.savez_compressed(
            path,
            global_index=data.metadata.iloc[data.indices["val"]].global_index.to_numpy(),
            score=np.asarray(values),
            logits=np.zeros((6, 3)),
        )
        complete_path = root / name / "complete.json"
        complete = json.loads(complete_path.read_text())
        complete["validation_predictions_sha256"] = file_sha256(path)
        write_json(complete_path, complete)
    quota = _freeze(fabricated)["quota_diagnostic"]
    assert quota["lambda_a"] == 1.0 and quota["lambda_b"] == 0.0
    np.testing.assert_allclose(quota["proportions"], [0.5, 0.25, 0.25])


def test_aggregate_bootstrap_draws_common_fields_across_fixed_seeds():
    y = np.array([0.0, 10.0, 3.0, 8.0, 1.0, 7.0])
    a = np.array([[2, 5, 3, 0, 1, 4], [3, 0, 5, 2, 4, 1]], dtype=float)
    b = -a
    groups = np.array([0, 1, 1, 2, 2, 2])
    repetitions, seed, fraction = 50, 31, 0.4
    result = paired_seed_field_bootstrap(
        y, a, b, groups, fraction, n_bootstrap=repetitions, seed=seed
    )
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    random = np.random.default_rng(seed)

    def differences(indices):
        k = int(np.ceil(fraction * len(indices)))
        return np.array(
            [
                y[indices[np.argsort(-sa[indices], kind="stable")[:k]]].mean()
                - y[indices[np.argsort(-sb[indices], kind="stable")[:k]]].mean()
                for sa, sb in zip(a, b)
            ]
        )

    draws = []
    for _ in range(repetitions):
        indices = np.concatenate([members[g] for g in random.integers(3, size=3)])
        draws.append(differences(indices).mean())
    observed = differences(np.arange(len(y)))
    assert result["difference_a_minus_b"] == pytest.approx(observed.mean())
    assert result["paired_seed_sd"] == pytest.approx(observed.std(ddof=1))
    np.testing.assert_allclose(
        [result["ci_low"], result["ci_high"]], np.quantile(draws, [0.025, 0.975])
    )


def test_real_full_model_training_freeze_and_test_complete_on_cpu(tmp_path):
    directory = tmp_path / "cohort"
    _write_fixture(directory)
    data = PreparedData(directory)
    root = tmp_path / "runs"
    spec = _spec()
    spec["common"].update(epochs=2, checkpoint_epochs=[1, 2], reference_epoch=1)
    for item in spec["runs"]:
        config = {**spec["common"], **item}
        name = config.pop("name")
        run_training(data, config, root / name)
    protocol_file = tmp_path / "protocol.json"
    protocol = evaluation.freeze_protocol(
        data,
        root,
        protocol_file,
        primary_fraction=1 / 3,
        fractions=(1 / 3, 0.5),
        expected_spec=spec,
    )
    output = tmp_path / "test_results"
    result = evaluation.evaluate_frozen(
        data, root, protocol_file, output, n_bootstrap=12, device="cpu"
    )
    assert result["n_test"] == 6 and result["n_test_fields"] == 2
    assert len(result["metrics"]) == 6
    assert protocol["checkpoint_epochs"] == [1, 2] and len(protocol["runs"]) == 6
    assert {row["epoch"] for row in result["metrics"]} == {1}
    assert len(result["paired_comparisons"]) == len(result["aggregate_comparisons"]) == 3
    assert {row["seed"] for row in result["paired_comparisons"]} == set(protocol["seeds"])
    assert all(row["bootstrap_seed"] == 20261021 for row in result["paired_comparisons"])
    assert all(
        row["bootstrap_seed"] == 20261021 and "seed" not in row
        for row in result["aggregate_comparisons"]
    )
    assert result["protocol_sha256"] == file_sha256(protocol_file)
    assert (output / "results.json").is_file()
    policy_choices = {
        "utility_selector": ("joint", protocol["selected_by_utility"]),
        "rmse_selector": ("joint", protocol["selected_by_rmse"]),
        "regression_baseline": ("joint", protocol["regression_baseline"]),
        "class_means_baseline": ("classifier", protocol["classifier_baseline"]),
    }
    selected_keys = {
        (mode, choice["lambda_class"], choice["epoch"]) for mode, choice in policy_choices.values()
    }
    for row in result["selected_metrics"]:
        mode, choice = policy_choices[row["policy"]]
        assert (row["mode"], row["lambda_class"], row["epoch"]) == (
            mode,
            choice["lambda_class"],
            choice["epoch"],
        )
    for run in protocol["runs"]:
        key = (run["config"]["mode"], run["config"]["lambda_class"], run["epoch"])
        expected = run["epoch"] == 1 or key in selected_keys
        assert (
            output / f"{run['name']}_epoch_{run['epoch']:03d}_test_predictions.npz"
        ).is_file() == expected
    protocol["primary_fraction"] = 0.5
    write_json(protocol_file, protocol)
    with pytest.raises(ValueError, match="different protocol"):
        evaluation.evaluate_frozen(data, root, protocol_file, output, n_bootstrap=12, device="cpu")
