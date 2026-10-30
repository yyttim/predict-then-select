"""Training integration tests using the full model and synthetic inputs on CPU."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import cellselect.training as training
from cellselect.model import build_model


def _write_fixture(directory: Path, poison_test=False):
    directory.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(71)
    images = random.normal(size=(24, 4, 28, 28)).astype(np.float32)
    # Every training field has all three classes, so nested field subsets remain
    # feasible without any label-dependent retry or resampling.
    rows = []
    for index in range(24):
        field = index // 3
        split = "train" if field < 4 else ("val" if field < 6 else "test")
        rows.append(
            {
                "global_index": 1000 + index,
                "class_id": index % 3,
                "CD16": float(3 * (index % 3) + field + 1),
                "fov_id": f"slide_a|field_{field}",
                "split_group": f"slide_a|field_{field}",
                "split": split,
            }
        )
    metadata = pd.DataFrame(rows)
    if poison_test:
        test = metadata.split.to_numpy() == "test"
        images[test] = np.nan
        metadata.loc[test, "CD16"] = np.nan
    np.save(directory / "images.npy", images)
    metadata.to_csv(directory / "metadata.csv", index=False)
    manifest = {
        "purpose": "synthetic engineering test fixture; not biological data",
        "prepared_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("images.npy", "metadata.csv")
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return images, metadata


def _config(**overrides):
    config = {
        "seed": 17,
        "device": "cpu",
        "cpu_threads": 2,
        "amp": False,
        "cache_on_gpu": True,
        "mode": "joint",
        "lambda_class": 0.5,
        "feature_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 2,
        "batch_size": 6,
        "warmup_epochs": 0,
        "min_lr_ratio": 0.05,
        "primary_fraction": 0.25,
        "eval_every": 1,
    }
    config.update(overrides)
    return config


@pytest.fixture(autouse=True)
def restore_cpu_thread_setting():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cohort(tmp_path):
    directory = tmp_path / "cohort"
    images, metadata = _write_fixture(directory)
    return directory, images, metadata


@pytest.fixture(scope="module")
def completed_fixture_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_training_audit")
    directory = root / "cohort"
    _write_fixture(directory)
    data = training.PreparedData(directory)
    config = _config()
    output = root / "run"
    previous = torch.get_num_threads()
    try:
        result = training.run_training(data, config, output)
    finally:
        torch.set_num_threads(previous)
    return directory, config, output, result


def test_normalization_uses_only_selected_training_fields(cohort):
    directory, images, metadata = cohort
    small = training.PreparedData(directory, train_fraction=0.25, subset_seed=3)
    medium = training.PreparedData(directory, train_fraction=0.5, subset_seed=3)
    full = training.PreparedData(directory, train_fraction=1.0, subset_seed=3)
    assert set(small.indices["train"]) <= set(medium.indices["train"]) <= set(full.indices["train"])
    for data in (small, medium, full):
        train = data.indices["train"]
        values = images[train].astype(np.float64)
        targets = metadata.iloc[train].CD16.to_numpy()
        np.testing.assert_allclose(data.image_mean, values.mean(axis=(0, 2, 3)), atol=1e-12)
        np.testing.assert_allclose(data.image_std, values.std(axis=(0, 2, 3)), atol=1e-12)
        assert data.target_mean == pytest.approx(targets.mean())
        assert data.target_std == pytest.approx(targets.std())
        for class_id in range(3):
            expected = targets[metadata.iloc[train].class_id.to_numpy() == class_id].mean()
            assert data.class_means[class_id] == pytest.approx(expected)
        included_fields = set(metadata.iloc[train].fov_id)
        assert set(train) == set(np.flatnonzero(metadata.fov_id.isin(included_fields)))
        x, y, classes, _ = data.partition("train")
        assert x.shape == (len(train), 4, 28, 28)
        assert y.shape == (len(train),)
        assert y.dtype == torch.float32
        assert classes.dtype == torch.int64
        assert y.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert y.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "objective,extra",
    [
        ("spo_plus", {}),
        ("pg_central", {"pg_h": 0.1}),
    ],
)
def test_decision_objective_training_keeps_test_sealed(tmp_path, objective, extra):
    directory = tmp_path / "decision_cohort"
    _write_fixture(directory, poison_test=True)
    data = training.PreparedData(directory)
    config = _config(objective=objective, lambda_class=0, **extra)
    result = training.run_training(data, config, tmp_path / "run")
    assert result["test_evaluated"] is False
    assert np.isfinite(result["final_validation"]["utility"])
    checkpoint = torch.load(tmp_path / "run" / "final.pt", weights_only=False)
    before = build_model(seed=config["seed"]).state_dict()
    # The decision objective must train encoder + marker head, not random logits.
    after = checkpoint["model"]
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("encoder."))
    assert any(
        not torch.equal(before[k], after[k]) for k in before if k.startswith("regression_head.")
    )


def test_pg_warmup_state_cannot_be_reported_as_a_pg_candidate(cohort, tmp_path):
    data = training.PreparedData(cohort[0])
    config = _config(
        objective="pg_central",
        pg_h=0.1,
        lambda_class=0,
        epochs=3,
        decision_warmup_epochs=1,
        checkpoint_epochs=[1, 3],
    )
    with pytest.raises(ValueError, match="after the SPO"):
        training.run_training(data, config, tmp_path / "invalid")


def test_training_does_not_access_heldout_pixels_or_compute_with_test_targets(
    tmp_path, monkeypatch
):
    directory = tmp_path / "cohort"
    _write_fixture(directory, poison_test=True)
    data = training.PreparedData(directory)
    original_images = data.images
    test_indices = set(data.indices["test"])

    class SealedTestPixels:
        def __getitem__(self, indices):
            assert test_indices.isdisjoint(np.asarray(indices).reshape(-1).tolist())
            return original_images[indices]

    data.images = SealedTestPixels()
    calls = []
    original_partition = data.partition

    def partition(split):
        calls.append(split)
        assert split in {"train", "val"}
        return original_partition(split)

    monkeypatch.setattr(data, "partition", partition)
    # GPU caching is requested in config but must be inert on a CPU device.
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda *a, **k: pytest.fail("CPU run queried GPU cache capacity"),
    )
    result = training.run_training(data, _config(epochs=1), tmp_path / "run")
    assert calls == ["train", "val"]
    assert result["test_evaluated"] is False
    assert np.isfinite(result["final_validation"]["rmse"])


def test_final_epoch_checkpoint_and_provenance_are_consistent(completed_fixture_run):
    directory, config, output, result = completed_fixture_run
    saved = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    provenance = json.loads((output / "provenance.json").read_text())
    assert saved["epoch"] == config["epochs"] == result["epochs"]
    assert [line["epoch"] for line in history] == [1, 2]
    assert result["final_validation"] == history[-1]["validation"]
    assert result["checkpoint_sha256"] == training.file_sha256(output / "final.pt")
    assert saved["config"] == config
    assert saved["normalization"] == json.loads((output / "normalization.json").read_text())
    assert provenance["manifest_sha256"] == training.file_sha256(directory / "manifest.json")
    assert provenance["n_train"] == 12 and provenance["n_val"] == 6
    assert provenance["test_evaluated"] is False
    assert "final epoch" in provenance["checkpoint_policy"]


def test_paired_cpu_runs_reproduce_parameters_and_validation_predictions(
    completed_fixture_run, tmp_path
):
    directory, config, output, _ = completed_fixture_run
    second = tmp_path / "repeat"
    training.run_training(training.PreparedData(directory), config, second)
    first_checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    second_checkpoint = torch.load(second / "final.pt", map_location="cpu", weights_only=True)
    for key, value in first_checkpoint["model"].items():
        torch.testing.assert_close(value, second_checkpoint["model"][key], atol=0, rtol=0)
    with (
        np.load(output / "validation_predictions.npz") as a,
        np.load(second / "validation_predictions.npz") as b,
    ):
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_classifier_baseline_uses_training_class_marker_means(cohort, tmp_path):
    directory, _, _ = cohort
    data = training.PreparedData(directory)
    config = _config(mode="classifier", lambda_class=0.0, epochs=1)
    output = tmp_path / "classifier"
    result = training.run_training(data, config, output)
    initial = build_model(seed=config["seed"]).state_dict()
    saved = torch.load(output / "final.pt", map_location="cpu", weights_only=True)["model"]
    for key, value in initial.items():
        if key.startswith("regression_head."):
            torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
    with np.load(output / "validation_predictions.npz") as predictions:
        probabilities = torch.softmax(torch.from_numpy(predictions["logits"]), dim=1).numpy()
        np.testing.assert_allclose(
            predictions["score"], probabilities @ data.class_means, atol=1e-6
        )
    assert "macro_f1" in result["final_validation"]


def test_regression_baseline_does_not_report_untrained_classification_metrics(cohort, tmp_path):
    directory, _, _ = cohort
    result = training.run_training(
        training.PreparedData(directory), _config(lambda_class=0, epochs=1), tmp_path / "regression"
    )
    assert "macro_f1" not in result["final_validation"]
    assert "accuracy" not in result["final_validation"]


def test_amp_training_still_uses_full_precision_validation(cohort, tmp_path, monkeypatch):
    directory, _, _ = cohort
    original = training.predict
    flags = []
    phase = {"predicting": False}
    original_autocast = torch.autocast

    def checked_autocast(*args, **kwargs):
        if phase["predicting"]:
            flags.append(kwargs.get("enabled", True))
        return original_autocast(*args, **kwargs)

    def checked_predict(model, images, mean, std, device, batch_size=512, amp=False):
        phase["predicting"] = True
        try:
            return original(model, images, mean, std, device, batch_size, amp)
        finally:
            phase["predicting"] = False

    monkeypatch.setattr(torch, "autocast", checked_autocast)
    monkeypatch.setattr(training, "predict", checked_predict)
    training.run_training(
        training.PreparedData(directory), _config(amp=True, epochs=1), tmp_path / "amp"
    )
    assert flags and not any(flags), (
        "Validation selection must not inherit BF16 output quantization"
    )


def test_regression_broadcast_shape_is_rejected(cohort, tmp_path, monkeypatch):
    directory, _, _ = cohort
    original = training.build_model

    def malformed_model(*args, **kwargs):
        model = original(*args, **kwargs)
        forward = model.forward

        def wrong_shape(x):
            output = forward(x)
            output["marker"] = output["marker"][:, None]
            return output

        model.forward = wrong_shape
        return model

    monkeypatch.setattr(training, "build_model", malformed_model)
    with pytest.raises(ValueError, match="[Ss]hape"):
        training.run_training(
            training.PreparedData(directory), _config(epochs=1), tmp_path / "bad_shape"
        )


def test_completed_run_reuse_rejects_different_training_subset(completed_fixture_run):
    directory, config, output, _ = completed_fixture_run
    subset = training.PreparedData(directory, train_fraction=0.5)
    with pytest.raises((ValueError, FileExistsError)):
        training.run_training(subset, config, output)


def test_completed_run_can_be_reused_when_config_data_and_artifacts_match(
    completed_fixture_run, monkeypatch
):
    directory, config, output, expected = completed_fixture_run
    data = training.PreparedData(directory)
    monkeypatch.setattr(
        data, "partition", lambda *a, **k: pytest.fail("Valid cached run retrained")
    )
    assert training.run_training(data, config, output) == expected


def test_completed_run_reuse_rejects_different_manifest(completed_fixture_run, tmp_path):
    directory, config, output, _ = completed_fixture_run
    other = tmp_path / "different_cohort"
    shutil.copytree(directory, other)
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["fixture_revision"] = 2
    (other / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        training.run_training(training.PreparedData(other), config, output)


def test_completed_run_reuse_rejects_modified_checkpoint(completed_fixture_run, tmp_path):
    directory, config, output, _ = completed_fixture_run
    modified = tmp_path / "modified_run"
    shutil.copytree(output, modified)
    with (modified / "final.pt").open("ab") as stream:
        stream.write(b"checkpoint unexpectedly changed")
    with pytest.raises((ValueError, FileExistsError)):
        training.run_training(training.PreparedData(directory), config, modified)


def test_completed_run_reuse_rejects_changed_training_implementation(
    completed_fixture_run, monkeypatch
):
    directory, config, output, _ = completed_fixture_run
    original_hash = training.file_sha256

    def changed_hash(path):
        if Path(path).name == "training.py":
            return "0" * 64
        return original_hash(path)

    monkeypatch.setattr(training, "file_sha256", changed_hash)
    with pytest.raises((ValueError, FileExistsError)):
        training.run_training(training.PreparedData(directory), config, output)


def test_partial_run_before_first_history_row_is_not_silently_overwritten(cohort, tmp_path):
    directory, _, _ = cohort
    output = tmp_path / "partial"
    output.mkdir()
    sentinel = {"seed": 999, "status": "interrupted before first epoch"}
    (output / "config.json").write_text(json.dumps(sentinel), encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        training.run_training(training.PreparedData(directory), _config(epochs=1), output)
    assert json.loads((output / "config.json").read_text()) == sentinel


@pytest.mark.parametrize(
    "change",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"eval_every": 0},
        {"lambda_class": float("nan")},
        {"loss_scale": 0},
    ],
)
def test_invalid_run_configuration_fails_before_writing_artifacts(cohort, tmp_path, change):
    directory, _, _ = cohort
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        training.run_training(training.PreparedData(directory), _config(**change), output)
    assert not output.exists() or not list(output.iterdir())


def test_predict_is_batch_partition_invariant_in_eval_mode(cohort):
    directory, _, _ = cohort
    data = training.PreparedData(directory)
    images, _, _, _ = data.partition("val")
    mean = torch.tensor(data.image_mean, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(data.image_std, dtype=torch.float32)[None, :, None, None]
    model = build_model(seed=2)
    first = training.predict(model, images, mean, std, torch.device("cpu"), batch_size=1)
    second = training.predict(model, images, mean, std, torch.device("cpu"), batch_size=4)
    assert not model.training
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b, rtol=5e-5, atol=2e-6)
        assert a.dtype == np.float32


@pytest.mark.parametrize("mode,weight", [("joint", 0.5), ("classifier", 0.0)])
def test_declared_candidates_save_matching_models_predictions_and_metrics(
    cohort, tmp_path, mode, weight
):
    directory, _, _ = cohort
    data = training.PreparedData(directory)
    config = _config(
        epochs=4,
        eval_every=4,
        checkpoint_epochs=[2, 4],
        reference_epoch=2,
        mode=mode,
        lambda_class=weight,
        gradient_every=0,
    )
    output = tmp_path / "candidates"
    complete = training.run_training(data, config, output)
    candidates = complete["candidates"]
    assert [item["epoch"] for item in candidates] == [2, 4]
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert [item["epoch"] for item in history if "validation" in item] == [1, 2, 4]
    x, _, _, metadata = data.partition("val")
    mean = torch.tensor(data.image_mean, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(data.image_std, dtype=torch.float32)[None, :, None, None]
    saved_states = []
    for item in candidates:
        epoch = item["epoch"]
        assert item["checkpoint_file"] == f"snapshots/epoch_{epoch:03d}.pt"
        assert item["predictions_file"] == f"snapshots/epoch_{epoch:03d}_validation.npz"
        checkpoint_path, predictions_path = (
            output / item["checkpoint_file"],
            output / item["predictions_file"],
        )
        assert item["checkpoint_sha256"] == training.file_sha256(checkpoint_path)
        assert item["predictions_sha256"] == training.file_sha256(predictions_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == epoch and checkpoint["config"] == config
        assert checkpoint["normalization"] == data.normalization()
        assert checkpoint["manifest_sha256"] == data.manifest_hash
        assert "optimizer" not in checkpoint
        saved_states.append(checkpoint["model"])
        model = build_model(seed=99)
        model.load_state_dict(checkpoint["model"])
        logits, marker = training.predict(
            model, x, mean, std, torch.device("cpu"), config["batch_size"]
        )
        scores = (
            torch.softmax(torch.from_numpy(logits), dim=1).numpy() @ data.class_means
            if mode == "classifier"
            else marker.astype(np.float64) * data.target_std + data.target_mean
        )
        with np.load(predictions_path) as predictions:
            assert predictions["score"].dtype == np.float64
            np.testing.assert_array_equal(
                predictions["global_index"], metadata.global_index.to_numpy()
            )
            np.testing.assert_array_equal(predictions["score"], scores)
            np.testing.assert_array_equal(predictions["logits"], logits)
        assert item["validation"] == next(
            row["validation"] for row in history if row["epoch"] == epoch
        )
    assert not torch.equal(
        saved_states[0]["encoder.cnn.stem.0.weight"], saved_states[1]["encoder.cnn.stem.0.weight"]
    )
    final = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    for key, value in final["model"].items():
        torch.testing.assert_close(value, saved_states[-1][key], atol=0, rtol=0)
    assert complete["final_validation"] == candidates[-1]["validation"]


def test_extra_candidate_evaluation_does_not_change_training_trajectory(cohort, tmp_path):
    directory, _, _ = cohort
    data = training.PreparedData(directory)
    states = []
    for grid in ([4], [2, 4]):
        output = tmp_path / f"grid_{len(grid)}"
        config = _config(epochs=4, eval_every=4, checkpoint_epochs=grid, gradient_every=0)
        training.run_training(data, config, output)
        states.append(
            torch.load(output / "final.pt", map_location="cpu", weights_only=True)["model"]
        )
    for key, value in states[0].items():
        torch.testing.assert_close(value, states[1][key], atol=0, rtol=0)


@pytest.mark.parametrize("artifact", ["checkpoint_file", "predictions_file"])
def test_completed_run_reuse_rejects_changed_candidate_artifact(
    completed_fixture_run, tmp_path, artifact
):
    directory, config, source, complete = completed_fixture_run
    output = tmp_path / "tampered_candidate"
    shutil.copytree(source, output)
    # Defaults preserve a single final-epoch candidate for older configurations.
    assert [item["epoch"] for item in complete["candidates"]] == [config["epochs"]]
    path = output / complete["candidates"][0][artifact]
    with path.open("ab") as stream:
        stream.write(b"changed candidate artifact")
    with pytest.raises(ValueError, match="candidate artifact"):
        training.run_training(training.PreparedData(directory), config, output)


def test_completed_run_reuse_rejects_candidate_grid_tampering(completed_fixture_run, tmp_path):
    directory, config, source, _ = completed_fixture_run
    output = tmp_path / "tampered_grid"
    shutil.copytree(source, output)
    complete = json.loads((output / "complete.json").read_text())
    complete["candidates"] = []
    (output / "complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="candidate grid"):
        training.run_training(training.PreparedData(directory), config, output)


@pytest.mark.parametrize(
    "changes",
    [
        {"checkpoint_epochs": []},
        {"checkpoint_epochs": [2, 1]},
        {"checkpoint_epochs": [1, 1]},
        {"checkpoint_epochs": [0, 2]},
        {"checkpoint_epochs": [1, 3]},
        {"checkpoint_epochs": [True, 2]},
        {"checkpoint_epochs": [1.0, 2]},
        {"checkpoint_epochs": None},
        {"reference_epoch": 1},
        {"reference_epoch": True},
    ],
)
def test_invalid_candidate_grid_fails_before_training_artifacts(cohort, tmp_path, changes):
    directory, _, _ = cohort
    output = tmp_path / "invalid_candidates"
    with pytest.raises(ValueError):
        training.run_training(training.PreparedData(directory), _config(**changes), output)
    assert not output.exists()
