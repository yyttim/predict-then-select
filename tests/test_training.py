"""CPU integration checks with synthetic images; no research outcomes are used."""

import copy
import csv
import json

import numpy as np
import pandas as pd
import pytest
import torch

from cellselect import training
from cellselect.data import PreparedData, normalization
from cellselect.model import build_model


@pytest.fixture(autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(4)
    images = rng.normal(size=(21, 4, 28, 28)).astype(np.float32)
    images[3:12] += 100  # Quarter normalization must not use other training fields.
    images[18:] = np.nan  # Any test-image access would fail.
    frame = pd.DataFrame(
        {
            "global_index": np.arange(100, 121),
            "target": np.tile([0.0, 10.0, 20.0], 7),
            "fov_id": [f"f{i // 3}" for i in range(21)],
            "slide_id": "synthetic",
            "split": ["train"] * 12 + ["val"] * 6 + ["test"] * 3,
            "train_quarter": [True] * 3 + [False] * 18,
        }
    )
    frame.loc[18:, "target"] = np.nan
    manifest = {"purpose": "synthetic test fixture", "threshold": {"value": 10.0, "operator": ">="}}
    path = tmp_path / "manifest.json"
    training.write_json(path, manifest)
    return PreparedData(
        images, frame, manifest, root=tmp_path, manifest_sha256=training.file_sha256(path)
    )


def config(data, **overrides):
    return {
        "method": "joint",
        "lambda_binary": 0.3,
        "seed": 17,
        "fraction": 0.25,
        "max_steps": 3,
        "batch_size": 4,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1,
        "min_lr_ratio": 0.05,
        "checkpoint_steps": [1, 3],
        "device": "cpu",
        "cpu_threads": 2,
        "source_sha256": {"fixture": "a" * 64},
        "data_manifest_sha256": data.manifest_sha256,
        **overrides,
    }


def test_cycled_batches_keep_every_example_and_pair_exactly():
    a, b = training.ShuffledBatches(5, 3, 71), training.ShuffledBatches(5, 3, 71)
    pieces = [next(a) for _ in range(5)]
    assert all(p.shape == (3,) for p in pieces)
    stream = torch.cat(pieces)
    for start in range(0, 15, 5):
        assert sorted(stream[start : start + 5].tolist()) == list(range(5))
    assert torch.equal(stream, torch.cat([next(b) for _ in range(5)]))


def test_update_schedule_has_exact_warmup_and_terminal_rates(data):
    cfg = config(data, max_steps=6, warmup_steps=2, checkpoint_steps=[3, 6])
    assert training.learning_rate(1, cfg) == cfg["lr"] / 2
    assert training.learning_rate(2, cfg) == training.learning_rate(3, cfg) == cfg["lr"]
    assert training.learning_rate(6, cfg) == cfg["lr"] * cfg["min_lr_ratio"]
    assert training.learning_rate(6, {**cfg, "fraction": 1.0}) == training.learning_rate(6, cfg)
    with pytest.raises(ValueError, match="outside"):
        training.learning_rate(7, cfg)


@pytest.mark.parametrize("method", ["mse", "bce", "joint"])
def test_training_saves_declared_states_and_only_active_scores_without_test_access(
    data, tmp_path, monkeypatch, method
):
    cfg = config(data, method=method)
    expected_norm = normalization(data, data.rows("train", fraction=0.25))
    called = []
    original_loss = training.objective_loss

    def checked_loss(outputs, target_z, high, selected, weight):
        raw = target_z * expected_norm["target_std"] + expected_norm["target_mean"]
        torch.testing.assert_close(high, (raw >= 10.0).float())
        called.append(len(high))
        return original_loss(outputs, target_z, high, selected, weight)

    monkeypatch.setattr(training, "objective_loss", checked_loss)
    output = tmp_path / method
    done = training.train_run(data, cfg, output)
    assert called == [4, 4, 4] and done["steps"] == 3 and done["examples_seen"] == 12
    assert [c["step"] for c in done["checkpoints"]] == [1, 3]
    assert json.loads((output / "config.json").read_text()) == cfg
    with (output / "history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert [int(r["step"]) for r in history] == [1, 2, 3]
    assert float(history[-1]["learning_rate"]) == cfg["lr"] * cfg["min_lr_ratio"]
    expected_fields = {"global_index"} | (
        {"mean"} if method == "mse" else {"hit_logit"} if method == "bce" else {"mean", "hit_logit"}
    )
    for candidate in done["checkpoints"]:
        saved = torch.load(
            output / candidate["checkpoint_file"], map_location="cpu", weights_only=True
        )
        assert saved["normalization"] == expected_norm and saved["config"] == cfg
        assert saved["threshold"] == 10 and saved["manifest_sha256"] == data.manifest_sha256
        assert "optimizer" not in saved
        model = build_model()
        model.load_state_dict(saved["model"])
        recomputed = training.predict(
            model, data.images[data.rows("val")], expected_norm, "cpu", batch_size=4
        )
        with np.load(output / candidate["predictions_file"], allow_pickle=False) as pred:
            assert set(pred.files) == expected_fields
            np.testing.assert_array_equal(
                pred["global_index"], data.metadata.iloc[data.rows("val")].global_index
            )
            for name in expected_fields - {"global_index"}:
                assert pred[name].dtype == np.float64
                np.testing.assert_array_equal(pred[name], recomputed[name])
    if method != "joint":
        inactive = "hit_head" if method == "mse" else "mean_head"
        initial = build_model(seed=cfg["seed"]).state_dict()
        for key, value in saved["model"].items():
            if key.startswith(inactive + "."):
                assert torch.equal(value, initial[key])
    monkeypatch.setattr(
        training, "build_model", lambda **_: pytest.fail("Completed run trained again")
    )
    assert training.train_run(data, cfg, output) == done
    changed = copy.deepcopy(cfg)
    changed["source_sha256"]["fixture"] = "b" * 64
    with pytest.raises(ValueError, match="different configuration"):
        training.train_run(data, changed, output)
    with (output / done["checkpoints"][0]["checkpoint_file"]).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        training.train_run(data, cfg, output)


def test_initial_forward_randomness_is_paired_across_methods(data, tmp_path, monkeypatch):
    original_loss, first_outputs = training.objective_loss, []

    def record_loss(outputs, *args):
        first_outputs.append({key: value.detach().clone() for key, value in outputs.items()})
        return original_loss(outputs, *args)

    monkeypatch.setattr(training, "objective_loss", record_loss)
    for method in ["mse", "bce", "joint"]:
        training.train_run(
            data,
            config(data, method=method, max_steps=1, warmup_steps=0, checkpoint_steps=[1]),
            tmp_path / method,
        )
    for output in first_outputs[1:]:
        for key in output:
            assert torch.equal(output[key], first_outputs[0][key])


def test_invalid_schedule_fails_before_creating_outputs(data, tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="checkpoint_steps"):
        training.train_run(data, config(data, checkpoint_steps=[1, 2]), output)
    assert not output.exists()
