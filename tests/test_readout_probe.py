"""Matched-readout tests using synthetic features and images."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cellselect.model import build_model
from cellselect.training import PreparedData, file_sha256, write_json
from test_training import _write_fixture

path = Path(__file__).resolve().parents[1] / "scripts" / "readout_probe.py"
spec = importlib.util.spec_from_file_location("readout_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def two_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_config():
    return {
        **probe.DEFAULT_CONFIG,
        "epochs": 2,
        "checkpoint_epochs": [1, 2],
        "warmup_epochs": 0,
        "batch_size": 6,
    }


def test_encoder_extraction_freezes_parameters_bn_and_dropout_but_features_train_heads():
    model = build_model(seed=3).train()
    before = probe.state_digest(model.encoder)
    x = torch.randn(6, 4, 28, 28)
    norm = {"image_mean": [0] * 4, "image_std": [1] * 4}
    first, recorded = probe.extract_features(model, x, norm, torch.device("cpu"), 3)
    second, _ = probe.extract_features(model, x, norm, torch.device("cpu"), 3)
    assert before == recorded == probe.state_digest(model.encoder)
    assert not model.encoder.training
    assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
    np.testing.assert_array_equal(first, second)
    head = probe.fresh_head(10003)
    loss = head(torch.from_numpy(first)).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert probe.state_digest(model.encoder) == before


def test_fresh_heads_are_paired_and_not_original_trained_head():
    first, second = probe.fresh_head(10016), probe.fresh_head(10016)
    assert probe.state_digest(first) == probe.state_digest(second)
    assert sum(p.numel() for p in first.parameters()) == 174209
    assert probe.state_digest(first) != probe.state_digest(probe.fresh_head(10026))
    assert probe.state_digest(first) != probe.state_digest(build_model(seed=16).regression_head)


def test_fit_head_saves_declared_candidates_and_rejects_tampering(tmp_path):
    rng = np.random.default_rng(7)
    train, val = rng.normal(size=(12, 256)), rng.normal(size=(6, 256))
    target, val_target = train[:, 0], val[:, 0]
    norm = {"target_mean": 0.0, "target_std": 1.0}
    config = {**small_config(), "readout_seed": 10016}
    binding = {"features_sha256": "synthetic"}
    args = (
        train,
        target,
        val,
        val_target,
        np.arange(6) % 3,
        np.arange(6),
        norm,
        config,
        binding,
        tmp_path / "head",
        torch.device("cpu"),
    )
    first = probe.fit_head(*args)
    assert [c["epoch"] for c in first["candidates"]] == [1, 2]
    assert first["trainable_parameters"] == 174209 and first["test_used"] is False
    assert probe.fit_head(*args) == first
    for candidate in first["candidates"]:
        with np.load(args[-2] / candidate["predictions_file"]) as predictions:
            assert predictions["score"].dtype == np.float64
            assert predictions["score"].shape == (6,)
    with (args[-2] / first["candidates"][0]["predictions_file"]).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="changed"):
        probe.fit_head(*args)


def _synthetic_core(tmp_path):
    data_root = tmp_path / "data"
    _write_fixture(data_root)
    data = PreparedData(data_root)
    runs, entries = tmp_path / "core_runs", []
    for family, (mode, weight) in probe.FAMILIES.items():
        name = family + "_seed17"
        directory = runs / name
        directory.mkdir(parents=True)
        config = {
            "seed": 17,
            "mode": mode,
            "lambda_class": weight,
            "epochs": 40,
            "synthetic_initial_weights_for_engineering_test": True,
        }
        model = build_model(seed=17)
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": 40,
                "config": config,
                "normalization": data.normalization(),
                "manifest_sha256": data.manifest_hash,
            },
            directory / "source.pt",
        )
        write_json(directory / "config.json", config)
        write_json(directory / "normalization.json", data.normalization())
        val = data.metadata.iloc[data.indices["val"]]
        np.savez_compressed(
            directory / "val.npz",
            global_index=val.global_index.to_numpy(),
            score=np.zeros(len(val)),
        )
        entries.append(
            {
                "name": name,
                "epoch": 40,
                "config": config,
                "checkpoint_file": "source.pt",
                "checkpoint_sha256": file_sha256(directory / "source.pt"),
                "config_sha256": file_sha256(directory / "config.json"),
                "normalization_sha256": file_sha256(directory / "normalization.json"),
                "predictions_file": "val.npz",
                "validation_predictions_sha256": file_sha256(directory / "val.npz"),
            }
        )
    protocol = tmp_path / "core_protocol.json"
    write_json(
        protocol,
        {
            "expected_experiment_verified": True,
            "test_used_to_choose_protocol": False,
            "manifest_sha256": data.manifest_hash,
            "seeds": [17],
            "runs": entries,
        },
    )
    return data, runs, protocol


def test_train_freeze_is_test_blind_and_evaluation_uses_locked_readouts(tmp_path, monkeypatch):
    data, runs, core = _synthetic_core(tmp_path)
    partition = data.partition
    calls = []

    def safe_partition(name):
        calls.append(name)
        assert name != "test"
        return partition(name)

    monkeypatch.setattr(data, "partition", safe_partition)
    output = tmp_path / "probe"
    protocol = probe.train_and_freeze(data, runs, core, output, torch.device("cpu"), small_config())
    assert calls == ["train", "val"]
    assert set(protocol["selected_head_epochs"]) == set(probe.FAMILIES)
    assert protocol["test_used_for_readout_selection"] is False
    assert len({r["initial_head_sha256"] for r in protocol["runs"]}) == 1
    assert len({r["binding"]["encoder_state_sha256"] for r in protocol["runs"]}) == 1
    monkeypatch.setattr(data, "partition", partition)
    result = probe.evaluate_probe(data, runs, output, torch.device("cpu"), n_bootstrap=8)
    assert result["n_test"] == 6 and len(result["metrics"]) == 18
    assert len(result["comparisons"]) == 5
    assert all(row["bootstrap_seed"] == 20261021 for row in result["comparisons"])
    assert all(row["encoder_epoch"] == 40 for row in result["metrics"])


def test_changed_readout_is_rejected_before_test_access(tmp_path, monkeypatch):
    data, runs, core = _synthetic_core(tmp_path)
    output = tmp_path / "probe"
    protocol = probe.train_and_freeze(data, runs, core, output, torch.device("cpu"), small_config())
    run = protocol["runs"][0]
    chosen = next(
        c
        for c in run["candidates"]
        if c["epoch"] == protocol["selected_head_epochs"][run["family"]]
    )
    with (output / run["name"] / chosen["checkpoint_file"]).open("ab") as stream:
        stream.write(b"tamper")
    monkeypatch.setattr(
        data, "partition", lambda *_: pytest.fail("Tampered freeze opened test data")
    )
    with pytest.raises(ValueError, match="artifact changed"):
        probe.evaluate_probe(data, runs, output, torch.device("cpu"), n_bootstrap=8)
