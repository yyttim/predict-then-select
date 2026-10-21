"""CPU integration tests using synthetic training data and two slide pools."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import softmax

import cellselect.study as study
from cellselect.external_data import load_external
from cellselect.training import PreparedData, run_training
from test_training import _config, _write_fixture


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_external_fixture(root):
    root.mkdir(parents=True)
    rng = np.random.default_rng(732)
    # Shifted images distinguish external from source-training normalization.
    images = rng.normal(loc=8.0, scale=3.0, size=(12, 4, 28, 28)).astype(np.float32)
    rows = []
    for i in range(12):
        slide = 0 if i < 5 else 1
        pool = f"synthetic_batch1_slide_{slide}"
        rows.append(
            {
                "global_index": 9000 + i,
                "batch": 1,
                "antibodies": "all",
                "slide_id": pool,
                "pool_id": pool,
                "fov_id": f"{pool}|field_{i // 2}",
                "split": "external",
                "class_id": -1,
                "class_label_available": False,
                "CD16": float(30 + i * 2),
                "CD45": float(100 + i * 3),
                "CD16_available": True,
                "CD45_available": True,
            }
        )
    metadata = pd.DataFrame(rows)
    np.save(root / "images.npy", images)
    metadata.to_csv(root / "metadata.csv", index=False)
    manifest = {
        "role": "external_acquisition_evaluation",
        "purpose": "synthetic test fixture; not biological data",
        "prepared_sha256": {name: _sha(root / name) for name in ("images.npy", "metadata.csv")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return metadata


@pytest.fixture(scope="module")
def frozen_external_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_external_study")
    main = root / "main"
    _write_fixture(main)
    metadata = pd.read_csv(main / "metadata.csv")
    metadata["CD45"] = metadata["CD16"] * 2 + np.arange(len(metadata)) / 10 + 3
    metadata.to_csv(main / "metadata.csv", index=False)
    manifest = json.loads((main / "manifest.json").read_text())
    manifest["prepared_sha256"]["metadata.csv"] = _sha(main / "metadata.csv")
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    external = root / "external"
    _write_external_fixture(external)
    common = _config(
        epochs=1,
        checkpoint_epochs=[1],
        reference_epoch=1,
        train_fraction=1.0,
        lambda_class=0.0,
        gradient_every=0,
        device="cpu",
        cpu_threads=2,
        primary_fraction=0.25,
    )
    methods = [
        dict(name="mse", mode="joint", objective="mse"),
        dict(name="joint", mode="joint", objective="mse", lambda_class=1.0),
        dict(name="ce", mode="classifier", objective="mse"),
        dict(name="spo", mode="joint", objective="spo_plus"),
        dict(name="pg", mode="joint", objective="pg_central", pg_h=0.1),
    ]
    overrides = []
    previous_threads = torch.get_num_threads()
    try:
        for marker in ("CD16", "CD45"):
            data = PreparedData(main, marker=marker)
            for method in methods:
                override = {**method, "marker": marker, "name": f"{marker}_{method['name']}"}
                config = {**common, **override}
                name = config.pop("name")
                run_training(data, config, root / "runs" / name)
                overrides.append(override)
        specification = root / "specification.json"
        specification.write_text(
            json.dumps(
                {
                    "common": common,
                    "runs": overrides,
                    "primary_fraction": 0.25,
                    "fractions": [0.25, 0.5],
                }
            )
        )
        protocol = root / "protocol.json"
        frozen = study.freeze_study(main, root / "runs", specification, protocol, external)
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "root": root,
        "main": main,
        "external": external,
        "runs": root / "runs",
        "protocol": protocol,
        "frozen": frozen,
    }


def test_two_external_pools_end_to_end_use_frozen_source_normalization(
    frozen_external_study, tmp_path, monkeypatch
):
    fixture = frozen_external_study
    main, external, runs = fixture["main"], fixture["external"], fixture["runs"]
    external_data = load_external(external)
    try:
        external_metadata = external_data.metadata.copy()
    finally:
        # Close the Windows memory map even if a later assertion retains this frame.
        external_data.images._mmap.close()
        del external_data
    assert (external_metadata["class_id"] == -1).all()
    assert np.isfinite(external_metadata[["CD16", "CD45"]]).all().all()
    protected_paths = [main / name for name in ("manifest.json", "metadata.csv", "images.npy")]
    protected_paths += [path / "normalization.json" for path in runs.iterdir()]
    original_hashes = {path: _sha(path) for path in protected_paths}
    source_data = PreparedData(main)
    original_predict = study.predict
    calls = []

    def checked_predict(model, images, mean, std, device, *args, **kwargs):
        assert str(device) == "cpu"
        assert next(model.parameters()).device.type == "cpu"
        np.testing.assert_allclose(
            mean.detach().cpu().numpy().reshape(-1), source_data.image_mean, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            std.detach().cpu().numpy().reshape(-1), source_data.image_std, rtol=1e-6, atol=1e-6
        )
        logits, marker = original_predict(model, images, mean, std, device, *args, **kwargs)
        calls.append({"n": len(images), "logits": logits.copy(), "marker": marker.copy()})
        return logits, marker

    monkeypatch.setattr(study, "predict", checked_predict)
    output = tmp_path / "evaluation"
    previous_threads = torch.get_num_threads()
    try:
        result = study.evaluate_study(
            main, runs, fixture["protocol"], output, external, device="cpu"
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert set(record["pool"] for record in result["predictions"]) == {
        "within_slide_test",
        "synthetic_batch1_slide_0",
        "synthetic_batch1_slide_1",
    }
    external_rows = [row for row in result["metrics"] if row["pool"] != "within_slide_test"]
    assert {row["marker"] for row in external_rows} == {"CD16", "CD45"}
    assert {row["policy"] for row in external_rows} == {
        "regression",
        "joint",
        "class_means",
        "predictive_selector",
        "utility_selector",
        "spo_plus",
        "pg_central",
    }
    forbidden = {
        "macro_f1",
        "accuracy",
        "selected_type_counts",
        "within_type_gap",
        "type_allocation_gap",
        "type_matched_random_utility",
        "within_type_enrichment",
        "quota_diagnostic",
    }
    for row in external_rows:
        assert forbidden.isdisjoint(row)
        expected_n = 5 if row["pool"].endswith("_0") else 7
        assert row["n_cells"] == expected_n
        assert row["k"] == math.ceil(row["fraction"] * expected_n)
        assert np.isfinite(row["utility"]) and np.isfinite(row["rmse"])
    # At q=.25 the independently selected budgets are 2+2, not ceil(.25*12)=3.
    assert {row["k"] for row in external_rows if row["fraction"] == 0.25} == {2}
    assert len(calls) == len(result["predictions"])
    main_metadata = pd.read_csv(main / "metadata.csv")
    for record, call in zip(result["predictions"], calls):
        pool = record["pool"]
        if pool == "within_slide_test":
            expected_ids = main_metadata.loc[
                main_metadata["split"] == "test", "global_index"
            ].to_numpy()
        else:
            expected_ids = external_metadata.loc[
                external_metadata["pool_id"] == pool, "global_index"
            ].to_numpy()
        path = output / record["path"]
        assert _sha(path) == record["sha256"]
        with np.load(path, allow_pickle=False) as predictions:
            np.testing.assert_array_equal(predictions["global_index"], expected_ids)
            assert call["n"] == len(expected_ids)
            norm = json.loads((runs / record["name"] / "normalization.json").read_text())
            config = json.loads((runs / record["name"] / "config.json").read_text())
            if config["mode"] == "classifier":
                # Use SciPy/Python arithmetic as an independent expectation reference.
                probabilities = softmax(call["logits"].astype(np.float64), axis=1)
                means = np.asarray(norm["class_means"], dtype=np.float64)
                expected_scores = np.array(
                    [
                        means[0]
                        + math.fsum(float(p) * float(mu - means[0]) for p, mu in zip(row, means))
                        for row in probabilities
                    ]
                )
            else:
                expected_scores = (
                    call["marker"].astype(float) * norm["target_std"] + norm["target_mean"]
                )
            np.testing.assert_allclose(predictions["score"], expected_scores, rtol=0, atol=1e-10)
    assert {path: _sha(path) for path in protected_paths} == original_hashes


@pytest.mark.parametrize(
    "artifact",
    ["external_images", "external_metadata", "external_manifest", "checkpoint", "normalization"],
)
def test_external_or_checkpoint_tamper_is_rejected_before_inference(
    frozen_external_study, tmp_path, monkeypatch, artifact
):
    fixture = frozen_external_study
    state = fixture["frozen"]["records"][0]
    paths = {
        "external_images": fixture["external"] / "images.npy",
        "external_metadata": fixture["external"] / "metadata.csv",
        "external_manifest": fixture["external"] / "manifest.json",
        "checkpoint": fixture["runs"] / state["name"] / state["checkpoint_file"],
        "normalization": fixture["runs"] / state["name"] / "normalization.json",
    }
    path = paths[artifact]
    original = path.read_bytes()
    called = []

    def forbidden_inference(*args, **kwargs):
        called.append(True)
        raise AssertionError("Inference started before tamper detection")

    monkeypatch.setattr(study, "predict", forbidden_inference)
    output = tmp_path / "rejected"
    try:
        path.write_bytes(original + b"\ntampered\n")
        with pytest.raises(ValueError, match="changed|checksum"):
            study.evaluate_study(
                fixture["main"],
                fixture["runs"],
                fixture["protocol"],
                output,
                fixture["external"],
                device="cpu",
            )
        assert not called
        assert not (output / "evaluation_started.json").exists()
        assert not (output / "predictions").exists()
    finally:
        path.write_bytes(original)
