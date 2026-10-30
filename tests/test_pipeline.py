from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cellselect.evaluation import evaluate_pool, freeze_selection
from cellselect.reporting import summarize
from cellselect.study import file_hash, make_spec, run_name, save_json
from cellselect.training import train_run


def test_complete_small_pipeline(tmp_path):
    target = [0, 10, 2, 8, 4, 6, 3, 7, 1, 11, 5, 9, 3, 12, 8, 10, 13, 7, 15, 4]
    meta = pd.DataFrame(
        {
            "global_index": np.arange(20),
            "target": target,
            "fov_id": [f"field_{i // 2}" for i in range(20)],
            "slide_id": ["slide"] * 16 + ["external_slide"] * 4,
            "split": ["train"] * 8 + ["val"] * 4 + ["test"] * 4 + ["external"] * 4,
            "train_quarter": [True] * 2 + [False] * 18,
        }
    )
    images = np.random.default_rng(10).normal(size=(20, 4, 28, 28)).astype(np.float32)

    def rows(split, fraction=1.0):
        if split == "train" and fraction == 0.25:
            return np.flatnonzero(meta.train_quarter)
        return np.flatnonzero(meta.split == split)

    data = SimpleNamespace(
        images=images,
        metadata=meta,
        rows=rows,
        manifest_sha256="a" * 64,
        manifest={"threshold": {"value": 9.0}},
    )
    settings = {
        "train_fractions": [0.25, 1.0],
        "seeds": [16],
        "joint_weights": [1.0],
        "batch_size": 4,
        "full_data_epochs": 2,
        "checkpoint_epochs": [1, 2],
        "warmup_epochs": 0,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "min_lr_ratio": 0.05,
        "cpu_threads": 2,
        "primary_fraction": 0.1,
        "selection_fractions": [0.05, 0.1, 0.2],
        "bootstrap_draws": 20,
        "bootstrap_seed": 15,
    }
    study = tmp_path / "study"
    spec = make_spec(data, settings)
    save_json(study / "study.json", spec)
    for config in spec["runs"]:
        train_run(data, {**config, "device": "cpu"}, study / "runs" / run_name(config))
    frozen = freeze_selection(data, study)
    assert len(frozen["selections"]) == 12
    metrics = evaluate_pool(data, study, device="cpu")
    assert metrics.purity.between(0, 1).all()
    assert metrics.loc[metrics.method == "bce", "rmse"].isna().all()
    assert metrics.loc[metrics.method == "mse", "brier"].isna().all()
    summary, paired = summarize(study, tmp_path / "report")
    assert len(summary) == 48 and len(paired) == 96
    assert (tmp_path / "report/figures/objectives.pdf").is_file()
    assert (tmp_path / "report/tables/budget_interactions.csv").is_file()
    selection_hash = file_hash(study / "selection.json")
    external = evaluate_pool(data, study, device="cpu", pool="external")
    assert file_hash(study / "selection.json") == selection_hash
    assert external.n_cells.eq(4).all()
    assert pd.read_csv(study / "external/cells.csv").global_index.tolist() == [16, 17, 18, 19]
    summarize(study, tmp_path / "external_report", pool="external")


def test_report_refuses_to_overwrite_a_different_pool(tmp_path):
    study, report = tmp_path / "study", tmp_path / "report"
    save_json(study / "selection.json", {"fixed": True})
    save_json(
        report / "summary.json",
        {"pool": "test", "selection_sha256": file_hash(study / "selection.json")},
    )
    with pytest.raises(FileExistsError, match="another pool"):
        summarize(study, report, pool="external")
