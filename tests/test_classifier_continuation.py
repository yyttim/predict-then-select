"""CPU tests for model and AdamW state restoration."""

import importlib.util
from pathlib import Path

import pytest
import torch

from cellselect.model import build_model
from cellselect.training import file_sha256

spec = importlib.util.spec_from_file_location(
    "continue_cd45_classifier",
    Path(__file__).resolve().parents[1] / "scripts/continue_cd45_classifier.py",
)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def test_restore_preserves_weights_moments_steps_and_replaces_only_post_scheduler_lr(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        config = {"seed": 16, "feature_dim": 256, "learning_rate": 3e-4, "weight_decay": 1e-4}
        model = build_model(seed=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        torch.nn.functional.cross_entropy(
            model(torch.randn(4, 4, 28, 28))["logits"], torch.tensor([0, 1, 2, 0])
        ).backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 1.500018e-5  # Stored after a scheduler step.
        path = tmp_path / "synthetic" / "final.pt"
        path.parent.mkdir()
        torch.save(
            {
                "epoch": 200,
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            path,
        )
        record = {
            "name": "synthetic",
            "config": config,
            "files_sha256": {"final.pt": file_sha256(path)},
        }
        restored, opt, _, lrs = continuation.restore_source(
            tmp_path, record, 1.5e-5, torch.device("cpu")
        )
        assert lrs == [1.500018e-5] and opt.param_groups[0]["lr"] == 1.5e-5
        assert continuation.optimizer_steps(opt) == 1
        for name, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[name])
        for source, target in zip(
            optimizer.state_dict()["state"].values(), opt.state_dict()["state"].values()
        ):
            assert source.keys() == target.keys()
            for key in source:
                assert torch.equal(source[key], target[key])
        with path.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="Bound source artifact changed"):
            continuation.restore_source(tmp_path, record, 1.5e-5, torch.device("cpu"))
    finally:
        torch.set_num_threads(previous)
