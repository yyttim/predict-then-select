"""Full reference architecture, paired initialization, and independent task gradients."""

import importlib.util
from pathlib import Path

import pytest
import torch

from cellselect.losses import objective_loss
from cellselect.model import build_model, parameter_counts


@pytest.fixture(autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_matches_reference_encoder_and_scalar_heads_without_gating():
    path = Path(__file__).resolve().parents[1] / "vendor/reference_model.py"
    spec = importlib.util.spec_from_file_location("reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    original = reference.create_model(num_classes=1, num_proteins=1, use_task_gating=False).eval()
    model = build_model(seed=3).eval()
    for name in ("shared_channel_attention", "cnn", "vit", "fusion"):
        getattr(model.encoder, name).load_state_dict(getattr(original, name).state_dict())
    model.mean_head.refinement.load_state_dict(original.reg_refinement.state_dict())
    model.mean_head.predictor.load_state_dict(original.regressor.regressor.state_dict())
    model.hit_head.refinement.load_state_dict(original.cls_refinement.state_dict())
    model.hit_head.predictor.load_state_dict(original.classifier.classifier.state_dict())
    x = torch.randn(3, 4, 28, 28)
    with torch.no_grad():
        logits, marker = original(x)
        actual = model(x)
    torch.testing.assert_close(actual["mean_z"], marker[:, 0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual["hit_logit"], logits[:, 0], rtol=2e-5, atol=2e-6)
    assert parameter_counts(model) == {"total": 1289583, "trainable": 1289583}


@pytest.mark.parametrize("method", ["mse", "bce", "joint"])
def test_active_gradients_reach_both_encoder_branches_and_leave_inactive_head_unchanged(method):
    model = build_model(seed=7).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    output = model(torch.randn(3, 4, 28, 28))
    loss, _, _ = objective_loss(
        output, torch.tensor([-0.5, 0.0, 1.0]), torch.tensor([0.0, 0.0, 1.0]), method, 0.3
    )
    loss.backward()
    for block in [model.encoder.cnn, model.encoder.vit, model.encoder.fusion]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
    inactive = "hit_head" if method == "mse" else "mean_head" if method == "bce" else None
    for name in ["mean_head", "hit_head"]:
        grads = [p.grad for p in getattr(model, name).parameters()]
        assert (
            all(g is None for g in grads)
            if name == inactive
            else all(g is not None and torch.isfinite(g).all() for g in grads)
        )
    optimizer.step()
    if inactive:
        for key, value in model.state_dict().items():
            if key.startswith(inactive + "."):
                assert torch.equal(value, before[key])


@pytest.mark.parametrize("batch", [1, 4])
def test_shapes_eval_determinism_and_seeded_initialization(batch):
    before = torch.random.get_rng_state().clone()
    first, second = build_model(seed=11).eval(), build_model(seed=11).eval()
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(
        torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items()
    )
    x = torch.randn(batch, 4, 28, 28)
    with torch.no_grad():
        a, b = first(x), first(x)
    assert set(a) == {"mean_z", "hit_logit"}
    for key in a:
        assert a[key].shape == (batch,) and torch.equal(a[key], b[key])


@pytest.mark.parametrize("shape", [(2, 3, 28, 28), (2, 4, 32, 32), (4, 28, 28), (0, 4, 28, 28)])
def test_invalid_image_shape_rejected(shape):
    with pytest.raises(ValueError):
        build_model(seed=4)(torch.randn(*shape))


def test_loss_rejects_target_broadcasting_and_keeps_unweighted_bce():
    outputs = {"mean_z": torch.tensor([0.0, 2.0]), "hit_logit": torch.tensor([0.0, 0.0])}
    total, mse, bce = objective_loss(
        outputs, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1.0]), "joint", 0.3
    )
    assert mse.item() == 2.0 and bce.item() == pytest.approx(0.69314718)
    assert total.item() == pytest.approx(2.0 + 0.3 * 0.69314718)
    with pytest.raises(ValueError, match="aligned"):
        objective_loss(outputs, torch.zeros(2, 1), torch.zeros(2), "mse")
