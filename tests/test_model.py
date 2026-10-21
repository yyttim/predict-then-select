"""Checks for preservation of the reference encoder and task separation."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from cellselect.model import build_model, parameter_counts


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _reference_model():
    path = Path(__file__).resolve().parents[1] / "vendor" / "reference_model.py"
    spec = importlib.util.spec_from_file_location("cellselect_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference.create_model(num_classes=3, num_proteins=1, use_task_gating=False)


def test_matches_ungated_released_model_after_copying_weights():
    """Encoder outputs agree with the upstream implementation."""
    reference = _reference_model().eval()
    model = build_model(seed=7).eval()
    for name in ("shared_channel_attention", "cnn", "vit", "fusion"):
        getattr(model.encoder, name).load_state_dict(getattr(reference, name).state_dict())
    model.classification_head.refinement.load_state_dict(reference.cls_refinement.state_dict())
    model.regression_head.refinement.load_state_dict(reference.reg_refinement.state_dict())
    model.classification_head.predictor.load_state_dict(
        reference.classifier.classifier.state_dict()
    )
    model.regression_head.predictor.load_state_dict(reference.regressor.regressor.state_dict())
    x = torch.randn(3, 4, 28, 28)
    with torch.no_grad():
        expected_cls, expected_marker, expected_features = reference(x, return_features=True)
        actual = model(x, return_features=True)
    torch.testing.assert_close(actual["logits"], expected_cls, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual["marker"], expected_marker[:, 0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        actual["features"], expected_features["shared_features"], rtol=2e-5, atol=2e-6
    )
    assert parameter_counts(model) == parameter_counts(reference)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_losses_reach_both_encoder_branches_but_not_other_head(task):
    model = build_model(seed=11).train()
    outputs = model(torch.randn(3, 4, 28, 28))
    if task == "classification":
        loss = F.cross_entropy(outputs["logits"], torch.tensor([0, 1, 2]))
        active = model.classification_head
        inactive = model.regression_head
    else:
        loss = F.mse_loss(outputs["marker"], torch.tensor([-1.0, 0.2, 0.9]))
        active = model.regression_head
        inactive = model.classification_head
    loss.backward()
    assert all(parameter.grad is None for parameter in inactive.parameters())
    for module in (
        model.encoder.shared_channel_attention,
        model.encoder.cnn,
        model.encoder.vit,
        model.encoder.fusion,
        active,
    ):
        grads = [parameter.grad for parameter in module.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)


def test_seeded_initialization_matches_and_preserves_cpu_rng_state():
    before = torch.random.get_rng_state().clone()
    first = build_model(seed=19)
    assert torch.equal(before, torch.random.get_rng_state())
    second = build_model(seed=19)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    different = build_model(seed=20)
    assert not torch.equal(first.encoder.cnn.stem[0].weight, different.encoder.cnn.stem[0].weight)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_output_contract_and_eval_determinism(batch_size):
    model = build_model(seed=21).eval()
    x = torch.randn(batch_size, 4, 28, 28)
    with torch.no_grad():
        outputs = model(x)
        repeated = model(x)
    assert set(outputs) == {"logits", "marker"}
    assert outputs["logits"].shape == (batch_size, 3)
    assert outputs["marker"].shape == (batch_size,)
    for key in outputs:
        assert torch.isfinite(outputs[key]).all()
        torch.testing.assert_close(outputs[key], repeated[key], rtol=0, atol=0)


def test_width_check_changes_fusion_and_heads_not_backbones():
    reference_width = build_model(seed=2)
    narrower = build_model(feature_dim=128, seed=2).eval()
    assert parameter_counts(reference_width.encoder.cnn) == parameter_counts(narrower.encoder.cnn)
    assert parameter_counts(reference_width.encoder.vit) == parameter_counts(narrower.encoder.vit)
    with torch.no_grad():
        assert narrower(torch.randn(1, 4, 28, 28), return_features=True)["features"].shape == (
            1,
            128,
        )


@pytest.mark.parametrize("shape", [(2, 3, 28, 28), (2, 4, 32, 32), (4, 28, 28), (0, 4, 28, 28)])
def test_rejects_wrong_input_shape(shape):
    model = build_model(seed=4)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_rejects_unprocessed_integer_input():
    with pytest.raises(TypeError):
        build_model(seed=4)(torch.zeros(2, 4, 28, 28, dtype=torch.uint8))


@pytest.mark.parametrize(
    "kwargs", [{"feature_dim": 0}, {"num_classes": 1}, {"img_size": 27}, {"dropout_rate": 1}]
)
def test_rejects_invalid_model_configuration(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)
