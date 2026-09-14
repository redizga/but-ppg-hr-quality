"""E4 tests: 1D-CNN / ResNet1D architecture (assignment section 4).

Random tensors only — no BUT PPG data needed. Checks shapes, both archs, the
optional ACC branch, error handling, config wiring, and that gradients
actually flow (a real training step, not just a forward pass).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # E4 is an optional "deep" dependency (pip install -e ".[deep]")

from butppg.config import load_config  # noqa: E402
from butppg.models.cnn1d import PPGModel, build_model_from_config  # noqa: E402

BATCH = 4
PPG_LEN = 300  # E1: 10s @ 30Hz


@pytest.mark.parametrize("arch", ["cnn1d", "resnet1d"])
def test_forward_ppg_only_shape(arch):
    model = PPGModel(arch=arch, use_acc=False)
    ppg = torch.randn(BATCH, 1, PPG_LEN)
    out = model(ppg)
    assert out.shape == (BATCH,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ["cnn1d", "resnet1d"])
def test_forward_with_acc_branch_different_length(arch):
    # ACC magnitude at its own length (e.g. native 100 Hz over 10s = 1000
    # samples) does not need to match the PPG branch's length -- that's the
    # point of the separate-branch design (see module docstring).
    model = PPGModel(arch=arch, use_acc=True)
    ppg = torch.randn(BATCH, 1, PPG_LEN)
    acc = torch.randn(BATCH, 1, 1000)
    out = model(ppg, acc)
    assert out.shape == (BATCH,)
    assert torch.isfinite(out).all()


def test_use_acc_true_requires_acc_tensor():
    model = PPGModel(use_acc=True)
    ppg = torch.randn(BATCH, 1, PPG_LEN)
    with pytest.raises(ValueError, match="use_acc=True"):
        model(ppg)


def test_use_acc_false_rejects_acc_tensor():
    model = PPGModel(use_acc=False)
    ppg = torch.randn(BATCH, 1, PPG_LEN)
    acc = torch.randn(BATCH, 1, 1000)
    with pytest.raises(ValueError, match="use_acc=False"):
        model(ppg, acc)


def test_rejects_wrong_ppg_shape():
    model = PPGModel()
    bad_ppg = torch.randn(BATCH, PPG_LEN)  # missing the channel dim
    with pytest.raises(ValueError, match="Expected ppg shape"):
        model(bad_ppg)


def test_build_model_from_config_matches_configs_models_cnn1d_yaml():
    cfg = load_config("models/cnn1d")
    assert cfg["model"]["arch"] == "resnet1d"
    assert cfg["model"]["channels"] == ["ppg"]

    model = build_model_from_config(cfg)
    assert isinstance(model, PPGModel)
    assert model.use_acc is False

    out = model(torch.randn(BATCH, 1, PPG_LEN))
    assert out.shape == (BATCH,)


def test_build_model_from_config_with_acc_override():
    cfg = load_config("models/cnn1d", overrides=["model.channels=[ppg, acc]"])
    model = build_model_from_config(cfg)
    assert model.use_acc is True


@pytest.mark.parametrize("task,loss_fn", [
    ("quality", torch.nn.BCEWithLogitsLoss()),
    ("hr", torch.nn.L1Loss()),
])
def test_gradients_flow_on_a_real_training_step(task, loss_fn):
    """One optimizer step should strictly reduce the loss on a fixed batch.

    Same architecture serves both tasks (module docstring): quality treats
    the scalar output as a BCE logit, HR treats it as a raw regression value.
    """
    torch.manual_seed(0)
    model = PPGModel(arch="resnet1d", use_acc=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    ppg = torch.randn(BATCH, 1, PPG_LEN)
    target = torch.randint(0, 2, (BATCH,)).float() if task == "quality" else torch.rand(BATCH) * 60 + 60

    out_before = model(ppg)
    loss_before = loss_fn(out_before, target).item()

    optimizer.zero_grad()
    loss = loss_fn(model(ppg), target)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "some parameters got no gradient"
    assert any(torch.any(g != 0) for g in grads), "all gradients are exactly zero"

    optimizer.step()
    loss_after = loss_fn(model(ppg), target).item()
    assert loss_after < loss_before
