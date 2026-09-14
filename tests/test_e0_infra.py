"""E0 smoke tests: config composition, seeding determinism, manifest round-trip.

These cover only the reproducibility spine. Split/metric/model tests arrive with
their epics (E1/E2/...).
"""

from __future__ import annotations

import numpy as np

from butppg.config import apply_overrides, load_config
from butppg.utils import RunManifest, seed_everything


def test_default_config_loads():
    cfg = load_config("default")
    assert cfg["seed"] == 42
    assert cfg["split"]["ratios"] == [0.6, 0.2, 0.2]
    assert cfg["window"]["ppg_len"] == 300


def test_config_compose_and_override():
    cfg = load_config("models/cnn1d", overrides=["train.epochs=1", "model.arch=cnn1d"])
    # deep-merged model config present alongside base keys
    assert cfg["model"]["name"] == "cnn1d"
    assert cfg["model"]["arch"] == "cnn1d"
    assert cfg["train"]["epochs"] == 1
    assert cfg["seed"] == 42  # base key survives


def test_override_casts_types():
    cfg = apply_overrides({"a": {"b": 0}}, ["a.b=5", "a.c=true"])
    assert cfg["a"]["b"] == 5 and isinstance(cfg["a"]["b"], int)
    assert cfg["a"]["c"] is True


def test_seed_is_deterministic():
    seed_everything(123)
    first = np.random.rand(5)
    seed_everything(123)
    second = np.random.rand(5)
    assert np.allclose(first, second)


def test_manifest_roundtrip(tmp_path):
    m = RunManifest(run_name="unit", task="quality", seed=42, config={"k": "v"})
    path = m.save(tmp_path / "manifest.json")
    loaded = RunManifest.load(path)
    assert loaded.seed == 42
    assert loaded.task == "quality"
    assert loaded.dataset_version == "BUT-PPG-v2.0.0"
