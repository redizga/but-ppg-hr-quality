"""Tests for the feature baseline, extended-input variants, and cascade metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from butppg.evaluation.evaluate import cascade_report
from butppg.features.extract import INPUT_VARIANTS, build_feature_matrix
from butppg.metrics.predictions import save_predictions


def _mini_registry(tmp_path, n=6):
    ppg_dir = tmp_path / "ppg"
    ppg_dir.mkdir()
    acc_dir = tmp_path / "acc"
    acc_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rid = f"10{i:03d}"
        t = np.arange(300) / 30.0
        np.save(ppg_dir / f"{rid}.npy", (np.sin(2 * np.pi * 1.2 * t) + 0.1 * rng.standard_normal(300)).astype("float32"))
        np.save(acc_dir / f"{rid}.npy", rng.standard_normal((3, 1000)).astype("float32"))
        rows.append({
            "record_id": rid, "subject_id": str(i), "quality_label": i % 2, "hr_ref": 70.0 + i,
            "has_acc": True, "ppg_path": str(ppg_dir / f"{rid}.npy"), "acc_path": str(acc_dir / f"{rid}.npy"),
            "sex": "M", "age": 30.0, "height": 175.0, "weight": 70.0, "measurement_site": "finger",
        })
    return pd.DataFrame(rows)


def test_feature_variants_widen(tmp_path):
    df = _mini_registry(tmp_path)
    widths = {}
    for v in INPUT_VARIANTS:
        x, names = build_feature_matrix(df, v, project_root=None)
        assert x.shape[0] == len(df)
        assert x.shape[1] == len(names)
        widths[v] = x.shape[1]
    # each extended variant adds columns
    assert widths["ppg"] < widths["ppg_acc"] < widths["ppg_acc_cov"]


def test_cascade_gate_metrics(tmp_path):
    # quality predictions on a full test set: 4 good (2 accepted), 2 bad (1 accepted)
    df = pd.DataFrame({
        "record_id": ["a", "b", "c", "d", "e", "f"],
        "subject_id": ["1"] * 6,
        "task": "quality",
        "y_true": [1, 1, 1, 1, 0, 0],
        "y_pred": [1, 1, 0, 0, 1, 0],
        "prob_good": [0.9, 0.8, 0.4, 0.3, 0.6, 0.2],
        "raw_response": pd.NA, "parse_status": pd.NA,
    })
    qpath = tmp_path / "q.csv"
    save_predictions(df, qpath)
    rep = cascade_report(qpath)
    assert rep["n_test"] == 6
    assert rep["n_accepted"] == 3  # a, b, e
    assert abs(rep["accepted_fraction"] - 0.5) < 1e-9
    assert abs(rep["false_reject_rate"] - 0.5) < 1e-9  # 2 of 4 good rejected (c, d)
    assert abs(rep["false_accept_rate"] - 0.5) < 1e-9  # 1 of 2 bad accepted (e)
