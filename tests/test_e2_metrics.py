"""E2 tests: metrics, prediction I/O, evaluator, trivial baselines.

Epic E2 (Budilov). All synthetic — no BUT PPG data needed, since none of this
epic's logic depends on the real dataset (E1), only on the fixed schemas
(registry columns, prediction columns) already pinned at E0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from butppg.evaluation.evaluate import evaluate_pipeline, evaluate_prediction_file
from butppg.metrics import PREDICTION_COLUMNS
from butppg.metrics.hr import hr_metrics
from butppg.metrics.predictions import load_predictions, save_predictions
from butppg.metrics.quality import quality_metrics, select_threshold
from butppg.models.trivial import DominantFrequencyHR, MajorityClassQuality, MedianHR


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_quality_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1]
    m = quality_metrics(y_true, y_pred, y_prob=[0.1, 0.2, 0.9, 0.8, 0.95])
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["confusion_matrix"] == [[2, 0], [0, 3]]


def test_quality_metrics_all_wrong():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    m = quality_metrics(y_true, y_pred)
    assert m["macro_f1"] == pytest.approx(0.0)
    assert m["roc_auc"] is None  # no y_prob supplied


def test_quality_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        quality_metrics([0, 1], [0, 1, 1])


def test_select_threshold_recovers_separable_classes():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    # probabilities cleanly separated around 0.5
    y_prob = np.concatenate([rng.uniform(0.0, 0.3, 50), rng.uniform(0.7, 1.0, 50)])
    t, f1 = select_threshold(y_true, y_prob)
    assert 0.3 <= t <= 0.7
    assert f1 == pytest.approx(1.0)


def test_hr_metrics_known_values():
    y_true = [60, 70, 80]
    y_pred = [62, 68, 85]
    m = hr_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx((2 + 2 + 5) / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 25) / 3))


# ---------------------------------------------------------------------------
# prediction I/O
# ---------------------------------------------------------------------------


def _quality_predictions_df(n=10, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    y_pred = y_true.copy()
    y_pred[0] = 1 - y_pred[0]  # one mistake
    return pd.DataFrame(
        {
            "record_id": [f"r{i:03d}" for i in range(n)],
            "subject_id": [f"S{i % 3:02d}" for i in range(n)],
            "task": "quality",
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_good": rng.uniform(0, 1, n),
            "raw_response": None,
            "parse_status": None,
        }
    )


def test_save_and_load_predictions_roundtrip(tmp_path):
    df = _quality_predictions_df()
    path = save_predictions(df, tmp_path / "preds.csv")
    loaded = load_predictions(path)
    assert list(loaded.columns) == PREDICTION_COLUMNS
    assert len(loaded) == len(df)
    assert (loaded["task"] == "quality").all()


def test_save_predictions_rejects_duplicate_record_id(tmp_path):
    df = _quality_predictions_df()
    df.loc[1, "record_id"] = df.loc[0, "record_id"]
    with pytest.raises(ValueError, match="Duplicate record_id"):
        save_predictions(df, tmp_path / "bad.csv")


def test_save_predictions_rejects_missing_columns(tmp_path):
    df = _quality_predictions_df().drop(columns=["prob_good"])
    with pytest.raises(ValueError, match="missing required columns"):
        save_predictions(df, tmp_path / "bad.csv")


def test_load_predictions_keeps_numeric_looking_record_id_as_string(tmp_path):
    """Regression test: BUT PPG record_id looks like a plain number ("100001").

    A bare ``pd.read_csv`` would silently turn that column into int64, which
    then breaks joins against a string-typed registry/other prediction file
    (e.g. ``evaluate_pipeline``'s merge). Found running against real data.
    """
    df = _quality_predictions_df()
    df["record_id"] = [f"1000{i:02d}" for i in range(len(df))]  # numeric-looking strings
    path = save_predictions(df, tmp_path / "preds.csv")
    loaded = load_predictions(path)
    assert not pd.api.types.is_integer_dtype(loaded["record_id"])
    assert loaded["record_id"].iloc[0] == "100000"


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------


def test_evaluate_prediction_file_quality(tmp_path):
    df = _quality_predictions_df(n=20)
    path = save_predictions(df, tmp_path / "quality.csv")
    result = evaluate_prediction_file(path)
    assert result["task"] == "quality"
    assert result["n_records"] == 20
    assert 0.0 <= result["metrics"]["macro_f1"] <= 1.0


def test_evaluate_prediction_file_hr(tmp_path):
    n = 15
    df = pd.DataFrame(
        {
            "record_id": [f"r{i:03d}" for i in range(n)],
            "subject_id": [f"S{i % 3:02d}" for i in range(n)],
            "task": "hr",
            "y_true": np.linspace(60, 90, n),
            "y_pred": np.linspace(60, 90, n) + 1.0,
            "prob_good": None,
            "raw_response": None,
            "parse_status": None,
        }
    )
    path = save_predictions(df, tmp_path / "hr.csv")
    result = evaluate_prediction_file(path)
    assert result["task"] == "hr"
    assert result["metrics"]["mae"] == pytest.approx(1.0)


def test_evaluate_pipeline_gating_metrics(tmp_path):
    # 4 windows: true quality [good, good, bad, bad]; predicted [good, bad, good, bad]
    # -> accepted = {0, 2}; false_reject = window 1 (true good, rejected); false_accept = window 2 (true bad, accepted)
    quality_df = pd.DataFrame(
        {
            "record_id": ["r0", "r1", "r2", "r3"],
            "subject_id": ["S0"] * 4,
            "task": "quality",
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 1, 0],
            "prob_good": [0.9, 0.4, 0.6, 0.1],
            "raw_response": None,
            "parse_status": None,
        }
    )
    hr_df = pd.DataFrame(
        {
            "record_id": ["r0", "r1", "r2", "r3"],
            "subject_id": ["S0"] * 4,
            "task": "hr",
            "y_true": [70, 72, 100, 40],
            "y_pred": [71, 73, 65, 41],  # only r0/r2 matter (accepted)
            "prob_good": None,
            "raw_response": None,
            "parse_status": None,
        }
    )
    q_path = save_predictions(quality_df, tmp_path / "q.csv")
    hr_path = save_predictions(hr_df, tmp_path / "hr.csv")

    result = evaluate_pipeline(q_path, hr_path)
    assert result["n_accepted"] == 2
    assert result["accepted_fraction"] == pytest.approx(0.5)
    assert result["false_reject_rate"] == pytest.approx(0.5)  # 1 of 2 truly-good windows rejected
    assert result["false_accept_rate"] == pytest.approx(0.5)  # 1 of 2 truly-bad windows accepted
    # HR MAE computed only on accepted windows {r0: |71-70|=1, r2: |65-100|=35}
    assert result["hr_metrics_on_accepted"]["mae"] == pytest.approx((1 + 35) / 2)


# ---------------------------------------------------------------------------
# trivial baselines
# ---------------------------------------------------------------------------


def _fake_registry(n_subjects=4, windows_per_subject=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for w in range(windows_per_subject):
            rows.append(
                {
                    "record_id": f"S{s}_w{w}",
                    "subject_id": f"S{s}",
                    "quality_label": int(rng.integers(0, 2)),
                    "hr_ref": float(rng.uniform(55, 95)),
                }
            )
    return pd.DataFrame(rows)


def test_majority_class_quality():
    df = _fake_registry()
    df.loc[:, "quality_label"] = [1, 1, 1, 0, 0] * 4  # majority = good (1)
    model = MajorityClassQuality()
    model.fit(df)
    preds = model.predict(df)
    assert (preds["y_pred"] == 1).all()
    assert preds["prob_good"].iloc[0] == pytest.approx(0.6)
    assert list(preds.columns) == PREDICTION_COLUMNS


def test_median_hr_uses_good_quality_only():
    df = _fake_registry()
    df["quality_label"] = 1
    df.loc[df.index[:3], "quality_label"] = 0  # first 3 rows excluded from the median
    df["hr_ref"] = list(range(20))[: len(df)]
    model = MedianHR()
    model.fit(df)
    expected_median = df[df["quality_label"] == 1]["hr_ref"].median()
    preds = model.predict(df)
    assert (preds["y_pred"] == expected_median).all()


def test_dominant_frequency_hr_recovers_known_bpm(tmp_path):
    fs = 30.0
    duration_s = 10.0
    true_bpm = 72.0
    t = np.arange(0, duration_s, 1 / fs)
    signal = np.sin(2 * np.pi * (true_bpm / 60.0) * t).astype(np.float32)

    ppg_path = tmp_path / "r0.npy"
    np.save(ppg_path, signal)

    records = pd.DataFrame(
        {
            "record_id": ["r0"],
            "subject_id": ["S0"],
            "hr_ref": [true_bpm],
            "ppg_path": [str(ppg_path)],
        }
    )
    model = DominantFrequencyHR()
    model.fit(records)
    preds = model.predict(records)
    # periodogram frequency resolution over a 10s window is 0.1 Hz = 6 bpm
    assert abs(preds["y_pred"].iloc[0] - true_bpm) < 6.0
