"""Smoke tests for section 8 (GalaxyPPG activity report) and 9 (low-data)."""

from __future__ import annotations

import pandas as pd

from butppg.evaluation.evaluate import activity_report
from butppg.metrics.predictions import save_predictions
from butppg.metrics import PREDICTION_COLUMNS
from butppg.reporting.galaxy_table import galaxy_markdown
from butppg.reporting.lowdata import _markdown, _mean_std


def _hr_pred_df(record_ids, y_true, y_pred):
    df = pd.DataFrame({
        "record_id": record_ids, "subject_id": [r.split("_")[0] for r in record_ids],
        "task": "hr", "y_true": y_true, "y_pred": y_pred,
        "prob_good": pd.NA, "raw_response": pd.NA, "parse_status": pd.NA,
    })
    return df[PREDICTION_COLUMNS]


def test_activity_report_overall_and_per_activity(tmp_path):
    ids = [f"P01_{i:03d}" for i in range(6)]
    y_true = [70, 72, 74, 150, 155, 160]
    y_pred = [71, 70, 76, 120, 130, 140]  # big error on the motion activity
    pred_path = tmp_path / "hr.csv"
    save_predictions(_hr_pred_df(ids, y_true, y_pred), pred_path)

    reg = pd.DataFrame({
        "record_id": ids,
        "activity": ["Standing"] * 3 + ["Running"] * 3,
    })
    reg_path = tmp_path / "registry.csv"
    reg.to_csv(reg_path, index=False)

    rep = activity_report(pred_path, reg_path)
    assert rep["n_scored"] == 6
    assert set(rep["per_activity"]) == {"Standing", "Running"}
    # motion activity should have the larger MAE
    assert rep["per_activity"]["Running"]["mae"] > rep["per_activity"]["Standing"]["mae"]
    assert rep["overall"]["mae"] > 0
    md = galaxy_markdown(rep)
    assert "GalaxyPPG" in md and "Running" in md


def test_lowdata_mean_std_and_markdown():
    assert _mean_std([1.0, 3.0])[0] == 2.0
    assert _mean_std([])[0] is None
    rows = [
        {"model": "Features + LogReg/XGBoost", "task": "quality", "fraction": 0.25,
         "n_train_subjects": 8, "n_seeds": 3, "metric": "macro_f1", "mean": 0.5, "std": 0.02},
        {"model": "Features + LogReg/XGBoost", "task": "hr", "fraction": 1.0,
         "n_train_subjects": 30, "n_seeds": 1, "metric": "mae", "mean": 9.7, "std": 0.0},
    ]
    md = _markdown(rows)
    assert "Macro-F1" in md and "MAE" in md and "25%" in md
