"""Shared training helpers: split-aware data access and result finalization.

Every trainer (baselines, CNN, SIGMA-PPG, OpenTSLM) ends the same way — write
the canonical test-set predictions, score them with the one evaluator, and
record the metrics on the run. :func:`finalize_predictions` is that shared tail
so all models land results in an identical, comparable shape.
"""

from __future__ import annotations

import json

import pandas as pd

from butppg.data.marts import load_registry_with_split
from butppg.evaluation.evaluate import evaluate_prediction_file
from butppg.metrics import PREDICTION_COLUMNS
from butppg.metrics.predictions import save_predictions
from butppg.orchestrator.runs import RunRecord


def build_prediction_frame(df, task: str, y_pred, prob_good=None):
    """Assemble a canonical prediction frame from a test slice + predictions."""
    out = pd.DataFrame(
        {
            "record_id": df["record_id"].values,
            "subject_id": df["subject_id"].values,
            "task": task,
            "y_true": df["quality_label"].values if task == "quality" else df["hr_ref"].values,
            "y_pred": y_pred,
            "prob_good": prob_good if prob_good is not None else pd.NA,
            "raw_response": pd.NA,
            "parse_status": pd.NA,
        }
    )
    return out[PREDICTION_COLUMNS]


def load_split_frames(
    registry_path, split_path, task: str, good_quality_only: bool = True, acc_subset: bool = False
):
    """Return (train_df, val_df, test_df) from the registry + split.

    For ``task='hr'`` all three folds are filtered to good-quality windows
    (assignment section 2B — the HR model operates only on signals the labels
    mark usable, at train AND test time). Evaluating HR on bad-quality windows
    would measure prediction on signals you'd never accept, and inflates MAE for
    every model. The separate quality->HR cascade (``evaluate_pipeline``) is
    where the quality gate is exercised instead.

    ``acc_subset=True`` restricts every fold to records that have ACC — the fair
    common subset for the extended-input variants (section 2A: PPG-only vs
    PPG+ACC vs +covariates must be compared on the same records).
    """
    registry = load_registry_with_split(registry_path, split_path)
    frames = {}
    for split_name in ("train", "val", "test"):
        df = registry[registry["split"] == split_name].copy()
        if task == "hr" and good_quality_only:
            df = df[df["quality_label"] == 1]
        if acc_subset:
            df = df[df["has_acc"] == True]  # noqa: E712 - pandas boolean mask
        frames[split_name] = df.reset_index(drop=True)
    return frames["train"], frames["val"], frames["test"]


def finalize_predictions(run: RunRecord, predictions: pd.DataFrame) -> dict:
    """Save test predictions, evaluate them, persist metrics, update the run.

    Returns the metrics dict. Called by every trainer so a run always ends with
    ``predictions/test_predictions.csv`` + ``metrics/metrics.json`` regardless
    of which model produced them.
    """
    run.ensure_dirs()
    pred_path = run.predictions_dir / "test_predictions.csv"
    save_predictions(predictions, pred_path)

    result = evaluate_prediction_file(pred_path)
    metrics = result["metrics"]

    (run.metrics_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    run.metrics = metrics
    run.save()
    return metrics
