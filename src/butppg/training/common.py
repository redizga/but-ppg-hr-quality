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
from butppg.metrics.predictions import save_predictions
from butppg.orchestrator.runs import RunRecord


def load_split_frames(registry_path, split_path, task: str, good_quality_only: bool = True):
    """Return (train_df, val_df, test_df) from the registry + split.

    For ``task='hr'`` the train/val folds are filtered to good-quality windows
    (assignment section 2B — HR is trained only on signals the labels mark
    usable). The **test** fold is left intact; filtering the test set is the
    caller's decision at evaluation time, not baked into the data here.
    """
    registry = load_registry_with_split(registry_path, split_path)
    frames = {}
    for split_name in ("train", "val", "test"):
        df = registry[registry["split"] == split_name].copy()
        if task == "hr" and good_quality_only and split_name in ("train", "val"):
            df = df[df["quality_label"] == 1]
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
