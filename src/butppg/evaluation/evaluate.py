"""Score saved prediction files on the fixed test set — E2 (Budilov).

One evaluator for both tasks (dispatch on the ``task`` column), so every
model is scored identically. Reads canonical prediction CSVs (see
``metrics.predictions``) — models need not be importable to be scored.

Also implements the end-to-end quality->HR gating scenario required by the
assignment (section 2B): first the quality model decides which windows are
usable, then the HR model predicts only on the accepted windows. The
assignment requires reporting, for that scenario: the accepted fraction, the
fraction of truly-good windows wrongly rejected, and the fraction of
truly-bad windows wrongly accepted. This lives here (not inside one model's
training script) so any model pair (baseline+baseline, CNN+CNN,
SIGMA-PPG+SIGMA-PPG, ...) can be scored on the same pipeline metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from butppg.metrics.hr import hr_metrics
from butppg.metrics.predictions import load_predictions
from butppg.metrics.quality import quality_metrics


def evaluate_prediction_file(path: str | Path) -> dict[str, Any]:
    """Evaluate one canonical prediction CSV; dispatches on its ``task`` column.

    The file must contain predictions for exactly one task (quality XOR hr) —
    that is how every model writes it (one CSV per task, see README's example
    ``artifacts/predictions/sigma_ppg_quality.csv``).
    """
    df = load_predictions(path)
    tasks = df["task"].unique()
    if len(tasks) != 1:
        raise ValueError(f"{path}: expected a single task per prediction file, found {sorted(tasks)}")
    task = tasks[0]

    n_invalid = int((df["parse_status"] == "invalid").sum())

    # Rows with no parseable y_pred (LLM answer-parsing failures, section 6 of the
    # assignment) cannot be scored as a class/number — they are excluded from the
    # metric computation itself, but never from the file (see predictions.py) and
    # never from n_records/invalid_response_fraction below, so the failure rate is
    # always visible even though it doesn't silently corrupt the score.
    scored = df[df["y_pred"].notna()]

    if len(scored) == 0:
        metrics = None
    elif task == "quality":
        y_prob = scored["prob_good"] if scored["prob_good"].notna().any() else None
        metrics = quality_metrics(scored["y_true"], scored["y_pred"], y_prob)
    elif task == "hr":
        metrics = hr_metrics(scored["y_true"], scored["y_pred"])
    else:  # pragma: no cover - guarded by predictions._validate already
        raise ValueError(f"Unknown task {task!r}")

    return {
        "file": str(path),
        "task": task,
        "n_records": len(df),
        "n_scored": len(scored),
        "n_invalid_responses": n_invalid,
        "invalid_response_fraction": n_invalid / len(df) if len(df) else 0.0,
        "metrics": metrics,
    }


def evaluate_pipeline(quality_path: str | Path, hr_path: str | Path) -> dict[str, Any]:
    """Score the sequential quality-gate -> HR-prediction scenario (section 2B).

    ``quality_path`` and ``hr_path`` must be predictions for the SAME window
    set (same ``record_id``s), typically the full test set: the HR file may
    carry HR predictions for every window (a model can always regress a
    number), while the *gate* is applied here using the quality file's
    predicted label, not by filtering the HR file upstream.
    """
    q = load_predictions(quality_path)
    hr = load_predictions(hr_path)
    if (q["task"] != "quality").any():
        raise ValueError(f"{quality_path} is not a quality-task prediction file")
    if (hr["task"] != "hr").any():
        raise ValueError(f"{hr_path} is not an hr-task prediction file")

    merged = q[["record_id", "y_true", "y_pred"]].merge(
        hr[["record_id", "y_true", "y_pred"]],
        on="record_id",
        suffixes=("_quality", "_hr"),
        how="inner",
    )
    if len(merged) == 0:
        raise ValueError("No overlapping record_id between quality and HR prediction files")

    accepted = merged["y_pred_quality"] == 1
    truly_good = merged["y_true_quality"] == 1
    truly_bad = merged["y_true_quality"] == 0

    accepted_fraction = float(accepted.mean())
    false_reject_rate = (
        float((~accepted & truly_good).sum() / truly_good.sum()) if truly_good.sum() else None
    )
    false_accept_rate = (
        float((accepted & truly_bad).sum() / truly_bad.sum()) if truly_bad.sum() else None
    )

    gated = merged[accepted]
    hr_on_accepted = (
        hr_metrics(gated["y_true_hr"], gated["y_pred_hr"]) if len(gated) else None
    )

    return {
        "quality_file": str(quality_path),
        "hr_file": str(hr_path),
        "n_windows": len(merged),
        "n_accepted": int(accepted.sum()),
        "accepted_fraction": accepted_fraction,
        "false_reject_rate": false_reject_rate,  # true-good windows wrongly rejected
        "false_accept_rate": false_accept_rate,  # true-bad windows wrongly accepted
        "hr_metrics_on_accepted": hr_on_accepted,
    }


def cascade_report(quality_path: str | Path, hr_path: str | Path | None = None) -> dict[str, Any]:
    """Quality-gate -> HR scenario (section 2B), gate metrics on the FULL test set.

    Unlike :func:`evaluate_pipeline`, the gate metrics (accepted fraction,
    false-reject, false-accept) are computed from the quality prediction file
    alone — which must cover the full test set — so they stay correct even when
    the HR prediction file only holds good-quality windows (our HR models train
    and predict on good windows). If ``hr_path`` is given, HR MAE/RMSE is also
    reported over the accepted windows that appear in that HR file (labelled so
    its scope is explicit).
    """
    q = load_predictions(quality_path)
    if (q["task"] != "quality").any():
        raise ValueError(f"{quality_path} is not a quality-task prediction file")

    accepted = q["y_pred"] == 1
    truly_good = q["y_true"] == 1
    truly_bad = q["y_true"] == 0
    result: dict[str, Any] = {
        "quality_file": str(quality_path),
        "n_test": int(len(q)),
        "n_accepted": int(accepted.sum()),
        "accepted_fraction": float(accepted.mean()),
        "false_reject_rate": float((~accepted & truly_good).sum() / truly_good.sum()) if truly_good.sum() else None,
        "false_accept_rate": float((accepted & truly_bad).sum() / truly_bad.sum()) if truly_bad.sum() else None,
    }
    if hr_path is not None:
        hr = load_predictions(hr_path)
        accepted_ids = set(q.loc[accepted, "record_id"])
        gated = hr[hr["record_id"].isin(accepted_ids)]
        result["hr_file"] = str(hr_path)
        result["hr_windows_scored"] = int(len(gated))
        result["hr_metrics_on_accepted"] = hr_metrics(gated["y_true"], gated["y_pred"]) if len(gated) else None
    return result
