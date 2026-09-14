"""Canonical prediction-file I/O — E2 (Budilov).

Schema lives in ``butppg.metrics.PREDICTION_COLUMNS``. Every model (baseline,
CNN, OpenTSLM, SIGMA-PPG) writes this exact format so the one evaluator in
``evaluation/evaluate.py`` scores all of them identically on the fixed test
set, and predictions can always be joined back to the registry by
``record_id``/``subject_id`` (assignment section 5).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from butppg.metrics import PREDICTION_COLUMNS

VALID_TASKS = {"quality", "hr"}
VALID_PARSE_STATUS = {"ok", "invalid", None}


def _validate(df: pd.DataFrame) -> None:
    missing = set(PREDICTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {sorted(missing)}")

    unknown_tasks = set(df["task"].unique()) - VALID_TASKS
    if unknown_tasks:
        raise ValueError(f"Unknown task value(s) {sorted(unknown_tasks)}, expected one of {VALID_TASKS}")

    if df["record_id"].duplicated().any():
        dupes = df.loc[df["record_id"].duplicated(), "record_id"].tolist()
        raise ValueError(f"Duplicate record_id in predictions: {dupes[:5]}{'...' if len(dupes) > 5 else ''}")

    bad_status = set(df["parse_status"].dropna().unique()) - {"ok", "invalid"}
    if bad_status:
        raise ValueError(f"Unknown parse_status value(s) {sorted(bad_status)}, expected 'ok'/'invalid'/empty")


def save_predictions(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write predictions in the canonical schema.

    ``df`` must already have all of ``PREDICTION_COLUMNS`` (missing optional
    fields like ``prob_good``/``raw_response``/``parse_status`` should be
    ``None``/``NaN``, not simply absent — this keeps every prediction file
    structurally identical regardless of which model produced it).
    """
    _validate(df)
    df = df[PREDICTION_COLUMNS].copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load and validate a canonical prediction CSV.

    ``record_id``/``subject_id`` are forced to string on read: BUT PPG's IDs
    are plain numbers ("100001", "100"), and pandas silently infers such a
    column as int64 on a bare ``read_csv`` -- which then breaks any join back
    to the registry, or between two prediction files (e.g.
    ``evaluate_pipeline``'s quality+HR merge), against a string-typed ID.
    Found by running this against real BUT PPG data, not synthetic IDs like
    "r0" (see ``docs/review/`` for the write-up).
    """
    df = pd.read_csv(Path(path), dtype={"record_id": str, "subject_id": str})
    _validate(df)
    return df
