"""Canonical prediction-file I/O — E2 (Budilov). Stub.

Schema lives in butppg.metrics.PREDICTION_COLUMNS. Every model writes this exact
format so one evaluator scores all of them on the fixed test set.
"""

from __future__ import annotations


def save_predictions(*args, **kwargs):  # pragma: no cover - E2
    raise NotImplementedError("E2 (Budilov): implement canonical prediction saving")


def load_predictions(*args, **kwargs):  # pragma: no cover - E2
    raise NotImplementedError("E2 (Budilov): implement prediction loading")
