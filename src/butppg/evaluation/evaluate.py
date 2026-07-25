"""Score saved prediction files on the fixed test set — E2 (Budilov). Stub.

One evaluator for both tasks (dispatch on the ``task`` column), so every model
is scored identically. Reads canonical prediction CSVs — models need not be
importable to be scored.
"""

from __future__ import annotations


def evaluate_prediction_file(*args, **kwargs):  # pragma: no cover - E2
    raise NotImplementedError("E2 (Budilov): implement evaluation over prediction files")
