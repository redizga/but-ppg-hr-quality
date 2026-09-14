"""HR-regression metrics (task B) — E2 (Budilov).

Primary: MAE in bpm (used for checkpoint selection, assignment section 5).
Secondary: RMSE in bpm.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def hr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) length mismatch")
    if len(y_true) == 0:
        raise ValueError("hr_metrics: empty input")

    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {"n": len(y_true), "mae": mae, "rmse": rmse}
