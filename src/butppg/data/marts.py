"""Shared helpers for building per-model input marts (OpenTSLM, SIGMA-PPG).

E5/E6 (Budilov / Golikov). Both training epics need the same starting point:
the E1 registry (one row per 10 s window, see ``registry.py``) joined with the
subject-wise split file, plus the processed per-window PPG array it points to.
This module is the shared read side of that contract so both mart builders
(``scripts/build_mart_sigma_ppg.py``, ``scripts/build_mart_opentslm.py``)
stay consistent and don't duplicate the join/resample/normalize logic.

Split-file reading goes through ``butppg.data.splits.load_split`` — the same
function E1 uses to write it (``save_split``) — rather than parsing the JSON
again here, so there is exactly one place that understands the split schema.
A mismatch between "what E1 writes" and "what E5/E6 read" is then a
contradiction (same code both ends), not a bug that only shows up once real
data exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import resample

from butppg.data.registry import load_registry
from butppg.data.splits import load_split

Split = Literal["train", "val", "test"]


def load_registry_with_split(registry_path: str | Path, split_path: str | Path) -> pd.DataFrame:
    """Join the E1 registry with the split file; adds a ``split`` column.

    Raises if any subject in the registry is not covered by the split file
    (``load_split`` already re-verifies the three folds are disjoint; the
    registry itself is validated by ``registry.load_registry``, which also
    forces record_id/subject_id to string -- see its docstring for why).
    """
    registry = load_registry(registry_path)
    split = load_split(split_path)
    subject_to_split: dict[str, str] = {}
    for split_name, key in (("train", "train_subjects"), ("val", "val_subjects"), ("test", "test_subjects")):
        for subject in split[key]:
            subject_to_split[subject] = split_name

    unknown = set(registry["subject_id"]) - subject_to_split.keys()
    if unknown:
        raise ValueError(f"Registry has subjects absent from the split file: {sorted(unknown)}")

    registry = registry.copy()
    registry["split"] = registry["subject_id"].map(subject_to_split)
    return registry


def load_ppg_window(ppg_path: str | Path, project_root: str | Path | None = None) -> np.ndarray:
    """Load one processed PPG window (shape ``(300,)`` at 30 Hz, per E1)."""
    path = Path(ppg_path)
    if not path.is_absolute() and project_root is not None:
        path = Path(project_root) / path
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D PPG window at {path}, got shape {arr.shape}")
    return arr


def resample_window(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample a window from ``orig_fs`` to ``target_fs`` Hz (same duration)."""
    if orig_fs == target_fs:
        return x.astype(np.float32)
    target_len = int(round(len(x) * target_fs / orig_fs))
    return resample(x, target_len).astype(np.float32)


def zscore(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """Z-score normalize a 1-D window; returns (normalized, mean, std)."""
    mean = float(np.mean(x))
    std = float(np.std(x))
    return (x - mean) / (std + eps), mean, std


def minmax_to_pm1(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [-1, 1], SIGMA-PPG's own pretraining convention."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (2 * (x - lo) / (hi - lo) - 1).astype(np.float32)
