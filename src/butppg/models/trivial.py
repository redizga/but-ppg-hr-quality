"""Trivial lower-bound baselines — E2 (Budilov), assignment section 4.

Three models, none of them "learn" in any interesting sense — they exist so
every real model has something to beat:

* ``MajorityClassQuality``  — always predicts the most common quality label
  seen in training.
* ``MedianHR``               — always predicts the median reference HR from
  the (good-quality) training windows.
* ``DominantFrequencyHR``    — classic signal-processing HR estimate: the
  frequency with the most power in the physiologically plausible band
  (30-220 bpm, same range the assignment fixes for OpenTSLM's answer
  parsing, section 6) becomes the HR estimate. No training at all.

All three follow the shared model contract from ``models/__init__.py``:
``fit(train, val, cfg)`` + ``predict(records) -> DataFrame`` in the canonical
prediction schema (``metrics.PREDICTION_COLUMNS``), so they run through the
exact same ``evaluate.py`` path as every other model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import periodogram

from butppg.data.marts import load_ppg_window
from butppg.metrics import PREDICTION_COLUMNS

HR_BAND_BPM = (30, 220)  # assignment section 6: physiologically plausible HR range
SOURCE_FS = 30.0  # BUT PPG, fixed by E1 (configs/default.yaml: window.ppg_fs)


def _empty_prediction_frame(records: pd.DataFrame, task: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": records["record_id"].values,
            "subject_id": records["subject_id"].values,
            "task": task,
            "y_true": pd.NA,
            "y_pred": pd.NA,
            "prob_good": pd.NA,
            "raw_response": pd.NA,
            "parse_status": pd.NA,
        }
    )[PREDICTION_COLUMNS]


class MajorityClassQuality:
    """Task A floor: always predict the majority class from training."""

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None, cfg: dict | None = None) -> None:
        if len(train) == 0:
            raise ValueError("MajorityClassQuality.fit: empty training set")
        counts = train["quality_label"].value_counts()
        self.majority_label_ = int(counts.idxmax())
        self.prob_good_ = float((train["quality_label"] == 1).mean())

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        out = _empty_prediction_frame(records, task="quality")
        out["y_true"] = records["quality_label"].values
        out["y_pred"] = self.majority_label_
        out["prob_good"] = self.prob_good_
        return out


class MedianHR:
    """Task B floor: always predict the median HR from good-quality training windows."""

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None, cfg: dict | None = None) -> None:
        good = train[train["quality_label"] == 1]
        if len(good) == 0:
            raise ValueError("MedianHR.fit: no quality_label==1 rows in training set")
        self.median_hr_ = float(good["hr_ref"].median())

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        out = _empty_prediction_frame(records, task="hr")
        out["y_true"] = records["hr_ref"].values
        out["y_pred"] = self.median_hr_
        return out


def _dominant_frequency_bpm(x: np.ndarray, fs: float, band_bpm: tuple[float, float] = HR_BAND_BPM) -> float:
    """Peak-power frequency within the plausible HR band, converted to bpm."""
    freqs, power = periodogram(x, fs=fs)
    lo_hz, hi_hz = band_bpm[0] / 60.0, band_bpm[1] / 60.0
    in_band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(in_band):
        # window too short / fs too low to resolve the band: fall back to band midpoint
        return float(np.mean(band_bpm))
    peak_freq = freqs[in_band][np.argmax(power[in_band])]
    return float(peak_freq * 60.0)


class DominantFrequencyHR:
    """Task B floor: HR from the dominant frequency of the raw PPG window.

    Stateless (no training) — ``fit`` is a no-op kept only so this model
    satisfies the same interface as every other one.
    """

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None, cfg: dict | None = None) -> None:
        pass

    def predict(self, records: pd.DataFrame, project_root=None) -> pd.DataFrame:
        preds = [
            _dominant_frequency_bpm(load_ppg_window(row["ppg_path"], project_root=project_root), fs=SOURCE_FS)
            for _, row in records.iterrows()
        ]
        out = _empty_prediction_frame(records, task="hr")
        out["y_true"] = records["hr_ref"].values
        out["y_pred"] = preds
        return out
