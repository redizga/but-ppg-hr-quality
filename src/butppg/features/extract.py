"""Hand-crafted features for the LogReg/XGBoost baseline (assignment section 4).

Turns a 10 s PPG window (optionally + ACC, + subject covariates) into a fixed
feature vector — the ~20-200 statistical/spectral features the assignment asks
for. The input variant controls which blocks are concatenated, covering the four
cases from section 2A:

    ppg              PPG time+frequency features only
    ppg_acc          + accelerometer-magnitude features
    ppg_acc_cov      + subject covariates (sex, age, height, weight, site)

Everything here is pure numpy/scipy so the baseline runs on CPU anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import kurtosis, skew

from butppg.data.marts import load_ppg_window
from butppg.utils.progress import pbar

PPG_FS = 30.0
ACC_FS = 100.0
HR_BAND_HZ = (0.5, 3.67)  # 30-220 bpm

INPUT_VARIANTS = ("ppg", "ppg_acc", "ppg_acc_cov")


def _time_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    diff = np.diff(x)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_range": float(np.ptp(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x**2))),
        f"{prefix}_skew": float(skew(x)) if np.std(x) > 1e-8 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(x)) if np.std(x) > 1e-8 else 0.0,
        f"{prefix}_mean_abs_diff": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        f"{prefix}_zero_cross": float(np.mean(np.abs(np.diff(np.sign(x - np.mean(x)))) > 0)),
    }


def _freq_features(x: np.ndarray, fs: float, prefix: str, band: tuple[float, float] = HR_BAND_HZ) -> dict[str, float]:
    freqs, power = periodogram(x, fs=fs)
    total = float(np.sum(power)) + 1e-12
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    band_power = float(np.sum(power[in_band]))
    if np.any(in_band) and np.sum(power[in_band]) > 0:
        dom_freq = float(freqs[in_band][np.argmax(power[in_band])])
    else:
        dom_freq = 0.0
    centroid = float(np.sum(freqs * power) / total)
    p_norm = power / total
    entropy = float(-np.sum(p_norm[p_norm > 0] * np.log(p_norm[p_norm > 0])))
    return {
        f"{prefix}_dom_freq": dom_freq,
        f"{prefix}_dom_freq_bpm": dom_freq * 60.0,
        f"{prefix}_band_power": band_power,
        f"{prefix}_band_power_ratio": band_power / total,
        f"{prefix}_spec_centroid": centroid,
        f"{prefix}_spec_entropy": entropy,
        f"{prefix}_total_power": total,
    }


def ppg_features(x: np.ndarray, fs: float = PPG_FS) -> dict[str, float]:
    """Time + frequency features from one PPG window (z-scored internally)."""
    xn = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return {**_time_features(xn, "ppg"), **_freq_features(xn, fs, "ppg")}


def acc_features(acc: np.ndarray, fs: float = ACC_FS) -> dict[str, float]:
    """Motion features from the ACC magnitude (L2 over the 3 axes)."""
    mag = np.linalg.norm(acc, axis=0) if acc.ndim == 2 else np.asarray(acc)
    mn = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
    return {**_time_features(mn, "acc"), **_freq_features(mn, fs, "acc", band=(0.5, 10.0))}


def covariate_features(row: pd.Series) -> dict[str, float]:
    """Subject covariates as numeric features (section 2A extended input)."""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    sex = str(row.get("sex", "")).strip().upper()
    site = str(row.get("measurement_site", "")).strip().lower()
    return {
        "cov_sex_male": 1.0 if sex in ("M", "MALE", "1") else 0.0,
        "cov_age": num(row.get("age")),
        "cov_height": num(row.get("height")),
        "cov_weight": num(row.get("weight")),
        "cov_site_finger": 1.0 if site == "finger" else 0.0,
    }


def _zero_acc_features() -> dict[str, float]:
    # keep a stable column set even when a record has no ACC
    return {k: 0.0 for k in {**_time_features(np.zeros(10), "acc"), **_freq_features(np.zeros(10), ACC_FS, "acc", band=(0.5, 10.0))}}


def build_feature_matrix(df: pd.DataFrame, variant: str, project_root: str | Path | None = None):
    """Build (X, feature_names) for a registry slice under the given input variant.

    ``variant`` in :data:`INPUT_VARIANTS`. Rows keep registry order so labels
    align. ACC blocks are zero-filled for records lacking ACC (stable columns).
    """
    if variant not in INPUT_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {INPUT_VARIANTS}")

    rows: list[dict[str, float]] = []
    for _, r in pbar(df.iterrows(), desc=f"features[{variant}]", total=len(df)):
        feats = ppg_features(load_ppg_window(r["ppg_path"], project_root=project_root))
        if variant in ("ppg_acc", "ppg_acc_cov"):
            if bool(r.get("has_acc")) and isinstance(r.get("acc_path"), str) and r["acc_path"]:
                path = Path(r["acc_path"])
                if not path.is_absolute() and project_root is not None:
                    path = Path(project_root) / path
                feats.update(acc_features(np.load(path)))
            else:
                feats.update(_zero_acc_features())
        if variant == "ppg_acc_cov":
            feats.update(covariate_features(r))
        rows.append(feats)

    matrix = pd.DataFrame(rows).fillna(0.0)
    return matrix.to_numpy(dtype=np.float32), list(matrix.columns)
