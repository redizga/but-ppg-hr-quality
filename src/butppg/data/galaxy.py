"""GalaxyPPG external-validation ETL (assignment section 8).

Turns the raw GalaxyPPG release (Zenodo DOI 10.5281/zenodo.14635823) into the
same registry + processed-window format as BUT PPG, so the models trained on
BUT PPG can be run on it unchanged (via ``orch predict``) and, if needed, its
SIGMA / OpenTSLM marts built with the usual mart builders.

Raw layout (per the data paper):

    <raw>/Data/P01/GalaxyWatch/*PPG*.csv      # PPG 25 Hz  (reflective sensor)
    <raw>/Data/P01/GalaxyWatch/*ACC*.csv      # ACC 25 Hz  (x,y,z)
    <raw>/Data/P01/Polar H10/*ECG*.csv        # ECG 130 Hz (reference HR source)
    <raw>/Data/P01/*Event*.csv                # activity ENTER/EXIT timestamps

Key GalaxyPPG specifics handled here:

* the Galaxy Watch PPG is a **reflective** sensor -> the raw signal is inverted
  before use (assignment requirement);
* reference HR is derived **from the Polar H10 ECG** (R-peak detection), never
  from PPG (leakage guard, same as BUT PPG's ECG-only-as-target rule);
* each 10 s window is labelled with the activity it falls in (from the event
  log), so section 8's per-activity metrics can be computed.

Column names vary slightly between releases, so signal CSVs are parsed
defensively (find the timestamp column + value column(s) by common names). If a
release differs, adjust :data:`COLS` — everything else stays.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from butppg.data.marts import resample_window
from butppg.data.registry import save_registry

# Protocol activities (data paper, phases 1-2). Motion ones matter most (8).
ACTIVITIES = [
    "Baseline", "SSST", "Meditation", "ScreenReading", "TSST",
    "KeyboardTyping", "MobileTyping", "Standing", "Walking", "Jogging", "Running",
]
MOTION_ACTIVITIES = {"Walking", "Jogging", "Running"}

PPG_FS = 25.0
ACC_FS = 25.0
ECG_FS = 130.0
WINDOW_S = 10.0
TARGET_PPG_FS = 30.0  # match BUT PPG (300-point windows for the registry-based models)

# candidate column names (case-insensitive substring match), tune per release
COLS = {
    "time": ("timestamp", "time", "unix", "ts"),
    "ppg": ("ppg", "green", "value"),
    "ecg": ("ecg", "value", "mv"),
    "acc": ("x", "y", "z"),
    "event": ("event", "session", "status", "activity", "marker"),
}


def _find(paths_glob: str) -> str | None:
    hits = sorted(glob.glob(paths_glob))
    return hits[0] if hits else None


def _pick_col(df: pd.DataFrame, keys: tuple[str, ...]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for k in keys:
        for lc, orig in low.items():
            if k in lc:
                return orig
    return None


def _read_series(path: str, value_keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_seconds, values) from a signal CSV."""
    df = pd.read_csv(path)
    tcol = _pick_col(df, COLS["time"])
    vcol = _pick_col(df, value_keys)
    if vcol is None:
        # last numeric column as a fallback
        num = df.select_dtypes("number").columns
        vcol = num[-1]
    vals = pd.to_numeric(df[vcol], errors="coerce").to_numpy(dtype=float)
    if tcol is not None:
        t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
        t = t - np.nanmin(t)
        if np.nanmax(t) > 1e6:  # milliseconds -> seconds
            t = t / 1000.0
    else:
        t = np.arange(len(vals)) / PPG_FS
    return t, vals


def ecg_hr_bpm(ecg: np.ndarray, fs: float = ECG_FS, hr_range=(30, 220)) -> float:
    """Reference HR (bpm) from an ECG window via R-peak detection.

    Uses neurokit2 when available (robust), else a scipy bandpass + find_peaks
    fallback. Returns NaN if too few beats are found to be trustworthy.
    """
    ecg = np.asarray(ecg, dtype=float)
    ecg = ecg[np.isfinite(ecg)]
    if len(ecg) < int(fs * 2):
        return float("nan")
    try:  # preferred
        import neurokit2 as nk

        _, info = nk.ecg_peaks(nk.ecg_clean(ecg, sampling_rate=fs), sampling_rate=fs)
        peaks = np.asarray(info["ECG_R_Peaks"])
    except Exception:  # scipy fallback
        from scipy.signal import butter, filtfilt, find_peaks

        b, a = butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
        filt = filtfilt(b, a, ecg)
        peaks, _ = find_peaks(filt, distance=int(fs * 60 / hr_range[1]), height=np.std(filt))
    if len(peaks) < 3:
        return float("nan")
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.median(rr)
    return float(hr) if hr_range[0] <= hr <= hr_range[1] else float("nan")


def _activity_of(t0: float, events: list[tuple[float, str]]) -> str:
    """Activity active at window-start time ``t0`` (last ENTER before t0)."""
    current = "unknown"
    for t, name in events:
        if t <= t0:
            current = name
        else:
            break
    return current


def _parse_events(path: str | None) -> list[tuple[float, str]]:
    if not path:
        return []
    df = pd.read_csv(path)
    tcol = _pick_col(df, COLS["time"])
    ncol = _pick_col(df, COLS["event"])
    if tcol is None or ncol is None:
        return []
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
    t = t - np.nanmin(t)
    if np.nanmax(t) > 1e6:
        t = t / 1000.0
    out = []
    for ti, ni in zip(t, df[ncol].astype(str)):
        name = ni.strip()
        # normalise "ENTER Walking" / "Walking_ENTER" -> "Walking"
        for tok in ("ENTER", "EXIT", "enter", "exit", ":", "_"):
            name = name.replace(tok, " ")
        name = name.strip().title().replace(" ", "")
        if name:
            out.append((float(ti), name))
    return sorted(out)


def build_galaxy_registry(
    raw_dir: str | Path,
    out_dir: str | Path,
    window_s: float = WINDOW_S,
    invert_ppg: bool = True,
) -> Path:
    """Build ``<out_dir>/registry.csv`` + 30 Hz PPG windows from raw GalaxyPPG.

    One row per 10 s window with: ``record_id, subject_id, activity, hr_ref``
    (from ECG), ``quality_label`` (placeholder 1 — GalaxyPPG has no quality
    label; only HR is validated in section 8), ``ppg_path`` (inverted, resampled
    to 30 Hz / 300 points), ``has_acc``/``acc_path``, ``measurement_site='wrist'``.
    """
    raw, out = Path(raw_dir), Path(out_dir)
    win_dir = out / "ppg"
    win_dir.mkdir(parents=True, exist_ok=True)
    n_ppg = int(window_s * PPG_FS)
    n_ecg = int(window_s * ECG_FS)

    rows: list[dict] = []
    participants = sorted(p for p in (raw / "Data").glob("P*") if p.is_dir()) if (raw / "Data").exists() else sorted(raw.glob("P*"))
    for pdir in participants:
        subject_id = pdir.name
        ppg_csv = _find(str(pdir / "GalaxyWatch" / "*PPG*.csv")) or _find(str(pdir / "**" / "*PPG*.csv"))
        ecg_csv = _find(str(pdir / "Polar H10" / "*ECG*.csv")) or _find(str(pdir / "**" / "*ECG*.csv"))
        ev_csv = _find(str(pdir / "*Event*.csv")) or _find(str(pdir / "**" / "*Event*.csv"))
        if not ppg_csv or not ecg_csv:
            continue
        _, ppg = _read_series(ppg_csv, COLS["ppg"])
        _, ecg = _read_series(ecg_csv, COLS["ecg"])
        events = _parse_events(ev_csv)

        n_windows = min(len(ppg) // n_ppg, len(ecg) // n_ecg)
        for w in range(n_windows):
            t0 = w * window_s
            ppg_w = ppg[w * n_ppg:(w + 1) * n_ppg].astype(np.float32)
            ecg_w = ecg[w * n_ecg:(w + 1) * n_ecg]
            if np.isnan(ppg_w).any() or len(ppg_w) < n_ppg:
                continue
            hr = ecg_hr_bpm(ecg_w)
            if not np.isfinite(hr):
                continue
            if invert_ppg:
                ppg_w = -ppg_w  # reflective sensor
            ppg_30 = resample_window(ppg_w, PPG_FS, TARGET_PPG_FS).astype(np.float32)
            record_id = f"{subject_id}_{w:04d}"
            ppg_path = win_dir / f"{record_id}.npy"
            np.save(ppg_path, ppg_30)
            rows.append({
                "record_id": record_id,
                "subject_id": subject_id,
                "activity": _activity_of(t0, events),
                "quality_label": 1,          # placeholder; only HR is validated on GalaxyPPG
                "hr_ref": float(hr),
                "has_acc": False,
                "ppg_path": str(ppg_path),
                "acc_path": "",
                "measurement_site": "wrist",
            })

    registry = pd.DataFrame(rows)
    reg_path = save_registry(registry, out / "registry.csv")
    print(
        f"[galaxy] registry -> {reg_path}  |  windows: {len(registry)}  "
        f"subjects: {registry['subject_id'].nunique() if len(registry) else 0}  "
        f"activities: {sorted(registry['activity'].unique()) if len(registry) else []}"
    )
    return reg_path
