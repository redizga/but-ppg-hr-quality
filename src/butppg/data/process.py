"""PROCESS layer: turn the raw cache into processed windows + the registry.

The fast, local, network-free half of the ETL. Reads ``data/raw`` (produced by
``raw.ingest_raw``), extracts the analysis PPG channel, and writes the processed
300-sample windows plus the record registry. Re-run this freely — changing the
channel choice or any preprocessing costs seconds, not a re-download.

Channel choice: most BUT PPG records are 3-channel RGB smartphone signals
(p_signal (300, 3), sig_names PPG_R/PPG_G/PPG_B) — we keep the GREEN channel
(strongest pulsatile SNR); a few early records use a non-standard single-channel
header (1, 300) which we just flatten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from butppg.data.raw import PPG_LEN
from butppg.data.registry import save_registry

PPG_CHANNEL = "green"  # deliberate default; switch here for red/averaged studies


class _RawRec:
    """Minimal stand-in for a wfdb record, rebuilt from the raw ``.npz``."""

    def __init__(self, p_signal: np.ndarray, sig_name: list[str]):
        self.p_signal = p_signal
        self.sig_name = sig_name


def extract_ppg(rec) -> np.ndarray:
    """Extract the analysis PPG channel as a ``(300,)`` float32 window.

    Handles both on-disk shapes: 3-channel RGB -> green column; the legacy
    single-channel header -> flatten.
    """
    sig = np.asarray(rec.p_signal, dtype=np.float32)
    names = [str(n).upper() for n in rec.sig_name]
    n_ch = sig.shape[1] if sig.ndim == 2 else 1

    if sig.ndim == 2 and n_ch == len(names) and n_ch >= 2:
        green = next((i for i, nm in enumerate(names) if "PPG_G" in nm or "GREEN" in nm), None)
        if green is None and n_ch == 3:
            green = 1  # RGB channel order fallback if names are unexpected
        ppg = sig[:, green] if green is not None else sig.flatten()
    else:
        ppg = sig.flatten()  # legacy single-channel header (1, 300) -> (300,)

    ppg = ppg.astype(np.float32)
    if ppg.shape != (PPG_LEN,):
        raise ValueError(f"unexpected PPG shape {ppg.shape} (expected ({PPG_LEN},))")
    return ppg


def _load_raw_ppg(npz_path: Path) -> _RawRec:
    data = np.load(npz_path, allow_pickle=True)
    return _RawRec(data["p_signal"], list(data["sig_name"]))


def process_records(raw_dir: str | Path = "data/raw", out_dir: str | Path = "data/processed") -> Path:
    """Build ``<out_dir>/registry.csv`` + processed PPG windows from the raw cache.

    Reads annotations and signals only from ``raw_dir`` — no network. Records
    that were never ingested (missing raw ``.npz``) are skipped. Returns the
    registry path.
    """
    raw = Path(raw_dir)
    out = Path(out_dir)
    ppg_out_dir = out / "ppg"
    ppg_out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = raw / "quality-hr-ann.csv"
    if not ann_path.exists():
        raise FileNotFoundError(f"raw annotations missing at {ann_path} — run `orch ingest` first")
    quality_hr = pd.read_csv(ann_path, encoding="utf-8-sig")
    subject_info = pd.read_csv(raw / "subject-info.csv", encoding="utf-8-sig")
    quality_hr["record_id"] = quality_hr["ID"].astype(str)
    quality_hr["subject_id"] = quality_hr["record_id"].str[:3]
    subject_info["record_id"] = subject_info["ID"].astype(str)
    subject_info = subject_info.set_index("record_id")

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for _, row in quality_hr.iterrows():
        record_id, subject_id = row["record_id"], row["subject_id"]
        raw_ppg = raw / "ppg" / f"{record_id}.npz"
        if not raw_ppg.exists():
            continue  # not ingested (or ingest failed) — skip silently
        try:
            ppg = extract_ppg(_load_raw_ppg(raw_ppg))
        except Exception as e:  # noqa: BLE001 - a bad record must not abort processing
            failures.append((record_id, str(e)))
            continue

        ppg_path = ppg_out_dir / f"{record_id}.npy"
        np.save(ppg_path, ppg)

        raw_acc = raw / "acc" / f"{record_id}.npy"
        has_acc = raw_acc.exists()
        acc_path = str(raw_acc) if has_acc else ""

        meta = subject_info.loc[record_id] if record_id in subject_info.index else None
        rows.append(
            {
                "record_id": record_id,
                "subject_id": subject_id,
                "dataset": "but_ppg",
                "quality_label": int(row["Quality"]),
                "hr_ref": float(row["HR"]),
                "has_acc": has_acc,
                "ppg_path": str(ppg_path),
                "acc_path": acc_path,
                "measurement_site": ("finger" if meta is not None and meta["Ear/finger"] == 1 else "ear"),
                "sex": meta["Gender"] if meta is not None else "",
                "age": meta["Age [years]"] if meta is not None else None,
                "height": meta["Height [cm]"] if meta is not None else None,
                "weight": meta["Weight [kg]"] if meta is not None else None,
                "activity": "",
            }
        )

    registry = pd.DataFrame(rows)
    registry_path = save_registry(registry, out / "registry.csv")
    print(
        f"[process] registry -> {registry_path}  |  records: {len(registry)}  "
        f"subjects: {registry['subject_id'].nunique() if len(registry) else 0}  "
        f"has_acc: {int(registry['has_acc'].sum()) if len(registry) else 0}"
    )
    if failures:
        print(f"[process] WARNING: {len(failures)} record(s) failed to process (e.g. {failures[0]})")
    return registry_path
