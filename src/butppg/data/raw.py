"""RAW layer: download BUT PPG v2.0.0 signals to local disk, untouched.

This is the slow, network-bound half of the ETL, split out so it runs once and
is cached. Everything here writes the signal *as PhysioNet serves it* — full
multichannel PPG (all RGB channels), raw ACC, and the two annotation CSVs — so
that re-processing (e.g. changing which PPG channel we keep) never re-downloads.

Layout::

    data/raw/
      quality-hr-ann.csv        reference HR + quality label per record
      subject-info.csv          per-subject covariates
      ppg/<record_id>.npz       { p_signal, sig_name }  — full raw PPG, all channels
      acc/<record_id>.npy       (3, T) raw accelerometer (where present)

BUT PPG is open access (CC-BY 4.0, DOI 10.13026/tn53-8153) — no credentials.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PN_DIR_ROOT = "butppg/2.0.0"
BASE_URL = "https://physionet.org/files/butppg/2.0.0"
PPG_LEN = 300  # 10 s @ 30 Hz
ANNOTATION_FILES = ("quality-hr-ann.csv", "subject-info.csv")


def fetch_csv(name: str) -> pd.DataFrame:
    """Read one of BUT PPG's annotation CSVs straight from PhysioNet."""
    with urllib.request.urlopen(f"{BASE_URL}/{name}") as resp:
        data = resp.read()
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")


def _download_acc(record_id: str) -> np.ndarray | None:
    import wfdb

    try:
        rec = wfdb.rdrecord(f"{record_id}_ACC", pn_dir=f"{PN_DIR_ROOT}/{record_id}")
    except Exception:
        return None
    acc = np.asarray(rec.p_signal, dtype=np.float32)
    if acc.ndim != 2 or 3 not in acc.shape:
        return None
    return acc.T if acc.shape[1] == 3 else acc  # -> (3, T)


def ingest_raw(
    raw_dir: str | Path = "data/raw",
    limit: int | None = None,
    include_acc: bool = True,
    skip_existing: bool = True,
) -> Path:
    """Download raw BUT PPG signals + annotations into ``raw_dir``.

    Resumable: already-downloaded records are skipped unless
    ``skip_existing=False``; a single bad record is logged and skipped, never
    fatal. Returns the raw dir. Network-bound — run once, then ``process``.
    """
    import wfdb

    raw = Path(raw_dir)
    ppg_dir = raw / "ppg"
    acc_dir = raw / "acc"
    ppg_dir.mkdir(parents=True, exist_ok=True)

    # cache the annotation CSVs locally so `process` needs no network at all
    for name in ANNOTATION_FILES:
        dest = raw / name
        if not (skip_existing and dest.exists()):
            with urllib.request.urlopen(f"{BASE_URL}/{name}") as resp:
                dest.write_bytes(resp.read())

    ann = pd.read_csv(raw / "quality-hr-ann.csv", encoding="utf-8-sig")
    ids = ann["ID"].astype(str).tolist()
    if limit is not None:
        ids = ids[:limit]

    failures: list[tuple[str, str]] = []
    for record_id in tqdm(ids, desc="ingest BUT PPG"):
        ppg_out = ppg_dir / f"{record_id}.npz"
        if not (skip_existing and ppg_out.exists()):
            try:
                rec = wfdb.rdrecord(f"{record_id}_PPG", pn_dir=f"{PN_DIR_ROOT}/{record_id}")
                np.savez(
                    ppg_out,
                    p_signal=np.asarray(rec.p_signal, dtype=np.float32),
                    sig_name=np.array([str(n) for n in rec.sig_name], dtype=object),
                )
            except Exception as e:  # noqa: BLE001 - one bad record must not abort the run
                failures.append((record_id, str(e)))
                continue

        if include_acc:
            acc_out = acc_dir / f"{record_id}.npy"
            if not (skip_existing and acc_out.exists()):
                acc = _download_acc(record_id)
                if acc is not None:
                    acc_dir.mkdir(parents=True, exist_ok=True)
                    np.save(acc_out, acc)

    n_ppg = len(list(ppg_dir.glob("*.npz")))
    n_acc = len(list(acc_dir.glob("*.npy"))) if acc_dir.exists() else 0
    print(f"[ingest] raw PPG records: {n_ppg}  |  raw ACC records: {n_acc}  -> {raw}")
    if failures:
        print(f"[ingest] WARNING: {len(failures)} record(s) failed to download (e.g. {failures[0]})")
    return raw
