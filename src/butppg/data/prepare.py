"""Download + preprocess BUT PPG v2.0.0 into processed windows + the registry.

Epic E1 (Budilov's verified script, ported to an importable function so the
orchestrator's ``mart`` command can run/refresh the registry on demand).
BUT PPG is open access (CC-BY 4.0, DOI 10.13026/tn53-8153) — no credentials.

Real-format notes (found against real data): PPG ``.hea`` headers are
non-standard (``wfdb`` reads n_sig=300/sig_len=1 -> ``.flatten()`` recovers the
300-point series); ACC is a normal record (3, 1000) transposed to (3, T);
record_id/subject_id are kept as strings to dodge pandas int64 coercion.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from butppg.data.registry import save_registry

PN_DIR_ROOT = "butppg/2.0.0"
BASE_URL = "https://physionet.org/files/butppg/2.0.0"
PPG_LEN = 300  # 10 s @ 30 Hz


def _fetch_csv(name: str) -> pd.DataFrame:
    with urllib.request.urlopen(f"{BASE_URL}/{name}") as resp:
        data = resp.read()
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")


def _download_ppg(record_id: str) -> np.ndarray:
    import wfdb

    rec = wfdb.rdrecord(f"{record_id}_PPG", pn_dir=f"{PN_DIR_ROOT}/{record_id}")
    ppg = rec.p_signal.flatten().astype(np.float32)  # (1, 300) -> (300,)
    if ppg.shape != (PPG_LEN,):
        raise ValueError(f"unexpected PPG shape {ppg.shape} (expected ({PPG_LEN},))")
    return ppg


def _download_acc(record_id: str) -> np.ndarray | None:
    import wfdb

    try:
        rec = wfdb.rdrecord(f"{record_id}_ACC", pn_dir=f"{PN_DIR_ROOT}/{record_id}")
    except Exception:
        return None
    acc = rec.p_signal.astype(np.float32)
    if acc.ndim != 2 or 3 not in acc.shape:
        return None
    return acc.T if acc.shape[1] == 3 else acc  # -> (3, T)


def prepare_but_ppg(
    out_dir: str | Path = "data/processed",
    limit: int | None = None,
    include_acc: bool = True,
    skip_existing: bool = True,
) -> Path:
    """Download/preprocess BUT PPG and write ``<out_dir>/registry.csv``.

    Returns the path to the registry CSV. Resumable: existing ``.npy`` files are
    reused unless ``skip_existing=False``; a single bad record is skipped, not
    fatal, so a multi-hour run survives one corrupt download.
    """
    out_dir = Path(out_dir)
    ppg_dir = out_dir / "ppg"
    acc_dir = out_dir / "acc"
    ppg_dir.mkdir(parents=True, exist_ok=True)

    quality_hr = _fetch_csv("quality-hr-ann.csv")
    subject_info = _fetch_csv("subject-info.csv")
    quality_hr["record_id"] = quality_hr["ID"].astype(str)
    quality_hr["subject_id"] = quality_hr["record_id"].str[:3]
    subject_info["record_id"] = subject_info["ID"].astype(str)
    subject_info = subject_info.set_index("record_id")

    records = quality_hr if limit is None else quality_hr.head(limit)

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for _, row in tqdm(records.iterrows(), total=len(records), desc="BUT PPG"):
        record_id, subject_id = row["record_id"], row["subject_id"]
        ppg_path = ppg_dir / f"{record_id}.npy"
        if not (skip_existing and ppg_path.exists()):
            try:
                np.save(ppg_path, _download_ppg(record_id))
            except Exception as e:  # noqa: BLE001 - one bad record must not abort the run
                failures.append((record_id, str(e)))
                continue

        has_acc, acc_path = False, ""
        if include_acc:
            acc_out = acc_dir / f"{record_id}.npy"
            if skip_existing and acc_out.exists():
                has_acc, acc_path = True, str(acc_out)
            else:
                acc = _download_acc(record_id)
                if acc is not None:
                    acc_dir.mkdir(parents=True, exist_ok=True)
                    np.save(acc_out, acc)
                    has_acc, acc_path = True, str(acc_out)

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
    registry_path = save_registry(registry, out_dir / "registry.csv")
    if failures:
        print(f"WARNING: {len(failures)} record(s) failed and were skipped (e.g. {failures[0]})")
    return registry_path
