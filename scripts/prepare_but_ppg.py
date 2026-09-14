#!/usr/bin/env python
"""Download and preprocess BUT PPG v2.0.0, build the record registry.

Epic E1 (Golikov). Downloads directly from PhysioNet — BUT PPG is open
access (CC-BY 4.0, DOI 10.13026/tn53-8153), no credentials needed. Writes
one ``.npy`` PPG window per record (plus ACC where available) and the
registry (``registry.py``) that ties everything together.

Real-format notes (found by running this against real data, see
``docs/review/done_real_data_smoketest.md``):

* BUT PPG's PPG ``.hea`` headers are non-standard: ``wfdb`` parses each
  record as ``n_sig=300, sig_len=1`` (one "signal" per sample) instead of
  the expected ``n_sig=1, sig_len=300``. ``rec.p_signal`` comes back shape
  ``(1, 300)`` — ``.flatten()`` recovers the real 300-point time series.
* ACC, by contrast, IS a normal WFDB record: ``n_sig=3, sig_len=1000``
  (10 s @ 100 Hz), ``p_signal`` shape ``(1000, 3)`` — transposed here to
  ``(3, 1000)`` to match the (channels, time) convention used elsewhere in
  this repo (e.g. ``opentslm_parsing``'s ACC-magnitude helper).
* ACC exists only for a subset of records; this script does not assume a
  fixed cutoff (the dataset docs mention "112001+", but that's not verified
  as a hard rule) — it just tries per record and sets ``has_acc`` from
  whether that succeeded.
* record_id/subject_id look like plain numbers ("100001", "100") — always
  written/read as strings (``registry.save_registry`` enforces the dtype on
  load) to avoid a pandas int64 auto-coercion bug found while testing this.

Usage
-----
    python scripts/prepare_but_ppg.py --out-dir data/processed
    python scripts/prepare_but_ppg.py --out-dir data/processed --limit 50   # quick smoke test

Re-running is safe and resumable: existing ``.npy`` files are reused unless
``--no-skip-existing`` is passed, so an interrupted run can just be restarted.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from butppg.data.registry import save_registry  # noqa: E402

PN_DIR_ROOT = "butppg/2.0.0"
BASE_URL = "https://physionet.org/files/butppg/2.0.0"
PPG_LEN = 300  # 10s @ 30 Hz, assignment section 3


def fetch_csv(name: str) -> pd.DataFrame:
    """Download one of BUT PPG's two small annotation CSVs."""
    with urllib.request.urlopen(f"{BASE_URL}/{name}") as resp:
        data = resp.read()
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")


def download_ppg(record_id: str) -> np.ndarray:
    rec = wfdb.rdrecord(f"{record_id}_PPG", pn_dir=f"{PN_DIR_ROOT}/{record_id}")
    ppg = rec.p_signal.flatten().astype(np.float32)  # see module docstring: (1, 300) -> (300,)
    if ppg.shape != (PPG_LEN,):
        raise ValueError(f"unexpected PPG shape {ppg.shape} (expected ({PPG_LEN},))")
    return ppg


def download_acc(record_id: str) -> np.ndarray | None:
    """Best effort: returns (3, T) or None if this record has no ACC signal."""
    try:
        rec = wfdb.rdrecord(f"{record_id}_ACC", pn_dir=f"{PN_DIR_ROOT}/{record_id}")
    except Exception:
        return None
    acc = rec.p_signal.astype(np.float32)
    if acc.ndim != 2 or 3 not in acc.shape:
        return None
    return acc.T if acc.shape[1] == 3 else acc  # normalize to (3, T)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/processed", help="where to write ppg/, acc/ and registry.csv")
    p.add_argument("--limit", type=int, default=None, help="only process the first N records (smoke testing)")
    p.add_argument("--include-acc", action="store_true", default=True)
    p.add_argument("--no-acc", dest="include_acc", action="store_false")
    p.add_argument("--skip-existing", action="store_true", default=True, help="reuse .npy files already on disk")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ppg_dir = out_dir / "ppg"
    acc_dir = out_dir / "acc"
    ppg_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching annotation files from PhysioNet...")
    quality_hr = fetch_csv("quality-hr-ann.csv")
    subject_info = fetch_csv("subject-info.csv")
    quality_hr["record_id"] = quality_hr["ID"].astype(str)
    quality_hr["subject_id"] = quality_hr["record_id"].str[:3]
    subject_info["record_id"] = subject_info["ID"].astype(str)
    subject_info = subject_info.set_index("record_id")

    records = quality_hr if args.limit is None else quality_hr.head(args.limit)
    print(f"Downloading {len(records)} PPG record(s) (ACC: {'on' if args.include_acc else 'off'})...")

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for _, row in tqdm(records.iterrows(), total=len(records)):
        record_id, subject_id = row["record_id"], row["subject_id"]
        ppg_path = ppg_dir / f"{record_id}.npy"

        if args.skip_existing and ppg_path.exists():
            pass
        else:
            try:
                np.save(ppg_path, download_ppg(record_id))
            except Exception as e:  # noqa: BLE001 - one bad record must not abort a multi-hour run
                failures.append((record_id, str(e)))
                continue

        has_acc, acc_path = False, ""
        if args.include_acc:
            acc_out = acc_dir / f"{record_id}.npy"
            if args.skip_existing and acc_out.exists():
                has_acc, acc_path = True, str(acc_out)
            else:
                acc = download_acc(record_id)
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

    print(f"\nRegistry written: {registry_path}")
    print(f"  records: {len(registry)}  subjects: {registry['subject_id'].nunique()}")
    print(f"  has_acc: {int(registry['has_acc'].sum())} / {len(registry)}")
    print(f"  quality_label mean (fraction good): {registry['quality_label'].mean():.3f}")
    if failures:
        print(f"\nWARNING: {len(failures)} record(s) failed to download and were skipped:")
        for record_id, err in failures[:20]:
            print(f"  {record_id}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
