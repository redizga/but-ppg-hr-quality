"""ETL layer test: process reads a raw cache and builds the registry offline.

Verifies the raw -> processed split: given a mock ``data/raw`` (npz PPG with all
RGB channels + annotation CSVs), ``process_records`` extracts the green channel
and writes a registry — with no network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from butppg.data.process import process_records


def _make_raw(tmp_path):
    raw = tmp_path / "raw"
    (raw / "ppg").mkdir(parents=True)
    (raw / "acc").mkdir()
    ids = [f"{100 + s}{r:02d}" for s in range(6) for r in range(4)]
    pd.DataFrame(
        {"ID": ids, "Quality": [i % 2 for i in range(len(ids))], "HR": [70 + i for i in range(len(ids))]}
    ).to_csv(raw / "quality-hr-ann.csv", index=False)
    pd.DataFrame(
        {"ID": ids, "Gender": ["M"] * len(ids), "Age [years]": [30] * len(ids),
         "Height [cm]": [180] * len(ids), "Weight [kg]": [75] * len(ids), "Ear/finger": [1] * len(ids)}
    ).to_csv(raw / "subject-info.csv", index=False)
    for rid in ids:
        sig = np.zeros((300, 3), dtype=np.float32)
        sig[:, 1] = np.arange(300)  # green ramp
        np.savez(raw / "ppg" / f"{rid}.npz", p_signal=sig,
                 sig_name=np.array(["PPG_R", "PPG_G", "PPG_B"], dtype=object))
        np.save(raw / "acc" / f"{rid}.npy", np.zeros((3, 1000), dtype=np.float32))
    return raw, ids


def test_process_builds_registry_from_raw(tmp_path):
    raw, ids = _make_raw(tmp_path)
    reg_path = process_records(raw_dir=raw, out_dir=tmp_path / "processed")
    reg = pd.read_csv(reg_path, dtype={"record_id": str, "subject_id": str})

    assert len(reg) == len(ids)
    assert reg["subject_id"].nunique() == 6
    assert reg["has_acc"].all()
    # green channel extracted (not red/blue zeros)
    ppg = np.load(reg.iloc[0]["ppg_path"])
    assert ppg.shape == (300,)
    assert np.allclose(ppg, np.arange(300))


def test_process_without_raw_annotations_errors(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        process_records(raw_dir=tmp_path / "does-not-exist", out_dir=tmp_path / "out")
