"""End-to-end orchestrator smoke tests on a tiny synthetic registry.

Covers the two halves the ``orch`` CLI wires together: mart building (both
model shapes) and a full trivial-baseline training run through the shared
finalize path. Uses absolute ``ppg_path`` so it does not depend on the project
data layout, and cleans up any run directory it creates.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest


def _make_registry(tmp_path):
    ppg_dir = tmp_path / "ppg"
    ppg_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    subjects = [f"{100 + i}" for i in range(6)]
    for s in subjects:
        for r in range(4):
            rid = f"{s}{r:02d}"
            t = np.arange(300) / 30.0
            hr = float(rng.uniform(55, 110))
            ppg = (np.sin(2 * np.pi * (hr / 60.0) * t) + 0.1 * rng.standard_normal(300)).astype("float32")
            np.save(ppg_dir / f"{rid}.npy", ppg)
            rows.append(
                {
                    "record_id": rid, "subject_id": s, "dataset": "but_ppg",
                    "quality_label": int(rng.random() < 0.6), "hr_ref": hr, "has_acc": False,
                    "ppg_path": str(ppg_dir / f"{rid}.npy"), "acc_path": "",
                    "measurement_site": "finger", "sex": "M", "age": 30.0,
                    "height": 180.0, "weight": 75.0, "activity": "",
                }
            )
    reg = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(reg, index=False)
    return reg, subjects


def _make_split(tmp_path, subjects):
    from butppg.data.splits import save_split

    path = tmp_path / "split.json"
    save_split(subjects[:4], subjects[4:5], subjects[5:6], path, seed=42)
    return path


def test_build_sigma_mart_shapes(tmp_path):
    from butppg.data.marts_build import build_sigma_mart

    reg, subjects = _make_registry(tmp_path)
    split = _make_split(tmp_path, subjects)
    out = tmp_path / "marts" / "sigma_ppg"
    manifest = build_sigma_mart(reg, split, out, task="quality", target_fs=50.0)

    assert manifest["target_fs"] == 50.0
    x_files = list((out / "quality" / "train").glob("*_x.npy"))
    assert x_files, "no per-subject arrays written"
    x = np.load(x_files[0])
    assert x.ndim == 3 and x.shape[1] == 1 and x.shape[2] == 500  # (N, 1, 10s@50Hz)


def test_build_opentslm_mart_records(tmp_path):
    from butppg.data.marts_build import build_opentslm_mart

    reg, subjects = _make_registry(tmp_path)
    split = _make_split(tmp_path, subjects)
    out = tmp_path / "marts" / "opentslm"
    build_opentslm_mart(reg, split, out, task="quality", acc_mode="none")

    line = (out / "quality" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    assert set(rec) >= {"record_id", "subject_id", "pre_prompt", "time_series", "post_prompt", "answer"}
    assert rec["answer"] in ("good", "bad")
    assert len(rec["time_series"][0]["series"]) == 300


def test_trivial_training_run(tmp_path):
    from butppg.orchestrator.runs import create_run, load_run
    from butppg.training.dispatch import run_training

    reg, subjects = _make_registry(tmp_path)
    split = _make_split(tmp_path, subjects)
    run = create_run("trivial", "quality", config={})
    try:
        run_training(run, {"registry": str(reg), "split": str(split), "device": "cpu"})
        reloaded = load_run(run.run_id)
        assert reloaded.status == "finished"
        assert "macro_f1" in reloaded.metrics
        assert (reloaded.predictions_dir / "test_predictions.csv").exists()
    finally:
        shutil.rmtree(run.dir, ignore_errors=True)
