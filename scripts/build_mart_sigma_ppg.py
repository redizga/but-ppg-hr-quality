#!/usr/bin/env python
"""Build the SIGMA-PPG input mart from the E1 registry (only-PPG mode).

Epic E6 (Golikov). This is the data-shaping step between "registry exists"
(E1) and "SIGMA-PPG can fine-tune" (train_sigma.py) — it does not touch the
SIGMA-PPG codebase itself.

Output format is deliberately modelled on the ``downstream/bidmc`` reference
implementation in the upstream SIGMA-PPG repo (github.com/ZonghengGuo/SigmaPPG),
since the repo's own ``downstream/butppg/`` module referenced by its
``downstream_main.py`` is NOT present in the public repo (verified 2026-09 —
only ``downstream/bidmc`` ships; ``butppg``/``wesad``/``ppgbp``/... are
imported but missing). We therefore write our own ``downstream/butppg/``
module against SIGMA-PPG's ``select_model`` / ``NeuralTransformer`` API later
in E6; this script only produces the per-subject ``.npy`` files that module
will load — same layout ``PreprocessBIDMC.preprocess_save`` produces:

    <out>/<task>/<split>/<subject_id>_x.npy       float32 (N, 1, L)
    <out>/<task>/<split>/<subject_id>_y_quality.npy   int64 (N,)     [task=quality]
    <out>/<task>/<split>/<subject_id>_y_hr.npy        float32 (N,)   [task=hr]
    <out>/<task>/manifest.json                     resample rate, normalization, seed, source files

Each window is independently resampled from 30 Hz (E1's ``ppg_len=300``) to
``--target-fs`` (default 50 Hz, SIGMA-PPG's own pretraining/default rsfreq)
and normalized (default z-score, matching the BIDMC downstream reference;
pass --normalize minmax for SIGMA-PPG's own pretraining convention instead).

Usage
-----
    python scripts/build_mart_sigma_ppg.py --task quality \\
        --registry data/processed/registry.csv \\
        --split splits/but_ppg_60_20_20.json \\
        --out artifacts/data_marts/sigma_ppg

    python scripts/build_mart_sigma_ppg.py --task hr \\
        --registry data/processed/registry.csv \\
        --split splits/but_ppg_60_20_20.json \\
        --out artifacts/data_marts/sigma_ppg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from butppg.data.marts import (  # noqa: E402
    load_ppg_window,
    load_registry_with_split,
    minmax_to_pm1,
    resample_window,
    zscore,
)
from butppg.paths import PROJECT_ROOT  # noqa: E402
from butppg.utils.seed import seed_everything  # noqa: E402

SOURCE_FS = 30.0  # BUT PPG, fixed by E1 (configs/default.yaml: window.ppg_fs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", required=True, help="path to registry.csv (E1 output)")
    p.add_argument("--split", required=True, help="path to the subject-wise split JSON (E1 output)")
    p.add_argument("--out", required=True, help="output directory for the mart")
    p.add_argument("--task", choices=["quality", "hr"], required=True)
    p.add_argument("--target-fs", type=float, default=50.0, help="SIGMA-PPG resample rate (default: 50 Hz)")
    p.add_argument("--normalize", choices=["zscore", "minmax"], default="zscore")
    p.add_argument(
        "--good-quality-only",
        action="store_true",
        default=True,
        help="task=hr only: train HR only on quality_label==1 windows (assignment section 2B). Default: on.",
    )
    p.add_argument("--no-good-quality-only", dest="good_quality_only", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_split_subset(df, out_dir: Path, task: str, target_fs: float, normalize: str) -> dict:
    stats = {"subjects": 0, "windows": 0}
    for subject_id, group in df.groupby("subject_id"):
        windows = []
        labels = []
        for _, row in group.iterrows():
            x = load_ppg_window(row["ppg_path"], project_root=PROJECT_ROOT)
            x = resample_window(x, orig_fs=SOURCE_FS, target_fs=target_fs)
            x = zscore(x)[0] if normalize == "zscore" else minmax_to_pm1(x)
            windows.append(x)
            labels.append(row["quality_label"] if task == "quality" else row["hr_ref"])

        if not windows:
            continue

        # (N, L) -> (N, 1, L), matching SIGMA-PPG's [Batch, Channel, Length] input.
        x_arr = np.stack(windows)[:, np.newaxis, :].astype(np.float32)
        y_arr = np.asarray(labels, dtype=np.int64 if task == "quality" else np.float32)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{subject_id}_x.npy", x_arr)
        np.save(out_dir / f"{subject_id}_y_{task}.npy", y_arr)

        stats["subjects"] += 1
        stats["windows"] += len(windows)
    return stats


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    registry = load_registry_with_split(args.registry, args.split)

    if args.task == "hr" and args.good_quality_only:
        before = len(registry)
        registry = registry[registry["quality_label"] == 1]
        print(f"[hr] filtered to quality_label==1: {before} -> {len(registry)} windows")

    out_root = Path(args.out) / args.task
    manifest = {
        "task": args.task,
        "target_fs": args.target_fs,
        "source_fs": SOURCE_FS,
        "normalize": args.normalize,
        "good_quality_only": args.good_quality_only if args.task == "hr" else None,
        "seed": args.seed,
        "registry": str(args.registry),
        "split_file": str(args.split),
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        subset = registry[registry["split"] == split_name]
        split_dir = out_root / split_name
        stats = build_split_subset(subset, split_dir, args.task, args.target_fs, args.normalize)
        manifest["splits"][split_name] = stats
        print(f"[{args.task}/{split_name}] {stats['subjects']} subjects, {stats['windows']} windows -> {split_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
