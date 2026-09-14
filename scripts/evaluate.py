#!/usr/bin/env python
"""Evaluate saved predictions on the fixed test set (all models or a selection).

Epic E2 (Budilov).

Usage
-----
    python scripts/evaluate.py --all
    python scripts/evaluate.py artifacts/predictions/sigma_ppg_quality.csv
    python scripts/evaluate.py --pipeline artifacts/predictions/sigma_ppg_quality.csv artifacts/predictions/sigma_ppg_hr.csv

Every evaluated prediction file gets its metrics written to
``artifacts/metrics/<file_stem>.json`` (the project's committed metrics dir,
see ``paths.py``) in addition to a summary table printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from butppg.evaluation.evaluate import evaluate_pipeline, evaluate_prediction_file  # noqa: E402
from butppg.paths import METRICS_DIR, PREDICTIONS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="specific prediction CSV(s) to evaluate")
    parser.add_argument("--all", action="store_true", help="evaluate every CSV under artifacts/predictions/")
    parser.add_argument(
        "--pipeline",
        nargs=2,
        metavar=("QUALITY_CSV", "HR_CSV"),
        help="score the end-to-end quality-gate -> HR scenario (assignment section 2B)",
    )
    return parser.parse_args()


def _format_metrics_line(result: dict) -> str:
    task = result["task"]
    m = result["metrics"]
    if m is None:
        return (
            f"{result['file']}  [{task}, n={result['n_records']}]  "
            f"no scorable predictions (all {result['n_invalid_responses']} responses invalid)"
        )
    if task == "quality":
        auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"
        return (
            f"{result['file']}  [{task}, n={result['n_records']}]  "
            f"macro_f1={m['macro_f1']:.4f}  accuracy={m['accuracy']:.4f}  roc_auc={auc}"
        )
    return (
        f"{result['file']}  [{task}, n={result['n_records']}]  "
        f"mae={m['mae']:.4f}  rmse={m['rmse']:.4f}"
    )


def evaluate_one(path: Path) -> dict:
    result = evaluate_prediction_file(path)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"{path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(_format_metrics_line(result))
    print(f"  -> {out_path}")
    return result


def main() -> None:
    args = parse_args()

    if args.pipeline:
        quality_csv, hr_csv = args.pipeline
        result = evaluate_pipeline(quality_csv, hr_csv)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = METRICS_DIR / "pipeline_quality_to_hr.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(
            f"pipeline: accepted={result['accepted_fraction']:.4f}  "
            f"false_reject={result['false_reject_rate']}  "
            f"false_accept={result['false_accept_rate']}"
        )
        print(f"  -> {out_path}")
        return

    targets: list[Path] = [Path(p) for p in args.files]
    if args.all:
        if not PREDICTIONS_DIR.exists():
            print(f"No predictions directory at {PREDICTIONS_DIR} — nothing to evaluate.")
            return
        targets = sorted(PREDICTIONS_DIR.glob("*.csv"))

    if not targets:
        print("Nothing to evaluate: pass file(s), --all, or --pipeline QUALITY_CSV HR_CSV.")
        return

    for path in targets:
        evaluate_one(path)


if __name__ == "__main__":
    main()
