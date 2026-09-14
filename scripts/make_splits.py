#!/usr/bin/env python
"""Create the subject-wise 60/20/20 split and verify no subject overlap.

Epic E1 (Golikov). Reads the registry built by ``prepare_but_ppg.py``,
splits by *subject* (never by record), balances on quality label and
measurement site by default (assignment section 3), and writes the result to
a committed JSON split file.

Usage
-----
    python scripts/make_splits.py \\
        --registry data/processed/registry.csv \\
        --out splits/but_ppg_60_20_20.json --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from butppg.data.registry import load_registry  # noqa: E402
from butppg.data.splits import make_subject_split, save_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", required=True, help="path to registry.csv (from prepare_but_ppg.py)")
    p.add_argument("--out", required=True, help="output path for the split JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", default="0.6,0.2,0.2", help="train,val,test (must sum to 1.0)")
    p.add_argument(
        "--balance-on",
        default="quality_label,measurement_site",
        help="comma-separated registry columns to balance subject folds on",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratios = tuple(float(x) for x in args.ratios.split(","))
    balance_on = tuple(c.strip() for c in args.balance_on.split(","))

    registry = load_registry(args.registry)
    print(f"Loaded registry: {len(registry)} records, {registry['subject_id'].nunique()} subjects")

    train, val, test = make_subject_split(registry, ratios=ratios, seed=args.seed, balance_on=balance_on)
    path = save_split(train, val, test, args.out, seed=args.seed, ratios=ratios)

    print(f"train: {len(train)} subjects, {registry['subject_id'].isin(train).sum()} records")
    print(f"val:   {len(val)} subjects, {registry['subject_id'].isin(val).sum()} records")
    print(f"test:  {len(test)} subjects, {registry['subject_id'].isin(test).sum()} records")
    for name, subjects in (("train", train), ("val", val), ("test", test)):
        subset = registry[registry["subject_id"].isin(subjects)]
        quality_rate = subset["quality_label"].mean()
        print(f"  {name}: quality_label mean={quality_rate:.3f}, mean HR={subset['hr_ref'].mean():.1f}")
    print(f"Split written: {path}")


if __name__ == "__main__":
    main()
