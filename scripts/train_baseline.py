#!/usr/bin/env python
"""Train feature-based baselines (LogReg / XGBoost) for quality and HR.

Epic E3 (Golikov). Stub — implemented in that epic. Kept as a stable entry
point so the README command surface and the pipeline wiring exist from day one.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="default", help="config name under configs/")
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    parser.parse_args()
    raise NotImplementedError("E3 (Golikov): implement train_baseline.py")


if __name__ == "__main__":
    main()
