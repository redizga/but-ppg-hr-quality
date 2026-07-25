#!/usr/bin/env python
"""Prepare GalaxyPPG windows (invert reflective PPG; reference HR from ECG).

Epic E7 (Budilov). Stub — implemented in that epic. Kept as a stable entry
point so the README command surface and the pipeline wiring exist from day one.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="default", help="config name under configs/")
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    parser.parse_args()
    raise NotImplementedError("E7 (Budilov): implement prepare_galaxy_ppg.py")


if __name__ == "__main__":
    main()
