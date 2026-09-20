"""One place for the progress-bar look so every trainer shows the same thing.

All trainers (CNN, SIGMA-PPG, OpenTSLM) and the feature/data loaders wrap their
loops with :func:`pbar` so a run always prints a live bar. Settings live here so
the bars stay consistent; ``BUTPPG_NO_PROGRESS=1`` turns them off (handy for
tests / non-tty logs) without touching call sites.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

from tqdm.auto import tqdm


def _disabled() -> bool:
    return os.environ.get("BUTPPG_NO_PROGRESS", "") not in ("", "0", "false", "False")


def pbar(iterable: Iterable, desc: str, total: Optional[int] = None, leave: bool = True):
    """Wrap ``iterable`` in a tqdm bar with the shared house style.

    ``leave=False`` for inner (per-epoch batch) loops so they don't pile up;
    ``leave=True`` for the outer/only loop so the final state stays visible.
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        dynamic_ncols=True,
        disable=_disabled(),
        file=sys.stdout,
    )
