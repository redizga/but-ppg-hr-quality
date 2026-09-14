"""BUT PPG preparation = RAW ingest + PROCESS, kept as one convenience call.

The ETL is now two independent stages (so re-processing never re-downloads):

* ``butppg.data.raw.ingest_raw``   — download raw signals to ``data/raw`` (slow, once);
* ``butppg.data.process.process_records`` — raw -> processed windows + registry (fast).

``prepare_but_ppg`` runs both, preserving the old one-call behaviour used by
``orch mart --prepare``. Prefer the explicit ``orch ingest`` / ``orch process``
commands when you want to re-run only the processing half.
"""

from __future__ import annotations

from pathlib import Path

# Re-exported for backward compatibility (tests and older imports).
from butppg.data.process import PPG_CHANNEL, extract_ppg as _extract_ppg, process_records
from butppg.data.raw import PPG_LEN, ingest_raw

__all__ = ["prepare_but_ppg", "_extract_ppg", "PPG_CHANNEL", "PPG_LEN"]


def prepare_but_ppg(
    out_dir: str | Path = "data/processed",
    limit: int | None = None,
    include_acc: bool = True,
    skip_existing: bool = True,
    raw_dir: str | Path | None = None,
) -> Path:
    """Ingest the raw layer then process it into ``<out_dir>/registry.csv``.

    ``raw_dir`` defaults to a ``raw`` sibling of ``out_dir`` (``data/raw`` when
    ``out_dir`` is ``data/processed``). Returns the registry path.
    """
    out_dir = Path(out_dir)
    raw_dir = Path(raw_dir) if raw_dir is not None else out_dir.parent / "raw"
    ingest_raw(raw_dir=raw_dir, limit=limit, include_acc=include_acc, skip_existing=skip_existing)
    return process_records(raw_dir=raw_dir, out_dir=out_dir)
