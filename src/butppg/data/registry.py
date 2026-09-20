"""Record registry — the flat table every model reads from.

Epic E1 (Golikov). ``save_registry``/``load_registry``/``validate_registry``
are implemented here already, on the same reasoning as
``butppg.data.splits``: the schema (``RECORD_COLUMNS``) was fixed at E0, so
writing both the read and write side now — rather than leaving E1 to
reinvent a loader against a schema it didn't write — removes a class of
"works on my end, breaks on the real file" bugs. ``build_registry`` (the
actual BUT PPG parsing/downloading) is orchestrated by
``scripts/prepare_but_ppg.py``, not this module.

The registry is the shared contract between data prep, the models and evaluation:
one row per 10-second PPG window; models read the registry + the processed arrays
it points to, never the raw dataset files. ECG-derived ``hr_ref`` is a
target/label only and must never be fed to a model as input.

Canonical columns (fixed here at E0 so downstream code can rely on them):

    record_id         unique 10s-window id
    subject_id        person id — the split unit
    dataset           'but_ppg' | 'galaxy_ppg'
    quality_label     1 = good / 0 = bad   (target, task A)
    hr_ref            reference HR in bpm from ECG (target, task B)
    has_acc           accelerometer available for this record
    ppg_path          processed PPG array (300 samples @ 30 Hz)
    acc_path          processed ACC array (nullable)
    measurement_site  where PPG was taken (balance checks)
    sex, age, height, weight   covariates (extended-input variant only)
    activity          GalaxyPPG activity label (external validation)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RECORD_COLUMNS = [
    "record_id", "subject_id", "dataset", "quality_label", "hr_ref", "has_acc",
    "ppg_path", "acc_path", "measurement_site", "sex", "age", "height", "weight",
    "activity",
]

# record_id/subject_id in BUT PPG look like plain numbers ("100001", "100") --
# force string on read so they never get silently coerced to int64 and then
# fail to match a (string) split file. Same fix as metrics/predictions.py;
# found running the real pipeline against real data (see docs/review/).
_ID_DTYPES = {"record_id": str, "subject_id": str}


def validate_registry(df: pd.DataFrame) -> None:
    missing = set(RECORD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Registry is missing columns: {sorted(missing)}")

    if df["record_id"].duplicated().any():
        dupes = df.loc[df["record_id"].duplicated(), "record_id"].tolist()
        raise ValueError(f"Duplicate record_id in registry: {dupes[:5]}{'...' if len(dupes) > 5 else ''}")

    bad_quality = set(df["quality_label"].dropna().unique()) - {0, 1}
    if bad_quality:
        raise ValueError(f"quality_label must be 0/1, found: {sorted(bad_quality)}")

    unknown_dataset = set(df["dataset"].unique()) - {"but_ppg", "galaxy_ppg"}
    if unknown_dataset:
        raise ValueError(f"Unknown dataset value(s) {sorted(unknown_dataset)}")


def save_registry(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write the registry (see module docstring for the schema)."""
    validate_registry(df)
    df = df[RECORD_COLUMNS].copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_registry(path: str | Path) -> pd.DataFrame:
    """Load and validate a registry CSV."""
    df = pd.read_csv(Path(path), dtype=_ID_DTYPES)
    validate_registry(df)
    return df


def build_registry(*args, **kwargs):  # pragma: no cover - E1
    raise NotImplementedError(
        "E1 (Golikov): orchestrated by scripts/prepare_but_ppg.py, which calls save_registry() directly"
    )
