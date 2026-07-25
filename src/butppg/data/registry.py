"""Record registry — the flat table every model reads from.

Epic E1 (Golikov). Stub — implemented in that epic.

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

RECORD_COLUMNS = [
    "record_id", "subject_id", "dataset", "quality_label", "hr_ref", "has_acc",
    "ppg_path", "acc_path", "measurement_site", "sex", "age", "height", "weight",
    "activity",
]


def build_registry(*args, **kwargs):  # pragma: no cover - E1
    raise NotImplementedError("E1 (Golikov): build the record registry from BUT PPG")


def load_registry(*args, **kwargs):  # pragma: no cover - E1
    raise NotImplementedError("E1 (Golikov): implement registry loading")


def save_registry(*args, **kwargs):  # pragma: no cover - E1
    raise NotImplementedError("E1 (Golikov): implement registry saving")


def validate_registry(*args, **kwargs):  # pragma: no cover - E1
    raise NotImplementedError("E1 (Golikov): implement registry validation")
