"""Canonical project paths — one source of truth so scripts never hardcode dirs."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

SPLITS_DIR = PROJECT_ROOT / "splits"          # committed: subject-wise split files
RESULTS_DIR = PROJECT_ROOT / "results"        # committed: final metrics & tables

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"    # gitignored (except metrics/)
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Per-model input marts (the `orch mart` output).
MARTS_DIR = ARTIFACTS_DIR / "data_marts"

# One directory per training run (the `orch train` output; `orch results` reads it).
RUNS_DIR = PROJECT_ROOT / "runs"

# Vendored upstream model repos, checked out next to this package.
OPENTSLM_DIR = PROJECT_ROOT / "OpenTSLM"
SIGMA_DIR = PROJECT_ROOT / "SigmaPPG"


def ensure_dirs() -> None:
    """Create the writable output dirs if missing (safe to call repeatedly)."""
    for d in (
        RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
        SPLITS_DIR, RESULTS_DIR / "tables",
        CHECKPOINTS_DIR, PREDICTIONS_DIR, METRICS_DIR, LOGS_DIR,
        MARTS_DIR, RUNS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
