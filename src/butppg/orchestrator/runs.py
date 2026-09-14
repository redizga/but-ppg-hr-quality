"""Run registry — one directory per training run, queried by ``orch results``.

A "run" is the unit the three commands hand off between each other: ``orch
train`` creates one and fills it in; ``orch results`` reads it. Everything a run
produces lives under ``runs/<run_id>/`` so a run is self-contained and portable
(you can tar one folder and have config + checkpoints + metrics + predictions):

    runs/<run_id>/
      run.json              status + summary (this module owns it)
      manifest.json         RunManifest (reproducibility receipt)
      train.log             training stdout/stderr
      checkpoints/          best_model.pt (+ any others)
      predictions/          test_predictions.csv (canonical PREDICTION_COLUMNS)
      metrics/              metrics.json

Status flow: ``created -> running -> finished`` (or ``failed``). The registry is
just JSON on disk — no daemon — so it works identically on a laptop and on the
GPU box the heavy training actually runs on.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from butppg.paths import RUNS_DIR

VALID_MODELS = ("trivial", "cnn1d", "baseline_features", "sigma_ppg", "opentslm")
VALID_TASKS = ("quality", "hr")
VALID_STATUS = ("created", "running", "finished", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(model: str, task: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{model}_{task}_{stamp}_{secrets.token_hex(2)}"


@dataclass
class RunRecord:
    """The ``run.json`` payload — status + everything ``orch results`` reports."""

    run_id: str
    model: str
    task: str
    status: str = "created"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    config: dict[str, Any] = field(default_factory=dict)
    mart_dir: str | None = None
    device: str | None = None
    best_checkpoint: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.run_id

    # --- lifecycle ------------------------------------------------------- #
    def save(self) -> Path:
        self.updated_at = _now()
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "run.json").write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.dir

    def set_status(self, status: str, error: str | None = None) -> None:
        if status not in VALID_STATUS:
            raise ValueError(f"invalid status {status!r}, expected one of {VALID_STATUS}")
        self.status = status
        if error is not None:
            self.error = error
        self.save()

    # --- convenience dirs ------------------------------------------------ #
    @property
    def checkpoints_dir(self) -> Path:
        return self.dir / "checkpoints"

    @property
    def predictions_dir(self) -> Path:
        return self.dir / "predictions"

    @property
    def metrics_dir(self) -> Path:
        return self.dir / "metrics"

    @property
    def log_file(self) -> Path:
        return self.dir / "train.log"

    def ensure_dirs(self) -> None:
        for d in (self.checkpoints_dir, self.predictions_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)


def create_run(model: str, task: str, config: dict[str, Any] | None = None) -> RunRecord:
    if model not in VALID_MODELS:
        raise ValueError(f"unknown model {model!r}, expected one of {VALID_MODELS}")
    if task not in VALID_TASKS:
        raise ValueError(f"unknown task {task!r}, expected one of {VALID_TASKS}")
    run = RunRecord(run_id=new_run_id(model, task), model=model, task=task, config=config or {})
    run.ensure_dirs()
    run.save()
    return run


def load_run(run_id: str) -> RunRecord:
    path = RUNS_DIR / run_id / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"no such run: {run_id} (looked in {path})")
    return RunRecord(**json.loads(path.read_text(encoding="utf-8")))


def list_runs() -> list[RunRecord]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        rj = d / "run.json"
        if rj.exists():
            try:
                runs.append(RunRecord(**json.loads(rj.read_text(encoding="utf-8"))))
            except Exception:
                continue
    return sorted(runs, key=lambda r: r.created_at, reverse=True)
