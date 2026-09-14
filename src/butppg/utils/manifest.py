"""Run manifest — the reproducibility receipt saved next to every result.

The assignment requires that we store, together with results: the run config,
dataset versions, RNG seed, split files, record registry, checkpoints (or links),
predictions, metrics and logs. The manifest is the single JSON that ties a set of
outputs back to the exact inputs that produced them.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_revision(short: bool = True) -> str | None:
    """Current git commit hash, or None if not in a git tree."""
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, check=True)
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}"
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "pandas", "sklearn", "torch", "xgboost"):
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except Exception:
            continue
    return versions


@dataclass
class RunManifest:
    """Everything needed to reproduce one run, serialised to ``manifest.json``."""

    run_name: str
    task: str  # "quality" | "hr"
    seed: int
    config: dict[str, Any]
    dataset_version: str = "BUT-PPG-v2.0.0"
    split_file: str | None = None
    registry_file: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_revision: str | None = field(default_factory=git_revision)
    platform: str = field(default_factory=platform.platform)
    package_versions: dict[str, str] = field(default_factory=_package_versions)
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
