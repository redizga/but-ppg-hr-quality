"""Config loading and composition.

A run config is a plain dict assembled from YAML files under ``configs/``.
Composition rules (kept deliberately simple, no extra deps):

* ``configs/default.yaml`` is always the base.
* A config may declare ``defaults: [data/but_ppg, models/cnn1d, ...]`` — each
  listed file is deep-merged in order.
* ``key.subkey=value`` command-line overrides win last.

This is enough for the study and stays dependency-free (no hydra/omegaconf),
so the reproducibility core installs in seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve(name: str) -> Path:
    """Resolve a config name (with or without .yaml, relative to configs/)."""
    p = Path(name)
    if p.suffix != ".yaml":
        p = p.with_suffix(".yaml")
    return p if p.is_absolute() else CONFIG_ROOT / p


def _cast(value: str) -> Any:
    """Cast a CLI override value using YAML rules (true/false/null/int/float/list/str).

    Using the YAML loader keeps override semantics identical to the config files
    themselves, so ``key=true`` becomes a bool exactly as it would in a .yaml.
    """
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` dotted overrides in place-ish (returns new dict)."""
    out = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = out
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override into non-dict at {k!r} in {dotted!r}")
        node[keys[-1]] = _cast(raw)
    return out


def load_config(name: str = "default", overrides: list[str] | None = None) -> dict[str, Any]:
    """Load and compose a config.

    Parameters
    ----------
    name:
        Config file name relative to ``configs/`` (``default``, ``models/cnn1d`` …).
    overrides:
        List of ``dotted.key=value`` strings applied last.
    """
    base = _load_yaml(_resolve("default"))
    config = base

    if name != "default":
        primary = _load_yaml(_resolve(name))
        # honour a `defaults:` list before merging the file's own keys
        for dep in primary.pop("defaults", []) or []:
            config = _deep_merge(config, _load_yaml(_resolve(dep)))
        config = _deep_merge(config, primary)

    for dep in base.pop("defaults", []) or []:
        # base-level defaults are merged under everything so explicit configs win
        config = _deep_merge(_load_yaml(_resolve(dep)), config)

    if overrides:
        config = apply_overrides(config, overrides)

    return config
