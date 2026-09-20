"""Subject-wise train/val/test splitting + the mandatory overlap check.

Epic E1 (Golikov). ``make_subject_split`` (the actual splitting/balancing
algorithm) and ``make_low_data_subsets`` (E8) are still stubs — those are
real design decisions (how to balance folds) that belong to E1/E8.

``save_split``/``load_split``/``verify_no_overlap`` are implemented here
already, on purpose: the JSON schema they read/write is exactly what
``butppg.data.marts`` (used by E5's and E6's mart builders) expects, and it
was fixed in that module's docstring before E1 existed. Writing both sides of
that contract here — instead of leaving E1 to reinvent a schema that then has
to match a guess made in E5/E6 — removes an entire class of "works on my
synthetic data, breaks on the real split file" bugs: there is now exactly one
implementation, not two independently-written ones that merely need to agree.

Contract this module satisfies:

* split is by *subject*, never by record — all records of a person in one fold;
* default ratios 60/20/20 -> for BUT PPG's 50 subjects that is 30/10/10;
* seed, ratios and the resulting subject lists are persisted to a JSON split
  file under ``splits/`` (committed), so the test fold is fixed for every model;
* an automatic check guarantees the three subject sets are pairwise disjoint
  (data-leakage guard) — enforced on both save and load, so a hand-edited or
  buggy split file is rejected immediately instead of silently leaking subjects;
* optional balancing keeps folds comparable on quality label / HR / site;
* nested low-data subsets (E8): subjects in 25% subset of 50% subset of 100%.

Split file schema::

    {
      "seed": 42,
      "ratios": [0.6, 0.2, 0.2],
      "train_subjects": ["S01", "S02", ...],
      "val_subjects": ["S03", ...],
      "test_subjects": ["S04", ...]
    }
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def verify_no_overlap(
    train_subjects: Sequence[str], val_subjects: Sequence[str], test_subjects: Sequence[str]
) -> None:
    """Raise if any subject appears in more than one fold (data-leakage guard)."""
    folds = {"train": set(train_subjects), "val": set(val_subjects), "test": set(test_subjects)}
    for (name_a, set_a), (name_b, set_b) in combinations(folds.items(), 2):
        overlap = set_a & set_b
        if overlap:
            raise ValueError(f"Subjects appear in both {name_a!r} and {name_b!r}: {sorted(overlap)}")


def save_split(
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
    path: str | Path,
    seed: int,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
) -> Path:
    """Persist a subject-wise split to JSON (see module docstring for schema).

    Refuses to write a split with overlapping subjects — better to fail here
    than to produce a committed ``splits/*.json`` that silently leaks data
    between folds.
    """
    verify_no_overlap(train_subjects, val_subjects, test_subjects)
    data = {
        "seed": seed,
        "ratios": list(ratios),
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_split(path: str | Path) -> dict:
    """Load a subject-wise split JSON, re-checking the disjointness guarantee."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    required = {"train_subjects", "val_subjects", "test_subjects"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Split file {path} is missing keys: {sorted(missing)}")

    verify_no_overlap(data["train_subjects"], data["val_subjects"], data["test_subjects"])
    return data


def _subject_strata(registry: pd.DataFrame, balance_on: Sequence[str]) -> pd.Series:
    """One stratification key per subject, combining ``balance_on`` columns.

    Numeric columns (e.g. ``hr_ref``) are bucketed above/below their overall
    median; categorical columns (e.g. ``measurement_site``) use each
    subject's most common value. Assignment section 3: "по возможности
    сделать разбиение так, чтобы части были примерно сбалансированы по
    качеству сигнала, распределению HR и месту измерения."
    """
    per_subject = registry.groupby("subject_id")
    parts: list[pd.Series] = []
    for col in balance_on:
        if col not in registry.columns:
            raise ValueError(f"balance_on column {col!r} not in registry")
        if pd.api.types.is_numeric_dtype(registry[col]):
            agg = per_subject[col].mean()
            median = agg.median()
            bucket = (agg > median).astype(int).astype(str)
        else:
            agg = per_subject[col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "na")
            bucket = agg.astype(str)
        parts.append(bucket)

    strata = parts[0].astype(str)
    for p in parts[1:]:
        strata = strata + "|" + p.astype(str)
    return strata


def make_subject_split(
    registry: pd.DataFrame,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 42,
    balance_on: Sequence[str] = ("quality_label", "measurement_site"),
) -> tuple[list[str], list[str], list[str]]:
    """Split subjects (not records) into train/val/test.

    Balances on subject-level aggregates of ``balance_on`` where the subject
    pool is large enough to stratify (falls back to a plain random split,
    same seed, if a stratum is too small for scikit-learn to split on — e.g.
    the handful of subjects in a quick smoke test).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    if len(ratios) != 3:
        raise ValueError(f"ratios must be (train, val, test), got {ratios}")

    subjects = sorted(registry["subject_id"].astype(str).unique())
    if len(subjects) < 3:
        raise ValueError(f"need at least 3 subjects to form train/val/test, got {len(subjects)}")

    strata = _subject_strata(registry.assign(subject_id=registry["subject_id"].astype(str)), balance_on)
    strata = strata.reindex(subjects)

    def _split(subj_list: list[str], strat_list: list[str], test_size: float) -> tuple[list[str], list[str]]:
        try:
            return train_test_split(subj_list, test_size=test_size, random_state=seed, stratify=strat_list)
        except ValueError:
            # a stratum too small to split on (typical with very few subjects) -- same seed, no stratification
            return train_test_split(subj_list, test_size=test_size, random_state=seed)

    train_ratio, val_ratio, test_ratio = ratios
    train_val, test = _split(subjects, strata.tolist(), test_ratio)
    relative_val = val_ratio / (train_ratio + val_ratio)
    train, val = _split(sorted(train_val), strata.loc[sorted(train_val)].tolist(), relative_val)

    verify_no_overlap(train, val, test)
    return sorted(train), sorted(val), sorted(test)


def make_low_data_subsets(*args, **kwargs):  # pragma: no cover - E8
    raise NotImplementedError("E8 (Both): implement nested low-data subsets")
