"""Subject-wise train/val/test splitting + the mandatory overlap check.

Epic E1 (Golikov). Stub — implemented in that epic.

Contract this module must satisfy (fixed here at E0 so everything downstream can
rely on it):

* split is by *subject*, never by record — all records of a person in one fold;
* default ratios 60/20/20 → for BUT PPG's 50 subjects that is 30/10/10;
* seed, ratios and the resulting subject lists are persisted to a JSON split file
  under ``splits/`` (committed), so the test fold is fixed for every model;
* an automatic check guarantees the three subject sets are pairwise disjoint
  (data-leakage guard);
* optional balancing keeps folds comparable on quality label / HR / site;
* nested low-data subsets (E8): subjects in 25% ⊂ 50% ⊂ 100%.
"""

from __future__ import annotations


def make_subject_split(*args, **kwargs):  # pragma: no cover - E1
    raise NotImplementedError("E1 (Golikov): implement subject-wise 60/20/20 split")


def verify_no_overlap(*args, **kwargs):  # pragma: no cover - E1
    raise NotImplementedError("E1 (Golikov): implement subject non-overlap check")


def save_split(*args, **kwargs):  # pragma: no cover - E1
    raise NotImplementedError("E1 (Golikov): implement split serialisation to splits/*.json")


def load_split(*args, **kwargs):  # pragma: no cover - E1
    raise NotImplementedError("E1 (Golikov): implement split loading")


def make_low_data_subsets(*args, **kwargs):  # pragma: no cover - E8
    raise NotImplementedError("E8 (Both): implement nested low-data subsets")
