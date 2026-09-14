"""Tests for the split-file I/O contract (``butppg.data.splits``).

Only ``save_split``/``load_split``/``verify_no_overlap`` are implemented so
far (the actual subject-assignment algorithm is still E1's job) — but this is
exactly the part ``butppg.data.marts`` (used by the E5/E6 mart builders)
depends on, so it is tested on its own here, independent of E1's eventual
splitting logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from butppg.data.marts import load_registry_with_split
from butppg.data.splits import load_split, make_subject_split, save_split, verify_no_overlap


def _synthetic_registry(n_subjects: int, seed: int = 0) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        subject_id = f"S{i:03d}"
        site = "ear" if i % 3 == 0 else "finger"  # constant per subject, like real BUT PPG
        for w in range(4):
            rows.append(
                {
                    "subject_id": subject_id,
                    "record_id": f"{subject_id}_{w}",
                    "quality_label": int(rng.integers(0, 2)),
                    "hr_ref": float(rng.uniform(55, 100)),
                    "measurement_site": site,
                }
            )
    return pd.DataFrame(rows)


def test_save_and_load_roundtrip(tmp_path):
    path = save_split(
        train_subjects=["S03", "S01"],
        val_subjects=["S04"],
        test_subjects=["S02"],
        path=tmp_path / "split.json",
        seed=42,
        ratios=(0.6, 0.2, 0.2),
    )
    loaded = load_split(path)
    assert loaded["seed"] == 42
    assert loaded["ratios"] == [0.6, 0.2, 0.2]
    # save_split sorts subject lists -- order shouldn't matter to callers, but
    # a deterministic file makes diffs in the committed splits/*.json readable
    assert loaded["train_subjects"] == ["S01", "S03"]
    assert loaded["val_subjects"] == ["S04"]
    assert loaded["test_subjects"] == ["S02"]


def test_verify_no_overlap_passes_on_disjoint_sets():
    verify_no_overlap(["S01", "S02"], ["S03"], ["S04"])  # must not raise


def test_verify_no_overlap_catches_leak_between_train_and_test():
    with pytest.raises(ValueError, match="S02"):
        verify_no_overlap(["S01", "S02"], ["S03"], ["S02", "S04"])


def test_save_split_refuses_to_write_overlapping_subjects(tmp_path):
    with pytest.raises(ValueError):
        save_split(
            train_subjects=["S01"],
            val_subjects=["S01"],  # leaked into val too
            test_subjects=["S02"],
            path=tmp_path / "bad_split.json",
            seed=42,
        )
    assert not (tmp_path / "bad_split.json").exists()


def test_load_split_rejects_hand_edited_file_with_a_leak(tmp_path):
    path = tmp_path / "tampered.json"
    path.write_text(
        '{"seed": 1, "ratios": [0.6, 0.2, 0.2], '
        '"train_subjects": ["S01"], "val_subjects": ["S01"], "test_subjects": ["S02"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="S01"):
        load_split(path)


def test_load_split_rejects_missing_keys(tmp_path):
    path = tmp_path / "incomplete.json"
    path.write_text('{"train_subjects": ["S01"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_split(path)


def test_marts_handles_numeric_looking_subject_ids(tmp_path):
    """Regression test: BUT PPG subject/record IDs are plain numbers.

    A bare ``pd.read_csv`` infers a column of "100", "101", ... as int64,
    while the split file always has them as JSON strings -- every subject
    then looks "unknown" even though the data is fine. Found by running the
    real pipeline against real BUT PPG data (see docs/review/).
    """

    registry_path = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {col: "" for col in ["record_id", "subject_id", "dataset", "quality_label", "hr_ref", "has_acc",
                                  "ppg_path", "acc_path", "measurement_site", "sex", "age", "height", "weight",
                                  "activity"]}
            | {"record_id": "100001", "subject_id": "100", "dataset": "but_ppg", "quality_label": 1, "hr_ref": 75.0}
        ]
    ).to_csv(registry_path, index=False)

    split_path = save_split(["100"], ["101"], ["102"], tmp_path / "split.json", seed=1)

    df = load_registry_with_split(registry_path, split_path)
    # stayed string-like (object or pandas StringDtype, depending on pandas
    # version), not silently coerced to int64 -- either way "100" must match
    # the split file's "100", not fail as an "unknown subject"
    assert not pd.api.types.is_integer_dtype(df["subject_id"])
    assert df.loc[0, "split"] == "train"


def test_marts_uses_the_same_split_loader(tmp_path):
    """load_registry_with_split must go through splits.load_split, not its own parser."""

    registry_path = tmp_path / "registry.csv"
    pd.DataFrame(
        {col: [] for col in ["record_id", "subject_id", "dataset", "quality_label", "hr_ref", "has_acc",
                              "ppg_path", "acc_path", "measurement_site", "sex", "age", "height", "weight",
                              "activity"]}
    ).to_csv(registry_path, index=False)

    split_path = save_split(["S01"], ["S02"], ["S03"], tmp_path / "split.json", seed=1)
    # an empty registry is fine here -- the point is that a leaking split file
    # is rejected before any registry row is even looked at
    tampered = tmp_path / "tampered2.json"
    tampered.write_text(
        '{"seed": 1, "ratios": [0.6, 0.2, 0.2], '
        '"train_subjects": ["S01"], "val_subjects": ["S01"], "test_subjects": ["S03"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="S01"):
        load_registry_with_split(registry_path, tampered)

    # sanity: the well-formed split loads fine through the same code path
    df = load_registry_with_split(registry_path, split_path)
    assert "split" in df.columns


# ---------------------------------------------------------------------------
# make_subject_split
# ---------------------------------------------------------------------------


def test_make_subject_split_matches_but_ppg_proportions():
    registry = _synthetic_registry(n_subjects=50)  # same subject count as real BUT PPG
    train, val, test = make_subject_split(registry, ratios=(0.6, 0.2, 0.2), seed=42)
    assert len(train) == 30
    assert len(val) == 10
    assert len(test) == 10
    verify_no_overlap(train, val, test)  # must not raise
    assert set(train) | set(val) | set(test) == set(registry["subject_id"].unique())


def test_make_subject_split_is_deterministic():
    registry = _synthetic_registry(n_subjects=50)
    split_a = make_subject_split(registry, seed=7)
    split_b = make_subject_split(registry, seed=7)
    assert split_a == split_b


def test_make_subject_split_different_seeds_differ():
    registry = _synthetic_registry(n_subjects=50)
    split_a = make_subject_split(registry, seed=1)
    split_b = make_subject_split(registry, seed=2)
    assert split_a != split_b


def test_make_subject_split_falls_back_gracefully_on_tiny_subject_count():
    # too few subjects per stratum to stratify -- must not crash, just fall
    # back to a plain (still seeded, still disjoint) random split
    registry = _synthetic_registry(n_subjects=6)
    train, val, test = make_subject_split(registry, ratios=(0.6, 0.2, 0.2), seed=42)
    assert len(train) + len(val) + len(test) == 6
    verify_no_overlap(train, val, test)


def test_make_subject_split_rejects_bad_ratios():
    registry = _synthetic_registry(n_subjects=10)
    with pytest.raises(ValueError, match="sum to 1.0"):
        make_subject_split(registry, ratios=(0.5, 0.2, 0.2))


def test_make_subject_split_rejects_unknown_balance_column():
    registry = _synthetic_registry(n_subjects=10)
    with pytest.raises(ValueError, match="not in registry"):
        make_subject_split(registry, balance_on=("does_not_exist",))
