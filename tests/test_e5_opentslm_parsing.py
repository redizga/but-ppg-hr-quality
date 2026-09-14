"""E5 tests: OpenTSLM answer parsing (assignment section 6).

Pure string parsing — no OpenTSLM/torch import, no BUT PPG data needed.
Also checks the round trip into E2's prediction schema/evaluator, since that
is the actual point of parsing the model's raw text in the first place.
"""

from __future__ import annotations

import pandas as pd
import pytest

from butppg.evaluation.evaluate import evaluate_prediction_file
from butppg.metrics import PREDICTION_COLUMNS
from butppg.metrics.predictions import save_predictions
from butppg.models.opentslm_parsing import (
    parse_hr_response,
    parse_quality_response,
    responses_to_predictions,
)

# ---------------------------------------------------------------------------
# quality parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("Answer: good", 1),
        ("Answer: bad", 0),
        ("Answer: 1", 1),
        ("Answer: 0", 0),
        ("The signal looks clean and usable.\nAnswer: good", 1),
        ("This PPG window is noisy.\nAnswer: bad", 0),
        ("Answer:good", 1),  # no space after colon
        ("ANSWER: GOOD", 1),  # case-insensitive
        ("Some reasoning about good vs bad features.\nAnswer: bad", 0),  # anchor wins over earlier mentions
    ],
)
def test_parse_quality_valid_cases(text, expected_label):
    y_pred, status = parse_quality_response(text)
    assert status == "ok"
    assert y_pred == expected_label


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        None,
        "I'm not sure about this signal.",  # no anchor, no good/bad/1/0 token
        "Answer: maybe",
        "Answer: excellent",
        "Answer: good or bad, hard to tell",  # both tokens present -> ambiguous
    ],
)
def test_parse_quality_invalid_cases(text):
    y_pred, status = parse_quality_response(text)
    assert status == "invalid"
    assert y_pred is None


# ---------------------------------------------------------------------------
# HR parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_bpm",
    [
        ("Answer: 72", 72.0),
        ("Answer: 72 bpm", 72.0),
        ("The heart rate is approximately 88.5 beats per minute.\nAnswer: 88.5", 88.5),
        ("Answer: 30", 30.0),  # inclusive lower bound
        ("Answer: 220", 220.0),  # inclusive upper bound
    ],
)
def test_parse_hr_valid_cases(text, expected_bpm):
    y_pred, status = parse_hr_response(text)
    assert status == "ok"
    assert y_pred == pytest.approx(expected_bpm)


def test_parse_hr_takes_first_in_range_number_after_anchor():
    # out-of-range number appears first in the anchor tail; assignment says
    # "first ADMISSIBLE number in range" -> 275 (out of range) must be skipped
    text = "Answer: 275 or maybe 90"
    y_pred, status = parse_hr_response(text)
    assert status == "ok"
    assert y_pred == pytest.approx(90.0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "I cannot estimate the heart rate.",
        "Answer: 400",  # out of physiological range
        "Answer: five",  # not a number at all
    ],
)
def test_parse_hr_invalid_cases(text):
    y_pred, status = parse_hr_response(text)
    assert status == "invalid"
    assert y_pred is None


# ---------------------------------------------------------------------------
# round trip into the canonical prediction schema
# ---------------------------------------------------------------------------


def test_responses_to_predictions_quality_roundtrip(tmp_path):
    records = pd.DataFrame(
        {
            "record_id": ["r0", "r1", "r2"],
            "subject_id": ["S0", "S0", "S1"],
            "quality_label": [1, 0, 1],
        }
    )
    raw_responses = ["Answer: good", "Answer: bad", "Answer: unclear"]  # last one invalid
    preds = responses_to_predictions(records, raw_responses, task="quality")

    assert list(preds.columns) == PREDICTION_COLUMNS
    assert preds["parse_status"].tolist() == ["ok", "ok", "invalid"]
    assert preds["y_pred"].tolist()[:2] == [1, 0]
    assert pd.isna(preds["y_pred"].iloc[2])

    # must be directly consumable by E2's save/evaluate path
    path = save_predictions(preds, tmp_path / "opentslm_quality.csv")
    result = evaluate_prediction_file(path)
    assert result["n_records"] == 3
    assert result["n_invalid_responses"] == 1
    assert result["invalid_response_fraction"] == pytest.approx(1 / 3)


def test_responses_to_predictions_hr_roundtrip(tmp_path):
    records = pd.DataFrame(
        {
            "record_id": ["r0", "r1"],
            "subject_id": ["S0", "S0"],
            "hr_ref": [72.0, 65.0],
        }
    )
    raw_responses = ["Answer: 70", "Answer: not sure"]
    preds = responses_to_predictions(records, raw_responses, task="hr")
    path = save_predictions(preds, tmp_path / "opentslm_hr.csv")
    result = evaluate_prediction_file(path)
    assert result["n_invalid_responses"] == 1


def test_evaluate_handles_all_invalid_gracefully(tmp_path):
    records = pd.DataFrame(
        {
            "record_id": ["r0", "r1"],
            "subject_id": ["S0", "S0"],
            "quality_label": [1, 0],
        }
    )
    preds = responses_to_predictions(records, ["Answer: unclear", "no idea"], task="quality")
    path = save_predictions(preds, tmp_path / "all_invalid.csv")
    result = evaluate_prediction_file(path)
    assert result["metrics"] is None
    assert result["n_invalid_responses"] == 2
    assert result["invalid_response_fraction"] == pytest.approx(1.0)


def test_responses_to_predictions_length_mismatch_raises():
    records = pd.DataFrame({"record_id": ["r0"], "subject_id": ["S0"], "quality_label": [1]})
    with pytest.raises(ValueError):
        responses_to_predictions(records, ["a", "b"], task="quality")
