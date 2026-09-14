"""OpenTSLM answer-parsing rules — E5 (Budilov), assignment section 6.

OpenTSLM answers in free text, not a class index or a float — so before its
output can go into the canonical prediction schema
(``metrics.PREDICTION_COLUMNS``), it must be turned into ``y_pred`` (+ an
audit trail). The assignment fixes the exact rules (section 6):

* quality: only ``good``/``bad`` or ``1``/``0`` are valid answers;
* HR: extract the first number in the response that falls inside a
  physiologically plausible range (30-220 bpm, matches ``HR_RANGE`` in
  ``configs/models/opentslm.yaml``);
* invalid answers must NEVER be silently dropped — the raw text, a parse
  status, and (in aggregate, see ``evaluation/evaluate.py``) the fraction of
  invalid answers must all be kept.

These two constraints are exactly why ``PREDICTION_COLUMNS`` carries both
``raw_response`` and ``parse_status`` alongside ``y_pred`` — an ``invalid``
row keeps its original text and a null ``y_pred`` rather than disappearing.

This module is intentionally standalone (pure string parsing, no OpenTSLM /
torch import) so it can be unit-tested and used to build a training set for
the OpenTSLM model itself using nothing but this and the marts already built
by ``scripts/build_mart_opentslm.py``.
"""

from __future__ import annotations

import re

import pandas as pd

from butppg.metrics import PREDICTION_COLUMNS

QUALITY_GOOD_TOKENS = {"good", "1"}
QUALITY_BAD_TOKENS = {"bad", "0"}
HR_RANGE = (30, 220)  # assignment section 6

_ANSWER_ANCHOR_RE = re.compile(r"answer\s*:\s*", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _text_after_last_anchor(raw_text: str) -> str:
    """Text following the last 'Answer:' anchor, or the whole text if absent.

    Our own prompts (``scripts/build_mart_opentslm.py``) always instruct the
    model to end with 'Answer: <...>', so this is the primary signal; falling
    back to the whole text keeps parsing usable for outputs that ignore the
    instruction (still graded as invalid if nothing parseable follows).
    """
    matches = list(_ANSWER_ANCHOR_RE.finditer(raw_text))
    if not matches:
        return raw_text
    return raw_text[matches[-1].end() :]


def parse_quality_response(raw_text: str) -> tuple[int | None, str]:
    """Parse a quality answer. Returns ``(y_pred, parse_status)``.

    ``y_pred`` is ``1`` (good), ``0`` (bad), or ``None`` if the response
    contains no unambiguous match; ``parse_status`` is ``'ok'``/``'invalid'``.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, "invalid"

    tail = _text_after_last_anchor(raw_text)
    tokens = {t.lower() for t in _WORD_RE.findall(tail)}

    is_good = bool(tokens & QUALITY_GOOD_TOKENS)
    is_bad = bool(tokens & QUALITY_BAD_TOKENS)

    if is_good and not is_bad:
        return 1, "ok"
    if is_bad and not is_good:
        return 0, "ok"
    return None, "invalid"  # nothing found, or both good/bad tokens present (ambiguous)


def parse_hr_response(raw_text: str, valid_range: tuple[float, float] = HR_RANGE) -> tuple[float | None, str]:
    """Parse an HR answer: first number in ``valid_range``. Returns ``(y_pred, parse_status)``."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, "invalid"

    lo, hi = valid_range
    for source in (_text_after_last_anchor(raw_text), raw_text):
        for match in _NUMBER_RE.finditer(source):
            value = float(match.group())
            if lo <= value <= hi:
                return value, "ok"
        # if the anchor text alone had numbers but none in range, don't silently
        # fall through to unrelated numbers elsewhere in a long response
        if _NUMBER_RE.search(source):
            break
    return None, "invalid"


def responses_to_predictions(records: pd.DataFrame, raw_responses: list[str], task: str) -> pd.DataFrame:
    """Turn raw OpenTSLM text output into the canonical prediction schema.

    Parameters
    ----------
    records:
        Must have ``record_id``, ``subject_id`` and the ground-truth column
        (``quality_label`` for task=quality, ``hr_ref`` for task=hr) — same
        rows, same order as ``raw_responses`` (typically the DataFrame built
        from the ``.jsonl`` mart for this split).
    raw_responses:
        One generated string per row of ``records``.
    task:
        ``'quality'`` or ``'hr'``.
    """
    if len(records) != len(raw_responses):
        raise ValueError(f"records ({len(records)}) and raw_responses ({len(raw_responses)}) length mismatch")
    if task not in ("quality", "hr"):
        raise ValueError(f"task must be 'quality' or 'hr', got {task!r}")

    parser = parse_quality_response if task == "quality" else parse_hr_response
    truth_col = "quality_label" if task == "quality" else "hr_ref"

    y_pred, parse_status = [], []
    for text in raw_responses:
        value, status = parser(text)
        y_pred.append(value)
        parse_status.append(status)

    return pd.DataFrame(
        {
            "record_id": records["record_id"].values,
            "subject_id": records["subject_id"].values,
            "task": task,
            "y_true": records[truth_col].values,
            "y_pred": y_pred,
            "prob_good": None,  # OpenTSLM emits text, not a calibrated probability
            "raw_response": list(raw_responses),
            "parse_status": parse_status,
        }
    )[PREDICTION_COLUMNS]
