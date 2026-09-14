"""Quality-classification metrics (task A) — E2 (Budilov).

Primary: Macro-F1 (used for checkpoint selection, assignment section 5). Also:
ROC-AUC, Accuracy, per-class precision/recall/F1, confusion matrix.

Also carries threshold selection: the assignment (section 3) requires the
classification threshold itself to be chosen on the validation split, not
hardcoded at 0.5 — ``select_threshold`` does that grid search.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

POSITIVE_LABEL = 1  # quality_label: 1 = good, 0 = bad (see registry.py)


def select_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, n_steps: int = 199
) -> tuple[float, float]:
    """Grid-search the decision threshold on Macro-F1 (validation split only).

    Returns ``(best_threshold, best_macro_f1)``. Never call this on the test
    split — the threshold is a hyperparameter, chosen on val like everything
    else (assignment section 3), then frozen for the test-set report.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        raise ValueError("select_threshold: empty y_true")

    thresholds = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]  # skip 0.0/1.0 (degenerate)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if len(np.unique(y_pred)) == 1 and len(np.unique(y_true)) > 1:
            # a threshold that collapses everything to one class still scores,
            # sklearn handles it fine (the missing class gets f1=0); no special case needed
            pass
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def quality_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict[str, Any]:
    """Full quality-task metric bundle for one prediction file.

    Parameters
    ----------
    y_true, y_pred:
        Binary labels (0/1, see ``registry.py``: 1 = good).
    y_prob:
        P(good), i.e. probability of the positive class. Optional — ROC-AUC is
        omitted (not an error) if not supplied, since some models (e.g. a
        hard-threshold-only baseline) may not expose a probability.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) length mismatch")
    if len(y_true) == 0:
        raise ValueError("quality_metrics: empty input")

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = float(np.mean(y_true == y_pred))

    labels = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    result: dict[str, Any] = {
        "n": len(y_true),
        "macro_f1": float(macro_f1),
        "accuracy": accuracy,
        "per_class": per_class,
        # rows = true label, cols = predicted label, both ordered [bad(0), good(1)]
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["bad", "good"],
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if len(np.unique(y_true)) < 2:
            # ROC-AUC is undefined with a single class present in y_true
            result["roc_auc"] = None
        else:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        result["roc_auc"] = None

    return result
