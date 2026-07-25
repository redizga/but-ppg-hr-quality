"""Metrics + prediction I/O — the shared evaluation contract.

Epic E2 (Budilov). Stub — implemented in that epic.

Quality (task A): Macro-F1 primary (checkpoint selection), plus ROC-AUC,
Accuracy, per-class precision/recall/F1, confusion matrix.
HR (task B): MAE primary (checkpoint selection), RMSE secondary.

Canonical prediction schema (fixed here at E0 so every model writes the same
file and one evaluator scores them all on the fixed test set):

    record_id, subject_id, task, y_true, y_pred,
    prob_good,      # nullable — P(good), quality only
    raw_response,   # nullable — OpenTSLM raw text
    parse_status    # 'ok' | 'invalid' — LLM answer-parsing audit (never drop silently)
"""

PREDICTION_COLUMNS = [
    "record_id", "subject_id", "task", "y_true", "y_pred",
    "prob_good", "raw_response", "parse_status",
]
