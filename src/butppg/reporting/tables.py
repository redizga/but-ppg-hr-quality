"""Assemble the section-10 comparison tables from finished runs.

Reads ``runs/<id>/run.json`` (model, task, config, metrics) and produces:

* the main **PPG-only** table — every model as a row, Quality (Macro-F1, ROC-AUC,
  Accuracy) and HR (MAE, RMSE) as columns;
* the **extended-input** table — Quality Macro-F1 for the feature baseline and the
  CNN across the PPG / PPG+ACC / PPG+ACC+covariates variants (section 2A).

Outputs Markdown (for the report) and CSV (for further processing) under
``results/``. Pure stdlib + the run registry — no torch/sklearn needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

from butppg.orchestrator.runs import RunRecord, list_runs
from butppg.paths import RESULTS_DIR

# Row order for the main PPG-only table (assignment section 10). The two
# OpenTSLM entries are placeholders — the actual row label carries the LLM size
# that was fine-tuned (e.g. "OpenTSLM-1B"), resolved from the run config.
MAIN_ORDER = [
    "Majority class",
    "Median HR",
    "Dominant-frequency HR",
    "Features + LogReg/XGBoost",
    "1D-CNN / ResNet1D",
    "OpenTSLM",
    "OpenTSLM (ECG)",
    "SIGMA-PPG",
]


def _llm_size(cfg: dict) -> str:
    """Human-readable LLM size from the run's llm_id (for the OpenTSLM label)."""
    llm = str(cfg.get("llm_id", "")).lower()
    for tag in ("270m", "1b", "3b", "7b", "8b"):
        if tag in llm:
            return tag.upper()
    if "gemma" in llm:
        return "Gemma"
    tail = llm.split("/")[-1]
    return tail or "?"


def run_label(run: RunRecord) -> str:
    cfg = run.config or {}
    m = run.model
    if m == "trivial":
        if run.task == "quality":
            return "Majority class"
        return "Dominant-frequency HR" if cfg.get("method") == "dominant_frequency" else "Median HR"
    if m == "baseline_features":
        return "Features + LogReg/XGBoost"
    if m == "cnn1d":
        return "1D-CNN / ResNet1D"
    if m == "sigma_ppg":
        return "SIGMA-PPG"
    if m == "opentslm":
        base = f"OpenTSLM-{_llm_size(cfg)}"
        return f"{base} (ECG)" if cfg.get("ecg_init") else base
    return m


def _label_matches(placeholder: str, label: str) -> bool:
    """Match a MAIN_ORDER placeholder to a concrete run label.

    OpenTSLM rows are size-tagged at runtime ("OpenTSLM-1B"), so the fixed
    placeholders match by family instead of exact string.
    """
    if placeholder == "OpenTSLM":
        return label.startswith("OpenTSLM-") and not label.endswith("(ECG)")
    if placeholder == "OpenTSLM (ECG)":
        return label.startswith("OpenTSLM-") and label.endswith("(ECG)")
    return label == placeholder


def _variant(run: RunRecord) -> str:
    return (run.config or {}).get("input_variant", "ppg")


def _latest_finished(runs, predicate):
    """Most recent finished run matching predicate, or None."""
    matches = [r for r in runs if r.status == "finished" and predicate(r)]
    return max(matches, key=lambda r: r.created_at) if matches else None


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def build_main_rows(runs) -> list[dict]:
    ppg = [r for r in runs if _variant(r) == "ppg"]
    rows = []
    for label in MAIN_ORDER:
        q = _latest_finished(ppg, lambda r: r.task == "quality" and run_label(r) == label)
        h = _latest_finished(ppg, lambda r: r.task == "hr" and run_label(r) == label)
        if q is None and h is None:
            rows.append({"model": label, "_missing": True})
            continue
        qm = (q.metrics if q else {}) or {}
        hm = (h.metrics if h else {}) or {}
        rows.append({
            "model": label,
            "quality_macro_f1": qm.get("macro_f1"),
            "quality_roc_auc": qm.get("roc_auc"),
            "quality_accuracy": qm.get("accuracy"),
            "hr_mae": hm.get("mae"),
            "hr_rmse": hm.get("rmse"),
        })
    return rows


def build_extended_rows(runs) -> list[dict]:
    """Quality Macro-F1 per input variant for the models that take extended inputs."""
    variants = ["ppg", "ppg_acc", "ppg_acc_cov"]
    rows = []
    for label in ("Features + LogReg/XGBoost", "1D-CNN / ResNet1D"):
        row = {"model": label}
        for v in variants:
            r = _latest_finished(runs, lambda r: r.task == "quality" and run_label(r) == label and _variant(r) == v)
            row[v] = (r.metrics or {}).get("macro_f1") if r else None
        rows.append(row)
    return rows


def _main_markdown(rows) -> str:
    head = "| Model | Quality Macro-F1 | ROC-AUC | Accuracy | HR MAE | HR RMSE |\n"
    head += "|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        if r.get("_missing"):
            body += f"| {r['model']} | — | — | — | — | — |\n"
            continue
        body += (
            f"| {r['model']} | {_fmt(r['quality_macro_f1'])} | {_fmt(r['quality_roc_auc'])} | "
            f"{_fmt(r['quality_accuracy'])} | {_fmt(r['hr_mae'], 3)} | {_fmt(r['hr_rmse'], 3)} |\n"
        )
    return head + body


def _extended_markdown(rows) -> str:
    head = "| Model | PPG | PPG+ACC | PPG+ACC+cov |\n|---|---|---|---|\n"
    body = ""
    for r in rows:
        body += f"| {r['model']} | {_fmt(r.get('ppg'))} | {_fmt(r.get('ppg_acc'))} | {_fmt(r.get('ppg_acc_cov'))} |\n"
    return head + body


def _write_csv(rows, fields, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_comparison(out_dir: str | Path | None = None) -> dict[str, Path]:
    """Build and save the main + extended comparison tables. Returns written paths."""
    out = Path(out_dir) if out_dir else RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    runs = list_runs()

    main_rows = build_main_rows(runs)
    ext_rows = build_extended_rows(runs)

    paths = {}
    (out / "comparison_main.md").write_text(
        "# Основная таблица (только PPG)\n\n" + _main_markdown(main_rows), encoding="utf-8"
    )
    paths["main_md"] = out / "comparison_main.md"
    _write_csv(main_rows, ["model", "quality_macro_f1", "quality_roc_auc", "quality_accuracy", "hr_mae", "hr_rmse"],
               out / "comparison_main.csv")
    paths["main_csv"] = out / "comparison_main.csv"

    (out / "comparison_extended_inputs.md").write_text(
        "# Расширенные входы — Quality Macro-F1\n\n" + _extended_markdown(ext_rows), encoding="utf-8"
    )
    paths["extended_md"] = out / "comparison_extended_inputs.md"
    _write_csv(ext_rows, ["model", "ppg", "ppg_acc", "ppg_acc_cov"], out / "comparison_extended_inputs.csv")
    paths["extended_csv"] = out / "comparison_extended_inputs.csv"

    return paths
