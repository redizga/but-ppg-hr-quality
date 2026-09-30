"""Low-data study aggregation (assignment section 9 / table in section 10).

Scans finished runs, finds those trained on a nested low-data split (the split
file carries a ``low_data`` block written by ``orch lowdata-subsets``), groups
them by (model, task, train-fraction), and reports the primary metric as
**mean +/- std** across seeds — plus ROC-AUC/Accuracy (quality) and RMSE (HR).
Produces Markdown + CSV tables and the section-9 curves.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from butppg.orchestrator.runs import list_runs
from butppg.paths import RESULTS_DIR
from butppg.reporting.tables import run_label


def _low_data_of(run) -> dict | None:
    split = (run.config or {}).get("split")
    if not split or not Path(split).exists():
        return None
    try:
        return json.loads(Path(split).read_text(encoding="utf-8")).get("low_data")
    except (OSError, ValueError):
        return None


def _mean_std(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None, None
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    return mean, std


def collect_rows(runs) -> list[dict]:
    grouped: dict[tuple, list] = {}
    for r in runs:
        if r.status != "finished":
            continue
        ld = _low_data_of(r)
        if not ld:
            continue
        key = (run_label(r), r.task, float(ld["fraction"]))
        grouped.setdefault(key, []).append((ld, r.metrics or {}))

    rows = []
    for (label, task, frac), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        primary = "macro_f1" if task == "quality" else "mae"
        mean, std = _mean_std([m.get(primary) for _, m in items])
        row = {
            "model": label, "task": task, "fraction": frac,
            "n_train_subjects": items[0][0].get("n_train_subjects"),
            "n_seeds": len(items), "metric": primary, "mean": mean, "std": std,
        }
        if task == "quality":
            row["roc_auc"], _ = _mean_std([m.get("roc_auc") for _, m in items])
            row["accuracy"], _ = _mean_std([m.get("accuracy") for _, m in items])
        else:
            row["rmse"], _ = _mean_std([m.get("rmse") for _, m in items])
        rows.append(row)
    return rows


def _markdown(rows) -> str:
    def fmt(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

    out = ["# Low-data: основная метрика (mean ± std по seed'ам), section 9\n"]
    for task, title in (("quality", "Quality — Macro-F1"), ("hr", "HR — MAE (bpm)")):
        trows = [r for r in rows if r["task"] == task]
        if not trows:
            continue
        out.append(f"## {title}\n")
        out.append("| Model | train % | #subj | seeds | mean ± std |")
        out.append("|---|---|---|---|---|")
        for r in sorted(trows, key=lambda r: (r["model"], r["fraction"])):
            pct = int(round(r["fraction"] * 100))
            out.append(f"| {r['model']} | {pct}% | {r['n_train_subjects']} | {r['n_seeds']} | "
                       f"{fmt(r['mean'], 3)} ± {fmt(r['std'], 3)} |")
        out.append("")
    return "\n".join(out) + "\n"


def write_low_data_report(out_dir: str | Path | None = None) -> dict[str, Path]:
    out = Path(out_dir) if out_dir else RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(list_runs())

    paths: dict[str, Path] = {}
    (out / "lowdata_main.md").write_text(_markdown(rows), encoding="utf-8")
    paths["md"] = out / "lowdata_main.md"

    fields = ["model", "task", "fraction", "n_train_subjects", "n_seeds", "metric", "mean", "std", "roc_auc", "accuracy", "rmse"]
    with open(out / "lowdata_main.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    paths["csv"] = out / "lowdata_main.csv"

    if rows:
        try:
            from butppg.reporting.plots import plot_low_data_curves

            for i, p in enumerate(plot_low_data_curves(rows, RESULTS_DIR / "figures")):
                paths[f"fig{i}"] = p
        except Exception as exc:  # noqa: BLE001 - plotting is optional
            print(f"[lowdata] skipped curves ({type(exc).__name__}: {exc}); install matplotlib to enable")
    return paths
