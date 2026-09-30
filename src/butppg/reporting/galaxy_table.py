"""GalaxyPPG external-validation table (assignment sections 8 & 10).

Renders the activity-stratified HR report (from
``evaluation.evaluate.activity_report``) as the Markdown table the assignment
asks for: overall MAE/RMSE, MAE/RMSE per activity (motion activities flagged),
and the fraction of windows the quality classifier accepted as good.
"""

from __future__ import annotations

from pathlib import Path

from butppg.data.galaxy import MOTION_ACTIVITIES


def _fmt(v, nd=3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def galaxy_markdown(report: dict) -> str:
    ov = report.get("overall") or {}
    lines = [
        "# Внешняя проверка на GalaxyPPG (HR, section 8)",
        "",
        f"Всего окон: {report.get('n_windows', '—')} | оценено: {report.get('n_scored', '—')}"
        + (f" | принято quality-классификатором: {_fmt(report.get('accepted_fraction'), 4)}"
           if "accepted_fraction" in report else ""),
        "",
        "| Активность | MAE | RMSE | n |",
        "|---|---|---|---|",
        f"| **Все (overall)** | {_fmt(ov.get('mae'))} | {_fmt(ov.get('rmse'))} | {report.get('n_scored','—')} |",
    ]
    for act, m in sorted(report.get("per_activity", {}).items()):
        star = " 🏃" if act in MOTION_ACTIVITIES else ""
        lines.append(f"| {act}{star} | {_fmt(m.get('mae'))} | {_fmt(m.get('rmse'))} | {m.get('n','—')} |")
    lines += ["", "🏃 — двигательные активности (walking/jogging/running): по заданию именно на них "
              "ожидается наибольшая деградация HR из-за артефактов движения."]
    return "\n".join(lines) + "\n"


def write_galaxy_table(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(galaxy_markdown(report), encoding="utf-8")
    return path
