"""Report figures (assignment section 9 curves + section 8 activity bars).

Matplotlib with the non-interactive Agg backend so it runs headless on a pod.
Figures are written as PNG under ``results/figures/``.
"""

from __future__ import annotations

from pathlib import Path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_low_data_curves(agg: list[dict], out_dir: str | Path) -> list[Path]:
    """One figure per task: primary metric vs #train-subjects, a line per model
    with mean +/- std error bars (assignment section 9)."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for task, ylabel, fname in (("quality", "Macro-F1", "lowdata_quality.png"),
                                ("hr", "MAE (bpm)", "lowdata_hr.png")):
        rows = [r for r in agg if r["task"] == task and r["mean"] is not None]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        models = sorted({r["model"] for r in rows})
        for model in models:
            pts = sorted((r for r in rows if r["model"] == model), key=lambda r: r["n_train_subjects"] or r["fraction"])
            xs = [r["n_train_subjects"] or r["fraction"] for r in pts]
            ys = [r["mean"] for r in pts]
            es = [r["std"] or 0.0 for r in pts]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=model)
        ax.set_xlabel("# обучающих испытуемых")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Low-data: {ylabel} vs объём обучения")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)
    return written


def plot_galaxy_activities(report: dict, out_path: str | Path) -> Path:
    """Bar chart of HR MAE per activity on GalaxyPPG (section 8)."""
    from butppg.data.galaxy import MOTION_ACTIVITIES

    plt = _plt()
    per = report.get("per_activity", {})
    acts = sorted(per)
    maes = [per[a]["mae"] for a in acts]
    colors = ["#d1495b" if a in MOTION_ACTIVITIES else "#4c72b0" for a in acts]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(acts, maes, color=colors)
    ov = (report.get("overall") or {}).get("mae")
    if isinstance(ov, (int, float)):
        ax.axhline(ov, color="k", ls="--", lw=1, label=f"overall {ov:.2f}")
        ax.legend(fontsize=8)
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("GalaxyPPG: HR MAE по активностям (красное — движение)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
