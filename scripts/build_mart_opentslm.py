#!/usr/bin/env python
"""Build the OpenTSLM input mart from the E1 registry.

Epic E5 (Budilov). Data-shaping step between "registry exists" (E1) and
"OpenTSLM can fine-tune" (train_opentslm.py). Does not touch the OpenTSLM
package itself — it produces one JSONL file per (task, split) whose records
carry exactly what OpenTSLM's abstract ``QADataset`` needs (see
``src/opentslm/time_series_datasets/QADataset.py`` in
github.com/OpenTSLM/OpenTSLM): a pre-prompt, a list of (text-label, series)
pairs, a post-prompt and an answer string. A thin ``QADataset`` subclass
(written in E5, inside a vendored/installed OpenTSLM checkout) reads these
JSONL files directly — see the docstring of ``_build_record`` for the exact
field mapping.

Prompt/answer format follows the assignment's fixed parsing rules (section 6):
  * quality: answer is exactly "good" or "bad".
  * hr: answer is the reference HR as an integer string, 30-220 bpm range.
Both post-prompts end with "Answer: " so response parsing has a fixed anchor,
matching the pattern used by OpenTSLM's own PAMAP2/TSQA dataset classes.

ACC handling: stage-1 default is "magnitude" (one extra time-series channel,
the L2 norm of the 3-axis signal), per configs/models/opentslm.yaml
`acc_mode: magnitude` and assignment section 6 ("подавать как модуль
ускорения, чтобы не усложнять архитектуру"). Pass --acc-mode none to build
the PPG-only variant, or --acc-mode axes for the (lower-priority, "if time
remains") 3-channel variant.

Output:
    <out>/<task>/<split>.jsonl

Usage
-----
    python scripts/build_mart_opentslm.py --task quality \\
        --registry data/processed/registry.csv \\
        --split splits/but_ppg_60_20_20.json \\
        --out artifacts/data_marts/opentslm

    python scripts/build_mart_opentslm.py --task hr --acc-mode magnitude \\
        --registry data/processed/registry.csv \\
        --split splits/but_ppg_60_20_20.json \\
        --out artifacts/data_marts/opentslm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from butppg.data.marts import load_ppg_window, load_registry_with_split, zscore  # noqa: E402
from butppg.paths import PROJECT_ROOT  # noqa: E402
from butppg.utils.seed import seed_everything  # noqa: E402

HR_RANGE = (30, 220)  # assignment section 6: physiologically plausible answer range

QUALITY_POST_PROMPT = (
    "Decide whether this 10-second PPG window is usable for further analysis.\n"
    "Answer ONLY with 'good' or 'bad'.\n"
    "You MUST end your response with 'Answer: <good|bad>'"
)

HR_POST_PROMPT = (
    f"Estimate the heart rate in beats per minute for this 10-second PPG window.\n"
    f"Answer with a single integer between {HR_RANGE[0]} and {HR_RANGE[1]}.\n"
    "You MUST end your response with 'Answer: <hr_bpm>'"
)

PRE_PROMPT = (
    "You are given a photoplethysmography (PPG) signal recorded at a fingertip, "
    "sampled at 30 Hz for 10 seconds (300 points)."
)


def acc_magnitude(acc_path: str, project_root: Path) -> np.ndarray | None:
    """L2 norm across the 3 ACC axes; returns None if acc_path is missing/empty."""
    if not isinstance(acc_path, str) or not acc_path:
        return None
    path = Path(acc_path)
    if not path.is_absolute():
        path = project_root / path
    arr = np.load(path).astype(np.float32)  # expected shape (3, T) per E1
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"Expected 3-axis ACC array (3, T) at {path}, got {arr.shape}")
    return np.linalg.norm(arr, axis=0)


def build_record(row, task: str, acc_mode: str) -> dict:
    """One QADataset-shaped sample.

    Field mapping to OpenTSLM's QADataset abstract methods:
        pre_prompt   -> _get_pre_prompt
        time_series  -> _get_text_time_series_prompt_list (each item becomes one
                         TextTimeSeriesPrompt(text, series))
        post_prompt  -> _get_post_prompt
        answer       -> _get_answer
    record_id/subject_id are extra fields (not part of QADataset's contract)
    kept so predictions can be joined back to the registry at eval time, per
    the assignment's required prediction format (section 5).
    """
    ppg = load_ppg_window(row["ppg_path"], project_root=PROJECT_ROOT)
    ppg_norm, mean, std = zscore(ppg)

    time_series = [
        {
            "text": f"This is the PPG signal, it has mean {mean:.4f} and std {std:.4f}.",
            "series": ppg_norm.tolist(),
        }
    ]

    if acc_mode != "none" and bool(row.get("has_acc")) and row.get("acc_path"):
        if acc_mode == "magnitude":
            mag = acc_magnitude(row["acc_path"], PROJECT_ROOT)
            if mag is not None:
                mag_norm, mag_mean, mag_std = zscore(mag)
                time_series.append(
                    {
                        "text": (
                            "This is the accelerometer magnitude (L2 norm of the 3 axes), "
                            f"it has mean {mag_mean:.4f} and std {mag_std:.4f}."
                        ),
                        "series": mag_norm.tolist(),
                    }
                )
        elif acc_mode == "axes":
            path = Path(row["acc_path"])
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            axes = np.load(path).astype(np.float32)
            for label, axis_series in zip(("x", "y", "z"), axes):
                axis_norm, axis_mean, axis_std = zscore(axis_series)
                time_series.append(
                    {
                        "text": (
                            f"This is the accelerometer data on the {label}-axis, "
                            f"it has mean {axis_mean:.4f} and std {axis_std:.4f}."
                        ),
                        "series": axis_norm.tolist(),
                    }
                )

    if task == "quality":
        answer = "good" if int(row["quality_label"]) == 1 else "bad"
        post_prompt = QUALITY_POST_PROMPT
    else:
        hr = int(round(float(row["hr_ref"])))
        answer = str(hr)
        post_prompt = HR_POST_PROMPT

    return {
        "record_id": row["record_id"],
        "subject_id": row["subject_id"],
        "pre_prompt": PRE_PROMPT,
        "time_series": time_series,
        "post_prompt": post_prompt,
        "answer": answer,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task", choices=["quality", "hr"], required=True)
    p.add_argument("--acc-mode", choices=["none", "magnitude", "axes"], default="magnitude")
    p.add_argument(
        "--good-quality-only",
        action="store_true",
        default=True,
        help="task=hr only: train HR only on quality_label==1 windows (assignment section 2B). Default: on.",
    )
    p.add_argument("--no-good-quality-only", dest="good_quality_only", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    registry = load_registry_with_split(args.registry, args.split)

    if args.task == "hr" and args.good_quality_only:
        before = len(registry)
        registry = registry[registry["quality_label"] == 1]
        print(f"[hr] filtered to quality_label==1: {before} -> {len(registry)} windows")

    out_dir = Path(args.out) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        subset = registry[registry["split"] == split_name]
        out_path = out_dir / f"{split_name}.jsonl"
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in subset.iterrows():
                record = build_record(row, task=args.task, acc_mode=args.acc_mode)
                f.write(json.dumps(record) + "\n")
                n += 1
        print(f"[{args.task}/{split_name}] {n} records -> {out_path}")


if __name__ == "__main__":
    main()
