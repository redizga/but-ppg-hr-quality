"""Build the per-model input marts from the E1 registry.

Two projections of the same shared registry + subject-wise split (see
``marts.py`` for the read side and the rationale):

* :func:`build_sigma_mart`   -> SIGMA-PPG: per-subject ``.npy`` tensors ``(N, 1, L)``
  resampled to SIGMA's rate, normalized, PPG-only.
* :func:`build_opentslm_mart` -> OpenTSLM: per-split ``.jsonl`` of QADataset-shaped
  prompt/series/answer records.

Ported from the standalone ``scripts/build_mart_*.py`` (Budilov's E5/E6 work,
verified on 32 real BUT PPG windows) into importable functions so the
orchestrator's ``mart`` command can call them directly instead of shelling out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from butppg.data.marts import (
    load_ppg_window,
    load_registry_with_split,
    minmax_to_pm1,
    resample_window,
    zscore,
)
from butppg.paths import PROJECT_ROOT
from butppg.utils.seed import seed_everything

SOURCE_FS = 30.0  # BUT PPG PPG rate, fixed by E1 (configs/default.yaml: window.ppg_fs)
HR_RANGE = (30, 220)  # assignment section 6: physiologically plausible HR answer range


# --------------------------------------------------------------------------- #
# SIGMA-PPG mart                                                              #
# --------------------------------------------------------------------------- #
def _sigma_split_subset(df, out_dir: Path, task: str, target_fs: float, normalize: str) -> dict:
    stats = {"subjects": 0, "windows": 0}
    for subject_id, group in df.groupby("subject_id"):
        windows, labels, record_ids = [], [], []
        for _, row in group.iterrows():
            x = load_ppg_window(row["ppg_path"], project_root=PROJECT_ROOT)
            x = resample_window(x, orig_fs=SOURCE_FS, target_fs=target_fs)
            x = zscore(x)[0] if normalize == "zscore" else minmax_to_pm1(x)
            windows.append(x)
            labels.append(row["quality_label"] if task == "quality" else row["hr_ref"])
            record_ids.append(str(row["record_id"]))
        if not windows:
            continue
        x_arr = np.stack(windows)[:, np.newaxis, :].astype(np.float32)  # (N, 1, L)
        y_arr = np.asarray(labels, dtype=np.int64 if task == "quality" else np.float32)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{subject_id}_x.npy", x_arr)
        np.save(out_dir / f"{subject_id}_y_{task}.npy", y_arr)
        # record_id kept so SIGMA's per-window predictions join back to the registry
        # (assignment section 5 requires record_id in the prediction file).
        np.save(out_dir / f"{subject_id}_record_ids.npy", np.asarray(record_ids, dtype=object), allow_pickle=True)
        stats["subjects"] += 1
        stats["windows"] += len(windows)
    return stats


def build_sigma_mart(
    registry_path: str | Path,
    split_path: str | Path,
    out_root: str | Path,
    task: str,
    target_fs: float = 50.0,
    normalize: str = "zscore",
    good_quality_only: bool = True,
    seed: int = 42,
) -> dict:
    """Write the SIGMA-PPG mart for one task. Returns the manifest dict.

    Layout (mirrors SIGMA-PPG's ``downstream/bidmc`` reference)::

        <out_root>/<task>/<split>/<subject_id>_x.npy        float32 (N, 1, L)
        <out_root>/<task>/<split>/<subject_id>_y_<task>.npy  int64|float32 (N,)
        <out_root>/<task>/manifest.json
    """
    seed_everything(seed)
    registry = load_registry_with_split(registry_path, split_path)
    if task == "hr" and good_quality_only:
        registry = registry[registry["quality_label"] == 1]

    out_task = Path(out_root) / task
    manifest = {
        "model": "sigma_ppg",
        "task": task,
        "target_fs": target_fs,
        "source_fs": SOURCE_FS,
        "normalize": normalize,
        "good_quality_only": good_quality_only if task == "hr" else None,
        "seed": seed,
        "registry": str(registry_path),
        "split_file": str(split_path),
        "splits": {},
    }
    for split_name in ("train", "val", "test"):
        subset = registry[registry["split"] == split_name]
        stats = _sigma_split_subset(subset, out_task / split_name, task, target_fs, normalize)
        manifest["splits"][split_name] = stats

    out_task.mkdir(parents=True, exist_ok=True)
    (out_task / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# OpenTSLM mart                                                               #
# --------------------------------------------------------------------------- #
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


def _acc_magnitude(acc_path: str, project_root: Path) -> np.ndarray | None:
    if not isinstance(acc_path, str) or not acc_path:
        return None
    path = Path(acc_path)
    if not path.is_absolute():
        path = project_root / path
    arr = np.load(path).astype(np.float32)  # expected (3, T)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"Expected 3-axis ACC array (3, T) at {path}, got {arr.shape}")
    return np.linalg.norm(arr, axis=0)


def _opentslm_record(row, task: str, acc_mode: str) -> dict:
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
            mag = _acc_magnitude(row["acc_path"], PROJECT_ROOT)
            if mag is not None:
                mag_norm, m_mean, m_std = zscore(mag)
                time_series.append(
                    {
                        "text": (
                            "This is the accelerometer magnitude (L2 norm of the 3 axes), "
                            f"it has mean {m_mean:.4f} and std {m_std:.4f}."
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
                a_norm, a_mean, a_std = zscore(axis_series)
                time_series.append(
                    {
                        "text": (
                            f"This is the accelerometer data on the {label}-axis, "
                            f"it has mean {a_mean:.4f} and std {a_std:.4f}."
                        ),
                        "series": a_norm.tolist(),
                    }
                )

    if task == "quality":
        answer = "good" if int(row["quality_label"]) == 1 else "bad"
        post_prompt = QUALITY_POST_PROMPT
    else:
        answer = str(int(round(float(row["hr_ref"]))))
        post_prompt = HR_POST_PROMPT

    return {
        "record_id": row["record_id"],
        "subject_id": row["subject_id"],
        "pre_prompt": PRE_PROMPT,
        "time_series": time_series,
        "post_prompt": post_prompt,
        "answer": answer,
    }


def build_opentslm_mart(
    registry_path: str | Path,
    split_path: str | Path,
    out_root: str | Path,
    task: str,
    acc_mode: str = "magnitude",
    good_quality_only: bool = True,
    seed: int = 42,
) -> dict:
    """Write the OpenTSLM mart for one task. Returns a small manifest dict.

    Layout::

        <out_root>/<task>/<split>.jsonl   (QADataset-shaped records)
        <out_root>/<task>/manifest.json
    """
    seed_everything(seed)
    registry = load_registry_with_split(registry_path, split_path)
    if task == "hr" and good_quality_only:
        registry = registry[registry["quality_label"] == 1]

    out_task = Path(out_root) / task
    out_task.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "opentslm",
        "task": task,
        "acc_mode": acc_mode,
        "good_quality_only": good_quality_only if task == "hr" else None,
        "seed": seed,
        "registry": str(registry_path),
        "split_file": str(split_path),
        "splits": {},
    }
    for split_name in ("train", "val", "test"):
        subset = registry[registry["split"] == split_name]
        out_path = out_task / f"{split_name}.jsonl"
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in subset.iterrows():
                f.write(json.dumps(_opentslm_record(row, task, acc_mode)) + "\n")
                n += 1
        manifest["splits"][split_name] = {"records": n}

    (out_task / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
