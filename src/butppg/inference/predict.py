"""Run a *trained* model on new data from its saved checkpoint (no retraining).

Assignment section 8 (external validation on GalaxyPPG) needs the models trained
on BUT PPG to predict on another dataset without any further training. This
module reloads a finished run's checkpoint, rebuilds the model, and writes
predictions + metrics for a chosen target — the fixed BUT PPG test fold (to
reproduce a run's numbers from its weights) or an external registry/mart such as
GalaxyPPG.

Each model reuses exactly the inference path its trainer uses, so
``orch predict`` and ``orch train`` score identically. Registry-based models
(trivial / baseline_features / cnn1d) read a registry slice; mart-based models
(sigma_ppg / opentslm) read a per-model mart directory.

An optional ``activity`` column on the target registry is carried into the
prediction frame so section 8's per-activity metrics can be computed downstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from butppg.data.marts import load_registry_with_split
from butppg.metrics import PREDICTION_COLUMNS
from butppg.metrics.predictions import save_predictions
from butppg.evaluation.evaluate import evaluate_prediction_file
from butppg.orchestrator.runs import RunRecord, create_run, load_run
from butppg.paths import PROJECT_ROOT


# --------------------------------------------------------------------------- #
# target loading
# --------------------------------------------------------------------------- #
def load_target_frame(registry_path, split_path=None, split: str = "test") -> pd.DataFrame:
    """Registry slice to predict on.

    With ``split_path`` -> that fold of a BUT PPG-style split (default ``test``).
    Without it -> the whole registry (external datasets like GalaxyPPG have no
    train/val/test split; every window is a test window).
    """
    reg = pd.read_csv(registry_path) if split_path is None else load_registry_with_split(registry_path, split_path)
    if split_path is not None and split != "all":
        reg = reg[reg["split"] == split].reset_index(drop=True)
    return reg.reset_index(drop=True)


def _frame(df: pd.DataFrame, task: str, y_pred, prob_good=None, raw=None, parse=None) -> pd.DataFrame:
    y_true = df["quality_label"].values if task == "quality" else df["hr_ref"].values
    out = pd.DataFrame(
        {
            "record_id": df["record_id"].values,
            "subject_id": df["subject_id"].values,
            "task": task,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_good": prob_good if prob_good is not None else pd.NA,
            "raw_response": raw if raw is not None else pd.NA,
            "parse_status": parse if parse is not None else pd.NA,
        }
    )[PREDICTION_COLUMNS]
    return out


# --------------------------------------------------------------------------- #
# per-model inference (mirrors each trainer's test-time path)
# --------------------------------------------------------------------------- #
def _predict_trivial(run: RunRecord, cfg: dict, target: pd.DataFrame) -> pd.DataFrame:
    """Trivial baselines carry no weights — refit the training statistic on the
    original train fold (from the run's config) and apply it to the target."""
    from butppg.models.trivial import DominantFrequencyHR, MajorityClassQuality, MedianHR
    from butppg.training.common import load_split_frames

    train_df, _, _ = load_split_frames(cfg["registry"], cfg["split"], run.task)
    if run.task == "quality":
        m = MajorityClassQuality(); m.fit(train_df); preds = m.predict(target)
    elif cfg.get("method") == "dominant_frequency":
        m = DominantFrequencyHR(); m.fit(train_df); preds = m.predict(target, project_root=PROJECT_ROOT)
    else:
        m = MedianHR(); m.fit(train_df); preds = m.predict(target)
    return preds


def _predict_features(run: RunRecord, cfg: dict, target: pd.DataFrame) -> pd.DataFrame:
    import joblib

    from butppg.features.extract import build_feature_matrix
    from butppg.metrics.quality import select_threshold

    bundle = joblib.load(run.best_checkpoint)
    model, scaler, names, variant = bundle["model"], bundle["scaler"], bundle["feature_names"], bundle["variant"]
    X, cols = build_feature_matrix(target, variant, PROJECT_ROOT)
    # align columns to the trained feature order (stable even if a block is absent)
    Xdf = pd.DataFrame(X, columns=cols).reindex(columns=names, fill_value=0.0)
    Xs = scaler.transform(Xdf.to_numpy(dtype=np.float32))
    if run.task == "quality":
        prob = model.predict_proba(Xs)[:, 1]
        thr = float(cfg.get("threshold", 0.5))
        return _frame(target, "quality", y_pred=(prob >= thr).astype(int), prob_good=prob)
    y = model.predict(Xs)
    return _frame(target, "hr", y_pred=np.asarray(y, dtype=float))


def _predict_cnn(run: RunRecord, cfg: dict, target: pd.DataFrame) -> pd.DataFrame:
    import torch

    from butppg.models.cnn1d import build_model_from_config
    from butppg.training.cnn import _load_acc_matrix, _load_ppg_matrix

    ckpt = torch.load(run.best_checkpoint, map_location="cpu", weights_only=False)
    ccfg = ckpt.get("cfg", cfg)
    model = build_model_from_config(ccfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    use_acc = "acc" in ccfg.get("model", {}).get("channels", ["ppg"])
    ppg = torch.from_numpy(_load_ppg_matrix(target)[:, np.newaxis, :].astype(np.float32))
    with torch.no_grad():
        out = (model(ppg, torch.from_numpy(_load_acc_matrix(target))) if use_acc else model(ppg)).cpu().numpy()
    if run.task == "quality":
        prob = 1 / (1 + np.exp(-out))
        return _frame(target, "quality", y_pred=(prob >= 0.5).astype(int), prob_good=prob)
    return _frame(target, "hr", y_pred=out)


def _predict_sigma(run: RunRecord, cfg: dict, target_mart: Path) -> pd.DataFrame:
    import sys

    import torch
    from einops import rearrange

    from butppg.adapters.sigma_ppg import _load_split_arrays, _require_sigma_repo

    _require_sigma_repo()
    from downstream.model_select import select_model  # type: ignore  # noqa: E402

    task = run.task
    manifest = cfg.get("mart_manifest", {})
    target_fs = int(manifest.get("target_fs", cfg.get("target_fs", 50)))
    input_len = int(manifest.get("input_len", cfg.get("input_len", 500)))
    patch_size = int(cfg.get("patch_size", target_fs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, use_patches = select_model(
        backbone=cfg.get("backbone", "sigma_ppg_pro"), num_classes=1, in_chans=1,
        pretrained=True, checkpoint_path=cfg["checkpoint_path"], freeze_backbone_flag=False,
        device=device, patch_size=patch_size, input_size=input_len,
    )
    ckpt = torch.load(run.best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    X, y, subs, rids = _load_split_arrays(target_mart, "test", task)
    X = torch.from_numpy(X.astype(np.float32))

    def maybe_patch(x):
        return rearrange(x, "b c (n t) -> b c n t", t=patch_size) if (use_patches and x.shape[-1] == input_len) else x

    with torch.no_grad():
        out = model(maybe_patch(X.to(device))).squeeze(-1).cpu().numpy()
    df = pd.DataFrame({"record_id": [str(r) for r in rids], "subject_id": subs, "hr_ref": y, "quality_label": y})
    if task == "quality":
        prob = 1 / (1 + np.exp(-out))
        return _frame(df, "quality", y_pred=(prob >= 0.5).astype(int), prob_good=prob)
    return _frame(df, "hr", y_pred=out)


def _predict_opentslm(run: RunRecord, cfg: dict, target_mart: Path) -> pd.DataFrame:
    import torch

    from butppg.adapters.opentslm import (
        _make_dataset_class, _read_test_rows, _require_opentslm, _reset_dataset_cache,
    )
    from butppg.models.opentslm_parsing import responses_to_predictions

    _require_opentslm()
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo  # noqa: E402
    from opentslm.model_config import PATCH_SIZE  # noqa: E402
    from opentslm.time_series_datasets.util import (  # noqa: E402
        extend_time_series_to_match_patch_size_and_aggregate,
    )
    from torch.utils.data import DataLoader

    task = run.task
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OpenTSLMFlamingo(device=device, llm_id=cfg.get("llm_id", "meta-llama/Llama-3.2-3B"))
    try:
        model.model.to(device=device, dtype=torch.float32)
        model.model.vision_encoder.visual.to(device=device, dtype=torch.float32)
    except AttributeError:
        pass
    # load our compact (trainable-only) checkpoint
    state = torch.load(run.best_checkpoint, map_location=device, weights_only=False)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            if n in params:
                params[n].data.copy_(v.to(device))
    model.eval()

    ds_cls = _make_dataset_class(); _reset_dataset_cache(ds_cls); ds_cls.JSONL_DIR = target_mart
    test_ds = ds_cls("test", EOS_TOKEN=model.get_eos_token())
    loader = DataLoader(test_ds, batch_size=int(cfg.get("train", {}).get("batch_size", 4)), shuffle=False,
                        collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=PATCH_SIZE))
    raw = []
    with torch.no_grad():
        for batch in loader:
            preds = model.generate(batch, max_new_tokens=int(cfg.get("max_new_tokens", 32)))
            raw.extend(preds if isinstance(preds, list) else [preds])
    rows = _read_test_rows(target_mart)
    truth = "quality_label" if task == "quality" else "hr_ref"
    if truth not in rows.columns or rows[truth].isna().all():
        rows[truth] = (rows["answer"].astype(str).str.strip().str.lower().eq("good").astype(int)
                       if task == "quality" else rows["answer"].astype(float))
    return responses_to_predictions(rows, raw, task=task)


_REGISTRY_MODELS = {"trivial": _predict_trivial, "baseline_features": _predict_features, "cnn1d": _predict_cnn}
_MART_MODELS = {"sigma_ppg": _predict_sigma, "opentslm": _predict_opentslm}


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def predict_run(
    source_run_id: str,
    *,
    registry: str | None = None,
    split_path: str | None = None,
    split: str = "test",
    mart_dir: str | None = None,
    tag: str = "predict",
) -> RunRecord:
    """Load ``source_run_id``'s weights and predict on the chosen target.

    Writes a NEW run ``<model>_<task>_<tag>_...`` with the canonical predictions
    and metrics, so ``orch results`` / ``orch table`` pick it up like any other.
    """
    src = load_run(source_run_id)
    cfg = dict(src.config or {})
    model_kind, task = src.model, src.task

    out_run = create_run(model_kind, task, config={**cfg, "predict_from": source_run_id, "predict_tag": tag})
    out_run.best_checkpoint = src.best_checkpoint
    out_run.set_status("running")
    try:
        if model_kind in _REGISTRY_MODELS:
            reg = registry or cfg.get("registry")
            target = load_target_frame(reg, split_path, split)
            preds = _REGISTRY_MODELS[model_kind](src, cfg, target)
        elif model_kind in _MART_MODELS:
            md = Path(mart_dir or cfg["mart_dir"]) / task
            preds = _MART_MODELS[model_kind](src, cfg, md)
        else:
            raise ValueError(f"no predict path for model {model_kind!r}")

        out_run.ensure_dirs()
        pred_path = out_run.predictions_dir / "test_predictions.csv"
        save_predictions(preds, pred_path)
        result = evaluate_prediction_file(pred_path)
        (out_run.metrics_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        out_run.metrics = result["metrics"]
        out_run.set_status("finished")
    except Exception as exc:  # noqa: BLE001
        out_run.set_status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    return out_run
