"""SIGMA-PPG fine-tuning adapter (assignment section 7).

Bridges our SIGMA-PPG mart (per-subject ``.npy`` at ``target_fs``) to the
vendored ``SigmaPPG`` repo's model. Deliberately does NOT reuse their
``downstream/bidmc`` trainer: that one merges train+test into a 5-fold CV and
piles on MixUp/TTA/SWA, which would break this assignment's fixed subject-wise
test set and fair-comparison protocol. We instead:

* load the model via their ``downstream.model_select.select_model`` (the public,
  documented entry point) with the released checkpoint;
* train on the **train** fold only, select the best epoch on the **val** fold
  (Macro-F1 for quality, MAE for hr), and report once on the fixed **test** fold;
* write predictions in the canonical schema so the same evaluator scores SIGMA
  exactly like every other model.

Runs where a GPU + SIGMA-PPG's deps + the pretrained checkpoint exist. On a
machine without them it fails with an actionable message rather than silently.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from butppg.metrics import PREDICTION_COLUMNS
from butppg.metrics.hr import hr_metrics
from butppg.metrics.quality import quality_metrics
from butppg.paths import SIGMA_DIR
from butppg.orchestrator.runs import RunRecord
from butppg.training.common import finalize_predictions


def _require_sigma_repo():
    if not (SIGMA_DIR / "downstream" / "model_select.py").exists():
        raise FileNotFoundError(
            f"SIGMA-PPG repo not found at {SIGMA_DIR}. Clone github.com/ZonghengGuo/SigmaPPG there."
        )
    if str(SIGMA_DIR) not in sys.path:
        sys.path.insert(0, str(SIGMA_DIR))


def _load_split_arrays(mart_task_dir, split: str, task: str):
    """Concatenate every subject's ``.npy`` in one split into (X, y, subjects, record_ids)."""
    split_dir = mart_task_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"mart split missing: {split_dir} — run `orch mart --model sigma_ppg` first")
    xs, ys, subs, rids = [], [], [], []
    for x_file in sorted(split_dir.glob("*_x.npy")):
        subject = x_file.name[: -len("_x.npy")]
        x = np.load(x_file)
        y = np.load(split_dir / f"{subject}_y_{task}.npy")
        rid_file = split_dir / f"{subject}_record_ids.npy"
        record_ids = (
            list(np.load(rid_file, allow_pickle=True))
            if rid_file.exists()
            else [f"{subject}_{i}" for i in range(len(y))]
        )
        xs.append(x)
        ys.append(y)
        subs.extend([subject] * len(y))
        rids.extend(str(r) for r in record_ids)
    if not xs:
        raise ValueError(f"no subjects found in {split_dir}")
    return np.concatenate(xs), np.concatenate(ys), subs, rids


def train_sigma_ppg(run: RunRecord, cfg: dict) -> None:
    import torch
    from einops import rearrange
    from torch.utils.data import DataLoader, TensorDataset

    _require_sigma_repo()
    from downstream.model_select import select_model  # type: ignore

    from pathlib import Path

    task = run.task
    mart_task_dir = Path(cfg["mart_dir"]) / task
    manifest = cfg.get("mart_manifest", {})
    target_fs = int(manifest.get("target_fs", cfg.get("target_fs", 50)))
    input_len = int(manifest.get("input_len", cfg.get("input_len", 500)))  # 10 s @ 50 Hz
    patch_size = int(cfg.get("patch_size", target_fs))  # SIGMA: one patch == one second
    checkpoint_path = cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError(
            "SIGMA-PPG needs the pretrained checkpoint: pass config.checkpoint_path "
            "(download from huggingface.co/zonhengu/sigmappg)."
        )

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("lr", 5e-4))

    model, use_patches = select_model(
        backbone=cfg.get("backbone", "sigma_ppg_pro"),
        num_classes=1,
        in_chans=1,
        pretrained=True,
        checkpoint_path=checkpoint_path,
        freeze_backbone_flag=cfg.get("freeze_backbone", False),
        device=device,
        patch_size=patch_size,
        input_size=input_len,
    )
    model = model.to(device)

    def load(split):
        X, y, subs, rids = _load_split_arrays(mart_task_dir, split, task)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        return X, y, subs, rids

    X_tr, y_tr, _, _ = load("train")
    X_va, y_va, _, _ = load("val")
    X_te, y_te, sub_te, rid_te = load("test")

    def maybe_patch(x):
        if use_patches and x.shape[-1] == input_len:
            return rearrange(x, "b c (n t) -> b c n t", t=patch_size)
        return x

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 0.05))
    criterion = torch.nn.BCEWithLogitsLoss() if task == "quality" else torch.nn.L1Loss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)

    best_score, best_state = -np.inf, None

    def val_score():
        model.eval()
        with torch.no_grad():
            out = model(maybe_patch(X_va.to(device))).squeeze(-1).cpu().numpy()
        if task == "quality":
            pred = (1 / (1 + np.exp(-out)) >= 0.5).astype(int)
            return quality_metrics(y_va.numpy().astype(int), pred)["macro_f1"]
        return -hr_metrics(y_va.numpy(), out)["mae"]

    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = maybe_patch(xb.to(device)), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        score = val_score()
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    run.ensure_dirs()
    ckpt = run.checkpoints_dir / "best_model.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "task": task}, ckpt)
    run.best_checkpoint = str(ckpt)

    model.eval()
    with torch.no_grad():
        out = model(maybe_patch(X_te.to(device))).squeeze(-1).cpu().numpy()

    if task == "quality":
        prob = 1 / (1 + np.exp(-out))
        y_pred, prob_good = (prob >= 0.5).astype(int), prob
    else:
        y_pred, prob_good = out, None

    preds = pd.DataFrame(
        {
            "record_id": rid_te,
            "subject_id": sub_te,
            "task": task,
            "y_true": y_te.numpy(),
            "y_pred": y_pred,
            "prob_good": prob_good if prob_good is not None else pd.NA,
            "raw_response": pd.NA,
            "parse_status": pd.NA,
        }
    )[PREDICTION_COLUMNS]

    finalize_predictions(run, preds)
    run.set_status("finished")
