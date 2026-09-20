"""1D-CNN / ResNet1D trainer (assignment section 4).

Trains directly off the registry's raw PPG windows (no mart needed — the CNN
eats the 300-point signal as-is). One architecture serves both tasks; the loss
and the checkpoint-selection metric are what differ:

* quality -> BCEWithLogitsLoss, select best epoch by validation Macro-F1;
* hr      -> L1Loss (MAE), select best epoch by validation MAE.

CPU-capable for the small default model, so it's runnable on a laptop for a
smoke run and on the GPU box for the real thing (``config['device']``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from butppg.data.marts import load_ppg_window
from butppg.metrics import PREDICTION_COLUMNS
from butppg.metrics.quality import quality_metrics
from butppg.metrics.hr import hr_metrics
from butppg.paths import PROJECT_ROOT
from butppg.training.common import finalize_predictions, load_split_frames
from butppg.orchestrator.runs import RunRecord
from butppg.utils.progress import pbar


ACC_LEN = 1000  # 10 s @ 100 Hz


def _load_ppg_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            load_ppg_window(r["ppg_path"], project_root=PROJECT_ROOT)
            for _, r in pbar(df.iterrows(), desc="load ppg", total=len(df), leave=False)
        ]
    )


def _load_acc_matrix(df: pd.DataFrame) -> np.ndarray:
    """ACC magnitude per record, padded/truncated to ACC_LEN -> (N, 1, ACC_LEN)."""
    from pathlib import Path

    out = []
    for _, r in pbar(df.iterrows(), desc="load acc", total=len(df), leave=False):
        path = Path(r["acc_path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        acc = np.load(path).astype(np.float32)  # (3, T)
        mag = np.linalg.norm(acc, axis=0)
        if len(mag) >= ACC_LEN:
            mag = mag[:ACC_LEN]
        else:
            mag = np.pad(mag, (0, ACC_LEN - len(mag)))
        out.append(mag)
    return np.stack(out)[:, np.newaxis, :].astype(np.float32)


def _predict_frame(df: pd.DataFrame, task: str, y_pred, prob_good=None) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "record_id": df["record_id"].values,
            "subject_id": df["subject_id"].values,
            "task": task,
            "y_true": df["quality_label"].values if task == "quality" else df["hr_ref"].values,
            "y_pred": y_pred,
            "prob_good": prob_good if prob_good is not None else pd.NA,
            "raw_response": pd.NA,
            "parse_status": pd.NA,
        }
    )
    return out[PREDICTION_COLUMNS]


def train_cnn(run: RunRecord, cfg: dict) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from butppg.models.cnn1d import build_model_from_config

    task = run.task
    requested = cfg.get("device", "cpu")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is False. "
            "Check the CUDA build of torch (python -c \"import torch; print(torch.version.cuda)\")."
        )
    device = torch.device(requested)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"[cnn] training on {device} ({dev_name})")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 30))
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))

    use_acc = "acc" in cfg.get("model", {}).get("channels", ["ppg"])
    # ACC needs the ACC subset (can't feed a channel that isn't there)
    train_df, val_df, test_df = load_split_frames(cfg["registry"], cfg["split"], task, acc_subset=use_acc)

    def tensors(df):
        ppg = torch.from_numpy(_load_ppg_matrix(df)[:, np.newaxis, :].astype(np.float32))  # (N,1,300)
        acc = torch.from_numpy(_load_acc_matrix(df)) if use_acc else None                   # (N,1,1000)
        y_col = "quality_label" if task == "quality" else "hr_ref"
        y = torch.from_numpy(df[y_col].to_numpy(dtype=np.float32))
        return ppg, acc, y

    ppg_tr, acc_tr, y_tr = tensors(train_df)
    ppg_va, acc_va, y_va = tensors(val_df)
    ppg_te, acc_te, y_te = tensors(test_df)

    model = build_model_from_config(cfg).to(device)
    print(f"[cnn] input variant: {'ppg+acc' if use_acc else 'ppg-only'}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss() if task == "quality" else torch.nn.L1Loss()

    dataset = TensorDataset(ppg_tr, acc_tr, y_tr) if use_acc else TensorDataset(ppg_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def forward(ppg, acc):
        return model(ppg.to(device), acc.to(device)) if use_acc else model(ppg.to(device))

    best_score = -np.inf  # we maximize: macro-F1 (quality) or -MAE (hr)
    best_state = None

    def val_score() -> float:
        model.eval()
        with torch.no_grad():
            out = forward(ppg_va, acc_va).cpu().numpy()
        if task == "quality":
            pred = (1 / (1 + np.exp(-out)) >= 0.5).astype(int)
            return quality_metrics(y_va.numpy().astype(int), pred)["macro_f1"]
        return -hr_metrics(y_va.numpy(), out)["mae"]

    metric_name = "macro_f1" if task == "quality" else "mae"

    def log(msg: str) -> None:
        print(msg)
        with open(run.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    for epoch in range(epochs):
        model.train()
        running, n_batches = 0.0, 0
        bar = pbar(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in bar:
            if use_acc:
                ppg_b, acc_b, yb = batch
            else:
                ppg_b, yb = batch
                acc_b = None
            yb = yb.to(device)
            opt.zero_grad()
            loss = criterion(forward(ppg_b, acc_b), yb)
            loss.backward()
            opt.step()
            running += float(loss)
            n_batches += 1
            bar.set_postfix(loss=f"{running / n_batches:.4f}")
        score = val_score()
        improved = score > best_score
        if improved:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        val_display = score if task == "quality" else -score
        log(
            f"[cnn] epoch {epoch + 1:>3}/{epochs}  loss={running / max(n_batches, 1):.4f}  "
            f"val_{metric_name}={val_display:.4f}{'  *best' if improved else ''}"
        )

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"[cnn] peak GPU memory: {peak_mb:.1f} MB  (non-zero confirms training ran on the GPU)")

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = run.checkpoints_dir / "best_model.pt"
    run.ensure_dirs()
    torch.save({"model": model.state_dict(), "cfg": cfg, "task": task}, ckpt_path)
    run.best_checkpoint = str(ckpt_path)

    model.eval()
    with torch.no_grad():
        out = forward(ppg_te, acc_te).cpu().numpy()
    if task == "quality":
        prob = 1 / (1 + np.exp(-out))
        preds = _predict_frame(test_df, task, y_pred=(prob >= 0.5).astype(int), prob_good=prob)
    else:
        preds = _predict_frame(test_df, task, y_pred=out)

    finalize_predictions(run, preds)
    run.set_status("finished")
