"""OpenTSLM fine-tuning adapter (assignment section 6).

Bridges our OpenTSLM mart (per-split ``.jsonl`` of QADataset-shaped records) to
the vendored ``OpenTSLM`` package. Two pieces:

* :class:`ButPPGQADataset` — a thin ``QADataset`` subclass that reads our JSONL
  instead of downloading one of OpenTSLM's built-in datasets. This is the
  "adapt OpenTSLM to BUT PPG" step.
* :func:`train_opentslm` — fine-tunes ``OpenTSLMFlamingo`` (Llama-3.2-3B by
  default) on one task, selects the best epoch by validation loss, then
  generates on the fixed test fold and parses answers with our fixed rules
  (``models/opentslm_parsing``: good/bad or an integer in 30-220), recording
  raw text + parse status so invalid answers are never silently dropped.

The **two runs** the assignment asks for are one flag apart:
* ``config['ecg_init'] = None`` -> fine-tune from the base LLM (no ECG);
* ``config['ecg_init'] = '<path/to/ecg_stage_checkpoint.pt>'`` -> start from the
  ECG-pretrained checkpoint (tests ECG->PPG transfer).

Runs where a GPU + the ``opentslm`` package + HF access to the LLM exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from butppg.models.opentslm_parsing import responses_to_predictions
from butppg.orchestrator.runs import RunRecord
from butppg.paths import OPENTSLM_DIR
from butppg.training.common import finalize_predictions
from butppg.utils.progress import pbar


def _require_opentslm():
    src = OPENTSLM_DIR / "src"
    if not (src / "opentslm").is_dir():
        raise FileNotFoundError(
            f"OpenTSLM repo not found at {OPENTSLM_DIR}. Clone github.com/OpenTSLM/OpenTSLM there "
            "and `pip install -e OpenTSLM` (or `uv sync`) so `opentslm` is importable."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _make_dataset_class():
    """Build the ButPPGQADataset subclass (imported lazily so opentslm is optional)."""
    from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
    from opentslm.time_series_datasets.QADataset import QADataset

    class ButPPGQADataset(QADataset):
        """Reads our OpenTSLM mart JSONL. ``JSONL_DIR`` must be set before use."""

        JSONL_DIR: Path | None = None

        def _load_splits(self):
            base = Path(type(self).JSONL_DIR)
            def read(name):
                path = base / f"{name}.jsonl"
                with open(path, encoding="utf-8") as f:
                    return [json.loads(line) for line in f]
            return read("train"), read("val"), read("test")

        def _get_answer(self, row) -> str:
            return str(row["answer"])

        def _get_pre_prompt(self, row) -> str:
            return row["pre_prompt"]

        def _get_post_prompt(self, row) -> str:
            return row["post_prompt"]

        def _get_text_time_series_prompt_list(self, row):
            return [TextTimeSeriesPrompt(ts["text"], ts["series"]) for ts in row["time_series"]]

    return ButPPGQADataset


def _reset_dataset_cache(ds_cls):
    """QADataset caches splits at class level; clear it between tasks/dirs."""
    for attr in ("loaded", "_train_dataset", "_validation_dataset", "_test_dataset"):
        if hasattr(ds_cls, attr):
            delattr(ds_cls, attr)


def _read_test_rows(jsonl_dir: Path) -> pd.DataFrame:
    with open(jsonl_dir / "test.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return pd.DataFrame(rows)


def train_opentslm(run: RunRecord, cfg: dict) -> None:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader

    _require_opentslm()
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )

    task = run.task
    jsonl_dir = Path(cfg["mart_dir"]) / task
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    llm_id = cfg.get("llm_id", "meta-llama/Llama-3.2-3B")
    ecg_init = cfg.get("ecg_init")  # None or a checkpoint path
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 5))
    batch_size = int(train_cfg.get("batch_size", 4))
    lr = float(train_cfg.get("lr", 1e-4))
    max_new_tokens = int(cfg.get("max_new_tokens", 32))

    model = OpenTSLMFlamingo(device=device, llm_id=llm_id)
    if ecg_init:
        # ECG->PPG transfer: warm-start from the ECG-stage checkpoint.
        model.load_from_file(ecg_init)

    ds_cls = _make_dataset_class()
    _reset_dataset_cache(ds_cls)
    ds_cls.JSONL_DIR = jsonl_dir
    eos = model.get_eos_token()
    train_ds = ds_cls("train", EOS_TOKEN=eos)
    val_ds = ds_cls("validation", EOS_TOKEN=eos)
    test_ds = ds_cls("test", EOS_TOKEN=eos)

    def collate(batch):
        return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=PATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    run.ensure_dirs()
    ckpt_path = run.checkpoints_dir / "best_model.pt"
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        running, n_tr = 0.0, 0
        bar = pbar(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in bar:
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            running += float(loss)
            n_tr += 1
            bar.set_postfix(loss=f"{running / n_tr:.4f}")

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in pbar(val_loader, desc=f"val {epoch + 1}/{epochs}", leave=False):
                val_loss += float(model.compute_loss(batch))
                n += 1
        val_loss = val_loss / max(n, 1)
        _log(run, f"epoch {epoch + 1}/{epochs}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            model.store_to_file(str(ckpt_path))
            run.best_checkpoint = str(ckpt_path)
            run.save()

    # Best checkpoint -> generate on the fixed test fold.
    if ckpt_path.exists():
        model.load_from_file(str(ckpt_path))
    model.eval()
    raw_responses: List[str] = []
    with torch.no_grad():
        for batch in pbar(test_loader, desc="generate (test)"):
            preds = model.generate(batch, max_new_tokens=max_new_tokens)
            raw_responses.extend(preds if isinstance(preds, list) else [preds])

    test_rows = _read_test_rows(jsonl_dir)
    predictions = responses_to_predictions(test_rows, raw_responses, task=task)
    finalize_predictions(run, predictions)
    run.set_status("finished")


def _log(run: RunRecord, msg: str) -> None:
    with open(run.log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)
