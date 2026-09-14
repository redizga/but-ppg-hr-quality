"""Map a model name to its trainer and run it with consistent run bookkeeping.

The orchestrator's ``train`` command calls :func:`run_training`; it flips the
run to ``running``, invokes the right trainer (each of which finalizes its own
predictions/metrics and flips the run to ``finished``), and on any exception
records the traceback and flips the run to ``failed`` before re-raising.
"""

from __future__ import annotations

import traceback

from butppg.orchestrator.runs import RunRecord


def run_training(run: RunRecord, cfg: dict) -> None:
    run.device = cfg.get("device")
    run.set_status("running")
    try:
        model = run.model
        if model == "trivial":
            from butppg.training.baselines import train_trivial

            train_trivial(run, cfg)
        elif model == "cnn1d":
            from butppg.training.cnn import train_cnn

            train_cnn(run, cfg)
        elif model == "sigma_ppg":
            from butppg.adapters.sigma_ppg import train_sigma_ppg

            train_sigma_ppg(run, cfg)
        elif model == "opentslm":
            from butppg.adapters.opentslm import train_opentslm

            train_opentslm(run, cfg)
        elif model == "baseline_features":
            raise NotImplementedError(
                "baseline_features (features + LogReg/XGBoost, E3) is not implemented yet."
            )
        else:
            raise ValueError(f"no trainer for model {model!r}")
    except Exception as exc:
        run.set_status("failed", error=f"{type(exc).__name__}: {exc}")
        with open(run.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + traceback.format_exc())
        raise
