"""Trivial baseline trainers (assignment section 4 lower bounds).

These need no GPU and run anywhere, so they're the ones that give the
orchestrator a real, comparable first row in the results table immediately
after the marts exist.

* quality -> MajorityClassQuality
* hr      -> MedianHR (learned) and DominantFrequencyHR (signal-processing);
             ``--model trivial`` uses MedianHR as the recorded baseline, since
             it is the "always predict the training statistic" floor the
             assignment names; DominantFrequencyHR is available too via
             ``config['method'] = 'dominant_frequency'``.
"""

from __future__ import annotations

from butppg.models.trivial import DominantFrequencyHR, MajorityClassQuality, MedianHR
from butppg.paths import PROJECT_ROOT
from butppg.training.common import finalize_predictions, load_split_frames
from butppg.orchestrator.runs import RunRecord


def train_trivial(run: RunRecord, cfg: dict) -> None:
    registry = cfg["registry"]
    split = cfg["split"]
    task = run.task
    train_df, _val_df, test_df = load_split_frames(registry, split, task)

    if task == "quality":
        model = MajorityClassQuality()
        model.fit(train_df)
        preds = model.predict(test_df)
    else:
        method = cfg.get("method", "median")
        if method == "dominant_frequency":
            model = DominantFrequencyHR()
            model.fit(train_df)
            preds = model.predict(test_df, project_root=PROJECT_ROOT)
        else:
            model = MedianHR()
            model.fit(train_df)
            preds = model.predict(test_df)

    finalize_predictions(run, preds)
    run.set_status("finished")
