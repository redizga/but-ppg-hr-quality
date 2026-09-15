"""Feature-based baseline trainer (assignment section 4).

Hand-crafted PPG (+ACC, +covariate) features -> a simple, fast, interpretable
model that sets the classical lower bound:

* quality -> LogisticRegression (default) or XGBoost; the decision threshold is
  chosen on the validation split (section 3), then frozen for the test report.
* hr      -> Ridge (default) or XGBoost regressor.

Input variant (section 2A) selects the feature blocks: ``ppg`` / ``ppg_acc`` /
``ppg_acc_cov``. CPU-only, no GPU needed.
"""

from __future__ import annotations

import joblib
import numpy as np

from butppg.features.extract import build_feature_matrix
from butppg.metrics.quality import select_threshold
from butppg.paths import PROJECT_ROOT
from butppg.orchestrator.runs import RunRecord
from butppg.training.common import build_prediction_frame, finalize_predictions, load_split_frames


def _fit_quality(x_tr, y_tr, estimator: str):
    if estimator == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=4,
        )
    else:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(x_tr, y_tr)
    return model


def _fit_hr(x_tr, y_tr, estimator: str):
    if estimator == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=4,
        )
    else:
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
    model.fit(x_tr, y_tr)
    return model


def train_features(run: RunRecord, cfg: dict) -> None:
    from sklearn.preprocessing import StandardScaler

    task = run.task
    variant = cfg.get("input_variant", "ppg")
    estimator = cfg.get("estimator", "logreg" if task == "quality" else "ridge")
    # extended-input variants live on the ACC subset for a fair comparison (2A)
    acc_subset = bool(cfg.get("acc_subset", False)) or variant in ("ppg_acc", "ppg_acc_cov")

    train_df, val_df, test_df = load_split_frames(cfg["registry"], cfg["split"], task, acc_subset=acc_subset)

    x_tr, names = build_feature_matrix(train_df, variant, PROJECT_ROOT)
    x_va, _ = build_feature_matrix(val_df, variant, PROJECT_ROOT)
    x_te, _ = build_feature_matrix(test_df, variant, PROJECT_ROOT)

    scaler = StandardScaler().fit(x_tr)
    x_tr, x_va, x_te = scaler.transform(x_tr), scaler.transform(x_va), scaler.transform(x_te)

    run.ensure_dirs()
    if task == "quality":
        y_tr = train_df["quality_label"].to_numpy(dtype=int)
        y_va = val_df["quality_label"].to_numpy(dtype=int)
        model = _fit_quality(x_tr, y_tr, estimator)
        prob_va = model.predict_proba(x_va)[:, 1]
        prob_te = model.predict_proba(x_te)[:, 1]
        threshold, val_f1 = select_threshold(y_va, prob_va)  # threshold chosen on val
        print(f"[features] variant={variant} est={estimator}  val macro_f1={val_f1:.4f} @thr={threshold:.3f}")
        preds = build_prediction_frame(
            test_df, "quality", y_pred=(prob_te >= threshold).astype(int), prob_good=prob_te
        )
    else:
        y_tr = train_df["hr_ref"].to_numpy(dtype=float)
        model = _fit_hr(x_tr, y_tr, estimator)
        y_pred = model.predict(x_te)
        print(f"[features] variant={variant} est={estimator}  (hr regression on {len(y_tr)} good windows)")
        preds = build_prediction_frame(test_df, "hr", y_pred=np.asarray(y_pred, dtype=float))

    ckpt = run.checkpoints_dir / "model.joblib"
    joblib.dump({"model": model, "scaler": scaler, "feature_names": names, "variant": variant}, ckpt)
    run.best_checkpoint = str(ckpt)

    finalize_predictions(run, preds)
    run.set_status("finished")
