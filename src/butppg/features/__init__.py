"""Hand-crafted PPG/ACC/covariate features for the LogReg/XGBoost baseline (E3)."""

from butppg.features.extract import INPUT_VARIANTS, build_feature_matrix

__all__ = ["INPUT_VARIANTS", "build_feature_matrix"]
