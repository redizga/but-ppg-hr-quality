"""Model implementations, added per epic.

* trivial baselines (majority / median-HR / dominant-frequency) — E2 (Budilov)
* feature baselines (LogReg / XGBoost)                           — E3 (Golikov)
* 1D-CNN / ResNet1D (quality + HR)                               — E4 (Budilov)
* OpenTSLM adapter (no-ECG / ECG-init)                           — E5 (Budilov)
* SIGMA-PPG adapter (fine-tune, PPG-only)                        — E6 (Golikov)

Each model exposes the same tiny contract: ``fit(train, val, cfg)`` and a
``predict(records) -> DataFrame`` in the canonical prediction schema, so the
one evaluator scores them all identically.
"""
