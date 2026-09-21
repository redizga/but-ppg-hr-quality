# Основная таблица (только PPG)

| Model | Quality Macro-F1 | ROC-AUC | Accuracy | HR MAE | HR RMSE |
|---|---|---|---|---|---|
| Majority class | 0.4494 | 0.5000 | 0.8162 | — | — |
| Median HR | — | — | — | 11.957 | 16.072 |
| Dominant-frequency HR | — | — | — | 26.986 | 36.794 |
| Features + LogReg/XGBoost | 0.6316 | 0.6611 | 0.7936 | 9.764 | 13.559 |
| 1D-CNN / ResNet1D | 0.6279 | 0.6720 | 0.8056 | 13.960 | 17.556 |
| OpenTSLM-1B | 0.4494 | — | 0.8162 | 11.406 | 14.926 |
| OpenTSLM (ECG) | — | — | — | — | — |
| SIGMA-PPG | 0.6195 | 0.7055 | 0.7936 | 11.760 | 15.721 |
