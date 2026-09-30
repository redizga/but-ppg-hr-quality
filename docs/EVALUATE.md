# Воспроизведение результатов

Как заново получить итоговые таблицы и метрики исследования **без переобучения**.
Для проверяющего: работает на чистой Ubuntu 24.04; **для таблиц GPU, датасет и
веса не нужны** — предсказания и метрики каждой модели уже лежат в репозитории
(папки `runs/<id>/`), а команды `orch table` / `orch cascade` просто агрегируют их.

Полное переобучение с нуля (если понадобится) — в [DEPLOY.md](DEPLOY.md).

---

## A. Пересобрать таблицы результатов (быстро, без GPU/датасета/весов)

```bash
git clone https://github.com/redizga/but-ppg-hr-quality.git
cd but-ppg-hr-quality
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # только ядро: numpy/scipy/pandas/scikit-learn (без torch)

# 1) основная таблица + таблица расширенных входов -> results/tables/
orch table

# 2) каскад quality->HR (раздел 2Б) -> results/tables/cascade_*.json
orch cascade \
  --quality-run baseline_features_quality_20260920-210038_1282 \
  --hr-run      baseline_features_hr_20260920-210224_29b0
```

Результат — файлы в `results/tables/`:
- `comparison_main.{md,csv}` — основная таблица «только PPG» (все модели × Quality/HR);
- `comparison_extended_inputs.{md,csv}` — Quality Macro-F1 по вариантам входа
  (PPG / PPG+ACC / PPG+ACC+cov);
- `cascade_*.json` — метрики сквозного сценария «качество → пульс».

Проверить, что совпало с зафиксированным в репозитории:
```bash
git status results/            # чисто = воспроизвелось байт-в-байт
cat results/tables/comparison_main.md
```

> Как это работает: каждый `orch train` сохраняет `runs/<id>/metrics/metrics.json`
> и `runs/<id>/predictions/test_predictions.csv`. `orch table` читает метрики
> завершённых прогонов, `orch cascade` — предсказания двух прогонов. Веса моделей
> для этого не требуются.
>
> RUN_ID всех прогонов — `orch results`. Если запустишь новый `orch train`, таблица
> подхватит **последний завершённый** прогон соответствующей модели/задачи.

---

## B. Скачать обученные веса (для запуска моделей / инспекции)

Веса в репозиторий не входят (тяжёлые) — они выложены отдельным архивом:

**Ссылка на архив весов:** `<ССЫЛКА_НА_АРХИВ_ВЕСОВ>`  (~1.3 ГБ)

```bash
# скачать архив по ссылке выше, затем распаковать в корень проекта:
tar xzf runs_full.tgz          # разворачивает чекпойнты в runs/<id>/checkpoints/
```

Что внутри `runs/<id>/checkpoints/`:
- `baseline_features_*` — `model.joblib` (модель + scaler + список признаков);
- `cnn1d_*` — `best_model.pt` (state_dict + конфиг);
- `sigma_ppg_*` — `best_model.pt` (state_dict + конфиг);
- `opentslm_*` — `best_model.pt` (только обучаемые слои; замороженная Llama тянется
  с HuggingFace при загрузке).

> Отдельной команды «загрузить веса → предсказать» (`orch predict`) в текущей
> версии нет: таблицы воспроизводятся из сохранённых предсказаний (раздел A), а
> полное воспроизведение через веса — это переобучение по [DEPLOY.md](DEPLOY.md).
> Если нужен именно инференс из чекпойнта одной командой — напишите, добавим.

---

## C. Полное воспроизведение с нуля (переобучение, GPU)

Если нужно проверить весь путь — загрузка датасета, витрины, обучение всех
моделей — см. [DEPLOY.md](DEPLOY.md): установка окружения, скачивание BUT PPG
(`orch ingest`/`process`/`mart`), обучение (`orch train`), сборка таблиц. Тяжёлые
модели (SIGMA-PPG, OpenTSLM) требуют NVIDIA GPU; тривиальные и feature-baseline
гоняются на CPU.
