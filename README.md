# BUT PPG — качество сигнала и предсказание HR

Воспроизводимое исследование на датасете **BUT PPG v2.0.0**: бинарная классификация
качества 10-секундного PPG-сигнала (задача A) и предсказание частоты сердечных
сокращений — HR (задача B). Сравниваются простые baseline-модели, **OpenTSLM** и
**SIGMA-PPG**; проверяется перенос знаний ECG→PPG, режим малого объёма обучающих
данных и внешняя устойчивость на **GalaxyPPG**.

> **Статус: E0 готов, E1/E2 реализованы и проверены (в т.ч. на реальных данных), часть
> E4/E5 готова.** E0 — каркас репозитория, окружение, конфиги, seed, манифест запуска.
> E1 — загрузка BUT PPG с PhysioNet, реестр, subject-wise разбиение (`prepare_but_ppg.py`,
> `make_splits.py`) — код рабочий и проверен на настоящих данных, полный прогон на всех
> 3888 записях ещё не выполнен. E2 — метрики, формат предсказаний, оценщик, тривиальные
> baseline — полностью готовы. E4 — архитектура 1D-CNN/ResNet1D готова, цикл обучения —
> нет. E5 — правила разбора ответа OpenTSLM и построение входной витрины готовы, сама
> адаптация модели — нет. Остальное (E3, E6 дообучение, E7, E8, E9) — стабы с понятными
> CLI, бросают `NotImplementedError` с указанием эпика и владельца. Разбор задания,
> сверка с планом и подробности по каждому куску — в [docs/review/](docs/review/)
> (начать с [docs/review/summary.md](docs/review/summary.md)).

## Установка окружения

```bash
# conda
conda env create -f environment.yml
conda activate butppg

# либо venv + pip
python -m venv .venv && source .venv/bin/activate
pip install -e .            # репро-ядро (numpy/scipy/pandas/sklearn/pyyaml)
# по мере необходимости подключать тяжёлые стеки:
pip install -e ".[baselines]"   # xgboost      (E3)
pip install -e ".[deep]"        # torch        (E4/E5/E6)
pip install -e ".[viz]"         # matplotlib   (E8)
```

## Полный запуск (команды по эпикам)

Порядок соответствует пайплайну. На E0 команды-стабы уже зафиксированы, чтобы
интерфейс не менялся при реализации.

```bash
# 1. Подготовка BUT PPG: загрузка, предобработка, реестр записей        (E1)
python scripts/prepare_but_ppg.py

# 2. Разбиение по испытуемым 60/20/20 (30/10/10) + проверка непересечения (E1)
python scripts/make_splits.py \
    --registry data/processed/registry.csv \
    --out splits/but_ppg_60_20_20.json --seed 42

# 3. Базовые модели: признаки + LogReg/XGBoost                          (E3)
python scripts/train_baseline.py --config models/baseline_features

# 4. Базовые модели: 1D-CNN / ResNet1D (качество и HR)                  (E4)
python scripts/train_cnn.py --config models/cnn1d

# 5. OpenTSLM: два запуска — без ECG и с ECG-инициализацией             (E5)
python scripts/train_opentslm.py --config models/opentslm model.ecg_init=false
python scripts/train_opentslm.py --config models/opentslm model.ecg_init=true

# 6. SIGMA-PPG: дообучение в режиме только-PPG                          (E6)
python scripts/train_sigma.py --config models/sigma_ppg

# 7. Эксперимент с малым объёмом данных (25/50/100%, >=3 сида)          (E8)
python scripts/run_low_data.py --config experiment/low_data

# 8. Подготовка GalaxyPPG (инверсия сигнала, окна, эталонный HR из ECG) (E7)
python scripts/prepare_galaxy_ppg.py --config data/galaxy_ppg

# 9. Внешняя проверка на GalaxyPPG (MAE общий и по активностям)         (E7)
python scripts/eval_external_galaxy.py --config data/galaxy_ppg

# Оценка на фиксированной тестовой части (все модели или выборочно)     (E2)
python scripts/evaluate.py --all
python scripts/evaluate.py artifacts/predictions/sigma_ppg_quality.csv
```

## Структура проекта

```
but-ppg-hr-quality/
├── configs/                 # YAML-конфиги (deep-merge + CLI-оверрайды)
│   ├── default.yaml         #   база: seed, окно, разбиение, выбор чекпоинта
│   ├── data/                #   but_ppg.yaml, galaxy_ppg.yaml
│   ├── models/              #   baseline_features, cnn1d, opentslm, sigma_ppg
│   └── experiment/          #   low_data.yaml
├── src/butppg/              # пакет
│   ├── config.py            #   [E0] загрузка/композиция конфигов
│   ├── paths.py             #   [E0] канонические пути проекта
│   ├── utils/               #   [E0] seed, логирование, манифест запуска
│   ├── data/                #   [E1] загрузка, предобработка, реестр, сплиты (стабы)
│   ├── features/            #   [E3] ручные признаки (стаб)
│   ├── models/              #   [E2-E6] реализации моделей (стабы)
│   ├── metrics/             #   [E2] метрики + формат предсказаний (стабы)
│   └── evaluation/          #   [E2] оценка по файлам предсказаний (стаб)
├── scripts/                 # CLI-точки входа (по одной на шаг пайплайна)
├── splits/                  # файлы разбиения по испытуемым (коммитятся)
├── data/                    # raw/interim/processed — В GIT НЕ ХРАНЯТСЯ
├── artifacts/               # checkpoints/predictions/logs (gitignore), metrics (коммитятся)
├── results/                 # итоговые метрики и таблицы (коммитятся)
├── docs/                    # PLAN.md — декомпозиция и календарный план
└── tests/                   # проверки инфраструктуры E0
```

## Принципы воспроизводимости (заложены на E0)

- **Единый seed** (`configs/default.yaml → seed`) применяется через
  `butppg.utils.seed_everything` к Python/NumPy/PyTorch; детерминированный режим.
- **Манифест запуска** (`butppg.utils.RunManifest`) сохраняется рядом с
  результатами: конфиг, версия датасета, seed, git-ревизия, версии библиотек,
  ссылки на файл разбиения и реестр.
- **Разбиение строго по испытуемым**, тестовая часть фиксирована и одинакова для
  всех моделей; выбор лучшего чекпоинта — только по валидации (Macro-F1 для
  задачи A, MAE для задачи B).
- **Сырые данные не коммитятся** — только скрипты загрузки/подготовки и
  инструкции. ECG используется лишь как источник эталонного HR, но не как вход
  модели (защита от утечки данных).

## Распределение работ

Полная декомпозиция, распределение между исполнителями (Голиков / Будилов) и
календарный план по недельным спринтам (27.07–20.08) — в [docs/PLAN.md](docs/PLAN.md)
и интерактивном плане [docs/plan.html](docs/plan.html).
