# BUT PPG — сервис-оркестратор

CLI-оркестратор для воспроизводимого исследования на **BUT PPG v2.0.0**: одна
точка входа с **тремя независимыми командами**, которые передают работу друг
другу через файлы (витрины и папки запусков), а не единым пайплайном.

| Команда | Что делает |
|---|---|
| `orch ingest` | **RAW-слой**: качает сырые сигналы BUT PPG в `data/raw` (медленно, один раз, кэшируется) |
| `orch process` | RAW → **обработанные окна + реестр + split** (быстро, локально, без сети) |
| `orch mart` | реестр → входные **витрины** для SIGMA-PPG и/или OpenTSLM |
| `orch train` | запускает **обучение** одной модели на одной задаче → `runs/<id>/` |
| `orch results` | **результат**: метрики запусков и выгрузка обученных весов |

ETL разделён на слои (`ingest` → `process` → `mart`), чтобы повторный прогон
обработки (смена канала PPG, параметров окна) не тянул датасет заново из сети.

Тяжёлые модели (SIGMA-PPG, OpenTSLM/Llama-3.2-3B) обучаются на сервере с GPU;
лёгкие baseline (trivial, 1D-CNN) гоняются где угодно, включая ноутбук.
`OpenTSLM/` и `SigmaPPG/` — вендоренные апстрим-репозитории рядом с пакетом.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # ядро (numpy/scipy/pandas/sklearn)
pip install -e ".[data]"         # wfdb — для загрузки BUT PPG
pip install -e ".[deep]"         # torch + einops — CNN / SIGMA-PPG / OpenTSLM
```

После установки доступна команда `orch` (или `python -m butppg.orchestrator.cli`).

### Доступ к Llama (для OpenTSLM)

OpenTSLM тянет gated-модель Llama с Hugging Face. Нужен токен и доступ к моделям:

1. Запроси доступ (base-версии, **не** Instruct): обязательно
   [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B),
   желательно ещё `meta-llama/Llama-3.2-1B` (дешёвый smoke-тест).
2. Создай токен со scope `read`: huggingface.co/settings/tokens.
3. Скопируй `.env.example` → `.env` и впиши `HF_TOKEN=hf_...`.

`.env` в git не коммитится; CLI подхватывает его сам при старте (переменные из
шелла имеют приоритет). Llama качается один раз и кэшируется в `~/.cache/huggingface`.

## 1. ETL: `orch ingest` → `orch process` → `orch mart`

Три слоя. Рекомендуемый порядок при первом запуске:

```bash
orch ingest                         # RAW: качает сырые сигналы в data/raw (медленно, один раз)
orch process --make-split           # RAW → реестр + окна + split 60/20/20 (быстро, без сети)
orch mart                           # реестр → обе витрины (sigma_ppg + opentslm)
```

Смысл разделения: `ingest` кэширует **полный сырой сигнал** (все RGB-каналы) на
диск, поэтому повторный `process` (сменил канал PPG, параметры окна) идёт за
секунды и **ничего не качает заново**.

```bash
# только RAW-слой, для быстрой проверки — первые 50 записей
orch ingest --limit 50

# переобработать локально (напр. после изменения логики) — сеть не трогается
orch process --make-split

# витрины выборочно
orch mart --model sigma_ppg --task hr
orch mart --model opentslm --task quality --acc-mode magnitude

# всё одной командой (ingest+process+mart), как раньше:
orch mart --prepare --make-split
```

Ключевые опции `mart`: `--model {sigma_ppg,opentslm,both}`, `--task {quality,hr,both}`,
`--target-fs` (ресемпл SIGMA, 50 Гц), `--normalize {zscore,minmax}`,
`--acc-mode {none,magnitude,axes}` (OpenTSLM). Витрины пишутся в
`artifacts/data_marts/<model>/<task>/`. RAW-слой — в `data/raw/`
(`ppg/<id>.npz` со всеми каналами + `acc/<id>.npy` + аннотации).

## 2. Обучение — `orch train`

Создаёт запуск `runs/<run_id>/` (конфиг, чекпоинты, предсказания, метрики, лог)
и обучает выбранную модель. Лучший чекпоинт выбирается по валидации (Macro-F1
для качества, MAE для HR), финальные метрики — на фиксированном тесте.

```bash
# baseline на ноутбуке (реестр читается напрямую, витрина не нужна)
orch train --model trivial --task quality
orch train --model cnn1d   --task hr --epochs 50 --device cpu

# SIGMA-PPG на GPU-сервере (нужна витрина sigma_ppg + предобученный чекпоинт)
orch train --model sigma_ppg --task hr --device cuda \
    --checkpoint-path /path/to/sigma.pth

# OpenTSLM — два запуска для переноса ECG→PPG (по заданию: Llama-3.2-3B)
orch train --model opentslm --task quality --device cuda            # без ECG
orch train --model opentslm --task quality --device cuda \
    --ecg-init /path/to/ecg_stage_checkpoint.pt                     # с ECG

# лёгкая debug-модель, чтобы отладить конвейер без тяжёлой 3B
orch train --model opentslm --task quality --device cuda --llm-id gemma-1b
```

**Алиасы `--llm-id`** (можно и полный HF-id): `llama-3b` (по умолчанию,
`meta-llama/Llama-3.2-3B`), `llama-1b`, `gemma-1b` (`google/gemma-3-1b-pt`),
`gemma-270m`. Gemma тоже gated на HF — доступ тем же `HF_TOKEN`.

Модели `trivial`/`cnn1d` читают реестр (`--registry`, `--split`); `sigma_ppg`/
`opentslm` читают витрину (`--mart-dir`, по умолчанию `artifacts/data_marts/<model>`).

## 3. Результат — `orch results`

```bash
orch results                       # список всех запусков с основной метрикой
orch results <run_id>              # метрики, статус, пути, ссылка на веса
orch results <run_id> --download exports/<run_id>   # выкачать веса (+метрики+предсказания)
```

`--download` копирует лучший чекпоинт вместе с `metrics/` и `predictions/` в
указанную папку — самодостаточный экспорт обученной модели.

## Как устроены витрины (BUT PPG → вход модели)

Сначала общий слой (загрузка + реестр + разбиение по испытуемым), затем две
проекции — модели едят принципиально разное:

| | SIGMA-PPG | OpenTSLM |
|---|---|---|
| Форма | `.npy` тензор `(N, 1, L)` по испытуемым | `.jsonl`: текст + ряд + ответ |
| PPG | ресемпл 300@30Гц → 500@50Гц, нормализация | остаётся 300@30Гц, z-score |
| ACC | нет (только PPG) | опционально, магнитудой |
| «Ответ» | метка в `y` (int64/float32) | языком (`good`/`bad` или число 30–220) |

ECG используется только как источник эталонного HR (метка), в модель не подаётся
(защита от утечки). HR-витрины фильтруются по `quality_label==1` (раздел 2Б).

## Структура проекта

```
Диссертация/
├── src/butppg/
│   ├── orchestrator/       # CLI (mart/train/results) + реестр запусков
│   ├── data/               # реестр, сплиты, подготовка BUT PPG, сборка витрин
│   ├── training/           # драйверы: baselines, CNN, общий finalize + dispatch
│   ├── adapters/           # мосты к вендоренным репо: sigma_ppg, opentslm
│   ├── models/             # trivial baselines, 1D-CNN, разбор ответов OpenTSLM
│   ├── metrics/            # Macro-F1/ROC-AUC/… , MAE/RMSE, формат предсказаний
│   └── evaluation/         # оценка по файлам предсказаний
├── configs/                # YAML-конфиги
├── tests/                  # инфраструктура + сквозные тесты оркестратора
├── OpenTSLM/  SigmaPPG/     # вендоренные апстрим-репозитории (в git не коммитятся)
├── data/  artifacts/  runs/ # входы/витрины/запуски (в git не коммитятся)
└── legacy/                  # предыдущие версии
```

Ядро (реестр, сплиты, витрины, метрики, тривиальные/CNN, разбор OpenTSLM)
перенесено из проверенной работы коллеги; поверх построен оркестратор.
