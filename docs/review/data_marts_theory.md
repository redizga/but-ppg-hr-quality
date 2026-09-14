# Что такое "витрины данных" и как с ними работать

Объяснение для тех, кто не работал с этим термином раньше. Смотреть вместе с
[data_marts.md](data_marts.md) (конкретные форматы под OpenTSLM/SIGMA-PPG) и
[next_steps.md](next_steps.md) (общий план по эпикам).

## 1. Откуда взялся термин

"Витрина данных" (data mart) — понятие из хранилищ данных (data warehousing), не
специфичное для ML. Стандартная трёхслойная схема:

```
raw data              →   единое хранилище (DWH)   →   витрины (data marts)
"как есть у источника"     "приведено к общему        "заточено под конкретного
                             виду, без потерь"            потребителя"
```

- **Raw** — сырые файлы источника, как они есть, с особенностями конкретной системы
  (свой формат, своя частота дискретизации, свои имена полей).
- **Хранилище** — единое, обезличенное от потребителей представление: одна строка =
  одна сущность, консистентные типы и единицы измерения, никаких артефактов
  конкретного получателя данных.
- **Витрина** — производная от хранилища, пересчитанная и переформатированная под
  ровно одного потребителя, чтобы этот потребитель мог взять её и сразу использовать,
  не занимаясь очисткой/пересборкой сам.

Разделение существует потому, что у разных потребителей разные требования к формату,
а пересчитывать эти требования из сырых данных каждый раз — дорого и чревато
рассинхроном (один потребитель случайно использует не ту версию логики очистки, что
другой).

## 2. Как это ложится на наш проект

| Слой | Что это здесь | Кто/где делает |
|---|---|---|
| Raw | Файлы BUT PPG с PhysioNet (сырые сигналы, разметка) | `scripts/prepare_but_ppg.py`, эпик E1 |
| Хранилище | `data/processed/registry.csv` + `.npy`-окна PPG/ACC — одна строка = одно 10-секундное окно, единые колонки (`record_id`, `subject_id`, `quality_label`, `hr_ref`, `ppg_path`, ...) | `src/butppg/data/registry.py`, эпик E1 |
| Витрины | Под каждую модель — свой формат: у SIGMA-PPG `.npy` (N,1,L) на испытуемого при 50 Гц; у OpenTSLM `.jsonl` с текстовыми промптами; у CNN/feature-baseline — обычно можно читать реестр напрямую, витрина не нужна | `scripts/build_mart_sigma_ppg.py`, `scripts/build_mart_opentslm.py` (эпики E5/E6) |

**Зачем не кормить моделям сразу реестр?** Потому что модели требуют физически
разные вещи:
- SIGMA-PPG — тензор `[Batch, 1, Length]` с сигналом на 50 Гц, нормализованным
  z-score, сохранённым в `.npy`;
- OpenTSLM — не тензор вообще, а текстовая строка-промпт с "врезанным" внутрь рядом
  чисел (это языковая модель — она принимает текст, не сигнал);
- баланс между "PPG на 30 Гц из реестра" и тем, что реально ожидает конкретная
  модель, у каждой модели свой.

Если бы каждый скрипт обучения сам читал `registry.csv`, ресемплил, нормализовал и
собирал батчи — эта логика дублировалась бы в 4-5 местах и легко разъехалась бы
(один скрипт нормализует z-score, другой забыл; один ресемплит на 50 Гц, другой
использует исходные 30 Гц). Витрина фиксирует это один раз, физически на диске, и
её можно посмотреть глазами (открыть `.jsonl`, открыть `.npy`) — важно для отладки,
когда модель "почему-то не учится".

## 3. Пайплайн целиком (куда встраиваются витрины)

```
PhysioNet (BUT PPG v2.0.0)
        │  scripts/prepare_but_ppg.py                              [E1]
        ▼
data/processed/registry.csv + data/processed/*.npy (PPG/ACC окна)
        │  scripts/make_splits.py → splits/but_ppg_60_20_20.json   [E1]
        ▼
registry.csv (со сплитами train/val/test по испытуемым)
        │
        ├─▶ scripts/build_mart_sigma_ppg.py  → artifacts/data_marts/sigma_ppg/...  [E6]
        │         │
        │         ▼  (нужно доп. дописать downstream/butppg/*.py внутри SigmaPPG)
        │      scripts/train_sigma.py → чекпоинт + предсказания на тесте
        │
        └─▶ scripts/build_mart_opentslm.py → artifacts/data_marts/opentslm/...    [E5]
                  │
                  ▼  (нужен свой класс QADataset поверх .jsonl)
               scripts/train_opentslm.py → чекпоинт + предсказания на тесте
        │
        ▼
scripts/evaluate.py --all  → метрики, итоговые таблицы                            [E2/E9]
```

Ключевое: **витрина — промежуточный, воспроизводимый артефакт**, а не разовый
скрипт "руками поправить csv". Она должна пересобираться командой из README, чтобы
при смене реестра (например, если добавили ещё записей или поправили разметку)
витрины под все модели обновились одинаково, детерминированно, по одному и тому же
seed.

## 4. Как работать с этим практически

### Шаг 0 — предпосылка

Витрину нельзя построить без `registry.csv` и файла сплита. Сейчас их нет
(`data/processed/`, `splits/` пустые) — сначала эпик E1.

### Шаг 1 — построить витрину (когда E1 готов)

```bash
# SIGMA-PPG, задача качества
python scripts/build_mart_sigma_ppg.py --task quality \
    --registry data/processed/registry.csv \
    --split splits/but_ppg_60_20_20.json \
    --out artifacts/data_marts/sigma_ppg

# SIGMA-PPG, задача HR (по умолчанию только quality_label==1, как требует задание)
python scripts/build_mart_sigma_ppg.py --task hr \
    --registry data/processed/registry.csv \
    --split splits/but_ppg_60_20_20.json \
    --out artifacts/data_marts/sigma_ppg

# OpenTSLM, обе задачи, ACC как magnitude (решение по умолчанию из configs/models/opentslm.yaml)
python scripts/build_mart_opentslm.py --task quality --acc-mode magnitude \
    --registry data/processed/registry.csv \
    --split splits/but_ppg_60_20_20.json \
    --out artifacts/data_marts/opentslm

python scripts/build_mart_opentslm.py --task hr --acc-mode magnitude \
    --registry data/processed/registry.csv \
    --split splits/but_ppg_60_20_20.json \
    --out artifacts/data_marts/opentslm
```

Каждый вызов пишет `manifest.json`/детерминированный вывод рядом — так видно, каким
seed и с какими параметрами (частота ресемплинга, нормализация) витрина была
построена. Это часть требования задания "сохранить всё для повторения экспериментов".

### Шаг 2 — проверить витрину глазами, не доверять молча

- SIGMA-PPG: `numpy.load(".../S01_x.npy").shape` — должно быть `(N, 1, L)`, где
  `L = 10 × target_fs` (500 при 50 Гц). Проверить, что `y_quality` в `{0,1}`, а `y_hr`
  — разумные числа уд/мин.
  Пример команды: `python -c "import numpy as np; x=np.load('artifacts/data_marts/sigma_ppg/quality/train/S01_x.npy'); print(x.shape, x.dtype)"`
- OpenTSLM: открыть первую строку `.jsonl` и прочитать её как текст — `pre_prompt` +
  `post_prompt` должны читаться как связный промпт, `answer` — то, что реально ожидаем
  (`good`/`bad` или число 30–220).
  Пример команды: `python -c "import json; print(json.loads(open('artifacts/data_marts/opentslm/quality/train.jsonl', encoding='utf-8').readline()))"`

### Шаг 3 — довести до обучения (то, что скрипты витрин НЕ делают)

Витрина — это данные на диске, а не обучение. Дальше нужно:
- **SIGMA-PPG:** написать `downstream/butppg/preprocess.py` (может быть тонкой
  обёрткой, читающей уже готовую витрину вместо собственной предобработки) и
  `downstream/butppg/train.py` — по образцу `downstream/bidmc/*.py` в их репозитории
  (этот файл у них есть и рабочий, наша витрина уже в его формате).
- **OpenTSLM:** установить пакет (`pip install opentslm`), написать небольшой класс
  `ButPPGQualityDataset(QADataset)` / `ButPPGHRDataset(QADataset)`, который читает наш
  `.jsonl` вместо `load_dataset(...)` — по образцу `TSQADataset`.

### Шаг 4 — предсказания в единый формат

После обучения любая модель должна отдать предсказания на тесте строго в формате
`record_id, subject_id, y_true, y_pred, proba_good` (задача A) или
`record_id, subject_id, hr_true, hr_pred` (задача B) — поэтому `record_id`/
`subject_id` протащены через витрину до самого конца (см. `build_record` в
`scripts/build_mart_opentslm.py`), чтобы потом можно было склеить предсказания
обратно с реестром для метрик/таблиц.

## 5. Итоговый чеклист "что сделать"

1. [ ] E1: скачать BUT PPG, собрать `registry.csv`, собрать `splits/but_ppg_60_20_20.json`.
2. [ ] Прогнать оба `build_mart_*.py` скрипта, глазами проверить один пример из каждой
       витрины (Шаг 2 выше).
3. [ ] SIGMA-PPG: написать `downstream/butppg/preprocess.py` + `train.py` поверх
       витрины, запустить `train_sigma.py`.
4. [ ] OpenTSLM: написать `ButPPGQualityDataset`/`ButPPGHRDataset`, запустить
       `train_opentslm.py` (два запуска — без/с ECG-инициализацией).
5. [ ] Предсказания обеих моделей на тесте — в единый CSV-формат, прогнать
       `scripts/evaluate.py`.
