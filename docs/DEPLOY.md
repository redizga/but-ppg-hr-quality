# Деплой и прогон на GPU-сервере (RunPod)

Инструкция для полного прогона на свежем поде: от авторизации GitHub до
выгрузки результатов. Проверено на **RunPod RTX A6000** (network volume
`/workspace`). Драйвер/CUDA у пода варьируется (встречались CUDA 13.0 и 12.8) —
torch ставим **под CUDA конкретного пода** (см. шаг 3).

Договорённости по путям:
- `/workspace` — сетевой том, **переживает перезапуск/миграцию пода**. Здесь код,
  витрины, `runs/`, venv312, секреты. Всё держим тут.
- `/root` — эфемерный, при остановке пода **обнуляется**: пропадают SSH-ключ,
  uv-питон (ломается venv312), кэш HuggingFace.
- Репозиторий: `github.com/redizga/but-ppg-hr-quality`, рабочая ветка `golikov/2026-09-14`.

> 🔁 **Под уже поднимался и перезапустился?** Не проходи всё заново — прыгай в
> раздел **[R. После перезапуска пода](#r-после-перезапуска-или-миграции-пода)**:
> код и venv на волюме целы, восстановить нужно только эфемерное из `/root`.

---

## 0. Поднять под

1. RunPod → Deploy → **RTX A6000**, шаблон **RunPod PyTorch** (Python 3.11).
2. Прикрепить **Network Volume** к `/workspace` (≥ 60 ГБ).
3. Открыть web-терминал (Jupyter → Terminal) или подключиться по SSH.

Проверить GPU:
```bash
nvidia-smi
```
Должны увидеть A6000 и версию CUDA в шапке (например `CUDA Version: 12.8`).
**Запомни эту версию** — под неё ставится torch (шаг 3): драйвер 12.x → колёса
`cu124`, драйвер 13.x → `cu13x`. Несовпадение даёт `cuda False` или
`NVIDIA driver too old`.

---

## 1. Авторизация GitHub (SSH-ключ)

> ⚠️ **Грабли `/workspace`:** сетевой том монтируется с правами `0666`, а ssh
> отказывается использовать ключ с такими правами. Поэтому ключ **генерируем и
> держим в `/root/.ssh`** (эфемерный, но с нормальными правами `600`), а git
> явно указываем этот ключ.

```bash
mkdir -p /root/.ssh && chmod 700 /root/.ssh
ssh-keygen -t ed25519 -C "rsclashgml@gmail.com" -f /root/.ssh/id_ed25519 -N ""
chmod 600 /root/.ssh/id_ed25519
cat /root/.ssh/id_ed25519.pub
```

Скопировать выведенный публичный ключ и добавить его на GitHub:
**Settings → SSH and GPG keys → New SSH key** (вставить, Save). Это **ручной шаг
в браузере** — без него `git pull`/`push` будут падать с `Permission denied (publickey)`.

Проверить хост:
```bash
ssh -o StrictHostKeyChecking=accept-new -T git@github.com
# ожидаемо: "Hi redizga! You've successfully authenticated..."
```

**Сразу сохрани ключ на волюм**, чтобы не перевыпускать после каждого рестарта:
```bash
mkdir -p /workspace/.secrets && cp /root/.ssh/id_ed25519 /workspace/.secrets/
```
После перезапуска пода ключ восстанавливается одной строкой (см. раздел R) —
заново на GitHub добавлять не нужно.

> Приватный репозиторий: HTTPS-пул попросит логин/пароль, так что для pull/push
> нужен именно SSH-ключ (или Personal Access Token).

---

## 2. Клонировать репозиторий

```bash
cd /workspace
git clone git@github.com:redizga/but-ppg-hr-quality.git
cd but-ppg-hr-quality
git checkout golikov/2026-09-14
```

Так как ключ лежит не в дефолтном месте — привязать его к этому репозиторию,
чтобы `pull`/`push` работали без вопросов:
```bash
git config core.sshCommand "ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes"
git config user.name  "redizga"
git config user.email "rsclashgml@gmail.com"
git pull   # проверка, что тянет
```

---

## 3. Основное окружение (Python 3.11 — всё, кроме OpenTSLM)

> 💡 **Рекомендация из практики:** проще держать **одно окружение — venv312**
> (шаг 5.5). Оно лежит на волюме и переживает рестарты пода, а Python 3.12
> подходит для всех моделей (не только OpenTSLM). Тогда после каждого рестарта не
> надо переустанавливать систему в эфемерном 3.11 — достаточно оживить
> интерпретатор (раздел R). Системный 3.11 ниже нужен, только если venv312
> ставить не хочешь. **Если идёшь по venv312-для-всего — переходи сразу к 5.5,**
> а команды из 5.1–5.4 запускай в активном `.venv312`.

Ставим пакет со всеми группами зависимостей + GPU-torch **под CUDA пода**
(смотри версию в `nvidia-smi`; для драйвера 12.x — `cu124`):
```bash
cd /workspace/but-ppg-hr-quality
pip install -e ".[data,deep,baselines]"
pip install "torch==2.6.*" "torchvision==0.21.*" --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('cuda', torch.cuda.is_available(), 'torch', torch.__version__)"
```
Ожидаем `cuda True`. Если `False` — torch собран под неподходящий CUDA:
переустанови колёса под свою версию (индекс `.../whl/cu124` или `.../whl/cu130`).
torch и torchvision ставь **одной парой** (2.6 ↔ 0.21), иначе будет
`torchvision::nms does not exist`.

Секреты — в `.env` (в git не попадает, см. `.gitignore`). На поде обычно нет
редактора (`nano`/`vi`), поэтому впиши токен одной командой (ввод скрыт):
```bash
cp .env.example .env
read -s -p "HF token: " HF && echo && sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$HF|" .env
grep -q '^HF_TOKEN=hf_' .env && echo "OK" || echo "пусто/неверный формат"
```
HF-CLI для скачивания весов (SIGMA-чекпойнт, gated-модели):
```bash
pip install -U "huggingface_hub[cli]"   # даёт команды `hf` и `huggingface-cli`
```

### 3.1 Vendored-репозитории моделей

`SigmaPPG` и `OpenTSLM` **не в git** (в `.gitignore`) — их клонируют отдельно
в корень проекта, туда, где их ждёт `paths.py`
(`PROJECT_ROOT/SigmaPPG`, `PROJECT_ROOT/OpenTSLM`):
```bash
cd /workspace/but-ppg-hr-quality
git clone https://github.com/ZonghengGuo/SigmaPPG.git   # для sigma_ppg (шаг 5.4)
git clone https://github.com/OpenTSLM/OpenTSLM.git       # для opentslm (шаг 5.5)
```
> Если SIGMA при запуске падает с `ModuleNotFoundError` на пакете из их репо
> (напр. `timm`) — доставь его `pip install <пакет>` и повтори команду.

---

## 4. Витрины (ETL раздельными шагами)

ETL слоёный: RAW-загрузка (сеть) → PROCESSED (локально, зелёный канал) →
витрины. Датасет качается **один раз** и потом переиспользуется.

```bash
orch ingest --workers 16      # параллельная загрузка BUT PPG (~секунды, latency-bound)
orch process                  # реестр из кэша: 3888 записей / 50 субъектов / 30·10·10
orch mart --model sigma_ppg              # per-subject .npy, ресемпл 30→50 Гц
orch mart --model opentslm --acc-mode none   # per-split JSONL, PPG-only
```
> `--acc-mode none` обязателен для OpenTSLM: с ACC у записей разное число рядов
> (PPG+ACC vs только PPG), и батч не собирается (`stack expects each tensor to
> be equal size`). OpenTSLM сравниваем по PPG.

---

## 5. Прогон моделей

Каждый `orch train` пишет в `runs/<id>/`: `run.json`, `checkpoints/`,
`predictions/test_predictions.csv`, `metrics/metrics.json`, `train.log`.
Прогресс-бары идут в консоль (отключить: `BUTPPG_NO_PROGRESS=1`).

### 5.1 Тривиальные бейзлайны (CPU, мгновенно)
```bash
orch train --model trivial --task quality
orch train --model trivial --task hr
orch train --model trivial --task hr --method dominant_frequency
```

### 5.2 Feature + LogReg/XGBoost (CPU)
```bash
orch train --model baseline_features --task quality --input-variant ppg         --estimator xgboost
orch train --model baseline_features --task quality --input-variant ppg_acc_cov --estimator xgboost
orch train --model baseline_features --task hr      --input-variant ppg         --estimator xgboost
orch train --model baseline_features --task hr      --input-variant ppg_acc_cov --estimator xgboost
```

### 5.3 1D-CNN / ResNet1D (GPU)
```bash
orch train --model cnn1d --task quality --device cuda --epochs 30
orch train --model cnn1d --task hr      --device cuda --epochs 30
orch train --model cnn1d --task quality --device cuda --epochs 30 --input-variant ppg_acc
orch train --model cnn1d --task hr      --device cuda --epochs 30 --input-variant ppg_acc
```

### 5.4 SIGMA-PPG (GPU, нужен предобученный чекпойнт)
Репо `SigmaPPG` уже склонирован на шаге 3.1. Скачать веса (публичные, токен не нужен):
```bash
hf download zonhengu/sigmappg sigma.pth --local-dir /workspace
orch train --model sigma_ppg --task quality --device cuda --checkpoint-path /workspace/sigma.pth
orch train --model sigma_ppg --task hr      --device cuda --checkpoint-path /workspace/sigma.pth
```
> Предупреждения о частично загруженных весах (q_bias/v_bias zero-init,
> интерполяция pos/time-эмбеддингов 120→10 патчей) — ожидаемы: наше окно 500
> точек короче претрейна SIGMA. Результат валиден, но с этой оговоркой.

### 5.5 OpenTSLM (отдельный venv 3.12)

OpenTSLM требует Python **3.12** (`requires-python >=3.12`), а образ пода — 3.11.
Поэтому отдельный venv. Ключевой нюанс: **uv создаёт venv без pip**, поэтому
делаем venv с флагом `--seed` (кладёт pip внутрь) — дальше обычный pip.

```bash
# uv (если ещё не стоит):
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env && hash -r      # добавить uv в PATH текущей сессии

uv venv --seed --python 3.12 /workspace/.venv312
source /workspace/.venv312/bin/activate
pip install -e ".[data,deep,baselines]"
pip install -e OpenTSLM            # репо склонирован на шаге 3.1
pip install "huggingface_hub[cli]"
# ВАЖНО: pip по умолчанию тянет torch под CUDA 13. Если драйвер пода 12.x
# (проверить: nvidia-smi), поставить torch+torchvision под cu124 одной парой
# (иначе "driver too old", а рассинхрон версий -> "torchvision::nms does not exist"):
pip install "torch==2.6.*" "torchvision==0.21.*" --index-url https://download.pytorch.org/whl/cu124
python -c "import torch, transformers; print('cuda', torch.cuda.is_available(), 'tf', transformers.__version__)"

# Патчи багов OpenTSLM (их репо несовместимо с open_flamingo 0.0.2):
# 1-2) обучаемый TS-энкодер лежит в .visual заглушки SimpleNamespace, а их код
#      обращается к самой заглушке (requires_grad_ и вызов в forward);
# 3)   Flamingo.generate() в 0.0.2 не принимает eos/pad_token_id.
sed -i 's/model\.vision_encoder\.requires_grad_(True)/model.vision_encoder.visual.requires_grad_(True)/' \
    OpenTSLM/src/opentslm/model/llm/OpenTSLMFlamingo.py
sed -i 's/self\.vision_encoder(/self.vision_encoder.visual(/g' \
    OpenTSLM/src/opentslm/model/llm/TimeSeriesFlamingoWithTrainableEncoder.py
sed -i '/eos_token_id=self.text_tokenizer.eos_token_id,/d; /pad_token_id=self.text_tokenizer.pad_token_id,/d' \
    OpenTSLM/src/opentslm/model/llm/OpenTSLMFlamingo.py
```
> Эти три `sed` правят **склонированный репозиторий OpenTSLM на волюме**, поэтому
> при перезапуске пода они **сохраняются** — повторять нужно только если репо
> склонировали заново (новый volume). Проверка, что патчи на месте:
> `grep -c vision_encoder.visual OpenTSLM/src/opentslm/model/llm/TimeSeriesFlamingoWithTrainableEncoder.py` → `2`.
>
> Остальное (перенос модели на GPU, единый dtype float32, компактный чекпойнт,
> восстановление лучшей эпохи, точные метки в витрине, preflight-проверка) уже
> **зашито в наш адаптер** `src/butppg/adapters/opentslm.py` — отдельных действий
> на поде не требует, только `git pull`.

Прогон (llama-1b — лёгкая; при желании llama-3b через `--llm-id llama-3b`):
```bash
orch train --model opentslm --task quality --device cuda --llm-id llama-1b --epochs 5
orch train --model opentslm --task hr      --device cuda --llm-id llama-1b --epochs 5
```
> **Эпохи:** 5 — рабочий оптимум. Больше (10+) переобучает модель на маленьком
> train и **вырождает генерацию** (hr начинает выдавать повторяющийся токен
> `7979…`, все ответы становятся invalid, MAE = null). Если поднимаешь эпохи —
> проверяй `metrics.json`: `invalid_response_fraction` должно быть низким.
>
> Gated-модели Llama/Gemma требуют одобренного доступа на HuggingFace и `HF_TOKEN`
> в `.env`. Вариант ECG→PPG (`--ecg-init <ckpt>`) требует ECG-претрейн чекпойнта
> (отдельный форк) — в этом прогоне не выполнялся.

Вернуться в основное окружение:
```bash
deactivate
```

---

## 6. Сводка результатов

```bash
orch results                  # список всех прогонов (RUN_ID, статус, primary-метрика)
orch table                    # сводные таблицы в results/tables/ (Markdown + CSV)
# каскад quality→HR (раздел 2Б): указать конкретные прогоны (лучший gate + лучший HR)
orch cascade --quality-run <QUALITY_RUN_ID> --hr-run <HR_RUN_ID>
```
> RUN_ID берутся из `orch results`. Таблица подхватывает **последний finished**
> прогон каждой модели/задачи — если последний прогон битый (например, hr с
> вырожденной генерацией и пустой метрикой), удали его каталог
> `rm -rf runs/<битый_run_id>` и пересобери `orch table`, чтобы вернулся валидный.
> Подпись OpenTSLM в таблице берёт реальный размер из `--llm-id` (`OpenTSLM-1B`).

---

## 7. Выгрузка результатов

Основной артефакт — папка **`results/`** (сводные таблицы + cascade json); она
**в git** и коммитится. `runs/` (веса/предсказания/метрики) — в `.gitignore`,
её при необходимости забираем архивом.

Вариант A (основной) — зафиксировать `results/` в git:
```bash
git add results/
git commit -m "Add comparison tables and quality->HR cascade results"
git push
```

Вариант B — бэкап метрик/предсказаний (их нет в git), архивом через File Browser:
```bash
cd /workspace/but-ppg-hr-quality
tar czf /workspace/results_backup.tgz results runs/*/metrics runs/*/predictions
# затем скачать /workspace/results_backup.tgz из RunPod → File Browser
```
> Чекпойнты теперь компактные: OpenTSLM пишет **только обучаемые слои (~1 ГБ)**,
> а не всю модель fp32 (было 6.4 ГБ) — квоту тома больше не пробивает.
> Веса лежат в `runs/<id>/checkpoints/`; если нужны — добавь их в архив отдельно.

После выгрузки под можно гасить: `/workspace` персистентный, при возврате —
раздел R.

---

## R. После перезапуска или миграции пода

Под может потухнуть/мигрировать (в т.ч. «угнали видяху» — тогда просто поднимай
новый под с **тем же network volume**). Код, витрины, `runs/`, venv312 и
`/workspace/.secrets` — на волюме, целы. Обнуляется только `/root`. Восстановление —
1–2 минуты, **без переустановки пакетов**:

```bash
cd /workspace/but-ppg-hr-quality

# 1. SSH-ключ (из бэкапа на волюме) — для git pull/push
mkdir -p /root/.ssh && cp /workspace/.secrets/id_ed25519 /root/.ssh/ && chmod 600 /root/.ssh/id_ed25519
git config core.sshCommand "ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes"

# 2. Оживить venv312: uv-питон жил в /root и пропал -> интерпретатор venv висит в пустоту.
#    Ставим ту же версию 3.12.14 (site-packages на волюме подхватятся как есть).
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env && hash -r
uv python install 3.12.14
hash -r

# 3. Активировать и проверить GPU
source /workspace/.venv312/bin/activate
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.__version__)"
```

Проверить, что всё на месте, и дообучать что осталось:
```bash
nvidia-smi                       # версия CUDA не изменилась? если да — переставь torch (шаг 3)
git pull                         # добрать возможные фиксы кода
grep -c vision_encoder.visual OpenTSLM/src/opentslm/model/llm/TimeSeriesFlamingoWithTrainableEncoder.py  # 2 = патчи на месте
orch results                     # что уже finished, что осталось
```

Заметки:
- **HF-кэш** (`/root/.cache/huggingface`) тоже обнулился — веса моделей (Llama,
  и т.д.) до-качаются на первом прогоне, это нормально.
- Если у нового пода **другая версия CUDA** (`nvidia-smi`) — переставь пару
  torch/torchvision под неё (шаг 3), иначе `cuda False`.
- CPU-only шаги (`orch results/cascade/table`, тривиальные/feature-модели) не
  требуют venv312 — достаточно `pip install -e .` в системный python пода.

---

## Шпаргалка по граблям

| Симптом | Причина | Фикс |
|---|---|---|
| `Permission denied (publickey)` при git | эфемерный `/root` обнулился (рестарт пода), ключ пропал | восстановить ключ из `/workspace/.secrets` (раздел R) |
| venv312 `bad interpreter: .../python: No such file` | uv-питон жил в `/root`, пропал при рестарте | `uv python install 3.12.14` (раздел R), venv оживёт |
| ssh: bad permissions on key | `/workspace` монтируется `0666` | держать ключ в `/root/.ssh`, `chmod 600` |
| `SIGMA-PPG repo not found` / OpenTSLM not found | vendored-репы в `.gitignore` | склонировать (шаг 3.1) |
| `huggingface-cli: command not found` | HF-CLI не установлен | `pip install "huggingface_hub[cli]"`, команда `hf download` |
| `nano: command not found` | на поде нет редактора | писать через `sed`/`read` (шаг 3) или `apt-get install -y nano` |
| `uv: command not found` после установки | uv не в PATH / кэш bash | `source /root/.local/bin/env && hash -r` |
| pip нет в venv312 | uv-venv без pip | пересоздать `uv venv --seed` |
| `cuda False` | torch не под нужный CUDA | переустановить cu121/cu130 колёса |
| `NVIDIA driver too old (found 12080)` | torch собран под CUDA 13, драйвер 12.8 | `pip install "torch==2.6.*" "torchvision==0.21.*" --index-url .../cu124` |
| `operator torchvision::nms does not exist` | torch и torchvision разных версий | ставить их одной парой (torch 2.6 ↔ torchvision 0.21) |
| `SimpleNamespace has no attribute requires_grad_` / `SimpleNamespace object is not callable` | баг OpenTSLM (TS-энкодер в `.visual`) | sed-патчи `OpenTSLMFlamingo.py` + `TimeSeriesFlamingoWithTrainableEncoder.py` (шаг 5.5) |
| OpenTSLM: `stack expects each tensor to be equal size` | витрина с ACC даёт разное число рядов | пересобрать `orch mart --model opentslm --acc-mode none` |
| OpenTSLM `No module named transformers` | стоит в 3.11, а не в venv312 | ставить внутри активного `.venv312` |
| OpenTSLM `Flamingo.generate() got ... eos_token_id` | open_flamingo 0.0.2 не принимает eos/pad | sed-патч generate (шаг 5.5) |
| OpenTSLM `KeyError: 'quality_label'` в конце | старая витрина без меток истины | пересобрать витрину / уже покрыто fallback (код `1232351`+) |
| OpenTSLM hr: `metrics=null`, ответы `7979…` | переобучение на большом числе эпох | вернуть 5 эпох; удалить битый run, `orch table` |
| `Disk quota exceeded` при сохранении чекпойнта | старый полный чекпойнт 6.4 ГБ | обновить код — чекпойнт теперь ~1 ГБ; чистить `runs/opentslm_*` |
| `NameError: RESULTS_DIR` в `orch cascade` | старый баг импорта | `git pull` (исправлено в `fb8ebf0`) |
| HR MAE у всех моделей ~median | HR считается только на good-quality окнах (раздел 2Б) — так и задумано | это валидный результат, не «баг» |
