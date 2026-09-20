# Деплой и прогон на GPU-сервере (RunPod)

Инструкция для полного прогона на свежем поде: от авторизации GitHub до
выгрузки обученных весов и таблиц. Проверено на **RunPod RTX A6000**
(network volume `/workspace`, драйвер 580 / CUDA 13.0).

Договорённости по путям:
- `/workspace` — сетевой том, переживает перезапуск пода. Всё держим здесь.
- `/root` — эфемерный, при остановке пода теряется.
- Репозиторий: `github.com/redizga/but-ppg-hr-quality`, рабочая ветка `golikov/2026-09-14`.

---

## 0. Поднять под

1. RunPod → Deploy → **RTX A6000**, шаблон **RunPod PyTorch** (Python 3.11).
2. Прикрепить **Network Volume** к `/workspace` (≥ 60 ГБ).
3. Открыть web-терминал (Jupyter → Terminal) или подключиться по SSH.

Проверить GPU:
```bash
nvidia-smi
```
Должны увидеть A6000 и версию CUDA. `torch` под cu13 работает с драйвером 580 —
даунгрейд не нужен.

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
**Settings → SSH and GPG keys → New SSH key** (вставить, Save).

Проверить и запомнить хост:
```bash
ssh -o StrictHostKeyChecking=accept-new -T git@github.com
# ожидаемо: "Hi redizga! You've successfully authenticated..."
```

> Альтернатива — HTTPS + Personal Access Token: `git clone https://<TOKEN>@github.com/redizga/but-ppg-hr-quality.git`.
> SSH предпочтительнее: токен не светится в истории и в `.git/config`.

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

Ставим пакет со всеми группами зависимостей + GPU-torch:
```bash
cd /workspace/but-ppg-hr-quality
pip install -e ".[data,deep,baselines]"
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('cuda', torch.cuda.is_available(), 'torch', torch.__version__)"
```
Ожидаем `cuda True`. Если `False` — переустановить torch под нужный CUDA
(cu121 или cu130 колёса).

Секреты — в `.env` (в git не попадает, см. `.gitignore`):
```bash
cp .env.example .env
# вписать HF_TOKEN=<токен Hugging Face>  (нужен для gated Llama/Gemma в OpenTSLM)
```

---

## 4. Витрины (ETL раздельными шагами)

ETL слоёный: RAW-загрузка (сеть) → PROCESSED (локально, зелёный канал) →
витрины. Датасет качается **один раз** и потом переиспользуется.

```bash
orch ingest --workers 16      # параллельная загрузка BUT PPG (~секунды, latency-bound)
orch process                  # реестр из кэша: 3888 записей / 50 субъектов / 30·10·10
orch mart --model sigma_ppg   # per-subject .npy, ресемпл 30→50 Гц
orch mart --model opentslm    # per-split JSONL (QADataset)
```

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
```bash
huggingface-cli download zonhengu/sigmappg sigma.pth --local-dir /workspace
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
pip install -e OpenTSLM
pip install "huggingface_hub[cli]"
python -c "import torch, transformers; print('cuda', torch.cuda.is_available(), 'tf', transformers.__version__)"
```

Прогон (llama-1b — лёгкая, для отладки; при желании llama-3b):
```bash
orch train --model opentslm --task quality --device cuda --llm-id llama-1b --epochs 5
orch train --model opentslm --task hr      --device cuda --llm-id llama-1b --epochs 5
```
> Gated-модели Llama/Gemma требуют одобренного доступа на Hugging Face и
> `HF_TOKEN` в `.env`. Вариант с ECG→PPG-переносом (`--ecg-init <ckpt>`) требует
> ECG-претрейн чекпойнта — отдельный шаг.

Вернуться в основное окружение:
```bash
deactivate
```

---

## 6. Сводка результатов

```bash
orch results                  # список всех прогонов
orch cascade                  # метрики каскада quality→HR (раздел 2Б)
orch table                    # сводные таблицы в results/tables/ (Markdown + CSV)
```

---

## 7. Выгрузка результатов и весов

Забираем `runs/` (веса, предсказания, метрики) и `results/tables/`.

Вариант A — через git (метрики/предсказания лёгкие; веса крупные, лучше не коммитить):
```bash
# зафиксировать только таблицы/метрики при необходимости — веса в .gitignore
```

Вариант B — скачать напрямую (из web-терминала RunPod → File Browser, или scp):
```bash
# на локальной машине:
scp -r -P <SSH_PORT> root@<POD_IP>:/workspace/but-ppg-hr-quality/runs ./runs
scp -r -P <SSH_PORT> root@<POD_IP>:/workspace/but-ppg-hr-quality/results ./results
```

Вариант C — упаковать и скачать одним архивом:
```bash
cd /workspace/but-ppg-hr-quality
tar czf /workspace/results.tgz runs results
# затем скачать /workspace/results.tgz через File Browser
```

---

## Шпаргалка по граблям

| Симптом | Причина | Фикс |
|---|---|---|
| ssh: bad permissions on key | `/workspace` монтируется `0666` | держать ключ в `/root/.ssh`, `chmod 600` |
| `uv: command not found` после установки | uv не в PATH / кэш bash | `source /root/.local/bin/env && hash -r` |
| pip нет в venv312 | uv-venv без pip | пересоздать `uv venv --seed` |
| `cuda False` | torch не под нужный CUDA | переустановить cu121/cu130 колёса |
| OpenTSLM `No module named transformers` | стоит в 3.11, а не в venv312 | ставить внутри активного `.venv312` |
| HR MAE у всех моделей ~median | HR считается только на good-quality окнах (раздел 2Б) — так и задумано | это валидный результат, не «баг» |
