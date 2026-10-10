# Репликация: «Sparse Attention as Compact Kernel Regression» (arXiv:2601.22766v3)

Самодостаточная реализация экспериментов статьи (Santos et al., 2026) для
одной GPU RTX 5070 Ti (16 ГБ). Теория проверена численно; эксперименты
(фазы 2–4) выполнены пресетом `quick`, доступны пресеты `quick`/`full`.
Отчёт с результатами: `hw2/REPORT.md`.

## Структура

```
src/
  kernels.py            # ядра внимания: softmax, sparsemax, entmax, normReLU,
                        # r-ReLUmax, top-k + Надарая-Ватсон (для теории)
  model.py              # Memory Mosaics (перенос официальной архитектуры)
  data_synthetic.py     # MQMTAR / Reverse / Sort + метрики exact match
  icl_dfa.py            # RegBench (PFA/DFA) — протокол berlino/seq_icl
  theory_checks.py      # фаза 1: числовые проверки теории статьи
  fig1_kernels.py       # фаза 1: перерисовка Figure 1
  train_synthetic.py    # фаза 2: length generalization (Table 2)
  train_icl.py          # фаза 3: RegBench ICL (Table 1)
  train_lm.py           # фаза 4: LM на BabiStories (Figure 2)
  prepare_babistories.py# подготовка датасета BabiStories (GPT-2 BPE)
  analyze.py            # сводка результатов в таблицы/графики
scripts/run_phase{2,3,4}_*.sh
scripts/run_phase5_supplement.sh # контрольные прогоны: ICL lr=5e-4, MQMTAR 5M, LM-сиды
scripts/rerun_entmax.sh         # запуск только biweight/triweight (quick-пресет, seed 1337)
results/                # metrics.json + summary_*.md
figures/                # fig1_kernels.png + fig_length_*.png, fig_icl.png, fig_lm.png
data/babistories/       # train.bin / val.bin (== Table 4 статьи) — НЕ в git, собирается (см. ниже)
external/               # клоны репозиториев авторов — НЕ в git, клонируются (см. ниже)
```

## Окружение (фаза 0)

```bash
python3 -m venv hw2/.venv
hw2/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
hw2/.venv/bin/pip install -r hw2/replication/requirements.txt
```

Все команды ниже запускаются из каталога `hw2/replication`:
```bash
cd hw2/replication
PY=../.venv/bin/python
```

## Фаза 1 — теория (~1 мин)

```bash
$PY -u -m src.theory_checks --out results/theory_checks.json   # 20/20 PASS (вкл. градиенты)
$PY -u -m src.fig1_kernels --out figures/fig1_kernels.png
```

## Внешние зависимости: `external/` и `data/` (не в git)

Обе директории не входят в репозиторий (крупные бинарники, см. `.gitignore`) —
они восстанавливаются из публичных источников:

**`external/`** — клоны репозиториев, из которых взяты архитектура, протоколы
и датасет (используются для справки и сборки данных; сама репликация в `src/`
самодостаточна):

```bash
git clone https://github.com/deep-spin/sparse_kernel_regression.git external/sparse_kernel_regression
git clone https://github.com/facebookresearch/MemoryMosaics.git external/MemoryMosaics
git clone https://github.com/berlino/seq_icl.git external/seq_icl
# seq_icl — на закреплённом коммите, как в ICLL-инструкции:
git -C external/seq_icl checkout 9b9223d15348b5a415fb453ed988ed5f7ab9fbdc
```

* `sparse_kernel_regression` (deep-spin) — официальный код статьи
  («Sparse Attention as Compact Kernel Regression»): оттуда перенесены
  `src/model.py` и `src/data_synthetic.py`;
* `MemoryMosaics` (facebookresearch) — архитектура Memory Mosaics и архив
  датасета BabiStories (`BabiStories/data/babistories-dataset.7z.001–005`);
* `seq_icl` (berlino) — протокол RegBench (PFA/DFA), перенесён в `src/icl_dfa.py`.

**`data/babistories/`** — `train.bin` (474 704 907 токенов) и `val.bin`
(4 749 107) — совпадают с Table 4 статьи до токена. Собираются из архива
`external/MemoryMosaics` (~3 мин, CPU):

```bash
$PY -u -m src.prepare_babistories \
  --archive-glob 'external/MemoryMosaics/BabiStories/data/babistories-dataset.7z.*' \
  --out-dir data/babistories
```

Фазы 1–3 (теория, синтетика, ICL) работают без `data/` и `external/` — данные
генерируются на лету; для фазы 4 (LM) нужно `data/babistories/`.

## Фаза 2 — length generalization (Table 2)

8 методов × 3 задачи (MQMTAR, Reverse, Sort); обучение до 64 токенов,
оценка exact match на длинах 1×–64×; логи в
`results/synthetic/<task>/<method>/seed*/train.log`.

```bash
bash scripts/run_phase2_synthetic.sh quick    # ~4 GPU-ч  (1M примеров, 1 сид)
bash scripts/run_phase2_synthetic.sh full     # ~11 GPU-ч (5M примеров, 1 сид)
bash scripts/run_phase2_synthetic.sh full 1337 1338 1339   # 3 сида как в статье (x3 ко времени)
# отдельный запуск:
$PY -u -m src.train_synthetic --task mqmtar --method softmax --seed 1337 \
    --train-examples 5000000 --out-dir results/synthetic
```

Методы: `softmax`, `epanechnikov` (=sparsemax), `biweight` (1.5-entmax),
`triweight` (4/3-entmax), `norm_relu`, `relumax` (=1-ReLUmax),
`top32_softmax`, `top32_uniform`; доп.: `relumax2`, `relumax3`.

## Фаза 3 — RegBench ICL (Table 1)

Гиперпараметры из официального README (n_layer=2, n_embd=128, n_head=8,
lr=1e-4, wd=0.01, batch 32); метрики: dfa-accuracy («Acc» в статье) и
TVD×100.

```bash
bash scripts/run_phase3_icl.sh quick    # ~6 GPU-ч (1k/5k, до 60 эпох)
bash scripts/run_phase3_icl.sh full     # ~7 GPU-ч (1k/5k/20k, до 200 эпох)
# отдельный запуск:
$PY -u -m src.train_icl --method epanechnikov --num-examples 1000 --seed 1337
```

## Фаза 4 — LM BabiStories (Figure 2)

1 слой (d=768, 12 голов, ctx 512, pmem 2688) как в Table 3; пресет quick —
8k итераций × batch 16 (в статье: 80k × 512 на H100).

```bash
bash scripts/run_phase4_lm.sh quick     # ~1 GPU-ч (3 метода, 8k итер)
bash scripts/run_phase4_lm.sh full      # ~5 GPU-ч (7 методов, 25k итер)
# отдельный запуск:
$PY -u -m src.train_lm --method softmax --n-layer 1 --batch-size 16 --max-iters 25000
```

## Фаза 5 — контрольные прогоны (~6 GPU-ч)

Уточняющие запуски, интерпретация — в `hw2/REPORT.md`, §6–§8:
* ICL n=1000 с lr=5e-4 (официальный конфиг) — контроль гиперпараметра;
* MQMTAR на 5M примеров (`results/synthetic5m/`) — контроль бюджета для Table 2;
* LM сиды 1338/1339 — разброс по сидам для Figure 2.

```bash
bash scripts/run_phase5_supplement.sh            # все три блока (~6 GPU-ч)
bash scripts/run_phase5_supplement.sh icl        # только ICL (~1.2 GPU-ч)
bash scripts/run_phase5_supplement.sh mqmtar5m   # только MQMTAR 5M (~3 GPU-ч)
bash scripts/run_phase5_supplement.sh lm         # только LM-сиды (~1.8 GPU-ч)
```

## Сводка результатов

```bash
$PY -u -m src.analyze --phase all      # или synthetic | icl | lm
```
Пишет `results/summary_*.md` (таблицы «мы» + «статья») и графики в `figures/`.

Текущие результаты (пресет quick, все фазы выполнены): `results/summary_*.md`
и отчёт `hw2/REPORT.md` (разделы 5–9).

## Примечания и troubleshooting

* **Память RAM.** Не оборачивать `MemmapChunks` (LM-датасет, ~474M стартовых
  позиций) в `DataLoader(shuffle=True)`: дефолтный `RandomSampler`
  материализует `randperm(474M).tolist()` ≈ 15 ГБ Python-целых и исчерпывает
  оперативную память. Батчи в `train_lm` семплируются напрямую из memmap —
  O(batch). Не собирать большие токенизированные корпуса в Python-список
  (`prepare_babistories` пишет потоково).
* **Память GPU.** Всё считается в bf16. Для LM batch 16 укладывается в 16 ГБ
  (~6.3 ГБ пик); batch 24–32 возможен на «чистой» GPU. Скрипты выставляют
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* **Ограничение памяти процесса** (для страховки при запуске):
  `systemd-run --user --scope -p MemoryMax=8G bash scripts/run_phase4_lm.sh quick` —
  при исчерпании памяти завершится только процесс обучения.
* **AdaSplash/triton** (официальная зависимость) не используется: вся
  репликация идёт на чистом PyTorch — при контексте ≤4k это не влияет на
  результат (статья также не фокусируется на throughput).
* **Формат MQMTAR** восстановлен из официального eval-кода (позиции
  1,2,4,5,7,8,10,11 в ответе); синтетические данные авторами не выложены
  («will be released soon»), поэтому генераторы реализованы по Appendix B.3.
* **Скорость** (RTX 5070 Ti, bf16): MQMTAR 4L/512d ≈ 9 it/s при batch 256;
  Sort/Reverse 2L/512d ≈ 15 it/s; LM 1L/768d ≈ 10 it/s при batch 16.