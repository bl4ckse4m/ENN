# HW1 — Аналитическая модель производительности маленькой CNN

Closed-form модели стоимости `FLOPs(S,B)`, `Memory(S,B)`, `Latency(S,B,θ)` и
`Energy(S,B,θ)` для заданной последовательной CNN: calibration и validation на
реальном GPU по заданной сетке из 132 точек `(S,B)`, а также дополнительный
stress-прогон, локализующий реальную границу CUDA OOM.

## 1. Hardware и software

| Компонент | Версия                                                                |
|---|-----------------------------------------------------------------------|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16303 MiB (15.8 GiB видно CUDA)           |
| NVIDIA driver | 580.95.05 (CUDA 13.0)                                                 |
| OS | Ubuntu 24.04.5 LTS (kernel-mode driver, без WDDM shared-memory spill) |
| Python | 3.12.3                                                                |
| PyTorch | 2.14.0+cu130                                                          |
| CUDA runtime / cuDNN | 13.0 / 9.24.0                                                         |
| NumPy / SciPy / Matplotlib | 2.5.2 / 1.18.1 / 3.11.2                                               |
| NVML bindings | nvidia-ml-py 13.615.71 (счётчик энергии всего GPU)                    |

## 2. Как воспроизвести

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install numpy scipy matplotlib nvidia-ml-py

# обязательная сетка из 132 точек -> results/measurements.csv + results/kernels.csv
.venv/bin/python hw1/measure.py

# дополнительные high-memory точки (пишет results/oom_stress.csv)
.venv/bin/python hw1/measure.py --stress-oom

# подгоняет θ, пишет results/theta.json и results/figures/*.png
.venv/bin/python hw1/calibrate.py
```

`measure.py` устанавливает обязательные флаги протокола (`cudnn.benchmark=False`,
`cudnn.allow_tf32=False`, `cuda.matmul.allow_tf32=False`, `eval()`,
`inference_mode()`, FP32, случайные входные тензоры), делает прогрев
(warmup) из 5 итераций, замеряет 30 синхронизированных forward проходов (медиана),
сбрасывает счётчики и читает `torch.cuda.max_memory_allocated()` за один forward
pass, измеряет энергию всей GPU по кумулятивному счётчику энергии NVML в окне
повторных forward-ов длительностью ≥1 с и профилирует CUDA kernels послойно через
`torch.profiler` в `results/kernels.csv`. Энергия измерена методом
`nvml_energy_counter` для всех 132 конфигураций.

Построение сетки (`seed=2026`): размеры изображений
`32,64,128,224,256,384,512` плюс случайные `48,112,352,400`; размеры batch
`1,2,4,8,16,32,64,128,256` плюс случайные `95,98,167`. Для калибровки
используются 63 точки базовой сетки (`is_validation=False`); 69 точек со
случайным размером/размером batch отложены как валидация
(`is_validation=True`).

## 3. Уравнения

Полные рукописные выводы — в `hw1_handwritten.pdf`. Кратко
(1 MAC = 2 FLOPs; считаются только MAC/линейные FLOPs; FP32 = 4 байта):

```text
F(S,B) = B (17712 S^2 + 313344)                    FLOPs
M(S,B) = 4 (1,040,324 + 13 B S^2)                  bytes   (ideal peak)
Q(S,B) = 4 (91 B S^2 + 2148 B + 1,040,324)         bytes   (ideal DRAM traffic)
L(S,B,θ) = t0 + max(F/Rc, Q/Rm)                    seconds
E(S,B,θ) = E0 + eF F + eQ Q                        joules
```

`M` считает параметры + вход, удерживаемый вызывающей стороной, + наибольшую
пару одновременно живых активаций (на шаге MaxPool — `3+8+2 = 13 B·S²`
элементов; ReLU in-place). `Q` считает одно чтение и одну запись каждой
активации, одно обращение за каждым параметром, включая трафик in-place ReLU.
`equations.py` реализует их; `Memory` и `FLOPs` — без параметров.

## 4. Оценённые параметры (`results/theta.json`)

| Параметр | Значение | Смысл                                                           |
|---|---|-----------------------------------------------------------------|
| `launch_overhead_s` | 158.2 µs | фиксированные накладные расходы framework + ~20 kernel launches |
| `compute_rate_flops_s` | 14.12 TFLOP/s | эффективный FP32 throughput (≈32% от пика ~44 TFLOP/s)          |
| `memory_bandwidth_bytes_s` | 340.8 GB/s | эффективная DRAM bandwidth (≈38% от пика ~896 GB/s)             |
| `fixed_energy_j` | 11.2 mJ | энергия всей GPU за окно запуска (≈ 67 W × 158 µs)              |
| `energy_per_flop_j` | 22.56 pJ/FLOP | ≈ 44.3 GFLOP/J                                                  |
| `energy_per_byte_j` | 1.3e-20 J/B | ≈ 0 — не идентифицируемо (см. обсуждение)                       |

θ для Latency подобраны через `scipy.optimize.least_squares` в log-пространстве
с log-невязками на 63 калибровочных точках; θ для Energy — через взвешенные по
относительной ошибке линейные МНК с неотрицательными ограничениями (модель
Energy линейна по θ). Machine balance подгонки: `Rc/Rm = 41.4 FLOP/byte`.

## 5. Итоги

Измеренные диапазоны на обязательной сетке (все 132 конфигурации помещаются в
16 GiB): latency 0.185 – 94.7 ms, peak memory 37 MiB – 4.29 GiB,
energy 0.012 – 27.0 J (средняя мощность всего GPU 67 W → 286 W).

| Величина | Выборка | MAPE | медиана APE | максимум APE |
|---|---|---|---|---|
| Latency | calibration (63) | 10.9 % | 10.0 % | 39.9 % |
| Latency | validation (69) | 9.4 % | 10.1 % | 41.8 % |
| Energy | calibration (63) | 9.4 % | 8.1 % | 45.6 % |
| Energy | validation (69) | 6.9 % | 4.4 % | 43.1 % |
| Memory | calibration (63) | 62.5 % | 68.5 % | 89.9 % |
| Memory | validation (69) | 53.1 % | 50.0 % | 90.1 % |

Latency и Energy почти одинаково хорошо обобщаются на 69 отложенных
точках (validation MAPE ≈ calibration MAPE), т.е. θ не переобучены.
Числа для Memory заведомо хуже: `Memory(S,B)` по построению не имеет
подогнанных параметров и является идеальной нижней границей (см. §7).

Рисунки в `results/figures/` (на каждом — измеренные точки вместе с
предсказанной кривой/поверхностью):

- `latency_pred_vs_measured.png`, `energy_pred_vs_measured.png`,
  `memory_pred_vs_measured.png` — log-log scatter относительно идеальной
  прямой, calibration (кружки) против validation (треугольники).
- `latency_curves.png`, `energy_curves.png`, `memory_curves.png` — предсказанные
  кривые по `B` для каждого `S` с измеренными маркерами; на memory-рисунке также
  показана workspace-inclusive диагностическая линия.
- `latency_error_heatmap.png`, `energy_error_heatmap.png`,
  `memory_error_heatmap.png` — относительная ошибка предсказания на плоскости
  `(S,B)`.
- `regime_map.png` — доминирующий член latency (панель a) и work regime с
  контуром launch-floor (панель b).
- `oom_boundary.png` — измеренные пики и OOM-точки против аналитической
  `Memory(S,B)` и workspace-inclusive диагностики.
- `memory_workspace_residual.png` — `measured − ideal` против `B·S²`.

## 6. Анализ OOM

Обязательная сетка из 132 точек целиком помещается на этот GPU с 16 GiB
(крупнейшая конфигурация `S=512, B=256` пикует на 4.29 GiB), поэтому внутри
обязательной сетки OOM не происходит. Чтобы отработать обязательную обработку
OOM, измерены 16 дополнительных точек (`--stress-oom`, строки с
`is_stress=True` в `measurements.csv`, сырой прогон в `results/oom_stress.csv`):
8 помещаются и 8 вызывают настоящий `torch.cuda.OutOfMemoryError`:

```text
S = 512 axis: OK through B = 816, OOM from B = 832
B = 256 axis: OK through S = 896, OOM from S = 928
```

Сравнение этих 8 OOM с предсказанием `Memory(S,B)` (порог = свободный VRAM
≈ 13.7 GiB до прогона):

| Модель | Предсказано OOM-точек | Верных решений |
|---|---|---|
| аналитическая `Memory(S,B)` | 2 из 8 | 10 / 16 |
| + workspace-член `1.30·M + 50 MiB` | 8 из 8 | 16 / 16 |

Таким образом, аналитическое уравнение memory занижает реальный пик, и его OOM
ложно отрицательные — это ровно то место, откуда берётся обсуждение в §7.

## 7. Обсуждение — где и почему уравнения ломаются

**Launch-bound → memory-bound → compute-bound.** Подогнанный roofline
(`t0 = 158 µs`, machine balance 41.4 FLOP/B) классифицирует 31 из 132 точек
сетки как launch-bound (левый нижний угол `regime_map.png`, панель a): весь их
forward pass ≈ 0.22 ms, т.е. ~20 kernel launches по ~8 µs каждый, и арифметика
(≤ 13 µs) незаметна. Оставшиеся 101 точки — compute-bound с эффективным темпом
14.12 TFLOP/s. Панель (b) показывает, где *работа* memory-bound: арифметическая интенсивность 
сети `F/Q` растёт от 4.1 FLOP/B при `(S=32,B=1)` (трафик parameters
доминирует в `Q`) до 48.6 FLOP/B при больших `S,B` (асимптота
`17712/(4·91) = 48.7`), поэтому малые батчи лежат ниже баланса 41.4 FLOP/B и их
работа memory-bound. Этот клин, однако, целиком ниже launch-floor
`max(F/Rc, Q/Rm) = t0`: batch size, при котором memory-время превысило бы `t0`
при `S = 32`, равен `B ≈ 131`, но уже к `B ≈ 65` работа пересекает machine
balance и становится compute-bound. Поэтому ни одна точка сетки не доминируется
memory-членом. Тем не менее все три режима существуют на уровне слоёв: семь
in-place ReLU, MaxPool и GlobalAvgPool имеют `I ≈ 0 FLOP/B` по MAC-конвенции и
memory-bound при любом размере, `conv2`/`conv5` достигают `I ≈ 230–270 FLOP/B`
и compute-bound, а `conv4`/`fc2` находятся около баланса 41.4 FLOP/B. Усреднение
`F/Q` по всей сети скрывает эту внутреннюю неоднородность — первое место, где
ломается latency-модель.

**Ошибки latency (MAPE ≈ 10 %, максимум ≈ 40 %).** Наихудшие точки —
launch-bound малые конфигурации, где постоянный `t0` неверен в деталях
(стоимость запуска не строго постоянна, а dispatch overhead PyTorch меняется);
и точки около переключений kernel-алгоритмов. `results/kernels.csv` показывает,
что cuDNN не держит один алгоритм на слой на всей сетке — только `conv2`
использует 11 различных семейств kernels (implicit-GEMM
`implicit_convolve_sgemm`, Winograd с data/filter/transform GEMM, CUTLASS
SGEMM), а `fc1/fc2` переключаются между `gemvx` (B=1) и small-GEMM kernels. У
каждого семейства свой достижимый throughput, который одна глобальная пара
`(Rc, Rm)` описать не может; структура невязок в `latency_error_heatmap.png`
следует за этими переключениями.

**Ошибки memory (predicted/measured ≈ 0.37 на сетке).** Вывод по liveness
считает только живые FP32 тензоры. `torch.cuda.max_memory_allocated()` также
считает *workspaces* cuDNN/cuBLAS, которые аллоцируются через PyTorch allocator
на время выполнения kernel. На всех 132 точках сетки превышение крайне
систематично:

```text
measured = 1.295 × ideal + 49.7 MiB     (max residual 190 MiB)
```

`memory_workspace_residual.png` показывает постоянную нижнюю границу (~50 MiB
shape-независимого workspace) плюс член, пропорциональный `B·S²`. Это не
округление allocator: при больших размерах
`max_memory_reserved / max_memory_allocated ≈ 1.00`. Это workspace бекэнда,
масштабирующийся с размером активаций, плюс переключения алгоритмов (разброс
±190 MiB на малых размерах). Поскольку задача требует, чтобы `Memory(S,B)` не
имела *калибруемых параметров*, closed-form остаётся идеальной нижней границей;
подогнанная линия `1.30·M + 50 MiB` сохранена как явно помеченная диагностика
(`memory_workspace_diagnostic` в `theta.json`) и именно она даёт успешное
предсказание OOM 16/16 в `oom_boundary.png`. Это второе и самое значимое место, где
уравнения ломаются.

**Модель energy.** Energy всей GPU предсказывается с MAPE ≈ 7 % на отложенных
точках, но подогнанный `energy_per_byte` схлопывается к ≈ 0, пока
`energy_per_flop` поглощает эффект. Это предел идентифицируемости, а не ошибка:
для этой сети `F` и `Q` почти идеально коллинеарны (`corr(F,Q) = 0.9999999994`,
число обусловленности матрицы дизайна 6.2e12), потому что обе величины
доминируются членом `B·S²`. Разделить joules-per-FLOP и joules-per-byte могли
бы только нагрузки с сильно меняющейся арифметической интенсивностью. 
Фиксированный член `E0 = 11.2 mJ` физически осмыслен (`≈ idle power 67 W × t0`), а измеренная
средняя мощность растёт с 67 W при `(32,1)` до 286 W при `(512,256)`, что
согласуется с compute-heavy большой конфигурацией на этой 300 W карте. Наконец,
счётчик энергии NVML опрашивается в окне повторных форвардов ≥ 1 с, чтобы
преодолеть его разрешение, поэтому энергии на каждый форвард самых крошечных
конфигураций несут наибольший относительный шум (виден как точки с
max APE ≈ 45 %).

## 8. Файлы

```text
hw1/
├── README.md                 # этот файл
├── hw1_handwritten.pdf       # рукописные выводы
├── models.py                 # сеть
├── equations.py              # flops(), memory(), bytes_moved(), latency(), energy()
├── measure.py                # измерения (сетка + --stress-oom)
├── calibrate.py              # подгонка θ, метрики, рисунки
└── results/
    ├── measurements.csv      # 132 строки сетки + 16 stress-строк (S,B,latency,memory/oom,energy,is_validation,...)
    ├── kernels.csv           # S,B,layer,имя kernel (2700 строк)
    ├── oom_stress.csv        # сырой дополнительный stress-прогон
    ├── theta.json            # подогнанные θ + метрики
    └── figures/*.png
```
