from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .equations import memory as predicted_memory
    from .models import CNN
else:
    from equations import memory as predicted_memory
    from models import CNN


BASE_IMAGE_SIZES = (32, 64, 128, 224, 256, 384, 512)
BASE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
OOM_STRESS_BATCHES = (512, 640, 768, 800, 816, 832, 896, 1024, 1152, 1280)
OOM_STRESS_IMAGE_SIZES = (640, 768, 896, 928, 960, 1024)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_measurement_grid(seed: int):
    rng = np.random.default_rng(seed)
    image_candidates = sorted(
        set(range(32, 513, 16)).difference(BASE_IMAGE_SIZES)
    )
    batch_candidates = [
        value for value in range(1, 257) if value & (value - 1) != 0
    ]

    extra_images = tuple(
        sorted(int(value) for value in rng.choice(image_candidates, 4, replace=False))
    )
    extra_batches = tuple(
        sorted(int(value) for value in rng.choice(batch_candidates, 3, replace=False))
    )
    image_sizes = tuple(sorted(BASE_IMAGE_SIZES + extra_images))
    batch_sizes = tuple(sorted(BASE_BATCH_SIZES + extra_batches))

    configurations = [
        {
            "S": image_size,
            "B": batch,
            "is_validation": (
                image_size in extra_images or batch in extra_batches
            ),
            "is_stress": False,
            "stress_axis": "",
        }
        for image_size in image_sizes
        for batch in batch_sizes
    ]
    return configurations, extra_images, extra_batches


def build_oom_stress_grid():
    configurations = [
        {
            "S": 512,
            "B": batch,
            "is_validation": False,
            "is_stress": True,
            "stress_axis": "batch",
        }
        for batch in OOM_STRESS_BATCHES
    ]
    configurations.extend(
        {
            "S": image_size,
            "B": 256,
            "is_validation": False,
            "is_stress": True,
            "stress_axis": "image_size",
        }
        for image_size in OOM_STRESS_IMAGE_SIZES
    )
    return sorted(
        configurations,
        key=lambda configuration: configuration["B"] * configuration["S"] ** 2,
    )


def configure_pytorch() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def warm_up(model, inputs, iterations: int, device) -> None:
    output = None
    with torch.inference_mode():
        for _ in range(iterations):
            output = model(inputs)
    torch.cuda.synchronize(device)
    del output


def measure_peak_memory(model, inputs, device) -> int:
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = model(inputs)
    torch.cuda.synchronize(device)

    expected_shape = (inputs.shape[0], 100)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"model returned shape {tuple(output.shape)}, expected {expected_shape}"
        )

    peak = int(torch.cuda.max_memory_allocated(device))
    del output
    return peak


def measure_latency(model, inputs, repeats: int, device) -> float:
    samples = []
    with torch.inference_mode():
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(inputs)
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - start)
            del output
    return float(statistics.median(samples))


def run_repeated_forwards(model, inputs, repeats: int) -> None:
    output = None
    with torch.inference_mode():
        for _ in range(repeats):
            output = model(inputs)
    del output


class NVMLEnergyMeter:
    def __init__(self, device_index: int, sample_interval_s: float):
        self.available = False
        self.method = "unavailable"
        self.reason = ""
        self._initialized = False
        self._sample_interval_s = sample_interval_s

        try:
            import pynvml
        except ImportError:
            self.reason = "install nvidia-ml-py to enable energy measurement"
            return

        self._nvml = pynvml
        try:
            pynvml.nvmlInit()
            self._initialized = True
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as error:
            self.reason = f"NVML initialization failed: {error}"
            self.close()
            return

        energy_function = getattr(
            pynvml, "nvmlDeviceGetTotalEnergyConsumption", None
        )
        if energy_function is not None:
            try:
                energy_function(self._handle)
                self._energy_function = energy_function
                self.available = True
                self.method = "nvml_energy_counter"
                return
            except Exception:
                pass

        try:
            self._power_watts()
            self.available = True
            self.method = "nvml_power_sampling"
        except Exception as error:
            self.reason = f"GPU exposes neither energy nor power through NVML: {error}"

    def close(self) -> None:
        if self._initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False

    def _power_watts(self) -> float:
        return float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1_000.0

    def _measure_with_counter(self, run, device) -> float:
        torch.cuda.synchronize(device)
        before_millijoules = float(self._energy_function(self._handle))
        run()
        torch.cuda.synchronize(device)
        after_millijoules = float(self._energy_function(self._handle))
        delta_joules = (after_millijoules - before_millijoules) / 1_000.0
        if delta_joules <= 0:
            raise RuntimeError("NVML energy counter did not advance")
        return delta_joules

    def _measure_with_power_samples(self, run, device) -> float:
        samples = []
        errors = []
        stop = threading.Event()
        ready = threading.Event()

        def sample_power() -> None:
            while not stop.is_set():
                try:
                    samples.append((time.perf_counter(), self._power_watts()))
                except Exception as error:
                    errors.append(error)
                    ready.set()
                    return
                ready.set()
                stop.wait(self._sample_interval_s)

        thread = threading.Thread(target=sample_power, daemon=True)
        thread.start()
        if not ready.wait(timeout=2.0):
            stop.set()
            thread.join(timeout=2.0)
            raise RuntimeError("timed out waiting for the NVML power sampler")
        if errors:
            stop.set()
            thread.join(timeout=2.0)
            raise errors[0]

        torch.cuda.synchronize(device)
        start_power = self._power_watts()
        start_time = time.perf_counter()
        try:
            run()
            torch.cuda.synchronize(device)
            end_time = time.perf_counter()
            end_power = self._power_watts()
        finally:
            stop.set()
            thread.join(timeout=2.0)

        if errors:
            raise errors[0]

        points = [(start_time, start_power)]
        points.extend(
            sample for sample in samples if start_time < sample[0] < end_time
        )
        points.append((end_time, end_power))
        points.sort(key=lambda sample: sample[0])

        energy_joules = 0.0
        for first, second in zip(points, points[1:]):
            elapsed = second[0] - first[0]
            energy_joules += elapsed * (first[1] + second[1]) / 2.0
        return energy_joules

    def measure(self, run, repeats: int, device):
        if not self.available:
            return None, self.method

        if self.method == "nvml_energy_counter":
            try:
                total_energy = self._measure_with_counter(run, device)
            except Exception:
                self.method = "nvml_power_sampling"
                total_energy = self._measure_with_power_samples(run, device)
        else:
            total_energy = self._measure_with_power_samples(run, device)

        return total_energy / repeats, self.method


def recorded_forward(model, inputs):

    def apply(label, operation, value):
        with torch.profiler.record_function(f"layer::{label}"):
            return operation(value)

    value = apply("conv1", model.conv1, inputs)
    value = apply("relu1", model.relu, value)
    value = apply("maxpool", model.pool, value)
    value = apply("conv2", model.conv2, value)
    value = apply("relu2", model.relu, value)
    value = apply("conv3", model.conv3, value)
    value = apply("relu3", model.relu, value)
    value = apply("conv4", model.conv4, value)
    value = apply("relu4", model.relu, value)
    value = apply("conv5", model.conv5, value)
    value = apply("relu5", model.relu, value)
    value = apply("conv6", model.conv6, value)
    value = apply("relu6", model.relu, value)
    value = apply("global_avg_pool", model.avgpool, value)
    value = torch.flatten(value, 1)
    value = apply("fc1", model.fc1, value)
    value = apply("relu7", model.relu, value)
    return apply("fc2", model.fc2, value)


def _event_layer(event) -> str:
    current = event
    while current is not None:
        name = str(getattr(current, "name", ""))
        if name.startswith("layer::"):
            return name.removeprefix("layer::")
        current = getattr(current, "cpu_parent", None)
    return "unknown"


def profile_kernels(model, inputs, image_size: int, batch: int, device):
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.inference_mode():
            output = recorded_forward(model, inputs)
        torch.cuda.synchronize(device)
    del output

    rows = []
    for event in profiler.events():
        layer = _event_layer(event)
        for kernel in getattr(event, "kernels", ()):
            rows.append(
                {
                    "S": image_size,
                    "B": batch,
                    "layer": layer,
                    "kernel_name": str(kernel.name),
                }
            )
    return rows


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


_OOM_ERROR_TYPES = tuple(
    error_type
    for error_type in (
        getattr(torch, "OutOfMemoryError", None),
        getattr(torch.cuda, "OutOfMemoryError", None),
    )
    if error_type is not None
)


def _is_cuda_oom(error: BaseException) -> bool:
    if _OOM_ERROR_TYPES and isinstance(error, _OOM_ERROR_TYPES):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return "out of memory" in message or "alloc_failed" in message


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument(
        "--energy-seconds",
        type=float,
        default=1.0,
        help="target duration for repeated forwards used by energy measurement",
    )
    parser.add_argument("--max-energy-repeats", type=int, default=10_000)
    parser.add_argument("--power-sample-interval", type=float, default=0.01)
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--skip-kernels", action="store_true")
    parser.add_argument(
        "--stress-oom",
        action="store_true",
        help=(
            "run supplemental high-memory points and write oom_stress.csv "
            "without replacing the required measurement grid"
        ),
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.latency_repeats <= 0:
        parser.error("--latency-repeats must be positive")
    if args.energy_seconds <= 0:
        parser.error("--energy-seconds must be positive")
    if args.max_energy_repeats <= 0:
        parser.error("--max-energy-repeats must be positive")
    if args.power_sample_interval <= 0:
        parser.error("--power-sample-interval must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-capable NVIDIA GPU is required for measurement")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise SystemExit(
            f"CUDA device {args.device} is unavailable; found {torch.cuda.device_count()} device(s)"
        )

    configure_pytorch()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = CNN().to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 1_040_324:
        raise RuntimeError(
            f"model has {parameter_count:,} parameters; expected 1,040,324"
        )

    if args.stress_oom:
        configurations = build_oom_stress_grid()
        measure_energy = False
        profile_kernel_names = False
        warmup_iterations = min(args.warmup, 1)
        latency_repeats = min(args.latency_repeats, 3)
        print("OOM stress mode: energy and kernel profiling are disabled")
        print(f"Stress points: {tuple((row['S'], row['B']) for row in configurations)}")
    else:
        configurations, extra_images, extra_batches = build_measurement_grid(
            args.seed
        )
        measure_energy = not args.skip_energy
        profile_kernel_names = not args.skip_kernels
        warmup_iterations = args.warmup
        latency_repeats = args.latency_repeats
        print(f"Validation image sizes: {extra_images}")
        print(f"Validation batch sizes: {extra_batches}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stress_oom:
        measurements_path = args.output_dir / "oom_stress.csv"
        kernels_path = args.output_dir / "oom_stress_kernels.csv"
    else:
        measurements_path = args.output_dir / "measurements.csv"
        kernels_path = args.output_dir / "kernels.csv"

    energy_meter = None
    if measure_energy:
        energy_meter = NVMLEnergyMeter(args.device, args.power_sample_interval)
        if energy_meter.available:
            print(f"Energy method: {energy_meter.method}")
        else:
            print(f"Energy measurement unavailable: {energy_meter.reason}")

    measurement_fields = (
        "S",
        "B",
        "latency_s",
        "memory_bytes",
        "energy_j",
        "is_validation",
        "is_stress",
        "stress_axis",
        "status",
        "predicted_memory_bytes",
        "gpu_free_memory_before_bytes",
        "gpu_total_memory_bytes",
        "predicted_oom_from_total",
        "energy_method",
        "oom_error",
    )
    kernel_fields = ("S", "B", "layer", "kernel_name")

    try:
        with ExitStack() as stack:
            measurements_file = stack.enter_context(
                measurements_path.open("w", newline="", encoding="utf-8")
            )
            measurement_writer = csv.DictWriter(
                measurements_file, fieldnames=measurement_fields
            )
            measurement_writer.writeheader()

            kernel_writer = None
            if profile_kernel_names:
                kernels_file = stack.enter_context(
                    kernels_path.open("w", newline="", encoding="utf-8")
                )
                kernel_writer = csv.DictWriter(kernels_file, fieldnames=kernel_fields)
                kernel_writer.writeheader()

            total = len(configurations)
            for index, configuration in enumerate(configurations, start=1):
                image_size = configuration["S"]
                batch = configuration["B"]
                print(f"[{index:03d}/{total}] S={image_size}, B={batch}", flush=True)

                free_memory, total_memory = torch.cuda.mem_get_info(device)
                predicted_memory_bytes = int(
                    round(predicted_memory(image_size, batch))
                )

                row = {
                    "S": image_size,
                    "B": batch,
                    "latency_s": "",
                    "memory_bytes": "",
                    "energy_j": "",
                    "is_validation": configuration["is_validation"],
                    "is_stress": configuration["is_stress"],
                    "stress_axis": configuration["stress_axis"],
                    "status": "OOM",
                    "predicted_memory_bytes": predicted_memory_bytes,
                    "gpu_free_memory_before_bytes": int(free_memory),
                    "gpu_total_memory_bytes": int(total_memory),
                    "predicted_oom_from_total": (
                        predicted_memory_bytes > total_memory
                    ),
                    "energy_method": "skipped" if not measure_energy else "unavailable",
                    "oom_error": "",
                }
                inputs = None

                try:
                    inputs = torch.randn(
                        batch,
                        3,
                        image_size,
                        image_size,
                        device=device,
                        dtype=torch.float32,
                    )
                    warm_up(model, inputs, warmup_iterations, device)
                    row["memory_bytes"] = measure_peak_memory(model, inputs, device)
                    latency_s = measure_latency(
                        model, inputs, latency_repeats, device
                    )
                    row["latency_s"] = f"{latency_s:.12g}"
                    row["status"] = "OK"

                    if energy_meter is not None and energy_meter.available:
                        energy_repeats = min(
                            args.max_energy_repeats,
                            max(1, math.ceil(args.energy_seconds / latency_s)),
                        )
                        try:
                            energy_j, energy_method = energy_meter.measure(
                                lambda: run_repeated_forwards(
                                    model, inputs, energy_repeats
                                ),
                                energy_repeats,
                                device,
                            )
                            row["energy_j"] = f"{energy_j:.12g}"
                            row["energy_method"] = energy_method
                        except Exception as error:
                            print(f"  Energy measurement failed: {error}")
                            row["energy_method"] = "error"

                    if kernel_writer is not None:
                        try:
                            kernel_rows = profile_kernels(
                                model, inputs, image_size, batch, device
                            )
                            kernel_writer.writerows(kernel_rows)
                            kernels_file.flush()
                        except Exception as error:
                            print(f"  Kernel profiling failed: {error}")

                except Exception as error:
                    if not _is_cuda_oom(error):
                        raise
                    row["oom_error"] = f"{type(error).__name__}: {str(error)[:200]}"
                    print(f"  OOM ({type(error).__name__})")
                finally:
                    del inputs
                    cleanup_cuda()

                measurement_writer.writerow(row)
                measurements_file.flush()
    finally:
        if energy_meter is not None:
            energy_meter.close()

    print(f"Measurements written to {measurements_path}")
    if profile_kernel_names:
        print(f"Kernels written to {kernels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
