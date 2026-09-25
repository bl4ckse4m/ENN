from collections.abc import Mapping, Sequence

import numpy as np


FP32_BYTES = 4.0
PARAMETER_COUNT = 1_040_324.0


def _inputs(image_size, batch):
    image_size = np.asarray(image_size, dtype=np.float64)
    batch = np.asarray(batch, dtype=np.float64)

    try:
        image_size, batch = np.broadcast_arrays(image_size, batch)
    except ValueError as error:
        raise ValueError("image_size and batch must be broadcast-compatible") from error

    if np.any(~np.isfinite(image_size)) or np.any(image_size <= 0):
        raise ValueError("image_size must contain positive, finite values")
    if np.any(~np.isfinite(batch)) or np.any(batch <= 0):
        raise ValueError("batch must contain positive, finite values")

    return image_size, batch


def _result(value):
    value = np.asarray(value, dtype=np.float64)
    return float(value) if value.ndim == 0 else value


def _parameters(theta, names):
    if isinstance(theta, Mapping):
        try:
            values = tuple(float(theta[name]) for name in names)
        except KeyError as error:
            raise ValueError(f"theta is missing parameter {error.args[0]!r}") from error
    elif isinstance(theta, Sequence) or isinstance(theta, np.ndarray):
        if len(theta) != len(names):
            raise ValueError(
                f"theta must contain {len(names)} values in this order: {', '.join(names)}"
            )
        values = tuple(float(value) for value in theta)
    else:
        raise TypeError("theta must be a mapping or a sequence")

    if not all(np.isfinite(value) for value in values):
        raise ValueError("theta values must be finite")
    return values


def flops(image_size, batch):
    image_size, batch = _inputs(image_size, batch)
    value = batch * (17_712.0 * image_size**2 + 313_344.0)
    return _result(value)


def memory(image_size, batch):
    image_size, batch = _inputs(image_size, batch)
    value = FP32_BYTES * (PARAMETER_COUNT + 13.0 * batch * image_size**2)
    return _result(value)


def bytes_moved(image_size, batch):
    image_size, batch = _inputs(image_size, batch)
    moved_elements = (
        91.0 * batch * image_size**2 + 2_148.0 * batch + PARAMETER_COUNT
    )
    return _result(FP32_BYTES * moved_elements)


def latency(image_size, batch, theta):
    launch_overhead, compute_rate, memory_bandwidth = _parameters(
        theta,
        (
            "launch_overhead_s",
            "compute_rate_flops_s",
            "memory_bandwidth_bytes_s",
        ),
    )
    if launch_overhead < 0:
        raise ValueError("launch_overhead_s must be non-negative")
    if compute_rate <= 0 or memory_bandwidth <= 0:
        raise ValueError("compute rate and memory bandwidth must be positive")

    compute_time = np.asarray(flops(image_size, batch)) / compute_rate
    memory_time = np.asarray(bytes_moved(image_size, batch)) / memory_bandwidth
    return _result(launch_overhead + np.maximum(compute_time, memory_time))


def energy(image_size, batch, theta_energy):
    fixed_energy, energy_per_flop, energy_per_byte = _parameters(
        theta_energy,
        ("fixed_energy_j", "energy_per_flop_j", "energy_per_byte_j"),
    )
    if fixed_energy < 0 or energy_per_flop < 0 or energy_per_byte < 0:
        raise ValueError("energy parameters must be non-negative")

    value = (
        fixed_energy
        + energy_per_flop * np.asarray(flops(image_size, batch))
        + energy_per_byte * np.asarray(bytes_moved(image_size, batch))
    )
    return _result(value)


__all__ = ["flops", "memory", "bytes_moved", "latency", "energy"]
