"""Fit the calibrated latency/energy parameters and validate all equations.

Reads ``results/measurements.csv``, fits theta on the base grid only, scores
calibration and held-out validation points, then writes ``results/theta.json``
and ``results/figures/*.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, lsq_linear

if __package__:
    from .equations import bytes_moved, energy, flops, latency, memory
else:
    from equations import bytes_moved, energy, flops, latency, memory


LATENCY_PARAMS = (
    "launch_overhead_s",
    "compute_rate_flops_s",
    "memory_bandwidth_bytes_s",
)
ENERGY_PARAMS = (
    "fixed_energy_j",
    "energy_per_flop_j",
    "energy_per_byte_j",
)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIB = 2**20


def _flag(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def load_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["S"] = int(raw["S"])
            row["B"] = int(raw["B"])
            row["is_validation"] = _flag(raw.get("is_validation", False))
            row["is_stress"] = _flag(raw.get("is_stress", False))
            row["status"] = raw.get("status", "OK")
            rows.append(row)
    return rows


def split_rows(rows):
    grid_ok = [r for r in rows if not r["is_stress"] and r["status"] == "OK"]
    calibration = [r for r in grid_ok if not r["is_validation"]]
    validation = [r for r in grid_ok if r["is_validation"]]
    oom = [r for r in rows if r["status"] == "OOM"]
    return calibration, validation, grid_ok, oom


def as_arrays(rows):
    return (
        np.array([r["S"] for r in rows], dtype=np.float64),
        np.array([r["B"] for r in rows], dtype=np.float64),
    )


def fit_log_model(model, param_names, initial, s_values, b_values, measured):
    """Fit exp(parameterized) with log residuals to weight relative error."""

    def residuals(log_params):
        theta = dict(zip(param_names, np.exp(log_params)))
        predicted = np.asarray(model(s_values, b_values, theta))
        return np.log(predicted) - np.log(measured)

    result = least_squares(residuals, np.log(np.asarray(initial, dtype=np.float64)))
    theta = dict(zip(param_names, (float(v) for v in np.exp(result.x))))
    return theta, result


def fit_energy_linear(s_values, b_values, measured):
    """Non-negative weighted LS for E = E0 + eF*F + eQ*Q (linear in theta)."""

    design = np.column_stack(
        [
            np.ones_like(s_values),
            np.asarray(flops(s_values, b_values)),
            np.asarray(bytes_moved(s_values, b_values)),
        ]
    )
    weighted_design = design / measured[:, None]
    weighted_target = np.ones_like(measured)
    result = lsq_linear(weighted_design, weighted_target, bounds=(0.0, np.inf))
    theta = dict(zip(ENERGY_PARAMS, (float(v) for v in result.x)))
    theta["_design_condition_number"] = float(np.linalg.cond(design))
    theta["_flops_bytes_correlation"] = float(
        np.corrcoef(design[:, 1], design[:, 2])[0, 1]
    )
    return theta, result


def fit_memory_workspace(grid_rows):
    """Least-squares diagnostic: measured = a_scale*ideal + b_bytes.

    This is NOT part of the required parameter-free Memory(S,B) function. It
    only quantifies the backend workspace overhead that the ideal liveness
    model omits.
    """

    ideal = np.array(
        [float(r["predicted_memory_bytes"]) for r in grid_rows], dtype=np.float64
    )
    measured = np.array(
        [float(r["memory_bytes"]) for r in grid_rows], dtype=np.float64
    )
    design = np.column_stack([ideal, np.ones_like(ideal)])
    a_scale, b_bytes = np.linalg.lstsq(design, measured, rcond=None)[0]
    fitted = a_scale * ideal + b_bytes
    return {
        "a_scale": float(a_scale),
        "b_bytes": float(b_bytes),
        "max_residual_bytes": float(np.max(np.abs(measured - fitted))),
        "note": (
            "measured = a_scale*ideal + b_bytes on the 132-point grid; "
            "diagnostic only, Memory(S,B) stays parameter-free"
        ),
    }


def error_metrics(measured, predicted):
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - measured
    ape = np.abs(error / measured) * 100.0
    return {
        "points": int(measured.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(np.mean(ape)),
        "median_ape_percent": float(np.median(ape)),
        "max_ape_percent": float(np.max(ape)),
        "predicted_over_measured_mean": float(np.mean(predicted / measured)),
    }


def predict_arrays(model, rows, theta, key):
    s_values, b_values = as_arrays(rows)
    measured = np.array([r[key] for r in rows], dtype=np.float64)
    predicted = np.asarray(model(s_values, b_values, theta))
    return measured, predicted


def score_split(model, rows, theta, key):
    measured, predicted = predict_arrays(model, rows, theta, key)
    return error_metrics(measured, predicted)


# ---------------------------------------------------------------- figures


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def fig_pred_vs_measured(path, rows_cal, rows_val, model, theta, name, unit, key):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for rows, marker, label, color in (
        (rows_cal, "o", "calibration (base grid)", "#1f77b4"),
        (rows_val, "^", "validation (unseen)", "#ff7f0e"),
    ):
        measured, predicted = predict_arrays(model, rows, theta, key)
        ax.scatter(measured, predicted, marker=marker, s=22, alpha=0.75,
                   label=label, color=color)
    limits = ax.get_xlim()
    low = min(limits[0], ax.get_ylim()[0])
    high = max(limits[1], ax.get_ylim()[1])
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="ideal: predicted = measured")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(ax, f"{name}: predicted vs measured", f"measured {name} ({unit})",
               f"predicted {name} ({unit})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_curves(path, rows, model, theta, name, unit, key, log_y=True,
               overlay_model=None, overlay_label=None):
    s_values = sorted({r["S"] for r in rows})
    b_values = np.array(sorted({r["B"] for r in rows}), dtype=np.float64)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for index, s_value in enumerate(s_values):
        color = cmap(index / max(1, len(s_values) - 1))
        batch_points = np.array(
            [r["B"] for r in rows if r["S"] == s_value], dtype=np.float64
        )
        measured_points = np.array(
            [r[key] for r in rows if r["S"] == s_value], dtype=np.float64
        )
        predicted_curve = np.asarray(
            model(np.full_like(b_values, s_value), b_values, theta)
        )
        ax.plot(b_values, predicted_curve, "-", color=color, linewidth=1.2,
                label=f"S={s_value}")
        ax.scatter(batch_points, measured_points, s=14, color=color, zorder=3)
        if overlay_model is not None:
            overlay_curve = np.asarray(
                overlay_model(np.full_like(b_values, s_value), b_values)
            )
            ax.plot(b_values, overlay_curve, "--", color="red", linewidth=1.0,
                    alpha=0.9, zorder=2)
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    style_axes(ax, f"{name}: predicted curves and measured points",
               "batch size B (images)", f"{name} ({unit})")
    legend = ax.legend(title="image size S", fontsize=7, ncol=2, loc="upper left")
    if overlay_model is not None:
        ax.plot([], [], "--", color="red", linewidth=1.0,
                label=overlay_label or "workspace-inclusive diagnostic")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-1:], labels[-1:], fontsize=7, loc="lower right")
        ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_error_heatmap(path, rows, model, theta, name, unit, key):
    s_values = sorted({r["S"] for r in rows})
    b_values = sorted({r["B"] for r in rows})
    lookup = {(r["S"], r["B"]): r for r in rows}
    s_values_arr = np.array(s_values, dtype=np.float64)
    b_values_arr = np.array(b_values, dtype=np.float64)
    mesh_s, mesh_b = np.meshgrid(s_values_arr, b_values_arr, indexing="ij")
    predicted = np.asarray(model(mesh_s, mesh_b, theta))
    measured = np.array(
        [[lookup[(s, b)][key] for b in b_values] for s in s_values],
        dtype=np.float64,
    )
    percent_error = 100.0 * (predicted / measured - 1.0)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    bound = max(1.0, float(np.max(np.abs(percent_error))))
    image = ax.imshow(percent_error, cmap="RdBu_r", vmin=-bound, vmax=bound,
                      aspect="auto", origin="lower")
    ax.set_xticks(range(len(b_values)))
    ax.set_xticklabels([str(b) for b in b_values], fontsize=7, rotation=45)
    ax.set_yticks(range(len(s_values)))
    ax.set_yticklabels([str(s) for s in s_values], fontsize=7)
    for i in range(len(s_values)):
        for j in range(len(b_values)):
            value = percent_error[i, j]
            ax.text(j, i, f"{value:+.0f}", ha="center", va="center", fontsize=4.8)
    style_axes(ax, f"{name}: prediction error (%) = 100*(pred/measured - 1)",
               "batch size B", "image size S (pixels)")
    fig.colorbar(image, ax=ax, label=f"{name} error (%)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_memory_error_heatmap(path, rows):
    fig_path = path
    s_values = sorted({r["S"] for r in rows})
    b_values = sorted({r["B"] for r in rows})
    lookup = {(r["S"], r["B"]): r for r in rows}
    percent_error = np.array(
        [
            [
                100.0
                * (
                    float(lookup[(s, b)]["predicted_memory_bytes"])
                    / float(lookup[(s, b)]["memory_bytes"])
                    - 1.0
                )
                for b in b_values
            ]
            for s in s_values
        ]
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    bound = max(1.0, float(np.max(np.abs(percent_error))))
    image = ax.imshow(percent_error, cmap="RdBu_r", vmin=-bound, vmax=bound,
                      aspect="auto", origin="lower")
    ax.set_xticks(range(len(b_values)))
    ax.set_xticklabels([str(b) for b in b_values], fontsize=7, rotation=45)
    ax.set_yticks(range(len(s_values)))
    ax.set_yticklabels([str(s) for s in s_values], fontsize=7)
    for i in range(len(s_values)):
        for j in range(len(b_values)):
            ax.text(j, i, f"{percent_error[i, j]:+.0f}", ha="center", va="center",
                    fontsize=4.8)
    style_axes(ax, "Peak memory: prediction error (%) vs measured",
               "batch size B", "image size S (pixels)")
    fig.colorbar(image, ax=ax, label="memory error (%)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def fig_oom_boundary(path, stress_rows, workspace):
    ok_rows = [r for r in stress_rows if r["status"] == "OK"]
    oom_rows = [r for r in stress_rows if r["status"] != "OK"]
    free_bytes = np.array(
        [float(r.get("gpu_free_memory_before_bytes", "nan")) for r in stress_rows]
    )
    free_gib = float(np.nanmean(free_bytes)) / 2**30

    def ideal_gib(rows):
        return np.array(
            [float(r["predicted_memory_bytes"]) for r in rows], dtype=np.float64
        ) / 2**30

    def empirical_gib(rows):
        return (
            workspace["a_scale"] * np.array(
                [float(r["predicted_memory_bytes"]) for r in rows], dtype=np.float64
            )
            + workspace["b_bytes"]
        ) / 2**30

    predicted_ok = ideal_gib(ok_rows)
    measured_ok = np.array(
        [float(r["memory_bytes"]) for r in ok_rows], dtype=np.float64
    ) / 2**30
    predicted_oom = ideal_gib(oom_rows)

    actual = np.array([r["status"] != "OK" for r in stress_rows])
    ideal_flags = ideal_gib(stress_rows) > free_gib
    emp_flags = empirical_gib(stress_rows) > free_gib
    ideal_correct = int(np.sum(ideal_flags == actual))
    emp_correct = int(np.sum(emp_flags == actual))
    total = len(stress_rows)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    limit = max(
        np.max(measured_ok) if measured_ok.size else free_gib,
        np.max(predicted_oom) if predicted_oom.size else free_gib,
    ) * 1.12
    grid_x = np.linspace(0, limit, 200)
    ax.plot(grid_x, grid_x, "k--", linewidth=1.0, label="analytic model: predicted = measured")
    ax.plot(
        grid_x,
        workspace["a_scale"] * grid_x + workspace["b_bytes"] / 2**30,
        "r--", linewidth=1.2,
        label=(
            "workspace-inclusive diagnostic: "
            f"{workspace['a_scale']:.2f}*M + {workspace['b_bytes'] / 2**20:.0f} MiB"
        ),
    )
    ax.axhline(free_gib, color="#555555", linestyle=":", linewidth=1.2,
               label=f"free VRAM before run = {free_gib:.1f} GiB")
    ax.scatter(predicted_ok, measured_ok, s=34, color="#2ca02c", zorder=3,
               label="measured peak (configuration fits)")
    ax.scatter(predicted_oom, np.full_like(predicted_oom, free_gib), s=58,
               marker="x", color="#d62728", linewidths=2, zorder=4,
               label="CUDA OOM (drawn at free VRAM)")
    ax.annotate(
        "OOM prediction quality over stress points:\n"
        f"  analytic Memory(S,B):     {ideal_correct}/{total} correct "
        f"(misses {int(np.sum(actual & ~ideal_flags))} OOMs)\n"
        "  with workspace term:      "
        f"{emp_correct}/{total} correct",
        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.5, va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    style_axes(ax, "Memory equation vs actual OOM boundary",
               "predicted peak memory (GiB)", "measured peak memory (GiB)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"ideal_correct": ideal_correct, "workspace_correct": emp_correct,
            "total": total}


def fig_memory_workspace_residual(path, grid_rows, workspace):
    work = np.array(
        [float(r["B"]) * float(r["S"]) ** 2 for r in grid_rows], dtype=np.float64
    )
    ideal = np.array(
        [float(r["predicted_memory_bytes"]) for r in grid_rows], dtype=np.float64
    )
    measured = np.array(
        [float(r["memory_bytes"]) for r in grid_rows], dtype=np.float64
    )
    residual = (measured - ideal) / 2**20

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(work, residual, s=18, color="#1f77b4", alpha=0.75,
               label="measured - ideal (grid points)")
    work_line = np.linspace(0, work.max(), 100)
    ideal_line = 4.0 * (1_040_324.0 + 13.0 * work_line)
    ax.plot(
        work_line,
        ((workspace["a_scale"] - 1.0) * ideal_line + workspace["b_bytes"]) / 2**20,
        "r--", linewidth=1.2,
        label=(
            f"fit: {workspace['a_scale']:.3f}*M + "
            f"{workspace['b_bytes'] / 2**20:.0f} MiB - M"
        ),
    )
    style_axes(ax, "Memory-model residual: cuDNN workspace + allocator overhead",
               "B * S^2 (pixel-images)", "measured - ideal (MiB)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_regime_map(path, theta, grid_rows):
    s_values = sorted({r["S"] for r in grid_rows})
    b_values = sorted({r["B"] for r in grid_rows})
    mesh_s, mesh_b = np.meshgrid(
        np.array(s_values, dtype=np.float64),
        np.array(b_values, dtype=np.float64),
        indexing="ij",
    )
    compute_time = np.asarray(flops(mesh_s, mesh_b)) / theta["compute_rate_flops_s"]
    memory_time = (
        np.asarray(bytes_moved(mesh_s, mesh_b)) / theta["memory_bandwidth_bytes_s"]
    )
    launch_time = theta["launch_overhead_s"]
    dominant = np.where(
        np.maximum(compute_time, memory_time) <= launch_time,
        0,
        np.where(compute_time >= memory_time, 2, 1),
    )
    work_regime = (compute_time >= memory_time).astype(float)
    overhead_ratio = np.maximum(compute_time, memory_time) / launch_time

    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    cmap_dominant = ListedColormap(["#c6dbef", "#fdd0a2", "#a1d99b"])
    image_a = axes[0].imshow(dominant, cmap=cmap_dominant, vmin=-0.5, vmax=2.5,
                             aspect="auto", origin="lower")
    cbar_a = fig.colorbar(image_a, ax=axes[0], ticks=[0, 1, 2], shrink=0.85)
    cbar_a.ax.set_yticklabels(["launch-bound", "memory-bound", "compute-bound"])
    style_axes(axes[0],
               "(a) dominant term of t0 + max(F/Rc, Q/Rm)",
               "batch size B", "image size S (pixels)")

    cmap_work = ListedColormap(["#fdd0a2", "#a1d99b"])
    image_b = axes[1].imshow(work_regime, cmap=cmap_work, vmin=-0.5, vmax=1.5,
                             aspect="auto", origin="lower")
    cbar_b = fig.colorbar(image_b, ax=axes[1], ticks=[0, 1], shrink=0.85)
    cbar_b.ax.set_yticklabels(["memory-bound work", "compute-bound work"])
    rows_i, cols_j = np.meshgrid(
        np.arange(len(s_values)), np.arange(len(b_values)), indexing="ij"
    )
    axes[1].contour(cols_j, rows_i, overhead_ratio, levels=[1.0],
                    colors="black", linewidths=1.4)
    axes[1].text(0.4, len(s_values) - 0.8, "below contour:\nlaunch-bound total",
                 fontsize=7, va="top",
                 bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    style_axes(axes[1],
               "(b) work regime (F/Rc vs Q/Rm) + launch floor",
               "batch size B", "image size S (pixels)")

    for ax in axes:
        ax.set_xticks(range(len(b_values)))
        ax.set_xticklabels([str(b) for b in b_values], fontsize=6, rotation=45)
        ax.set_yticks(range(len(s_values)))
        ax.set_yticklabels([str(s) for s in s_values], fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return dominant


# ------------------------------------------------------------------ main


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(results_dir / "measurements.csv")
    stress_path = results_dir / "oom_stress.csv"
    if stress_path.exists():
        stress_rows = load_rows(stress_path)
    else:
        stress_rows = [r for r in rows if r["is_stress"]]

    calibration, validation, grid_ok, oom_rows = split_rows(rows)
    if not calibration or not validation:
        raise SystemExit("measurements.csv lacks calibration/validation rows")

    s_cal, b_cal = as_arrays(calibration)
    latency_measured = np.array([r["latency_s"] for r in calibration], dtype=np.float64)
    energy_measured = np.array([r["energy_j"] for r in calibration], dtype=np.float64)

    latency_theta, latency_fit = fit_log_model(
        latency,
        LATENCY_PARAMS,
        initial=(1e-4, 1e13, 3e11),
        s_values=s_cal,
        b_values=b_cal,
        measured=latency_measured,
    )
    energy_theta, energy_fit = fit_energy_linear(s_cal, b_cal, energy_measured)
    energy_diagnostics = {
        "design_condition_number": energy_theta.pop("_design_condition_number"),
        "flops_bytes_correlation": energy_theta.pop("_flops_bytes_correlation"),
        "method": "scipy.optimize.lsq_linear (non-negative, relative-error weighted)",
    }

    def latency_model(s_values, b_values, theta):
        return latency(s_values, b_values, theta)

    def energy_model(s_values, b_values, theta):
        return energy(s_values, b_values, theta)

    def memory_model(s_values, b_values, theta):
        return memory(s_values, b_values)

    workspace = fit_memory_workspace(grid_ok)

    def memory_workspace_model(s_values, b_values):
        return workspace["a_scale"] * np.asarray(memory(s_values, b_values)) + workspace["b_bytes"]

    metrics = {
        "latency_calibration": score_split(latency_model, calibration, latency_theta, "latency_s"),
        "latency_validation": score_split(latency_model, validation, latency_theta, "latency_s"),
        "energy_calibration": score_split(energy_model, calibration, energy_theta, "energy_j"),
        "energy_validation": score_split(energy_model, validation, energy_theta, "energy_j"),
        "memory_calibration": score_split(memory_model, calibration, None, "memory_bytes"),
        "memory_validation": score_split(memory_model, validation, None, "memory_bytes"),
    }
    if stress_rows:
        stress_ok = [r for r in stress_rows if r["status"] == "OK"]
        if stress_ok:
            ratio = np.array(
                [float(r["memory_bytes"]) / float(r["predicted_memory_bytes"])
                 for r in stress_ok]
            )
            metrics["memory_measured_over_predicted_stress"] = {
                "mean": float(ratio.mean()),
                "min": float(ratio.min()),
                "max": float(ratio.max()),
            }
        actual_oom = np.array([r["status"] != "OK" for r in stress_rows])
        ideal_demand = np.array(
            [float(r["predicted_memory_bytes"]) for r in stress_rows]
        )
        free_bytes = np.array(
            [float(r.get("gpu_free_memory_before_bytes", np.inf)) for r in stress_rows]
        )
        ideal_flags = ideal_demand > free_bytes
        emp_flags = (
            workspace["a_scale"] * ideal_demand + workspace["b_bytes"]
        ) > free_bytes
        metrics["oom_comparison"] = {
            "stress_points": len(stress_rows),
            "actual_oom": int(actual_oom.sum()),
            "analytic_predicted_oom": int(ideal_flags.sum()),
            "analytic_correct": int((ideal_flags == actual_oom).sum()),
            "workspace_predicted_oom": int(emp_flags.sum()),
            "workspace_correct": int((emp_flags == actual_oom).sum()),
        }

    payload = {
        "latency": latency_theta,
        "energy": energy_theta,
        "memory_workspace_diagnostic": workspace,
        "fit": {
            "latency_method": "scipy.optimize.least_squares (log-space, log residuals)",
            "energy": energy_diagnostics,
            "residual": "relative / log error",
            "calibration_points": len(calibration),
            "validation_points": len(validation),
            "latency_cost": float(latency_fit.cost),
        },
        "metrics": metrics,
    }
    (results_dir / "theta.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    fig_pred_vs_measured(
        figures_dir / "latency_pred_vs_measured.png",
        calibration, validation, latency_model, latency_theta, "latency", "s",
        "latency_s",
    )
    fig_pred_vs_measured(
        figures_dir / "energy_pred_vs_measured.png",
        calibration, validation, energy_model, energy_theta, "energy", "J",
        "energy_j",
    )
    fig_pred_vs_measured(
        figures_dir / "memory_pred_vs_measured.png",
        calibration, validation, memory_model, None, "peak memory", "bytes",
        "memory_bytes",
    )
    fig_curves(
        figures_dir / "latency_curves.png",
        grid_ok, latency_model, latency_theta, "latency", "s", "latency_s",
    )
    fig_curves(
        figures_dir / "energy_curves.png",
        grid_ok, energy_model, energy_theta, "energy", "J", "energy_j",
    )
    fig_curves(
        figures_dir / "memory_curves.png",
        grid_ok, memory_model, None, "peak memory", "bytes", "memory_bytes",
        overlay_model=memory_workspace_model,
        overlay_label=(
            f"workspace-inclusive diagnostic: "
            f"{workspace['a_scale']:.2f}*M + {workspace['b_bytes'] / 2**20:.0f} MiB"
        ),
    )
    fig_error_heatmap(
        figures_dir / "latency_error_heatmap.png",
        grid_ok, latency_model, latency_theta, "latency", "s", "latency_s",
    )
    fig_error_heatmap(
        figures_dir / "energy_error_heatmap.png",
        grid_ok, energy_model, energy_theta, "energy", "J", "energy_j",
    )
    fig_memory_error_heatmap(figures_dir / "memory_error_heatmap.png", grid_ok)
    fig_memory_workspace_residual(
        figures_dir / "memory_workspace_residual.png", grid_ok, workspace
    )
    if stress_rows:
        fig_oom_boundary(figures_dir / "oom_boundary.png", stress_rows, workspace)
    fig_regime_map(figures_dir / "regime_map.png", latency_theta, grid_ok)

    print(json.dumps({"latency": latency_theta, "energy": energy_theta}, indent=2))
    for section, values in metrics.items():
        print(f"{section}: {values}")
    print(f"theta written to {results_dir / 'theta.json'}")
    print(f"figures written to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
