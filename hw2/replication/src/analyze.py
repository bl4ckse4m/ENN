"""Aggregate replication results into tables and figures.

  python -m src.analyze --phase synthetic   # Table 2 style + length curves
  python -m src.analyze --phase icl         # Table 1 style
  python -m src.analyze --phase lm          # Figure 2 style val-loss curves
  python -m src.analyze --phase all

Outputs: figures/fig_*.png, results/summary_*.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRETTY = {
    "softmax": "Gaussian (softmax)",
    "top32_softmax": "Gaussian kNN k=32",
    "top16_softmax": "Gaussian kNN k=16",
    "top32_uniform": "Uniform kNN k=32",
    "top16_uniform": "Uniform kNN k=16",
    "norm_relu": "Epanech. (normReLU)",
    "relumax": "Epanech. (1-ReLUmax)",
    "relumax2": "Biweight (2-ReLUmax)",
    "relumax3": "Triweight (3-ReLUmax)",
    "epanechnikov": "Epanech. (sparsemax)",
    "biweight": "Biweight (1.5-entmax)",
    "triweight": "Triweight (4/3-entmax)",
}

# paper Table 2 (mean over 3 seeds) for side-by-side comparison
PAPER_TABLE2 = {
    "mqmtar": {
        "Gaussian (softmax)": [0.99, 0.96, 0.93, 0.85, 0.64, 0.14, 0.00],
        "Gaussian kNN k=32": [0.99, 0.97, 0.93, 0.84, 0.68, 0.27, 0.05],
        "Uniform kNN k=32": [0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        "Epanech. (normReLU)": [1.00, 1.00, 0.89, 0.66, 0.44, 0.33, 0.31],
        "Epanech. (1-ReLUmax)": [1.00, 1.00, 0.99, 0.92, 0.74, 0.42, 0.12],
        "Epanech. (sparsemax)": [1.00, 1.00, 1.00, 0.98, 0.86, 0.63, 0.31],
        "Biweight (2-ReLUmax)": [0.99, 0.97, 0.93, 0.85, 0.69, 0.27, 0.04],
        "Biweight (1.5-entmax)": [1.00, 1.00, 0.99, 0.95, 0.75, 0.51, 0.23],
        "Triweight (3-ReLUmax)": [0.99, 0.97, 0.93, 0.85, 0.71, 0.37, 0.08],
        "Triweight (4/3-entmax)": [1.00, 1.00, 0.99, 0.97, 0.89, 0.61, 0.25],
    },
}

MULT = {"mqmtar": ["1x", "2x", "4x", "8x", "16x", "32x", "64x"],
        "reverse": ["1x", "1.5x"], "sort": ["1x", "2x", "4x"]}


def load_metrics(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            out.append(json.load(f))
    return out


def phase_synthetic(out_dir="results/synthetic", fig_dir="figures", tag=""):
    suffix = f"_{tag}" if tag else ""
    rows = {}  # (task, method) -> {mult: [accs]}
    for m in load_metrics(os.path.join(out_dir, "*", "*", "*", "metrics.json")):
        task, method, seed = m["args"]["task"], m["args"]["method"], m["args"]["seed"]
        rows.setdefault((task, method), {}).setdefault(seed, m["results_by_length"])

    lines = ["# Length generalization (paper Table 2)\n"]
    for task in ("mqmtar", "reverse", "sort"):
        if not any(t == task for t, _ in rows):
            continue
        mults = MULT[task]
        lines.append(f"\n## {task}\n")
        lines.append("| method | " + " | ".join(mults) + " |")
        lines.append("|---" * (len(mults) + 1) + "|")
        plt.figure(figsize=(7, 4.2))
        for (t, method), by_seed in sorted(rows.items()):
            if t != task:
                continue
            means = []
            for mu in mults:
                accs = [s[mu]["exact_acc"] for s in by_seed.values() if mu in s]
                means.append(float(np.mean(accs)) if accs else float("nan"))
            label = PRETTY.get(method, method)
            lines.append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in means) + " |")
            plt.plot(range(len(mults)), means, marker="o", label=label)
        paper = PAPER_TABLE2.get(task)
        if paper:
            lines.append("\npaper (Table 2, 3 seeds):\n")
            lines.append("| method | " + " | ".join(mults) + " |")
            lines.append("|---" * (len(mults) + 1) + "|")
            for label, vals in paper.items():
                lines.append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        plt.xticks(range(len(mults)), mults)
        plt.xlabel("sequence length multiplier (train = 64 tokens)")
        plt.ylabel("exact match accuracy")
        plt.title(f"Length generalization — {task}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7, ncol=2)
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f"fig_length_{task}{suffix}.png"), bbox_inches="tight")
        plt.close()

    os.makedirs("results", exist_ok=True)
    with open(f"results/summary_synthetic{suffix}.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote results/summary_synthetic{suffix}.md and figures/fig_length_*{suffix}.png")


def phase_icl(out_dir="results/icl", fig_dir="figures"):
    rows = {}
    for m in load_metrics(os.path.join(out_dir, "*", "*", "*", "metrics.json")):
        n, method = m["args"]["num_examples"], m["args"]["method"]
        rows.setdefault(method, {})[n] = m["test"]
    sizes = sorted({n for by_n in rows.values() for n in by_n})
    lines = ["# RegBench ICL (paper Table 1)\n",
             "| method | " + " | ".join(f"acc@{n//1000}k" for n in sizes) + " | "
             + " | ".join(f"tvd@{n//1000}k" for n in sizes) + " |",
             "|" + "---|" * (2 * len(sizes) + 1)]
    plt.figure(figsize=(7, 4.2))
    for method, by_n in sorted(rows.items()):
        accs = [by_n[n]["acc"] if n in by_n else float("nan") for n in sizes]
        tvds = [by_n[n]["tvd"] if n in by_n else float("nan") for n in sizes]
        label = PRETTY.get(method, method)
        lines.append(f"| {label} | " + " | ".join(f"{a:.3f}" for a in accs) + " | "
                     + " | ".join(f"{t:.1f}" for t in tvds) + " |")
        plt.plot(sizes, accs, marker="o", label=label)
    plt.xscale("log")
    plt.xlabel("training examples")
    plt.ylabel("DFA accuracy")
    plt.title("RegBench in-context learning")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "fig_icl.png"), bbox_inches="tight")
    plt.close()
    lrs = {m["args"]["num_examples"]: m["args"].get("lr")
           for m in load_metrics(os.path.join(out_dir, "*", "*", "*", "metrics.json"))}
    lines.append("\nlr: " + ", ".join(f"{n // 1000}k={lr:g}" for n, lr in sorted(lrs.items())))
    with open("results/summary_icl.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/summary_icl.md and figures/fig_icl.png")


def phase_lm(out_dir="results/lm", fig_dir="figures"):
    rows = {}  # (layer, method) -> [metrics per seed]
    for m in load_metrics(os.path.join(out_dir, "*", "*", "*", "metrics.json")):
        method, layer = m["args"]["method"], m["args"]["n_layer"]
        rows.setdefault((layer, method), []).append(m)
    plt.figure(figsize=(7, 4.2))
    lines = ["# LM BabiStories (paper Figure 2, scaled budget)\n",
             "| method | params | val loss (mean±std) | per-seed | time/run (min) |",
             "|---|---|---|---|---|"]
    for (layer, method), ms in sorted(rows.items()):
        ms = sorted(ms, key=lambda x: x["args"].get("seed", 0))
        label = f"{PRETTY.get(method, method)} ({layer}L)"
        finals = np.array([x.get("final_val_loss") for x in ms], dtype=float)
        std = float(finals.std(ddof=1)) if len(finals) > 1 else 0.0
        per_seed = ", ".join(f"{x['args'].get('seed', '?')}:{x.get('final_val_loss'):.3f}" for x in ms)
        lines.append(f"| {label} | {ms[0]['params'] / 1e6:.1f}M | {finals.mean():.3f}±{std:.3f} | "
                     f"{per_seed} | {np.mean([x['wall_time_s'] / 60 for x in ms]):.0f} |")
        curves = [x["history"].get("val_loss", []) for x in ms]
        curves = [c for c in curves if c]
        if curves:
            for c in curves:  # thin per-seed curves
                plt.plot([v["iter"] for v in c], [v["val_loss"] for v in c], alpha=0.25, linewidth=1)
            k = min(len(c) for c in curves)  # mean curve over seeds
            mean = np.mean([[v["val_loss"] for v in c[:k]] for c in curves], axis=0)
            plt.plot([v["iter"] for v in curves[0][:k]], mean, linewidth=2, label=label)
    plt.xlabel("iteration")
    plt.ylabel("validation loss")
    plt.title("BabiStories language modeling (scaled budget)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "fig_lm.png"), bbox_inches="tight")
    plt.close()
    with open("results/summary_lm.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/summary_lm.md and figures/fig_lm.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["synthetic", "icl", "lm", "all"], default="all")
    args = ap.parse_args()
    if args.phase in ("synthetic", "all"):
        phase_synthetic()
        if os.path.isdir("results/synthetic5m"):
            phase_synthetic(out_dir="results/synthetic5m", tag="5m")
    if args.phase in ("icl", "all"):
        phase_icl()
    if args.phase in ("lm", "all"):
        phase_lm()


if __name__ == "__main__":
    main()
