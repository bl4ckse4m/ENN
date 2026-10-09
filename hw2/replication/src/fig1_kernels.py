"""Phase 1 — reproduce Figure 1 of the paper.

Left : normalized 1-D kernels used in kernel regression.
Right: attention maps of the corresponding activations on a 3-D score vector
       u = [0, u1, u2]; colour shows the weight p0 of the first entry, which
       visualises where each mechanism keeps the entry in its support.

Run:  python -m src.fig1_kernels --out figures/fig1_kernels.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import kernels as K


def normalized_kernel(u: np.ndarray, kind: str) -> np.ndarray:
    """1-D kernel values, normalized to peak 1 for shape comparison."""
    d2 = u**2
    if kind == "Gaussian":
        v = np.exp(-d2 / 2)
    elif kind == "Triweight":
        v = K.rectified_poly_kernel(torch.tensor(d2), 1.0, 3).numpy()
    elif kind == "Biweight":
        v = K.rectified_poly_kernel(torch.tensor(d2), 1.0, 2).numpy()
    elif kind == "Epanechnikov":
        v = K.epanechnikov_kernel(torch.tensor(d2), 1.0).numpy()
    elif kind == "Gaussian kNN":
        v = np.exp(-d2 / 2) * (np.abs(u) <= 1.25)
    elif kind == "Uniform kNN":
        v = (np.abs(u) <= 1.25).astype(float)
    else:
        raise ValueError(kind)
    m = v.max()
    return v / m if m > 0 else v


def attn_weight_p0(u1: np.ndarray, u2: np.ndarray, method: str, gamma: float = 1.0) -> np.ndarray:
    """p0 of activation([0, u1, u2]) on a grid."""
    out = np.zeros_like(u1)
    flat = torch.tensor(np.stack([u1.ravel(), u2.ravel()], axis=-1), dtype=torch.float32)
    zeros = torch.zeros(flat.shape[0], 1)
    scores = torch.cat([zeros, flat], dim=-1)  # (N, 3)
    p = K.attention(scores, method=method, gamma=gamma)
    out = p[:, 0].reshape(u1.shape).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/fig1_kernels.png")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig = plt.figure(figsize=(15.5, 6.2))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.35] + [1] * 5, hspace=0.32, wspace=0.28)

    # ---- left: kernels -----------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    u = np.linspace(-2, 2, 800)
    for kind, style in [
        ("Gaussian", "-"),
        ("Triweight", "-"),
        ("Biweight", "-"),
        ("Epanechnikov", "-"),
        ("Gaussian kNN", "--"),
        ("Uniform kNN", "--"),
    ]:
        ax.plot(u, normalized_kernel(u, kind), style, lw=1.8, label=kind)
    ax.set_title("Kernels", fontsize=11)
    ax.set_xlabel("u")
    ax.set_ylabel("K(u)  (peak-normalized)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    # ---- right: attention maps -------------------------------------------
    methods = [
        ("softmax", "1-entmax (Softmax)"),
        ("entmax43", "4/3-entmax"),
        ("entmax15", "1.5-entmax"),
        ("sparsemax", "2-entmax (Sparsemax)"),
        ("top2_softmax", "Top-2 Softmax"),
        ("top2_uniform", "Top-2 Uniform"),
        ("norm_relu", "NormReLU"),
        ("relumax1", "ReLUmax (r=1)"),
        ("relumax2", "ReLUmax (r=2)"),
        ("relumax3", "ReLUmax (r=3)"),
    ]
    g = np.linspace(-2, 2, 220)
    U1, U2 = np.meshgrid(g, g)
    for i, (meth, title) in enumerate(methods):
        row, col = divmod(i, 5)
        ax = fig.add_subplot(gs[row, 1 + col])
        Z = attn_weight_p0(U1, U2, meth)
        im = ax.imshow(
            Z,
            extent=[-2, 2, -2, 2],
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("u1", fontsize=8)
        ax.set_ylabel("u2", fontsize=8)
        ax.tick_params(labelsize=7)

    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cax, label="p0 = weight of the entry with score 0")

    fig.suptitle(
        "Figure 1 replication: classical kernels and their attention activations\n"
        "(score vector u = [0, u1, u2]; compact support shows up as zero-weight regions)",
        fontsize=12,
    )
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
