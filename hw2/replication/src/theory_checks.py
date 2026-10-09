"""Phase 1 — numerical verification of the paper's theoretical claims.

Checks (all on random unit-norm keys / queries):
  1. Proposition 1:  alpha-entmax attention == Nadaraya-Watson regression with
     rectified polynomial kernel [1 - ||u||^2/h^2]_+^r, h = sqrt(2 - 2 r gamma tau).
  2. sparsemax == auto-normalized (adaptive bandwidth) Epanechnikov, r = 1.
  3. normalized ReLU == fixed-bandwidth Epanechnikov (Eq. 6).
  4. top-k softmax == truncated-Gaussian kNN; top-k uniform == uniform-kNN (Eq. 4, 10).
  5. r-ReLUmax support == {i : k_i^T q > m - b h^2} (Eq. 12); no 0/0 degeneracy.
  6. cross-validation of our sparsemax/entmax against the `entmax` reference package.
  7. sanity: simplex constraints, sparsity pattern, softmax/Gaussian limit.
  8. gradients of sparsemax / alpha-entmax: analytic Jacobian vs. finite
     differences and vs. the `entmax` package backward (regression guard for
     the entmax autograd.Function).

Run:  python -m src.theory_checks --out results/theory_checks.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from . import kernels as K

RESULTS = []


def record(name: str, passed: bool, detail: dict):
    RESULTS.append({"name": name, "passed": bool(passed), **detail})
    status = "PASS" if passed else "FAIL"
    extra = "  ".join(f"{k}={v}" for k, v in detail.items())
    print(f"[{status}] {name}:  {extra}")


def unit_rows(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, d, generator=gen)
    return x / x.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# 1 + 2. Proposition 1 (and its r = 1 specialization: sparsemax/Epanechnikov)
# ---------------------------------------------------------------------------


def check_prop1(n_trials: int = 20, d: int = 32, seed: int = 0):
    alphas = {"4/3 (triweight, r=3)": 4 / 3, "1.5 (biweight, r=2)": 1.5, "2.0 (Epanechnikov, r=1)": 2.0}
    max_diffs = {}
    for label, alpha in alphas.items():
        worst = 0.0
        for t in range(n_trials):
            gen = torch.Generator().manual_seed(seed * 1000 + t)
            n = int(torch.randint(2, 40, (1,), generator=gen))
            keys = unit_rows(n - 1, d, gen)
            q = unit_rows(1, d, gen)[0]
            vals = torch.randn(n - 1, d, generator=gen)
            gamma = float(torch.empty(1).uniform_(0.3, 3.0, generator=gen))

            scores = keys @ q  # (n-1,)
            p_ent = K.entmax(scores / gamma, alpha=alpha)  # attention weights

            # implicit bandwidth of Prop. 1
            r = 1.0 / (alpha - 1.0)
            tau = K.entmax_threshold(scores, gamma, alpha)
            h2 = 2.0 - 2.0 * r * gamma * tau
            assert (h2 > 0).all(), f"h^2 must be positive, got {h2}"
            h = torch.sqrt(h2)

            # Nadaraya-Watson with rectified polynomial kernel
            dist2 = 2.0 * (1.0 - scores)  # ||k_i - q||^2 for unit-norm vectors
            kw = K.rectified_poly_kernel(dist2, h, r=int(round(r)) if abs(r - round(r)) < 1e-6 else r)
            p_nw = kw / kw.sum(-1, keepdim=True).clamp_min(1e-12)

            worst = max(worst, (p_ent - p_nw).abs().max().item())
        max_diffs[label] = worst
        record(
            f"Prop.1: alpha-entmax == NW[rectpoly r=1/(alpha-1)]  (alpha={label})",
            worst < 1e-4,
            {"max_abs_diff": f"{worst:.2e}", "trials": n_trials},
        )
    return max_diffs


def check_sparsemax_epanechnikov(n_trials: int = 20, d: int = 32, seed: int = 7):
    """Eq. 8: sparsemax(Kq/gamma) is auto-normalized Epanechnikov; sum of
    rectified scores equals 1 and h = sqrt(2 - 2 gamma tau)."""
    worst_w, worst_h, worst_sum = 0.0, 0.0, 0.0
    for t in range(n_trials):
        gen = torch.Generator().manual_seed(seed * 1000 + t)
        n = int(torch.randint(2, 40, (1,), generator=gen))
        keys = unit_rows(n - 1, d, gen)
        q = unit_rows(1, d, gen)[0]
        gamma = float(torch.empty(1).uniform_(0.3, 3.0, generator=gen))

        scores = keys @ q
        p_sp = K.sparsemax(scores / gamma)
        tau = K.entmax_threshold(scores, gamma, alpha=2.0)

        # auto-normalization: sum_i [k_i^T q / gamma - tau]_+ = 1
        rect = torch.relu(scores / gamma - tau)
        worst_sum = max(worst_sum, abs(rect.sum().item() - 1.0))

        # adaptive bandwidth h = sqrt(2 - 2 gamma tau) -> Epanechnikov NW
        h = torch.sqrt(2.0 - 2.0 * gamma * tau)
        dist2 = 2.0 * (1.0 - scores)
        kw = K.epanechnikov_kernel(dist2, h)
        p_nw = kw / kw.sum(-1, keepdim=True).clamp_min(1e-12)
        worst_w = max(worst_w, (p_sp - p_nw).abs().max().item())

        # direct identity: sparsemax weights = [s/gamma - tau]_+ (Eq. 1 / 8)
        worst_h = max(worst_h, (p_sp - rect).abs().max().item())

    record("Eq.8: sparsemax == [s/gamma - tau]_+ (auto-normalized Epanechnikov)",
           worst_h < 1e-4, {"max_abs_diff": f"{worst_h:.2e}"})
    record("Eq.8: sum_i [s_i/gamma - tau]_+ = 1 (auto-normalization)",
           worst_sum < 1e-4, {"max_abs_err": f"{worst_sum:.2e}"})
    record("Eq.8: sparsemax == NW[Epanechnikov, h=sqrt(2-2 gamma tau)]",
           worst_w < 1e-4, {"max_abs_diff": f"{worst_w:.2e}"})


# ---------------------------------------------------------------------------
# 3. normalized ReLU == fixed-bandwidth Epanechnikov (Eq. 6)
# ---------------------------------------------------------------------------


def check_norm_relu(n_trials: int = 20, d: int = 32, seed: int = 11):
    worst = 0.0
    ok_degenerate = True
    for t in range(n_trials):
        gen = torch.Generator().manual_seed(seed * 1000 + t)
        n = int(torch.randint(2, 40, (1,), generator=gen))
        keys = unit_rows(n - 1, d, gen)
        q = unit_rows(1, d, gen)[0]
        h = float(torch.empty(1).uniform_(0.8, 3.0, generator=gen))
        gamma, b = h * h / 2.0, 1.0 - 2.0 / (h * h)

        scores = keys @ q
        w_relu = K.norm_relu(scores, gamma, b)

        dist2 = 2.0 * (1.0 - scores)
        kw = K.epanechnikov_kernel(dist2, h)  # [1 - ||u||^2/h^2]_+
        if kw.sum() > 0:  # non-degenerate row: compare directly
            p_nw = kw / kw.sum(-1, keepdim=True)
            worst = max(worst, (w_relu - p_nw).abs().max().item())
        else:  # degenerate row: official code falls back to uniform
            ok_degenerate &= bool(torch.allclose(w_relu, torch.ones_like(w_relu) / w_relu.numel()))

    record("Eq.6: normReLU == NW[Epanechnikov, fixed h] (gamma=h^2/2, b=1-2/h^2)",
           worst < 1e-4, {"max_abs_diff": f"{worst:.2e}", "degenerate_fallback_uniform": ok_degenerate})


# ---------------------------------------------------------------------------
# 4. top-k variants (Eq. 4 / 10)
# ---------------------------------------------------------------------------


def check_topk(n_trials: int = 20, d: int = 32, seed: int = 13):
    worst_soft, worst_unif = 0.0, 0.0
    for t in range(n_trials):
        gen = torch.Generator().manual_seed(seed * 1000 + t)
        n = int(torch.randint(6, 40, (1,), generator=gen))
        keys = unit_rows(n, d, gen)
        q = unit_rows(1, d, gen)[0]
        kk = int(torch.randint(2, min(8, n), (1,), generator=gen))
        h = float(torch.empty(1).uniform_(0.5, 2.0, generator=gen))

        scores = keys @ q
        dist2 = 2.0 * (1.0 - scores)

        # top-k softmax == truncated Gaussian kNN with h=1 (softmax(s) ∝ exp(-dist^2/2))
        p_soft = K.topk_softmax(scores, kk)
        kw = K.topk_gaussian_kernel(dist2, 1.0, kk)
        p_nw = kw / kw.sum(-1, keepdim=True).clamp_min(1e-12)
        worst_soft = max(worst_soft, (p_soft - p_nw).abs().max().item())

        # top-k uniform == uniform-kNN NW
        p_uni = K.topk_uniform(scores, kk)
        near = dist2.topk(kk, largest=False).indices
        ref = torch.zeros(n)
        ref[near] = 1.0 / kk
        worst_unif = max(worst_unif, (p_uni - ref).abs().max().item())

    record("Eq.10: top-k softmax == NW[truncated Gaussian kNN]",
           worst_soft < 1e-5, {"max_abs_diff": f"{worst_soft:.2e}"})
    record("top-k uniform == uniform-kNN NW (Eq. 4)",
           worst_unif < 1e-5, {"max_abs_diff": f"{worst_unif:.2e}"})


# ---------------------------------------------------------------------------
# 5. r-ReLUmax: Eq. 11 / 12
# ---------------------------------------------------------------------------


def check_relumax(n_trials: int = 20, d: int = 32, seed: int = 17):
    worst, support_ok, nonempty_ok = 0.0, True, True
    for t in range(n_trials):
        gen = torch.Generator().manual_seed(seed * 1000 + t)
        n = int(torch.randint(3, 40, (1,), generator=gen))
        keys = unit_rows(n, d, gen)
        q = unit_rows(1, d, gen)[0]
        h2 = float(torch.empty(1).uniform_(0.25, 4.0, generator=gen))
        b = float(torch.empty(1).uniform_(0.1, 2.0, generator=gen))
        r = int(torch.randint(1, 4, (1,), generator=gen))

        scores = keys @ q
        m = scores.max()
        w = K.relu_max(scores, h2, r=r, b=b)

        # Eq. 12: support == {i : s_i > m - b h^2}
        pred_support = scores > (m - b * h2) - 1e-6
        got_support = w > 0
        support_ok &= bool(torch.equal(pred_support, got_support))
        nonempty_ok &= bool((w.sum(-1) > 0).all())

        # Eq. 11: NW with max-anchored kernel K ∝ [b + (s - m)/h^2]_+^r
        kw = torch.relu(b + (scores - m) / h2) ** r
        p_nw = kw / kw.sum(-1, keepdim=True).clamp_min(1e-12)
        worst = max(worst, (w - p_nw).abs().max().item())

    record("Eq.11/12: r-ReLUmax == NW[max-anchored poly kernel] and support == {s > m - b h^2}",
           worst < 1e-5 and support_ok and nonempty_ok,
           {"max_abs_diff": f"{worst:.2e}", "support_matches": support_ok, "no_empty_support": nonempty_ok})


# ---------------------------------------------------------------------------
# 6. cross-validation against the reference `entmax` package
# ---------------------------------------------------------------------------


def check_against_entmax_package(n_trials: int = 10, d: int = 16, seed: int = 23):
    try:
        from entmax import sparsemax as ref_sparsemax, entmax_bisect
    except Exception as e:  # pragma: no cover
        record("cross-check vs `entmax` package", True, {"skipped": f"package unavailable: {e}"})
        return
    worst_sp, worst_e = 0.0, 0.0
    for t in range(n_trials):
        gen = torch.Generator().manual_seed(seed * 1000 + t)
        u = torch.randn(4, int(torch.randint(3, 20, (1,), generator=gen)), generator=gen)
        worst_sp = max(worst_sp, (K.sparsemax(u) - ref_sparsemax(u, dim=-1)).abs().max().item())
        for alpha in (1.5, 4 / 3):
            worst_e = max(worst_e, (K.entmax(u, alpha=alpha) - entmax_bisect(u, alpha=alpha, dim=-1)).abs().max().item())
    record("our sparsemax == entmax.sparsemax", worst_sp < 1e-5, {"max_abs_diff": f"{worst_sp:.2e}"})
    record("our entmax(1.5/4/3) == entmax.entmax_bisect", worst_e < 1e-4, {"max_abs_diff": f"{worst_e:.2e}"})


# ---------------------------------------------------------------------------
# 7. sanity: simplex constraints, sparsity, softmax limit
# ---------------------------------------------------------------------------


def check_sanity(seed: int = 31):
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(256, 64, generator=gen) * 2.0
    ok = True
    detail = {}
    for name, fn in [
        ("softmax", lambda x: K.softmax(x)),
        ("sparsemax", lambda x: K.sparsemax(x)),
        ("entmax15", lambda x: K.entmax(x, 1.5)),
        ("entmax43", lambda x: K.entmax(x, 4 / 3)),
        ("norm_relu", lambda x: K.norm_relu(x, 1.0, 0.0)),
        ("relumax1", lambda x: K.relu_max(x, 1.0, r=1)),
        ("relumax3", lambda x: K.relu_max(x, 1.0, r=3)),
        ("top8_softmax", lambda x: K.topk_softmax(x, 8)),
        ("top8_uniform", lambda x: K.topk_uniform(x, 8)),
    ]:
        p = fn(u)
        sums_ok = torch.allclose(p.sum(-1), torch.ones(256), atol=1e-4)
        nonneg = bool((p >= -1e-8).all())
        sparsity = float((p == 0).float().mean())
        detail[name] = {"sums_to_1": bool(sums_ok), "nonneg": nonneg, "zero_frac": round(sparsity, 3)}
        ok &= bool(sums_ok) and nonneg

    # softmax limit: entmax(alpha->1) -> softmax
    diff = (K.entmax(u, alpha=1.0 + 1e-3) - K.softmax(u)).abs().max().item()
    detail["entmax_alpha_1.001_vs_softmax"] = f"{diff:.2e}"
    ok &= diff < 5e-2
    record("sanity: simplex constraints for all activations", ok, detail)


# ---------------------------------------------------------------------------
# 8. gradients: backward of sparsemax / alpha-entmax (analytic Jacobian)
# ---------------------------------------------------------------------------


def _finite_diff_grad(fn, u: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """grad of sum(fn(u) * w) w.r.t. u via central differences (float64)."""
    g = torch.zeros_like(u)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            up, um = u.clone(), u.clone()
            up[i, j] += eps
            um[i, j] -= eps
            g[i, j] = ((fn(up) * w).sum() - (fn(um) * w).sum()) / (2 * eps)
    return g


def check_gradients(n_trials: int = 5, seed: int = 41):
    """Backward of sparsemax / alpha-entmax against finite differences and the
    `entmax` reference package.

    Guards the analytic Jacobian of src/kernels._EntmaxBisect (it cannot be
    obtained by differentiating the bisection loop through autograd: the sum
    enters only the boolean of torch.where, so d tau / d z is lost).
    """
    gen = torch.Generator().manual_seed(seed)
    cases = [
        ("sparsemax (alpha=2.0)", 2.0),
        ("entmax alpha=1.2", 1.2),
        ("entmax alpha=1.5 (biweight)", 1.5),
        ("entmax alpha=4/3 (triweight)", 4 / 3),
        ("entmax alpha=1.8", 1.8),
    ]
    for label, alpha in cases:
        worst = 0.0
        for _ in range(n_trials):
            n = int(torch.randint(4, 12, (1,), generator=gen))
            u = torch.randn(3, n, generator=gen, dtype=torch.float64) * 2.0
            w = torch.randn(3, n, generator=gen, dtype=torch.float64)
            ur = u.clone().requires_grad_(True)
            g_auto, = torch.autograd.grad((K.entmax(ur, alpha=alpha) * w).sum(), ur)
            g_fd = _finite_diff_grad(lambda x: K.entmax(x, alpha=alpha), u, w)
            # Floor the denominator: for (near-)one-hot rows the true gradient is
            # exactly 0 and both estimates are finite-difference noise, so a pure
            # relative metric reports O(1) for an essentially zero vector.
            # 1e-5 >> FD roundoff noise (~1e-10 at eps=1e-6) but still small
            # compared to any real gradient error; the exact check against the
            # `entmax` package Jacobian below is the primary correctness test.
            den = max(float(g_fd.norm()), float(g_auto.norm()), 1e-5)
            worst = max(worst, float((g_auto - g_fd).norm()) / den)
        record(f"grad: {label} backward == finite differences", worst < 1e-3,
               {"max_rel_err": f"{worst:.2e}", "trials": n_trials})

    # J @ 1 = 0 (shift invariance of entmax) <=> row sums of the gradient vanish
    u = torch.randn(64, 16, generator=gen, dtype=torch.float64) * 2.0
    w = torch.randn(64, 16, generator=gen, dtype=torch.float64)
    worst_sum = 0.0
    for alpha in (2.0, 1.5, 4 / 3, 2.5):   # 2.5: also guards inf-free s = p^(2-a)
        ur = u.clone().requires_grad_(True)
        g, = torch.autograd.grad((K.entmax(ur, alpha=alpha) * w).sum(), ur)
        assert torch.isfinite(g).all(), f"non-finite gradient at alpha={alpha}"
        worst_sum = max(worst_sum, g.sum(-1).abs().max().item())
    record("grad: shift invariance (row sums of grad == 0)", worst_sum < 1e-9,
           {"max_abs_row_sum": f"{worst_sum:.2e}"})

    try:
        from entmax import entmax_bisect
    except Exception as e:  # pragma: no cover
        record("grad: our entmax backward == entmax.entmax_bisect backward", True,
               {"skipped": f"package unavailable: {e}"})
        return
    worst_ref = 0.0
    for alpha in (1.2, 1.5, 4 / 3, 1.8):
        for _ in range(n_trials):
            n = int(torch.randint(6, 24, (1,), generator=gen))
            u32 = torch.randn(4, n, generator=gen) * 2.0
            w32 = torch.randn(4, n, generator=gen)
            u1, u2 = u32.clone().requires_grad_(True), u32.clone().requires_grad_(True)
            g1, = torch.autograd.grad((K.entmax(u1, alpha=alpha) * w32).sum(), u1)
            g2, = torch.autograd.grad((entmax_bisect(u2, alpha=alpha, dim=-1) * w32).sum(), u2)
            worst_ref = max(worst_ref, (g1 - g2).abs().max().item())
    record("grad: our entmax backward == entmax.entmax_bisect backward", worst_ref < 1e-4,
           {"max_abs_diff": f"{worst_ref:.2e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/theory_checks.json")
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    print("=" * 78)
    print("Phase 1: numerical verification of the kernel-regression correspondences")
    print("=" * 78)
    torch.manual_seed(0)

    check_prop1(n_trials=args.trials)
    check_sparsemax_epanechnikov(n_trials=args.trials)
    check_norm_relu(n_trials=args.trials)
    check_topk(n_trials=args.trials)
    check_relumax(n_trials=args.trials)
    check_against_entmax_package()
    check_sanity()
    check_gradients()

    n_pass = sum(r["passed"] for r in RESULTS)
    print("-" * 78)
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"saved: {args.out}")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
