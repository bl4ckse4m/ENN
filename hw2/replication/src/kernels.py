"""Kernel-regression view of attention (Santos et al., arXiv:2601.22766v3).

This module implements, in one place:
  * classical kernels used in kernel regression (Gaussian, Epanechnikov,
    biweight, triweight, uniform / truncated variants);
  * the Nadaraya-Watson estimator;
  * attention activations: softmax, sparsemax, alpha-entmax, normalized ReLU,
    r-ReLUmax, top-k softmax, top-k uniform.

All activations operate row-wise on a score matrix u of shape (..., n) and are
implemented with differentiable pure-PyTorch ops (no custom CUDA), so the same
code is used both for the numerical theory checks and for training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Classical kernels (argument u is a distance-like quantity; see NW below)
# ---------------------------------------------------------------------------


def gaussian_kernel(dist2: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """K(u) ∝ exp(-||u||^2 / (2 h^2)); dist2 = ||u||^2."""
    return torch.exp(-dist2 / (2.0 * h * h))


def epanechnikov_kernel(dist2: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """K(u) = [1 - ||u||^2 / h^2]_+  (r = 1)."""
    return F.relu(1.0 - dist2 / (h * h))


def rectified_poly_kernel(dist2: torch.Tensor, h: float, r: int) -> torch.Tensor:
    """K(u) = [1 - ||u||^2 / h^2]_+^r : r=1 Epanechnikov, 2 biweight, 3 triweight."""
    return F.relu(1.0 - dist2 / (h * h)) ** r


def biweight_kernel(dist2: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    return rectified_poly_kernel(dist2, h, r=2)


def triweight_kernel(dist2: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    return rectified_poly_kernel(dist2, h, r=3)


def uniform_kernel(dist2: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """K(u) = 1{||u|| <= h}."""
    return (dist2 <= h * h).to(dist2.dtype)


def topk_gaussian_kernel(dist2: torch.Tensor, h: float, k: int) -> torch.Tensor:
    """Gaussian truncated to the k nearest neighbours (Eq. 4)."""
    n = dist2.shape[-1]
    kk = min(k, n)
    idx = dist2.topk(kk, dim=-1, largest=False).indices
    mask = torch.zeros_like(dist2).scatter_(-1, idx, 1.0)
    return gaussian_kernel(dist2, h) * mask


# ---------------------------------------------------------------------------
# Nadaraya-Watson estimator (Eq. 2)
# ---------------------------------------------------------------------------


def nadaraya_watson(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """weights: (..., n) unnormalized kernel values; values: (..., n, d)."""
    w = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return w.unsqueeze(-1) * values


def nw_from_scores(
    scores: torch.Tensor,
    values: torch.Tensor,
    kernel: str = "epanechnikov",
    h: float = 1.0,
    r: int = 1,
    k: int = 0,
) -> torch.Tensor:
    """NW estimate for unit-norm keys/queries with s_i = k_i^T q.

    Uses 0.5 ||k_i - q||^2 = 1 - s_i  =>  ||k_i - q||^2 = 2 (1 - s_i).
    """
    dist2 = 2.0 * (1.0 - scores)
    if kernel == "gaussian":
        w = gaussian_kernel(dist2, h)
    elif kernel == "epanechnikov":
        w = rectified_poly_kernel(dist2, h, 1)
    elif kernel == "biweight":
        w = rectified_poly_kernel(dist2, h, 2)
    elif kernel == "triweight":
        w = rectified_poly_kernel(dist2, h, 3)
    elif kernel == "rectpoly":
        w = rectified_poly_kernel(dist2, h, r)
    elif kernel == "uniform":
        w = uniform_kernel(dist2, h)
    elif kernel == "topk_gaussian":
        w = topk_gaussian_kernel(dist2, h, k)
    else:
        raise ValueError(f"unknown kernel {kernel}")
    return (w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)).unsqueeze(-1) * values


# ---------------------------------------------------------------------------
# Attention activations (row-wise on score rows u ∈ R^n)
# ---------------------------------------------------------------------------


def softmax(u: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1-entmax = softmax ↔ Gaussian kernel."""
    return F.softmax(u, dim=dim)


def sparsemax(u: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """2-entmax = sparsemax ↔ auto-normalized Epanechnikov (Eq. 8).

    Exact sort-based projection onto the probability simplex (Martins &
    Astudillo, 2016), differentiable a.e.
    """
    u = u.transpose(dim, -1) if dim != -1 else u
    orig_shape = u.shape
    z = u.reshape(-1, orig_shape[-1])
    n = z.shape[-1]
    zs, _ = torch.sort(z, dim=-1, descending=True)
    cssv = zs.cumsum(dim=-1) - 1.0
    idx = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    cond = zs - cssv / idx > 0
    k = cond.sum(dim=-1, keepdim=True).clamp_min(1)
    tau = cssv.gather(-1, k - 1) / k.to(z.dtype)
    p = F.relu(z - tau)
    p = p.reshape(orig_shape)
    return p.transpose(dim, -1) if dim != -1 else p


class _EntmaxBisect(torch.autograd.Function):
    """alpha-entmax via bisection, with the analytic Jacobian as backward.

    Forward:  p(z) = [(alpha-1) z - tau]_+^{1/(alpha-1)}, tau set by sum_i p_i = 1.
    Backward: J = diag(s) - s s^T / sum(s),  s_i = p_i^(2-alpha)   (Peters et al., 2019),
    i.e.  grad_z = s * (grad_p - <s, grad_p> / sum(s)).

    Note that the Jacobian cannot be obtained by differentiating the bisection
    loop through autograd: the sum enters the update of tau only through the
    boolean inside torch.where, so d tau / d z would be lost.  J @ 1 = 0, so
    the constant row shift used for numerical range does not change the
    gradient either.
    """

    @staticmethod
    def forward(ctx, z, alpha, iters):
        a = alpha - 1.0
        zz = z - (z.amax(dim=-1, keepdim=True) - 1.0)  # p invariant to row shift
        # s(tau) = sum_i [a z_i - tau]_+^(1/a) is decreasing in tau:
        #   s(lo) >= 1, s(hi) <= 1  ->  bisection
        hi = a * zz.amax(dim=-1, keepdim=True)              # s(hi) = 0
        lo = a * zz.amin(dim=-1, keepdim=True) - 1.0        # every term >= 1 -> s >= n
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            p = torch.relu(a * zz - mid) ** (1.0 / a)
            s = p.sum(dim=-1, keepdim=True)
            lo = torch.where(s > 1.0, mid, lo)   # need larger tau
            hi = torch.where(s > 1.0, hi, mid)   # need smaller tau
        p = torch.relu(a * zz - hi) ** (1.0 / a)
        ctx.save_for_backward(p)
        ctx.alpha = float(alpha)
        ctx.in_dtype = z.dtype
        return p

    @staticmethod
    def backward(ctx, grad_p):
        (p,) = ctx.saved_tensors
        # apply the Jacobian in float32 or wider (inputs may be bf16 under autocast)
        work = p if p.dtype in (torch.float32, torch.float64) else p.float()
        gp = grad_p.to(work.dtype)
        # s_i = p_i^(2-alpha); zero outside the support (for alpha > 2 the
        # exponent is negative and 0^(2-alpha) would be inf)
        exp = 2.0 - ctx.alpha
        s = torch.where(work > 0, work.clamp_min(1e-30).pow(exp), torch.zeros_like(work))
        denom = s.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        grad_z = s * (gp - (s * gp).sum(dim=-1, keepdim=True) / denom)
        return grad_z.to(ctx.in_dtype), None, None


def entmax(u: torch.Tensor, alpha: float = 1.5, dim: int = -1, iters: int = 50) -> torch.Tensor:
    """alpha-entmax (Eq. 1) via bisection on the threshold tau.

    alpha=1 -> softmax limit, alpha=2 -> sparsemax; alpha=1.5 <-> biweight
    (r=2), alpha=4/3 <-> triweight (r=3).  Fully differentiable: forward is the
    bisection of _EntmaxBisect, backward its closed-form Jacobian.
    """
    if abs(alpha - 1.0) < 1e-8:
        return softmax(u, dim=dim)
    if abs(alpha - 2.0) < 1e-8:
        return sparsemax(u, dim=dim)
    u = u.transpose(dim, -1) if dim != -1 else u
    p = _EntmaxBisect.apply(u, float(alpha), int(iters))
    return p.transpose(dim, -1) if dim != -1 else p


def norm_relu(u: torch.Tensor, gamma: float | torch.Tensor, b: float | torch.Tensor) -> torch.Tensor:
    """Normalized ReLU ↔ fixed-bandwidth Epanechnikov (Eq. 6).

    weights = [u/gamma + b]_+ / sum; degenerate (all-zero) rows fall back to
    uniform over the row (as in the official implementation).
    """
    scores = F.relu(u / gamma + b)
    row_sum = scores.sum(dim=-1, keepdim=True)
    uniform = torch.ones_like(scores) / scores.shape[-1]
    return torch.where(row_sum > 0, scores / row_sum.clamp_min(1e-12), uniform)


def relu_max(
    u: torch.Tensor,
    gamma: float | torch.Tensor,
    r: int = 1,
    b: float = 1.0,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """r-ReLUmax ↔ max-anchored rectified polynomial kernel (Eq. 11).

    K ∝ [b + (u_i - max_j u_j)/h^2]_+^r with gamma = h^2.  Because b > 0 at
    least the argmax entry is active, so no 0/0 degeneracy.  `valid` (bool)
    marks admissible keys (e.g. causal / non-padding); invalid entries are
    excluded from max and get zero weight.
    """
    if valid is not None:
        neg = -1e9
        u = u.masked_fill(~valid, neg)
    m = u.amax(dim=-1, keepdim=True)
    scores = F.relu((u - m) / gamma + b)
    if valid is not None:
        scores = scores * valid.to(scores.dtype)
    if r != 1:
        scores = scores ** r
    row_sum = scores.sum(dim=-1, keepdim=True)
    return scores / row_sum.clamp_min(1e-12)


def topk_softmax(u: torch.Tensor, k: int, dim: int = -1, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Top-k softmax ↔ kNN-truncated Gaussian kernel (Eq. 10)."""
    if valid is not None:
        neg = -1e9
        u = u.masked_fill(~valid, neg)
    kk = min(k, u.shape[dim])
    top_vals, top_idx = u.topk(kk, dim=dim)
    mask = torch.full_like(u, float("-inf")).scatter_(dim, top_idx, top_vals)
    p = F.softmax(mask, dim=dim)
    if valid is not None:
        p = p * valid.to(p.dtype)
    return p


def topk_uniform(u: torch.Tensor, k: int, dim: int = -1, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Top-k average pooling ↔ uniform-kNN kernel regression."""
    if valid is not None:
        neg = -1e9
        u = u.masked_fill(~valid, neg)
    kk = min(k, u.shape[dim])
    top_idx = u.topk(kk, dim=dim).indices
    mask = torch.zeros_like(u).scatter_(dim, top_idx, 1.0)
    if valid is not None:
        mask = mask * valid.to(mask.dtype)
    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return mask / denom


# ---------------------------------------------------------------------------
# Unified dispatcher (method names follow the official code)
# ---------------------------------------------------------------------------

# official name -> paper notation
METHOD_ALIASES = {
    "softmax": "gaussian",
    "gaussian": "gaussian",
    "epanechnikov": "sparsemax",       # official name for sparsemax
    "sparsemax": "sparsemax",
    "biweight": "biweight",            # 1.5-entmax (r = 2)
    "triweight": "triweight",          # 4/3-entmax (r = 3)
    "norm_relu": "norm_relu",
    "relumax": "relumax1",
    "relumax1": "relumax1",
    "relumax2": "relumax2",
    "relumax3": "relumax3",
}


def resolve_method(name: str) -> str:
    base = name
    if name.startswith("top") and "softmax" in name:
        return name  # e.g. top16_softmax / top32_softmax
    if name.startswith("top") and "uniform" in name:
        return name  # e.g. top16_uniform / top32_uniform
    if name in METHOD_ALIASES:
        return METHOD_ALIASES[name]
    return base


def attention(
    scores: torch.Tensor,
    method: str = "softmax",
    valid: torch.Tensor | None = None,
    gamma: float | torch.Tensor | None = None,
    b: float = 1.0,
    k: int = 16,
) -> torch.Tensor:
    """Apply an attention activation to a score matrix (any leading dims).

    scores: (..., n).  valid: optional bool mask of the same shape (True =
    admissible key).  gamma: kernel bandwidth squared (h^2) used by norm_relu /
    relu_max; if None a default of 1.0 is used (the model passes its learned
    temperature).
    """
    m = resolve_method(method)
    g = 1.0 if gamma is None else gamma

    # scores may contain -inf for masked keys; sparsemax/entmax/relu_max need
    # finite inputs (sort/cumsum and (u - max) overflow otherwise)
    scores = scores.masked_fill(~torch.isfinite(scores), -1e9)

    if m == "gaussian":
        if valid is not None:
            neg = -1e9
            scores = scores.masked_fill(~valid, neg)
        p = softmax(scores)
        return p * valid.to(p.dtype) if valid is not None else p
    if m == "sparsemax":
        if valid is not None:
            neg = -1e9
            scores = scores.masked_fill(~valid, neg)
        p = sparsemax(scores)
        return p * valid.to(p.dtype) if valid is not None else p
    if m == "biweight":
        return attention(scores, "entmax15", valid=valid)
    if m == "triweight":
        return attention(scores, "entmax43", valid=valid)
    if m == "entmax15":
        if valid is not None:
            neg = -1e9
            scores = scores.masked_fill(~valid, neg)
        p = entmax(scores, alpha=1.5)
        return p * valid.to(p.dtype) if valid is not None else p
    if m == "entmax43":
        if valid is not None:
            neg = -1e9
            scores = scores.masked_fill(~valid, neg)
        p = entmax(scores, alpha=4.0 / 3.0)
        return p * valid.to(p.dtype) if valid is not None else p
    if m == "norm_relu":
        # fixed-bandwidth Epanechnikov; official code: gamma=h^2/2, b=1-2/h^2
        if valid is not None:
            scores = scores.masked_fill(~valid, 0.0)
        h2 = g if not torch.is_tensor(g) else g
        gamma_half = h2 * 0.5
        b_official = 1.0 - 2.0 / h2
        p = norm_relu(scores, gamma_half, b_official)
        return p * valid.to(p.dtype) if valid is not None else p
    if m.startswith("relumax"):
        r = int(m[-1]) if len(m) > 1 else 1
        return relu_max(scores, g, r=r, b=b, valid=valid)
    if "top" in m and "softmax" in m:
        kk = int(m.split("top")[1].split("_")[0])
        return topk_softmax(scores, kk, valid=valid)
    if "top" in m and "uniform" in m:
        kk = int(m.split("top")[1].split("_")[0])
        return topk_uniform(scores, kk, valid=valid)
    raise ValueError(f"unknown attention method: {method}")


def entmax_threshold(scores: torch.Tensor, gamma: float, alpha: float, dim: int = -1) -> torch.Tensor:
    """Return tau of alpha-entmax(scores/gamma) (Eq. 1) — needed to recover the
    implicit bandwidth h = sqrt(2 - 2 r gamma tau) of Proposition 1."""
    z = scores / gamma
    u = z.transpose(dim, -1) if dim != -1 else z
    a = alpha - 1.0
    hi = a * u.amax(dim=-1, keepdim=True)            # s(hi) = 0
    lo = a * u.amin(dim=-1, keepdim=True) - 1.0      # s(lo) >= n
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p = F.relu(a * u - mid) ** (1.0 / a)
        s = p.sum(dim=-1, keepdim=True)
        lo = torch.where(s > 1.0, mid, lo)
        hi = torch.where(s > 1.0, hi, mid)
    tau = hi
    return tau.transpose(dim, -1) if dim != -1 else tau
