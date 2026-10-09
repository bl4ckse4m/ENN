"""Memory Mosaics (Zhang et al., ICLR 2025) with pluggable kernel attention.

Faithful pure-PyTorch port of the official implementation from
deep-spin/sparse_kernel_regression (Library/models/memory_mosaics.py and
memory_mosaics_eft.py): leaky-average keys, one-step-ahead values, contextual
memory (kernel-regression attention over past keys) + persistent memory
(learned slots).  Attention activations come from .kernels and are the ones
derived in the paper (softmax / sparsemax / entmax / normReLU / ReLUmax / top-k).

Differences from the official code (documented in the report):
  * no adasplash/triton dependency (pure PyTorch, works on any GPU);
  * padding is handled with an explicit key mask (same semantics as the eft
    variant's task-aware mask) instead of flash-attention varlen;
  * `method` names follow the official code ("softmax", "epanechnikov" =
    sparsemax, "biweight" = 1.5-entmax, "triweight" = 4/3-entmax, "norm_relu",
    "relumax" = 1-ReLUmax, "relumax2", "relumax3", "top{k}_softmax",
    "top{k}_uniform").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import attention as kernel_attention


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class FeatureExtractor(nn.Module):
    """Convolutional / linear feature extractor for keys and values."""

    def __init__(self, fe_type, shift, norm, kernel_size, n_embd, n_head, bias):
        super().__init__()
        self.fe_type = fe_type
        self.norm = norm
        self.n_head = n_head
        self.n_embd = n_embd
        if self.fe_type == "linearconv":
            self.shiftpad = nn.ZeroPad1d((kernel_size - 1 - shift, shift))
            self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=kernel_size, bias=bias)
        elif self.fe_type == "lowrlinearconv":
            assert kernel_size == 2, "lowrlinearconv only supports kernel_size=2"
            self.shiftpad = nn.ZeroPad1d((-shift, shift))
            self.linear = nn.Linear(n_embd, n_embd, bias=bias)
            self.coef = nn.Parameter(torch.ones(n_head).uniform_(0, 1))
        else:
            raise NotImplementedError(self.fe_type)

    def forward(self, x):
        B, T, C = x.size()
        nh, hs = self.n_head, self.n_embd // self.n_head
        if self.fe_type == "lowrlinearconv":
            x = self.linear(x).transpose(1, 2).view(B, C, T)
            y = self.shiftpad(x).view(B, nh, hs, T).transpose(2, 3) * (
                1 - self.coef.view(1, nh, 1, 1)
            ) + x.view(B, nh, hs, T).transpose(2, 3) * (self.coef.view(1, nh, 1, 1))
        else:
            x = x.transpose(1, 2)
            y = self.conv(self.shiftpad(x)).view(B, nh, hs, T).transpose(2, 3)
        if self.norm:
            y = y / y.norm(dim=-1, keepdim=True)
        return y


class Pmem(nn.Module):
    """Persistent memory: attention over learned key/value slots."""

    def __init__(self, pmem_count, pmem_size, pmem_dim, pmem_nhead, dropout, chunk_size=8):
        super().__init__()
        self.pmem_size, self.pmem_dim = pmem_size, pmem_dim
        self.pmem_nhead, self.pmem_count = pmem_nhead, pmem_count
        self.dropout = dropout
        self.chunk_size = chunk_size
        for i in range(pmem_count):
            M_k = nn.Parameter(torch.zeros(1, pmem_nhead, pmem_size, pmem_dim))
            M_v = nn.Parameter(torch.zeros(1, pmem_nhead, pmem_size, pmem_dim))
            torch.nn.init.normal_(M_k, mean=0.0, std=1 / np.sqrt(pmem_dim))
            torch.nn.init.normal_(M_v, mean=0.0, std=1 / np.sqrt(pmem_dim))
            setattr(self, f"M_k{i}", M_k)
            setattr(self, f"M_v{i}", M_v)

    def forward(self, key):
        # key: (B, nh, T, d); .expand() is a view (no copy) and keeps the SDPA
        # batch shapes equal so the memory-efficient kernel is selected.
        # Batch is processed in chunks so the backward pass never materialises
        # a (B, nh, pmem_size, d) gradient at once (16 GB friendly).
        B = key.size(0)
        chunk = max(1, min(B, self.chunk_size))
        outs = []
        for start in range(0, B, chunk):
            k = key[start : start + chunk]
            b = k.size(0)
            y = 0
            for i in range(self.pmem_count):
                y = y + F.scaled_dot_product_attention(
                    k,
                    getattr(self, f"M_k{i}").expand(b, self.pmem_nhead, self.pmem_size, self.pmem_dim),
                    getattr(self, f"M_v{i}").expand(b, self.pmem_nhead, self.pmem_size, self.pmem_dim),
                    is_causal=False,
                    scale=1,
                    dropout_p=self.dropout if self.training else 0,
                )
            outs.append(y / self.pmem_count)
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def _leaky_weights(T, beta, device, dtype):
    """Leaky-average weights w[h,i,j] = exp(-(i-j) beta_h) (1 - exp(-beta_h)), j <= i.

    beta: (n_head,).  Equivalent to the official `leakyavg_coef` buffer but
    materialised for the current length only (allows evaluating sequences
    longer than the training block size).
    """
    i = torch.arange(T, device=device, dtype=dtype).view(1, T, 1)
    j = torch.arange(T, device=device, dtype=dtype).view(1, 1, T)
    delta = (i - j).clamp_min(0)
    b = beta.to(device=device, dtype=dtype).view(-1, 1, 1)  # (nh, 1, 1)
    w = torch.exp(-delta * b) * (1 - torch.exp(-b))
    return w.masked_fill(j > i, 0)  # (nh, T, T)


def _batched_leaky_matmul(w, k, chunk_size=8):
    """(nh, T, T) weights applied per batch chunk (keeps backward memory low)."""
    B, nh, T, _ = k.shape
    outs = []
    for start in range(0, B, chunk_size):
        kc = k[start : start + chunk_size]
        b = kc.size(0)
        outs.append(w.unsqueeze(0).expand(b, nh, T, T) @ kc)
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


class InContextAssoMemBlock(nn.Module):
    """Contextual memory: kernel-regression attention over past keys (Eq. 3, 5)."""

    def __init__(
        self,
        method="softmax",
        n_head=12,
        n_embd=768,
        v_shift=1,
        block_size=512,
        k_fe_type="linearconv",
        v_fe_type="lowrlinearconv",
        k_kernel_size=1,
        v_kernel_size=2,
        ic_dropout=0.05,
        hd_dropout=0.05,
        bias=False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.method = method
        self.v_shift = v_shift
        self.block_size = block_size
        self.att_shift = 0
        self.exp_param_scale = 10
        self.kernel_beta_upbound = 5

        self.k_featurizer = FeatureExtractor(
            k_fe_type, shift=0, norm=False, kernel_size=k_kernel_size, n_embd=n_embd, n_head=n_head, bias=bias
        )
        self.v_featurizer = FeatureExtractor(
            v_fe_type, shift=v_shift, norm=True, kernel_size=v_kernel_size, n_embd=n_embd, n_head=n_head, bias=bias
        )
        self.ic_dropout = nn.Dropout(ic_dropout)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(hd_dropout)

        self.leaky_key_beta = nn.Parameter(torch.linspace(0.5, 5, n_head) / self.exp_param_scale)
        self.kernel_beta = nn.Parameter(torch.ones(n_head) / self.exp_param_scale)
        self.value_beta = nn.Parameter(torch.ones(n_head) * (-0.5) / self.exp_param_scale)

    def _leaky_average(self, k, T, nh, B):
        leaky_key_beta = self.leaky_key_beta.abs() * self.exp_param_scale
        w = _leaky_weights(T, leaky_key_beta, k.device, k.dtype)  # (nh, T, T)
        return _batched_leaky_matmul(w, k)

    def forward(self, x, key_valid=None):
        B, T, C = x.size()
        hs, nh = self.n_embd // self.n_head, self.n_head

        k = self._leaky_average(self.k_featurizer(x), T, nh, B)
        k = k / k.norm(dim=-1, keepdim=True)
        beta = torch.exp(torch.clamp(self.kernel_beta * self.exp_param_scale, max=self.kernel_beta_upbound))
        k = k * beta.view(1, nh, 1, 1).expand(B, nh, T, hs)

        v = self.v_featurizer(x) * torch.exp(self.value_beta * self.exp_param_scale).view(1, nh, 1, 1)

        att = k @ k.transpose(-1, -2)  # (B, nh, T, T), q ≡ k (Memory Mosaics design)

        # causal + padding validity for keys; the main diagonal is excluded
        # (attention sums run up to n-1, as in the paper / official code)
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril(diagonal=-1)
        valid = causal.view(1, 1, T, T).expand(B, nh, T, T)
        if key_valid is not None:  # (B, T) bool: non-pad positions
            valid = valid & key_valid.view(B, 1, 1, T)

        att = att.masked_fill(~valid, float("-inf"))
        att = att.clone()
        sl = slice(self.v_shift + self.att_shift, T)
        att[:, :, sl] = kernel_attention(
            att[:, :, sl],
            method=self.method,
            valid=valid[:, :, sl],
            gamma=beta.view(1, nh, 1, 1) ** -2,  # h^2 = 1/beta^2
        )
        att[:, :, : self.v_shift + self.att_shift] = 0  # no attention for the first v_shift rows

        att = self.ic_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class PersistentAssoMemBlock(nn.Module):
    def __init__(
        self,
        n_head=12,
        n_embd=768,
        block_size=512,
        k_fe_type="linearconv",
        k_kernel_size=1,
        ic_dropout=0.05,
        hd_dropout=0.05,
        bias=False,
        pmem_count=1,
        pmem_size=2688,
    ):
        super().__init__()
        self.n_head, self.n_embd, self.block_size = n_head, n_embd, block_size
        self.exp_param_scale = 10
        self.kernel_beta_upbound = 5

        self.pmem = Pmem(pmem_count, pmem_size, n_embd // n_head, n_head, ic_dropout)
        self.k_featurizer = FeatureExtractor(
            k_fe_type, shift=0, norm=False, kernel_size=k_kernel_size, n_embd=n_embd, n_head=n_head, bias=bias
        )
        self.ic_dropout = nn.Dropout(ic_dropout)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(hd_dropout)

        self.leaky_key_beta = nn.Parameter(torch.linspace(0.5, 5, n_head) / self.exp_param_scale)
        self.kernel_beta = nn.Parameter(torch.ones(n_head) / self.exp_param_scale)
        self.value_beta = nn.Parameter(torch.ones(n_head) * (-0.5) / self.exp_param_scale)

    def forward(self, x):
        B, T, C = x.size()
        hs, nh = self.n_embd // self.n_head, self.n_head
        k = self.k_featurizer(x)
        leaky_key_beta = self.leaky_key_beta.abs() * self.exp_param_scale
        w = _leaky_weights(T, leaky_key_beta, k.device, k.dtype)  # (nh, T, T)
        k = _batched_leaky_matmul(w, k)
        k = k / k.norm(dim=-1, keepdim=True)
        beta = torch.exp(torch.clamp(self.kernel_beta * self.exp_param_scale, max=self.kernel_beta_upbound))
        k = k * (beta**2).view(1, nh, 1, 1).expand(B, nh, T, hs)

        y = self.pmem(k) * torch.exp(self.value_beta * self.exp_param_scale).view(1, nh, 1, 1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class AttnOnlyBlock(nn.Module):
    def __init__(self, method="softmax", n_head=12, n_embd=768, v_shift=1, block_size=512, **kw):
        super().__init__()
        self.attn_only = kw.pop("attn_only", False)
        pre_ln = kw.pop("pre_ln", True)
        pmem_count = kw.pop("pmem_count", 1)
        pmem_size = kw.pop("pmem_size", 2688)
        self.pre_ln = pre_ln
        self.ln1 = LayerNorm(n_embd, bias=kw.get("bias", False))
        self.attn = InContextAssoMemBlock(
            method=method, n_head=n_head, n_embd=n_embd, v_shift=v_shift, block_size=block_size, **kw
        )
        if not self.attn_only:
            self.ln2 = LayerNorm(n_embd, bias=kw.get("bias", False))
            self.pmem = PersistentAssoMemBlock(
                n_head=n_head,
                n_embd=n_embd,
                block_size=block_size,
                k_fe_type=kw.get("k_fe_type", "linearconv"),
                k_kernel_size=kw.get("k_kernel_size", 1),
                ic_dropout=kw.get("ic_dropout", 0.05),
                hd_dropout=kw.get("hd_dropout", 0.05),
                bias=kw.get("bias", False),
                pmem_count=pmem_count,
                pmem_size=pmem_size,
            )

    def forward(self, x, key_valid=None):
        if self.pre_ln:
            x = x + self.attn(self.ln1(x), key_valid=key_valid)
            if not self.attn_only:
                x = x + self.pmem(self.ln2(x))
        else:
            x = self.ln1(x)
            x = x + self.attn(x, key_valid=key_valid)
            if not self.attn_only:
                x = self.ln2(x)
                x = x + self.pmem(x)
        return x


@dataclass
class ModelConfig:
    vocab_size: int = 263
    method: str = "softmax"
    n_head: int = 8
    n_embd: int = 512
    n_layer: int = 4
    v_shift: int = 1
    block_size: int = 512
    k_fe_type: str = "linearconv"
    v_fe_type: str = "lowrlinearconv"
    k_kernel_size: int = 1
    v_kernel_size: int = 2
    ic_dropout: float = 0.05
    hd_dropout: float = 0.05
    bias: bool = False
    pmem_count: int = 1
    pmem_size: int = 2688
    pre_ln: bool = True
    weight_tying: bool = True
    attn_only: bool = False
    skip_tokens: int = 0
    loss_on: str = "after_sep"  # "after_sep" (synthetic) | "all" (LM)
    pad_id: int = -1
    sep_id: int = -1


class StackAssoMem(nn.Module):
    """Full Memory Mosaics stack: contextual + persistent memories per block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        kw = dict(
            k_fe_type=config.k_fe_type,
            v_fe_type=config.v_fe_type,
            k_kernel_size=config.k_kernel_size,
            v_kernel_size=config.v_kernel_size,
            ic_dropout=config.ic_dropout,
            hd_dropout=config.hd_dropout,
            bias=config.bias,
            pre_ln=config.pre_ln,
            attn_only=config.attn_only,
            pmem_count=config.pmem_count,
            pmem_size=config.pmem_size,
        )
        self.blocks = nn.ModuleList(
            [
                AttnOnlyBlock(
                    method=config.method,
                    n_head=config.n_head,
                    n_embd=config.n_embd,
                    v_shift=config.v_shift,
                    block_size=config.block_size,
                    **kw,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_out = LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.emb.weight = self.head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                fanin = p.shape[1]
                torch.nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(fanin) / math.sqrt(config.n_layer))
        self.eval_status = []

    @property
    def device(self):
        return self.emb.weight.device

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fanin = module.weight.shape[1]
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / np.sqrt(fanin))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            fanin = module.weight.shape[1] * module.weight.shape[2]
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / np.sqrt(fanin))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1)

    def forward(self, idx, targets=None, logits_only=False, return_all_logits=True):
        B, T = idx.size()
        key_valid = idx != self.config.pad_id if self.config.pad_id >= 0 else None
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, key_valid=key_valid)
        x = self.ln_out(x)
        logits_scale = 1.0 / math.sqrt(self.n_embd)
        logits = logits_scale * self.head(x[:, self.config.skip_tokens :])

        if logits_only:
            return logits

        loss = None
        if targets is not None:
            tgt = targets[:, self.config.skip_tokens :]
            if self.config.loss_on == "after_sep" and self.config.sep_id >= 0 and self.config.pad_id >= 0:
                sep_pos = (idx == self.config.sep_id).int().argmax(dim=1)  # first SEP per row
                pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
                loss_mask = (pos >= sep_pos.unsqueeze(1)) & (idx != self.config.pad_id)
            else:
                loss_mask = idx != self.config.pad_id if self.config.pad_id >= 0 else torch.ones_like(idx, dtype=torch.bool)
            loss_mask = loss_mask & (tgt != -100)
            per_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="none"
            )
            m = loss_mask.reshape(-1)
            loss = (per_tok * m).sum() / m.sum().clamp_min(1)
            with torch.no_grad():
                self.eval_status = [loss.detach().item()]
        if not return_all_logits:
            logits = logits[:, [-1], :]
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond, targets=None, return_all_logits=False)
            idx = torch.cat([idx, logits.argmax(dim=-1)], dim=1)
        return idx
