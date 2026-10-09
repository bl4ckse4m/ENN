"""Synthetic tasks for length generalization: MQMTAR, Reverse, Sort.

Sequence format follows the official code
(deep-spin/sparse_kernel_regression: Library/data/dataset_synthetic.py):

    seq = [BOS] + src + [SEP] + trg + [EOS],   xs = seq[:-1], ys = seq[1:]

Token tables (official SPECIAL dict):
    mqmtar : vocab 263, data symbols 3..258 (256 ints), utility tokens 0,1,2,
             <bos>=259 <sep>=260 <eos>=261 <pad>=262
    reverse/sort : vocab 36, data symbols 0..31,
             <bos>=32 <sep>=33 <eos>=34 <pad>=35

MQMTAR layout (reconstructed from the official evaluation code, which scores
exact match on trg positions 1,2, 4,5, 7,8, 10,11 — i.e. four 2-token values
interleaved with a marker):
    src = [1, k1a, k1b, 2, v1a, v1b] * n_pairs + [0, q1a, q1b, ..., q4a, q4b]
    trg = [0, a1, a2, 0, b1, b2, 0, c1, c2, 0, d1, d2]
(1 = key marker, 2 = value marker, 0 = query/answer marker; keys/values/
queries are 2-token symbols from the 256-symbol alphabet).

Reverse / Sort layout: src = x_1..x_n, trg = x_n..x_1 (reverse) or
sorted(x) (sort).

Exact-match metrics are ports of compute_assoc_and_exact_accuracy /
compute_reverse_exact_accuracy from the official Library/test_synthetic.py.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

SPECIAL = {
    "mqmtar": {"<bos>": 259, "<sep>": 260, "<eos>": 261, "<pad>": 262},
    "reverse": {"<bos>": 32, "<sep>": 33, "<eos>": 34, "<pad>": 35},
    "copy": {"<bos>": 32, "<sep>": 33, "<eos>": 34, "<pad>": 35},
    "sort": {"<bos>": 32, "<sep>": 33, "<eos>": 34, "<pad>": 35},
}

TASK_VOCAB = {t: SPECIAL[t]["<pad>"] + 1 for t in SPECIAL}

# MQMTAR utility markers
MK_KEY, MK_VAL, MK_QRY = 1, 2, 0
MQMTAR_SYM_LO, MQMTAR_SYM_HI = 3, 258   # 256 data symbols
N_QUERIES = 4


# ---------------------------------------------------------------------------
# Generators (src, trg as lists of ints; no special tokens)
# ---------------------------------------------------------------------------


def gen_mqmtar(n_pairs: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """One MQMTAR example with n_pairs key-value pairs and 4 queries."""
    # 2 tokens per key + 2 tokens per value for every pair
    syms = rng.integers(MQMTAR_SYM_LO, MQMTAR_SYM_HI + 1, size=4 * n_pairs)
    keys = [(int(syms[2 * i]), int(syms[2 * i + 1])) for i in range(n_pairs)]
    # distinct keys (associative recall): resample collisions
    seen = set()
    for i in range(n_pairs):
        while keys[i] in seen:
            keys[i] = (
                int(rng.integers(MQMTAR_SYM_LO, MQMTAR_SYM_HI + 1)),
                int(rng.integers(MQMTAR_SYM_LO, MQMTAR_SYM_HI + 1)),
            )
        seen.add(keys[i])
    vals_off = 2 * n_pairs
    vals = [
        (int(syms[vals_off + 2 * i]), int(syms[vals_off + 2 * i + 1])) for i in range(n_pairs)
    ]
    qidx = rng.integers(0, n_pairs, size=N_QUERIES)

    src = []
    for (ka, kb), (va, vb) in zip(keys, vals):
        src += [MK_KEY, ka, kb, MK_VAL, va, vb]
    src += [MK_QRY]
    for qi in qidx:
        src += [keys[qi][0], keys[qi][1]]

    trg = []
    for qi in qidx:
        trg += [MK_QRY, vals[qi][0], vals[qi][1]]
    return src, trg


def gen_reverse(n: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    xs = [int(v) for v in rng.integers(0, 32, size=n)]
    return xs, xs[::-1]


def gen_sort(n: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    xs = [int(v) for v in rng.integers(0, 32, size=n)]
    return xs, sorted(xs)


def gen_example(task: str, size_param: int, rng: np.random.Generator):
    if task == "mqmtar":
        return gen_mqmtar(size_param, rng)
    if task == "reverse":
        return gen_reverse(size_param, rng)
    if task == "sort":
        return gen_sort(size_param, rng)
    raise ValueError(task)


def n_pairs_for_total_len(total_len: int) -> int:
    """MQMTAR: total = 6*n_pairs + 24  ->  n_pairs = (L - 24) / 6."""
    return max(1, round((total_len - 24) / 6))


def n_items_for_total_len(total_len: int) -> int:
    """Reverse/Sort: total = 2*n + 3."""
    return max(1, (total_len - 3) // 2)


def size_param_for_length(task: str, total_len: int) -> int:
    return n_pairs_for_total_len(total_len) if task == "mqmtar" else n_items_for_total_len(total_len)


def total_len_of(task: str, src, trg) -> int:
    return len(src) + len(trg) + 3  # BOS + SEP + EOS


# ---------------------------------------------------------------------------
# In-memory dataset with deterministic per-index generation
# ---------------------------------------------------------------------------


@dataclass
class SeqExample:
    xs: torch.Tensor
    ys: torch.Tensor
    ctx_len: int  # len([BOS] src [SEP] trg [EOS]) as in the official code


class SyntheticDataset(Dataset):
    """Deterministic on-the-fly generation; no data files needed.

    `size_param` = n_pairs (mqmtar) or n_items (reverse/sort).  Pass a list to
    sample variable sizes per example ("up to 64 tokens" training protocol).
    """

    def __init__(self, task: str, n_examples: int, size_param: int | list[int], seed: int = 0):
        self.task = task
        self.n_examples = n_examples
        self.seed = seed
        if isinstance(size_param, int):
            self.size_param = [size_param] * n_examples
        else:
            assert len(size_param) == n_examples
            self.size_param = list(size_param)
        self.SPECIAL = SPECIAL[task]

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx: int) -> SeqExample:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        src, trg = gen_example(self.task, self.size_param[idx], rng)
        seq = [self.SPECIAL["<bos>"]] + src + [self.SPECIAL["<sep>"]] + trg + [self.SPECIAL["<eos>"]]
        xs = torch.tensor(seq[:-1], dtype=torch.long)
        ys = torch.tensor(seq[1:], dtype=torch.long)
        return SeqExample(xs, ys, len(seq))


def make_collate(task: str):
    """Official make_indexedpair_collate semantics: pad to max_len with <pad>."""
    PAD = SPECIAL[task]["<pad>"]

    def collate(batch: list[SeqExample]):
        max_len = max(ex.ctx_len for ex in batch) - 1
        bx, by = [], []
        for ex in batch:
            pad_x = torch.full((max_len - len(ex.xs),), PAD, dtype=torch.long)
            pad_y = torch.full((max_len - len(ex.ys),), PAD, dtype=torch.long)
            bx.append(torch.cat([ex.xs, pad_x]))
            by.append(torch.cat([ex.ys, pad_y]))
        ctx = torch.tensor([ex.ctx_len for ex in batch], dtype=torch.long) - 1
        return torch.stack(bx), torch.stack(by), ctx

    return collate


# ---------------------------------------------------------------------------
# Exact-match metrics (verbatim semantics from the official test_synthetic.py)
# ---------------------------------------------------------------------------


def compute_assoc_and_exact_accuracy(pred_seq: torch.Tensor, tgt_seq: torch.Tensor, eos_token: int):
    """MQMTAR: per-pair recall and exact match over trg positions 1,2,4,5,7,8,10,11."""
    if pred_seq[-1] == eos_token:
        pred_seq = pred_seq[:-1]
    if tgt_seq[-1] == eos_token:
        tgt_seq = tgt_seq[:-1]
    kv_indices = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11], device=pred_seq.device)
    ncorrect, npairs = 0, 0
    for i in range(0, len(kv_indices) - 1, 2):
        idx1, idx2 = kv_indices[i], kv_indices[i + 1]
        if idx1 < len(pred_seq) and idx2 < len(pred_seq):
            npairs += 1
            if pred_seq[idx1] == tgt_seq[idx1] and pred_seq[idx2] == tgt_seq[idx2]:
                ncorrect += 1
    valid_idx = kv_indices[kv_indices < len(pred_seq)]
    exact_acc = bool(torch.all(pred_seq[valid_idx] == tgt_seq[valid_idx]).item()) if valid_idx.numel() else False
    return npairs, ncorrect, exact_acc


def compute_reverse_exact_accuracy(pred_seq: torch.Tensor, tgt_seq: torch.Tensor, sep_token: int):
    """Reverse/Sort: exact match of the tail after SEP (EOS removed)."""
    tgt_seq = tgt_seq[:-1]
    pred_seq = pred_seq[:-1]
    sep_idx = (tgt_seq == sep_token).nonzero(as_tuple=True)[0]
    if len(sep_idx) != 1:
        return 0.0, 0, 0
    sep_idx = sep_idx.item()
    tgt_tail = tgt_seq[sep_idx + 1 :]
    pred_tail = pred_seq[sep_idx + 1 :]
    if tgt_tail.numel() == 0 or tgt_tail.shape != pred_tail.shape:
        return 0.0, 0, 0
    token_matches = (pred_tail == tgt_tail).sum().item()
    num_tokens = tgt_tail.numel()
    return float(token_matches == num_tokens), token_matches, num_tokens


@torch.no_grad()
def evaluate_exact_match(model, task: str, size_param: int, n_examples: int, seed: int, batch_size: int, device: str):
    """Teacher-forced argmax exact match on a fixed-length test set.

    All sequences share the same length (as the official per-shard eval), so no
    padding is involved.  Returns a dict with exact_acc, token_acc, n.
    """
    model.eval()
    ds = SyntheticDataset(task, n_examples, size_param, seed=seed)
    collate = make_collate(task)
    eos = SPECIAL[task]["<eos>"]
    sep = SPECIAL[task]["<sep>"]
    total_exact, total_tok_match, total_tok, total_samples = 0.0, 0, 0, 0
    total_pairs, total_correct = 0, 0

    for start in range(0, n_examples, batch_size):
        batch = [ds[i] for i in range(start, min(start + batch_size, n_examples))]
        X, Y, ctx = collate(batch)
        X, Y = X.to(device), Y.to(device)
        logits, _ = model(X, Y)
        preds = logits.argmax(dim=-1)
        for b in range(preds.size(0)):
            if task == "mqmtar":
                pred_seq, tgt_seq = preds[b, -13:], Y[b, -13:]
                npairs, ncorrect, exact = compute_assoc_and_exact_accuracy(pred_seq, tgt_seq, eos_token=eos)
                total_pairs += npairs
                total_correct += ncorrect
                total_exact += int(exact)
            else:
                exact, tok_match, ntok = compute_reverse_exact_accuracy(preds[b], Y[b], sep)
                total_exact += exact
                total_tok_match += tok_match
                total_tok += ntok
            total_samples += 1

    out = {
        "exact_acc": total_exact / max(1, total_samples),
        "n": total_samples,
        "seq_len": total_len_of(task, *gen_example(task, size_param, np.random.default_rng(0))),
    }
    if task == "mqmtar":
        out["pair_acc"] = total_correct / max(1, total_pairs)
    else:
        out["token_acc"] = total_tok_match / max(1, total_tok)
    return out


# ---------------------------------------------------------------------------
# On-disk writer in the official format (src / trg text + .idx)
# ---------------------------------------------------------------------------


def write_official_split(out_dir: str, task: str, n_examples: int, size_param: int | list[int], seed: int, split: str):
    """Write files readable by the official IndexedPairDataset."""
    os.makedirs(out_dir, exist_ok=True)
    src_path = os.path.join(out_dir, f"{split}.src")
    trg_path = os.path.join(out_dir, f"{split}.trg")
    if isinstance(size_param, int):
        size_param = [size_param] * n_examples
    with open(src_path, "w") as fs, open(trg_path, "w") as ft, \
         open(src_path + ".idx", "wb") as isrc, open(trg_path + ".idx", "wb") as itrg:
        off_s = off_t = 0
        for i in range(n_examples):
            rng = np.random.default_rng(seed * 1_000_003 + i)
            src, trg = gen_example(task, size_param[i], rng)
            ls = " ".join(map(str, src)) + "\n"
            lt = " ".join(map(str, trg)) + "\n"
            isrc.write(struct.pack("Q", off_s))
            itrg.write(struct.pack("Q", off_t))
            fs.write(ls)
            ft.write(lt)
            off_s += len(ls)
            off_t += len(lt)
    return src_path, trg_path
