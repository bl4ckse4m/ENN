"""Phase 4 — language modeling on BabiStories (paper Section 4.1 / Figure 2).

Scaled-down budget vs the paper (which used 80k iterations x global batch 512
on H100/H200): default here is 20k iterations x batch 32 on one RTX 5070 Ti.
Model / optimizer follow Table 3 (d=768, 12 heads, ctx 512, pmem 2688, AdamW,
cosine LR, warmup 2k); only the depth and the training budget are scaled.

Example:
  python -m src.train_lm --method softmax --n-layer 1
  python -m src.train_lm --method epanechnikov --n-layer 1 --max-iters 25000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from .model import ModelConfig, StackAssoMem


class MemmapChunks:
    """Random 512-token chunks from a uint16 memmap (official StoriesDataset protocol).

    IMPORTANT: do not wrap this in a DataLoader with `shuffle=True` — the
    default RandomSampler materialises `randperm(n).tolist()` over all ~474M
    chunk offsets (~15 GB of Python ints) and kills the machine.  Batches are
    sampled here directly (with replacement, like the official
    InfiniteDataLoader), which is O(batch_size) in memory.
    """

    def __init__(self, path: str, block_size: int = 512):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    @property
    def n_chunks(self) -> int:
        return max(1, len(self.data) - self.block_size - 2)

    def batch(self, rng: np.random.Generator, batch_size: int):
        offs = rng.integers(0, self.n_chunks, size=batch_size)
        cols = np.arange(self.block_size)[None, :]
        x = torch.from_numpy(np.asarray(self.data[offs[:, None] + cols], dtype=np.int64))
        y = torch.from_numpy(np.asarray(self.data[offs[:, None] + cols + 1], dtype=np.int64))
        return x, y


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="softmax")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--data-dir", default="data/babistories")
    ap.add_argument("--n-layer", type=int, default=1, help="paper Figure 2: 1 / 8 / 12")
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--pmem-size", type=int, default=2688)
    ap.add_argument("--batch-size", type=int, default=16, help="16 fits 16 GB at ctx 512 (24-32 on a clean GPU)")
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-3, help="paper Table 3: 5e-3 (1L/8L), 1e-3 (12L)")
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup-iters", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--out-dir", default="results/lm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    return ap.parse_args()


def get_lr(it, max_iters, warmup, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / max(1, warmup)
    t = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def estimate_loss(model, data: MemmapChunks, iters, batch_size, rng, ac, device):
    model.eval()
    losses = []
    for _ in range(iters):
        X, Y = data.batch(rng, batch_size)
        with ac:
            _, loss = model(X.to(device, non_blocking=True), Y.to(device, non_blocking=True))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = os.path.join(args.out_dir, args.method, f"{args.n_layer}L", f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")

    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")
    if not os.path.isfile(train_path):
        raise SystemExit(f"{train_path} not found — run: python -m src.prepare_babistories")

    print(f"== LM BabiStories  method={args.method}  {args.n_layer}L  seed={args.seed} ==")
    t0 = time.time()
    train_ds = MemmapChunks(train_path, args.block_size)
    val_ds = MemmapChunks(val_path, args.block_size)
    print(f"data: train={train_ds.n_chunks:,} chunks  val={val_ds.n_chunks:,} chunks")

    cfg = ModelConfig(
        vocab_size=50257,
        method=args.method,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        pmem_size=args.pmem_size,
        pad_id=-1,
        loss_on="all",
    )
    model = StackAssoMem(cfg).to(args.device)
    print(f"model: {model.num_params() / 1e6:.1f}M params")
    log(json.dumps(vars(args)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    ac = torch.amp.autocast(args.device, dtype=torch.bfloat16) if args.dtype == "bfloat16" else torch.amp.autocast(args.device, enabled=False)

    history = {"iter": [], "train_loss": [], "val_loss": [], "lr": []}
    data_rng = np.random.default_rng(args.seed + 1)
    val_rng = np.random.default_rng(args.seed + 2)
    pbar = tqdm(range(1, args.max_iters + 1), desc=f"lm {args.method}/{args.n_layer}L", unit="it")
    model.train()
    for it in pbar:
        lr = get_lr(it, args.max_iters, args.warmup_iters, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        X, Y = train_ds.batch(data_rng, args.batch_size)
        X, Y = X.to(args.device, non_blocking=True), Y.to(args.device, non_blocking=True)
        with ac:
            _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if it % 50 == 0:
            history["iter"].append(it)
            history["train_loss"].append(float(loss.item()))
            history["lr"].append(lr)
            pbar.set_postfix(train=f"{loss.item():.3f}", lr=f"{lr:.1e}")

        if it % args.eval_interval == 0 or it == args.max_iters:
            val_loss = estimate_loss(model, val_ds, args.eval_iters, args.batch_size, val_rng, ac, args.device)
            history["val_loss"].append({"iter": it, "val_loss": val_loss})
            pbar.set_postfix(train=f"{loss.item():.3f}", val=f"{val_loss:.3f}")
            log(f"iter={it} train={loss.item():.4f} val={val_loss:.4f}")
            pbar.set_postfix(train=f"{loss.item():.3f}", val=f"{val_loss:.3f}")

    pbar.close()
    final_val = history["val_loss"][-1]["val_loss"] if history["val_loss"] else None
    metrics = {
        "args": vars(args),
        "params": model.num_params(),
        "wall_time_s": time.time() - t0,
        "final_val_loss": final_val,
        "history": history,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"final val loss: {final_val:.4f}" if final_val is not None else "no eval")
    print(f"done in {metrics['wall_time_s'] / 60:.1f} min -> {run_dir}/metrics.json")


if __name__ == "__main__":
    main()
