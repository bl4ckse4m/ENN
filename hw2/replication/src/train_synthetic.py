"""Phase 2 — length generalization (paper Table 2): MQMTAR / Reverse / Sort.

Trains a Memory Mosaics model (paper Table 6 hyperparameters) on synthetic
sequences up to 64 tokens and evaluates exact-match accuracy at increasing
sequence lengths (1x ... 64x).  Scaled-down budget vs the paper (50M/30M/40M
training examples): use --train-examples to control the size.

Examples:
  python -m src.train_synthetic --task mqmtar --method softmax --seed 1337 \
      --train-examples 2000000 --out-dir results/synthetic

  # paper Table 2 methods, one at a time:
  --method {softmax, top32_softmax, top32_uniform, norm_relu, relumax,
            relumax2, relumax3, epanechnikov, biweight, triweight}
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

from .data_synthetic import (
    SPECIAL,
    TASK_VOCAB,
    SyntheticDataset,
    evaluate_exact_match,
    make_collate,
    size_param_for_length,
)
from .model import ModelConfig, StackAssoMem

# evaluation multipliers used in the paper (Table 2)
MULTIPLIERS = {
    "mqmtar": [1, 2, 4, 8, 16, 32, 64],
    "reverse": [1, 1.5],
    "sort": [1, 2, 4],
}

# paper Table 6 model sizes
PAPER_MODEL = {
    "mqmtar": dict(n_layer=4, n_head=8, n_embd=512),
    "reverse": dict(n_layer=4, n_head=8, n_embd=512),
    "sort": dict(n_layer=2, n_head=8, n_embd=512),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["mqmtar", "reverse", "sort"], required=True)
    ap.add_argument("--method", default="softmax",
                    help="softmax | epanechnikov(=sparsemax) | biweight | triweight | "
                         "norm_relu | relumax | relumax2 | relumax3 | "
                         "top{k}_softmax | top{k}_uniform")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-examples", type=int, default=2_000_000,
                    help="training set size (paper: 50M/30M/40M); sampled with replacement")
    ap.add_argument("--batch-size", type=int, default=256, help="paper Table 6: 256")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-layer", type=int, default=None, help="default: paper Table 6")
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--n-embd", type=int, default=None)
    ap.add_argument("--pmem-size", type=int, default=2688)
    ap.add_argument("--eval-examples", type=int, default=256, help="test sequences per length")
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--out-dir", default="results/synthetic")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--save-ckpt", action="store_true")
    return ap.parse_args()


def get_lr(it: int, max_iters: int, warmup: int, lr: float, min_lr: float) -> float:
    if it < warmup:
        return lr * (it + 1) / max(1, warmup)
    t = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, t)))


def train_size_params(task: str, n: int, seed: int) -> list[int]:
    """Variable sizes "up to 64 tokens" (paper Table 6)."""
    rng = np.random.default_rng(seed)
    if task == "mqmtar":
        return [int(v) for v in rng.integers(4, 7, size=n)]   # 4..6 pairs -> 48..60 tokens
    return [int(v) for v in rng.integers(8, 31, size=n)]      # 8..30 items -> 19..63 tokens


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mdl = PAPER_MODEL[args.task]
    n_layer = args.n_layer or mdl["n_layer"]
    n_head = args.n_head or mdl["n_head"]
    n_embd = args.n_embd or mdl["n_embd"]

    cfg = ModelConfig(
        vocab_size=TASK_VOCAB[args.task],
        method=args.method,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=512,
        pmem_size=args.pmem_size,
        pad_id=SPECIAL[args.task]["<pad>"],
        sep_id=SPECIAL[args.task]["<sep>"],
    )
    model = StackAssoMem(cfg).to(args.device)
    if args.compile:
        model = torch.compile(model)

    max_iters = max(1, args.train_examples // args.batch_size)
    warmup = int(max_iters * args.warmup_frac)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    run_dir = os.path.join(args.out_dir, args.task, args.method, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")
    with open(log_path, "w") as f:
        f.write(json.dumps(vars(args), indent=2) + "\n")

    def log(msg: str):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    print(f"== {args.task} / {args.method} / seed {args.seed} ==")
    print(f"model: {model.num_params() / 1e6:.1f}M params, {n_layer}L/{n_head}H/{n_embd}d")
    print(f"training: {max_iters} iters x batch {args.batch_size} (~{args.train_examples} examples)")
    log(f"params={model.num_params()} max_iters={max_iters}")

    size_params = train_size_params(args.task, max(args.train_examples, 4096), args.seed + 1)
    train_ds = SyntheticDataset(args.task, len(size_params), size_params, seed=args.seed + 2)
    collate = make_collate(args.task)
    ac = torch.amp.autocast(args.device, dtype=torch.bfloat16) if args.dtype == "bfloat16" else torch.amp.autocast(args.device, enabled=False)

    history = {"iter": [], "train_loss": [], "lr": [], "eval": []}
    t0 = time.time()
    data_rng = np.random.default_rng(args.seed + 3)
    pbar = tqdm(range(1, max_iters + 1), desc=f"train {args.task}/{args.method}", unit="it")
    model.train()
    for it in pbar:
        lr = get_lr(it, max_iters, warmup, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        idx = data_rng.integers(0, len(train_ds), size=args.batch_size)
        X, Y, _ = collate([train_ds[int(i)] for i in idx])
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
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.1e}")

        if it % args.eval_interval == 0 or it == max_iters:
            em = evaluate_exact_match(model, args.task, size_param_for_length(args.task, 64),
                                      args.eval_examples, seed=10_000 + args.seed,
                                      batch_size=min(args.batch_size, 64), device=args.device)
            history["eval"].append({"iter": it, **em})
            pbar.set_postfix(loss=f"{loss.item():.3f}", exact=f"{em['exact_acc']:.2f}")
            log(f"iter={it} loss={loss.item():.4f} exact={em['exact_acc']:.4f}")
            model.train()

    # ---- final evaluation at all sequence lengths -------------------------
    pbar.close()
    print("evaluating at increasing sequence lengths ...")
    results = {}
    for mu in MULTIPLIERS[args.task]:
        total_len = int(64 * mu)
        sp = size_param_for_length(args.task, total_len)
        # scale eval batch to keep the O(T^2) attention inside 16 GB
        eval_bs = max(1, min(64, 8192 // max(64, total_len)))
        em = evaluate_exact_match(model, args.task, sp, args.eval_examples,
                                  seed=20_000 + args.seed, batch_size=eval_bs, device=args.device)
        results[f"{mu}x"] = em
        print(f"  {mu:>4}x  (len~{em['seq_len']:>4}):  exact={em['exact_acc']:.3f}"
              + (f"  pair={em['pair_acc']:.3f}" if "pair_acc" in em else f"  tok={em['token_acc']:.3f}"))
        log(f"len={mu}x {em}")

    metrics = {
        "args": vars(args),
        "params": model.num_params(),
        "wall_time_s": time.time() - t0,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "results_by_length": results,
        "history": history,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.save_ckpt:
        raw = getattr(model, "_orig_mod", model)
        torch.save({"model": raw.state_dict(), "config": vars(cfg), "args": vars(args)},
                   os.path.join(run_dir, "ckpt.pt"))

    print(f"done in {metrics['wall_time_s'] / 60:.1f} min  ->  {run_dir}/metrics.json")


if __name__ == "__main__":
    main()
