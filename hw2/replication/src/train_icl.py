"""Phase 3 — RegBench in-context learning (paper Table 1).

Trains Memory Mosaics variants on the RegBench PFA benchmark (train sizes
1k..40k examples) and reports:
  * Accuracy  = fraction of valid DFA transitions (paper's "Acc" column)
  * TVD x100  = mean L1 distance to the uniform outgoing distribution
  * next-token exact match (extra diagnostic)

Hyperparameters follow the official ICLL config
(ICLL/configs/experiment/dfa/mm.yaml: n_layer=2, n_embd=128, n_head=8, batch 32,
lr=5e-4, wd=0.1) with lr/wd scaled down here to 1e-4/0.01 (documented in the
report).  Deviations
from the paper (documented in the report): max-epoch cap with early stopping
on validation loss instead of the full 200-epoch grid.

Example:
  python -m src.train_icl --method softmax --num-examples 1000
  python -m src.train_icl --method epanechnikov --num-examples 5000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from .icl_dfa import ICLConfig, ICLDFADataset, build_icl_dfa, dfa_accuracy, make_loader, tvd_metric
from .model import ModelConfig, StackAssoMem


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="softmax")
    ap.add_argument("--num-examples", type=int, default=1000, help="1k..40k (paper Table 1)")
    ap.add_argument("--num-test-examples", type=int, default=500, help="half test / half val")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=12, help="early stop after N epochs w/o val improvement")
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--pmem-size", type=int, default=448)
    ap.add_argument("--block-size", type=int, default=1000)
    ap.add_argument("--out-dir", default="results/icl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, vocab, device, ac):
    model.eval()
    losses, accs, tvds, exacts = [], [], [], []
    for X, Y, dfas in loader:
        X, Y = X.to(device), Y.to(device)
        with ac:
            logits, loss = model(X, Y)
        losses.append(float(loss.item()))
        accs.append(dfa_accuracy(logits, X, dfas, vocab))
        tvds.append(tvd_metric(logits, X, dfas, vocab))
        preds = logits.argmax(-1)
        mask = Y != -100
        exacts.append(float((preds[mask] == Y[mask]).float().mean().item()))
    model.train()
    return {
        "loss": float(np.mean(losses)),
        "acc": float(np.mean(accs)),
        "tvd": float(np.mean(tvds)),
        "next_token_acc": float(np.mean(exacts)),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = os.path.join(args.out_dir, args.method, f"n{args.num_examples}", f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")

    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    print(f"== ICL/RegBench  method={args.method}  n={args.num_examples}  seed={args.seed} ==")
    t0 = time.time()
    train_ds, val_ds, test_ds, vocab = build_icl_dfa(
        ICLConfig(num_examples=args.num_examples, num_test_examples=args.num_test_examples, seed=args.seed)
    )
    print(f"data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} vocab={vocab.vocab_size}")
    log(json.dumps(vars(args)))

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)

    cfg = ModelConfig(
        vocab_size=vocab.vocab_size,
        method=args.method,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        pmem_size=args.pmem_size,
        pad_id=vocab.get_id(vocab.noop),
        loss_on="all",
    )
    model = StackAssoMem(cfg).to(args.device)
    print(f"model: {model.num_params() / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    ac = torch.amp.autocast(args.device, dtype=torch.bfloat16) if args.dtype == "bfloat16" else torch.amp.autocast(args.device, enabled=False)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for epoch in tqdm(range(1, args.max_epochs + 1), desc=f"icl {args.method} n={args.num_examples}", unit="ep"):
        model.train()
        for X, Y, _ in train_loader:
            X, Y = X.to(args.device, non_blocking=True), Y.to(args.device, non_blocking=True)
            with ac:
                _, loss = model(X, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val = evaluate(model, val_loader, vocab, args.device, ac)
        history.append({"epoch": epoch, **val})
        tqdm.write(f"epoch {epoch:3d}  val_loss={val['loss']:.4f}  val_acc={val['acc']:.3f}  val_tvd={val['tvd']:.1f}")
        log(f"epoch={epoch} {val}")

        if val["loss"] < best_val - 1e-4:
            best_val = val["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch} (best val_loss={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, vocab, args.device, ac)
    print(f"TEST (best val epoch):  acc={test['acc']:.3f}  tvd={test['tvd']:.1f}  "
          f"next_token={test['next_token_acc']:.3f}  loss={test['loss']:.4f}")

    metrics = {
        "args": vars(args),
        "params": model.num_params(),
        "wall_time_s": time.time() - t0,
        "best_val_loss": best_val,
        "test": test,
        "history": history,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"done in {metrics['wall_time_s'] / 60:.1f} min -> {run_dir}/metrics.json")


if __name__ == "__main__":
    main()
