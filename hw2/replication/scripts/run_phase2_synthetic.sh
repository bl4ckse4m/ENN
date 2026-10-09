#!/usr/bin/env bash
# Phase 2 — length generalization (paper Table 2).
# Run manually; each run logs with tqdm and writes results/<task>/<method>/seed<N>/metrics.json.
#
# Usage:
#   bash scripts/run_phase2_synthetic.sh            # preset "full"  (~11 GPU-h, 1 seed)
#   bash scripts/run_phase2_synthetic.sh quick      # preset "quick" (~4 GPU-h, 1 seed)
#   bash scripts/run_phase2_synthetic.sh full 1337 1338 1339   # paper's 3-seed protocol (x3 time)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET=${1:-full}
shift || true
SEEDS=${*:-"1337"}

if [[ "$PRESET" == "quick" ]]; then
  EXAMPLES=1000000      # ~1M examples per run
else
  EXAMPLES=5000000      # paper: 50M/30M/40M; 5M is our scaled budget
fi

# Table 2 methods (official names: epanechnikov == sparsemax)
METHODS=(
  softmax
  top32_softmax
  top32_uniform
  norm_relu
  relumax
  epanechnikov
  biweight
  triweight
)
# optional extra: relumax2 relumax3 (paper's 2-/3-ReLUmax)

for TASK in mqmtar reverse sort; do
  for METHOD in "${METHODS[@]}"; do
    for SEED in $SEEDS; do
      echo "=================================================================="
      echo " task=$TASK method=$METHOD seed=$SEED examples=$EXAMPLES"
      echo "=================================================================="
      $PY -u -m src.train_synthetic \
        --task "$TASK" --method "$METHOD" --seed "$SEED" \
        --train-examples "$EXAMPLES" \
        --out-dir results/synthetic
    done
  done
done

echo "all done; aggregate with:  $PY -u -m src.analyze --phase synthetic"
