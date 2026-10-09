#!/usr/bin/env bash
# Phase 4 — language modeling on BabiStories (paper Section 4.1 / Figure 2), scaled.
# Prepare data first (CPU, ~15 min):  ../.venv/bin/python -u -m src.prepare_babistories
#
# Usage:
#   bash scripts/run_phase4_lm.sh            # preset "full"  (~4-6 GPU-h)
#   bash scripts/run_phase4_lm.sh quick      # preset "quick" (~1.5 GPU-h)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET=${1:-full}

# paper Figure 2 methods (official names: epanechnikov == sparsemax)
METHODS=(softmax top32_softmax norm_relu epanechnikov relumax biweight triweight)

if [[ "$PRESET" == "quick" ]]; then
  ITERS=8000
  METHODS=(softmax epanechnikov biweight)
else
  ITERS=25000
fi

for METHOD in "${METHODS[@]}"; do
  echo "=================================================================="
  echo " LM method=$METHOD iters=$ITERS"
  echo "=================================================================="
  $PY -u -m src.train_lm \
    --method "$METHOD" --n-layer 1 --seed 1337 \
    --batch-size 16 --max-iters "$ITERS" \
    --out-dir results/lm
done

echo "all done; aggregate with:  $PY -u -m src.analyze --phase lm"
