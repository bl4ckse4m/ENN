#!/usr/bin/env bash
# Phase 3 — RegBench in-context learning (paper Table 1).
# Run manually; tqdm shows epoch progress, results land in results/icl/<method>/n<N>/seed<S>/metrics.json
#
# Usage:
#   bash scripts/run_phase3_icl.sh            # preset "full"  (~4-6 GPU-h)
#   bash scripts/run_phase3_icl.sh quick      # preset "quick" (~1.5 GPU-h)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET=${1:-full}

METHODS=(softmax epanechnikov biweight triweight top16_softmax top16_uniform norm_relu relumax)

if [[ "$PRESET" == "quick" ]]; then
  SIZES=(1000 5000)
  MAXEPOCHS=60
else
  SIZES=(1000 5000 20000)   # paper Table 1: 1k..40k; 20k is our scaled budget
  MAXEPOCHS=200
fi

for METHOD in "${METHODS[@]}"; do
  for N in "${SIZES[@]}"; do
    echo "=================================================================="
    echo " ICL method=$METHOD num_examples=$N"
    echo "=================================================================="
    $PY -u -m src.train_icl \
      --method "$METHOD" --num-examples "$N" --seed 1337 \
      --max-epochs "$MAXEPOCHS" --patience 12 \
      --out-dir results/icl
  done
done

echo "all done; aggregate with:  $PY -u -m src.analyze --phase icl"
