#!/bin/bash
# for LLaDA-8B-Instruct and Dream-v0-Instruct-7B. Each ReCon row runs on the drafts stored by its draft row.
#   bash run.sh                    # all at once
#   bash run.sh dream              # one model
#   bash run.sh llada gsm8k        # one model, one dataset
#   LIMIT=5 bash run.sh llada math # smoke test on the first 5 examples

set -euo pipefail
cd "$(dirname "$0")"
source ./venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
EXTRA=(); [ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")

for model in ${1:-llada dream}; do
  for ds in ${2:-gsm8k arcc truthfulqa math}; do
    for draft in semi_ar abd; do
      python run_eval.py --model "$model" --dataset "$ds" --method "$draft" "${EXTRA[@]}"
      python run_eval.py --model "$model" --dataset "$ds" --method "${draft}_recon" "${EXTRA[@]}"
    done
    python run_eval.py --model "$model" --dataset "$ds" --method abd_recon --N 6 "${EXTRA[@]}"
  done
done
python run_eval.py --table
