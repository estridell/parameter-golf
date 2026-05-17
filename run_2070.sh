#!/bin/bash
# Quick run script for RTX 2070 (sm_75)
# Usage: ./run_2070.sh [wallclock_seconds] [seed]
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

WALLCLOCK=${1:-240}
SEED=${2:-42}

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SEED=$SEED
export MAX_WALLCLOCK_SECONDS=$WALLCLOCK
export WARMUP_STEPS=0
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export TRAIN_BATCH_TOKENS=65536
export VAL_LOSS_EVERY=0

echo "[rtx2070] seed=$SEED wallclock=${WALLCLOCK}s batch=65536 expandable_segments warmup=0"
python3 -u train_gpt.py
