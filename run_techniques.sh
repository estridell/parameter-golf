#!/bin/bash
set -euo pipefail
cd ~/projects/parameter-golf
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SEED=42
export MAX_WALLCLOCK_SECONDS=600
export WARMUP_STEPS=0
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export TRAIN_BATCH_TOKENS=65536
export VAL_LOSS_EVERY=0
export SMEAR_GATE_ENABLED=1
export CASEOPS_ENABLED=1
export ASYMLOGIT_ENABLED=1
export EMA_ENABLED=1

echo "[TECHNIQUES] Starting at $(date)"
python3 -u train_gpt.py
echo "[TECHNIQUES] Done at $(date)"
