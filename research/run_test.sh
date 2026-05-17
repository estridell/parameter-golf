#!/bin/bash
# Standardized 200-step test with step timing
# Usage: ./research/run_test.sh "TEST_NAME" [ENV1=val1 ENV2=val2 ...]
set -euo pipefail
cd ~/projects/parameter-golf
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SEED=42
export ITERATIONS=200
export MAX_WALLCLOCK_SECONDS=900
export WARMUP_STEPS=5
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export TRAIN_BATCH_TOKENS=65536
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=20
# Disable all techniques by default
export SMEAR_GATE_ENABLED=0
export CASEOPS_ENABLED=0
export ASYMLOGIT_ENABLED=0
export EMA_ENABLED=0
export BIGRAM_HASH_ENABLED=0

# Parse test name
TEST_NAME="$1"; shift
# Apply overrides
for arg in "$@"; do export "$arg"; done

echo "=== $TEST_NAME ==="
echo "Start: $(date)"
python3 -u train_gpt.py
echo "End: $(date)"
