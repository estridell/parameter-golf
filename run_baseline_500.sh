#!/bin/bash
set -e
cd ~/projects/parameter-golf
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SEED=42
export TRAIN_BATCH_TOKENS=65536
export TRAIN_SEQ_LEN=1024
export EVAL_SEQ_LEN=1024
export MAX_STEPS=500
export WARMUP_STEPS=5
export VAL_LOSS_EVERY=100
export VAL_BATCH_TOKENS=65536
export TRAIN_LOG_EVERY=10
export GRADIENT_CHECKPOINT_ENABLED=0
export TTT_ENABLED=0
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export CASEOPS_ENABLED=0
export SMEAR_GATE_ENABLED=0
export LQER_ASYM_ENABLED=0

python3 -u train_gpt.py
