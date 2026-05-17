#!/bin/bash
# Ultra-fast 20-step tests for step time measurement
set -euo pipefail
cd ~/projects/parameter-golf
source .venv/bin/activate

R=~/projects/parameter-golf/research/quick_results.txt
echo "=== QUICK TESTS (20 steps) ===" > "$R"
echo "Started: $(date)" >> "$R"

run() {
    local name="$1"; shift
    echo "" >> "$R"
    echo "--- $name ---" >> "$R"
    
    # Kill GPU hogs
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null || true)
        if echo "$cmd" | grep -q "python"; then kill -9 "$pid" 2>/dev/null || true; fi
    done
    sleep 2
    
    export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export SEED=42 ITERATIONS=20 MAX_WALLCLOCK_SECONDS=120 WARMUP_STEPS=2
    export DATA_PATH=./data/datasets/fineweb10B_sp1024
    export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
    export TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=5
    export SMEAR_GATE_ENABLED=0 CASEOPS_ENABLED=0 ASYMLOGIT_ENABLED=0 EMA_ENABLED=0 BIGRAM_HASH_ENABLED=0
    for arg in "$@"; do export "$arg"; done
    
    echo "Start: $(date)" >> "$R"
    python3 -u train_gpt.py >> "$R" 2>&1
    echo "End: $(date)" >> "$R"
    sync
}

run "BASELINE"
run "SmearGate" "SMEAR_GATE_ENABLED=1"
run "CaseOps+AsymLogit" "CASEOPS_ENABLED=1" "ASYMLOGIT_ENABLED=1"
run "EMA" "EMA_ENABLED=1"
run "BigramHash" "BIGRAM_HASH_ENABLED=1"
run "ALL" "SMEAR_GATE_ENABLED=1" "CASEOPS_ENABLED=1" "ASYMLOGIT_ENABLED=1" "EMA_ENABLED=1" "BIGRAM_HASH_ENABLED=1"

echo "" >> "$R"
echo "=== ALL DONE ===" >> "$R"
echo "Finished: $(date)" >> "$R"
sync
