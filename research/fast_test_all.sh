#!/bin/bash
# Fast 50-step tests for all techniques
set -euo pipefail
cd ~/projects/parameter-golf
source .venv/bin/activate

RESULTS=~/projects/parameter-golf/research/fast_results.txt
echo "=== FAST TECHNIQUE TESTS (50 steps) ===" > "$RESULTS"
echo "Started: $(date)" >> "$RESULTS"

kill_gpu() {
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null || true)
        if echo "$cmd" | grep -q "python"; then
            kill -9 "$pid" 2>/dev/null || true
            echo "Killed GPU hog $pid" >> "$RESULTS"
        fi
    done
    sleep 3
}

run_test() {
    local name="$1"; shift
    echo "" >> "$RESULTS"
    echo "--- $name ---" >> "$RESULTS"
    echo "Start: $(date)" >> "$RESULTS"
    sync
    kill_gpu
    
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export SEED=42
    export ITERATIONS=50
    export MAX_WALLCLOCK_SECONDS=300
    export WARMUP_STEPS=3
    export DATA_PATH=./data/datasets/fineweb10B_sp1024
    export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
    export TRAIN_BATCH_TOKENS=65536
    export VAL_LOSS_EVERY=0
    export TRAIN_LOG_EVERY=10
    export SMEAR_GATE_ENABLED=0
    export CASEOPS_ENABLED=0
    export ASYMLOGIT_ENABLED=0
    export EMA_ENABLED=0
    export BIGRAM_HASH_ENABLED=0
    
    for arg in "$@"; do export "$arg"; done
    python3 -u train_gpt.py >> "$RESULTS" 2>&1
    sync
    echo "End: $(date)" >> "$RESULTS"
    echo "DONE: $name"
}

run_test "BASELINE"
run_test "SmearGate" "SMEAR_GATE_ENABLED=1"
run_test "CaseOps+AsymLogit" "CASEOPS_ENABLED=1" "ASYMLOGIT_ENABLED=1"
run_test "EMA" "EMA_ENABLED=1"
run_test "BigramHash" "BIGRAM_HASH_ENABLED=1"
run_test "ALL" "SMEAR_GATE_ENABLED=1" "CASEOPS_ENABLED=1" "ASYMLOGIT_ENABLED=1" "EMA_ENABLED=1" "BIGRAM_HASH_ENABLED=1"

echo "" >> "$RESULTS"
echo "=== ALL TESTS DONE ===" >> "$RESULTS"
echo "Finished: $(date)" >> "$RESULTS"
sync
