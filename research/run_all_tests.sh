#!/bin/bash
# Run all technique tests sequentially - no buffering
set -euo pipefail
cd ~/projects/parameter-golf
source .venv/bin/activate

RESULTS=~/projects/parameter-golf/research/test_results.txt
echo "=== TECHNIQUE TEST SUITE ===" > "$RESULTS"
echo "Started: $(date)" >> "$RESULTS"
sync

kill_gpu_hogs() {
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null || true)
        if echo "$cmd" | grep -q "python"; then
            kill -9 "$pid" 2>/dev/null || true
            echo "Killed stale GPU process $pid"
        fi
    done
    sleep 3
}

run_test() {
    local name="$1"
    shift
    echo "" >> "$RESULTS"
    echo "--- $name ---" >> "$RESULTS"
    echo "Start: $(date)" >> "$RESULTS"
    sync
    
    kill_gpu_hogs
    
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
    export SMEAR_GATE_ENABLED=0
    export CASEOPS_ENABLED=0
    export ASYMLOGIT_ENABLED=0
    export EMA_ENABLED=0
    export BIGRAM_HASH_ENABLED=0
    
    for arg in "$@"; do export "$arg"; done
    
    # Use tee for real-time output and log, with unbuffered python
    python3 -u train_gpt.py 2>&1 | tee -a "$RESULTS"
    sync
    echo "End: $(date)" >> "$RESULTS"
    sync
    echo "=== Completed: $name ==="
}

# 1. BASELINE
run_test "BASELINE (all techniques OFF)"

# 2. SmearGate only
run_test "SmearGate only" "SMEAR_GATE_ENABLED=1"

# 3. CaseOps + AsymLogit only
run_test "CaseOps+AsymLogit only" "CASEOPS_ENABLED=1" "ASYMLOGIT_ENABLED=1"

# 4. EMA only
run_test "EMA only" "EMA_ENABLED=1"

# 5. BigramHash only
run_test "BigramHash only" "BIGRAM_HASH_ENABLED=1"

# 6. All combined
run_test "ALL techniques" "SMEAR_GATE_ENABLED=1" "CASEOPS_ENABLED=1" "ASYMLOGIT_ENABLED=1" "EMA_ENABLED=1" "BIGRAM_HASH_ENABLED=1"

echo "" >> "$RESULTS"
echo "=== ALL TESTS COMPLETE ===" >> "$RESULTS"
echo "Finished: $(date)" >> "$RESULTS"
sync
echo "ALL DONE"
