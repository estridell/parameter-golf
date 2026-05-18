#!/bin/bash
# 2^4 factorial: SmearGate × CaseOps × AsymLogit × EMA
# 300s each, seed=42, training loss only (VAL_LOSS_EVERY=0)
# Then targeted validation+roundtrip for key combos
set -euo pipefail
cd ~/projects/parameter-golf
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export TRAIN_BATCH_TOKENS=16384
export SEED=42
export MAX_WALLCLOCK_SECONDS=300
export WARMUP_STEPS=0
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=5

LOGDIR=./logs/factorial_ema_isolation
mkdir -p "$LOGDIR"

# 16 combos: SG=SmearGate, CO=CaseOps, AL=AsymLogit, EM=EMA
# Label: {on/off}{SG}{CO}{AL}{EM}
declare -a COMBOS=(
    "0000:0:0:0:0"
    "1000:1:0:0:0"
    "0100:0:1:0:0"
    "0010:0:0:1:0"
    "0001:0:0:0:1"
    "1100:1:1:0:0"
    "1010:1:0:1:0"
    "1001:1:0:0:1"
    "0110:0:1:1:0"
    "0101:0:1:0:1"
    "0011:0:0:1:1"
    "1110:1:1:1:0"
    "1101:1:1:0:1"
    "1011:1:0:1:1"
    "0111:0:1:1:1"
    "1111:1:1:1:1"
)

echo "=== FACTORIAL EMA ISOLATION EXPERIMENT ==="
echo "Starting $(date)"
echo "16 combos × 300s = ~80 min training"
echo ""

for combo in "${COMBOS[@]}"; do
    IFS=':' read -r label sg co al em <<< "$combo"
    
    echo "--- Running combo $label (SG=$sg CO=$co AL=$al EM=$em) ---"
    
    SMEAR_GATE_ENABLED=$sg \
    CASEOPS_ENABLED=$co \
    ASYMLOGIT_ENABLED=$al \
    EMA_ENABLED=$em \
    python3 -u train_gpt.py 2>&1 | tee "$LOGDIR/run_${label}.log"
    
    echo "--- Combo $label done ---"
    echo ""
done

echo "=== ALL 16 TRAINING RUNS COMPLETE ==="
echo "Finished $(date)"
echo ""
echo "Now running targeted validation for key combos..."

# Key combos for validation: baseline(0000), EMA-only(0001), all-on(1111), all-but-EMA(1110)
VALIDATE_COMBOS=("0000" "0001" "1110" "1111")

for label in "${VALIDATE_COMBOS[@]}"; do
    IFS=':' read -r _ sg co al em <<< "$(printf '%s\n' "${COMBOS[@]}" | grep "^${label}:")"
    
    echo "--- Validating combo $label (SG=$sg CO=$co AL=$al EM=$em) ---"
    
    SMEAR_GATE_ENABLED=$sg \
    CASEOPS_ENABLED=$co \
    ASYMLOGIT_ENABLED=$al \
    EMA_ENABLED=$em \
    VAL_LOSS_EVERY=1 \
    MAX_WALLCLOCK_SECONDS=300 \
    python3 -u train_gpt.py 2>&1 | tee "$LOGDIR/validate_${label}.log"
    
    echo "--- Validation $label done ---"
    echo ""
done

echo "=== ALL DONE ==="
echo "Finished $(date)"
