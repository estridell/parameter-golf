#!/bin/bash
# RTX 2070: Download sp8192 data, prepare CaseOps, then run validation
set -e
cd ~/projects/parameter-golf
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== Step 1: Download docs_selected.jsonl ==="
if [ ! -f data/docs_selected.jsonl ]; then
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download('willdepueoai/parameter-golf', 'datasets/docs_selected.jsonl', repo_type='dataset')
shutil.copy(p, 'data/docs_selected.jsonl')
print(f'Downloaded: {p}')
"
else
    echo "Already exists, skipping"
fi

echo "=== Step 2: Prepare CaseOps data ==="
if [ ! -d data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved ]; then
    python3 records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/prepare_caseops_data.py \
        --docs data/docs_selected.jsonl \
        --sp-model data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
        --out-dir data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
        --out-tok-dir data/tokenizers \
        2>&1
else
    echo "Already exists, skipping"
fi

echo "=== Step 3: Verify data ==="
ls -la data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/ | head -5
ls data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin | wc -l
echo "val shards found"

echo "=== Step 4: Run validation (sp8192 CaseOps) ==="
CASEOPS_ENABLED=1 \
TRAIN_BATCH_TOKENS=16384 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=600 \
GRADIENT_CHECKPOINT_ENABLED=1 \
TTT_ENABLED=0 \
SEED=42 \
python3 -u train_gpt.py 2>&1 | tee /tmp/sota_validation.log

echo "=== DONE ==="
