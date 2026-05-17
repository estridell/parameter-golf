# CaseOps Pipeline Run Notes

**Date:** 2026-05-18
**Machine:** desktoparch (RTX 2070, 16 cores, 31GB RAM)

## Pipeline Script Issues (`run_sota_pipeline.sh`)

### 1. Wrong argument names in step 2
The script passes `--sp-model`, `--out-dir`, `--out-tok-dir` but `prepare_caseops_data.py`
accepts `--sp`, `--out`, `--val-docs`. Running step 2 as-is would fail with argparse error.

**Fix:** Change the step 2 command to:
```bash
python3 records/.../prepare_caseops_data.py \
    --docs data/docs_selected.jsonl \
    --sp data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    --out data
```

### 2. Default caseops path mismatch in train_gpt.py
`train_gpt.py` line 393-400 defines the default caseops data path as:
`data_dir/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`

This has a double `datasets/datasets/` nesting. The prep script writes to:
`data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/`

**Fix:** Use explicit env vars in step 4:
`DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`
`TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`

### 3. SHARD_TOKENS
Each shard is ~19MB (header 1024 bytes + 10M uint16 tokens = ~20MB).
`SHARD_TOKENS = 10_000_000` (10M tokens per shard).

For 15.37M docs, expect ~1,450 train shards + 1-2 val shards.

### 4. Performance
- Rate: ~3.5 shards/min (single-threaded Python, 100% CPU on 1 core)
- Total time estimate: ~7 hours for full tokenization
- Process is CPU-bound (tokenization + SentencePiece encoding)

## Corrected Step 4 Validation Command
```bash
CASEOPS_ENABLED=1 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
TRAIN_BATCH_TOKENS=65536 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=600 \
GRADIENT_CHECKPOINT_ENABLED=1 \
TTT_ENABLED=0 \
SEED=42 \
python3 -u train_gpt.py
```
