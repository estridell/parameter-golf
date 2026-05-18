# CaseOps Data Validation + 200-Step Benchmark

**Date:** 2026-05-18
**Machine:** desktoparch (RTX 2070, 8GB VRAM)
**Branch:** rtx2070

## Summary

CaseOps data loads and trains correctly. The SOTA model (36M params) requires significant
settings adjustments for the RTX 2070 — default settings OOM immediately.

## Phase 1: Data Validation PASS

- **Data path:** `fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` — CORRECT
- **Tokenizer:** `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — loaded OK
- **Train shards:** 1502 files
- **Val tokens:** 9,662,464
- **Training starts:** YES (with adjusted settings)
- **Loss decreases:** YES (9.01 -> 5.97 by step 50)

### Required 2070 Adaptations (broken defaults)

The default CaseOps path in train_gpt.py has a double `datasets/datasets/` nesting that
doesn't match where `prepare_caseops_data.py` writes output. Must override with:

    DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
    TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model

The SOTA model (36M params with loop architecture) OOMs at default settings on 8GB.
Working settings for 2070:

    TRAIN_BATCH_TOKENS=4096    (default 65536 OOM)
    TRAIN_SEQ_LEN=512          (default 2048 OOM)
    EVAL_SEQ_LEN=512
    GRADIENT_CHECKPOINT_ENABLED=1
    TTT_ENABLED=0

## Phase 2: 200-Step Benchmark

### Loss Trajectory

| Step | Loss    | Notes |
|------|---------|-------|
| 1    | 9.0155  | Random init (ln 8192 = 9.01) |
| 10   | 9.1095  | Initial spike from zero warmup |
| 50   | 5.9736  | Rapid descent |
| 100  | 5.5701  | Pre-looping steady state |
| 200  | 4.7427  | Post-looping convergence |

Initial loss spike (steps 2-5: 15.4->16.3->14.5->13.4) is gradient noise from zero warmup
with small batch. Expected behavior, recovers quickly.

### Performance

| Metric | Value |
|--------|-------|
| Model params | 35,944,615 (36M) |
| Step time (pre-loop, ~step 1-95) | ~0.65s/step |
| Step time (post-loop, ~step 95-200) | ~0.78s/step |
| Throughput (pre-loop) | ~6,200 tok/s |
| Throughput (post-loop) | ~5,200 tok/s |
| Peak VRAM | 1,932 MiB / 8,192 MiB (24%) |
| Effective batch | 4,096 tokens x 8 grad accum = 32,768 tokens |

### Layer Looping

Looping activated at step 95 (enable_looping_at=0.35):

    encoder: [0, 1, 2, 3, 4, 5, 3, 4]
    decoder: [5, 3, 4, 5, 6, 7, 8, 9, 10]

This reuses layers 3-5 in the encoder, adding ~20% compute overhead but enabling
deeper processing. Loss continued to improve after activation (5.57 -> 4.74).

## Known Bug: Validation Crash

`compiled_forward_logits` (line 3376) is a lambda wrapping `compiled_model()`. Since
RTX 2070 patches set `compiled_model = base_model`, this calls `GPT.forward()` which
requires `target_ids`. But `eval_val` calls it with only `(x, cu_seqlens, max_seqlen)`.

**Trigger:** Any run that hits MAX_WALLCLOCK_SECONDS will crash at final validation.
**Workaround:** Set iterations low enough to complete before wallclock, or set
MAX_WALLCLOCK_SECONDS very high. VAL_LOSS_EVERY=0 does not help — last_step always
triggers validation.
**Fix needed:** Line 3376 should use `compiled_model.forward_logits` instead of
`compiled_model`.

## Config Used

    CASEOPS_ENABLED=1
    SMEAR_GATE_ENABLED=1
    LQER_ASYM_ENABLED=1
    GRADIENT_CHECKPOINT_ENABLED=1
    TTT_ENABLED=0
    SEED=42
    TRAIN_BATCH_TOKENS=4096
    TRAIN_SEQ_LEN=512
    EVAL_SEQ_LEN=512
    VAL_LOSS_EVERY=0
    TRAIN_LOG_EVERY=10
