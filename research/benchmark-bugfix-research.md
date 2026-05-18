# Benchmark Bug Fixes — Research Notes

**Date:** 2026-05-18
**Branch:** rtx2070
**Machine:** RTX 2070 (8GB VRAM, sm_75)

## Summary

Fixed 3 bugs (val_bytes guard, VAL_BATCH_TOKENS default, cu_seqlens document-boundary attention). The critical Bug 2 fix uses a hybrid approach: block-diagonal causal mask for small sequences (≤4096 tokens, single SDPA call) and per-document loop for large sequences (memory-efficient). Bug 3 (TRAIN_BATCH_TOKENS=65536 OOM) is NOT fully resolved by Bug 2 alone — the OOM is from activation memory, not attention memory.

## Bugs Fixed

### Bug 1: eval_val val_bytes crash for non-CaseOps runs
**Fix:** Added `if val_data.val_bytes is not None:` guard around sidecar access in eval_val.
**Status:** Fixed. Trivially correct — guards against None access.

### Bug 2: cu_seqlens packed attention ignores document boundaries
**Fix:** Hybrid per-document SDPA:
- `n_seg <= 1`: plain causal SDPA (single document)
- `total_len <= 4096`: block-diagonal causal mask (single SDPA call, fast for warmup)
- `total_len > 4096`: per-document loop (memory-efficient for eval)

**Key insight:** The original code did `F.scaled_dot_product_attention(q, k, v, is_causal=True)` which creates a full causal attention matrix over ALL tokens in the micro-batch, ignoring document boundaries. This is:
1. **Incorrect** — tokens in different documents can attend to each other
2. **Memory-wasteful** — full T² attention matrix instead of per-document

**Performance note:** The initial per-document loop approach was extremely slow for the cu_bucket warmup (which creates many tiny segments of 64 tokens). The mask approach solves this — it's a single SDPA call regardless of segment count. The threshold of 4096 tokens keeps the mask under 16MB (4096² × 1 bool).

### Bug 4: VAL_BATCH_TOKENS default too large
**Fix:** Changed default from 524288 to 65536.
**Status:** Fixed.

## Bug 3: TRAIN_BATCH_TOKENS=65536 OOM — NOT a pure attention issue

**Finding:** The task description states "If Bug 2 is fixed (per-document attention), batch=65536 should work." This is **incorrect**. The OOM at batch=65536 is caused by **activation memory**, not attention memory.

**Evidence:**
- With TRAIN_BATCH_TOKENS=65536 and grad_accum_steps=8, each micro-batch is 8192 tokens
- The per-document attention fix reduces attention memory (1024² per document vs 8192² total)
- But the model still needs to store activations for 8192 tokens through 11 layers
- With gradient checkpointing, peak memory for 8192 tokens exceeds 7.6GB available
- TRAIN_BATCH_TOKENS=16384 (micro-batch=2048) works fine with grad checkpointing

**Practical maximum:**
- 32M model, grad_ckpt=1: micro-batch ≤ 2048 → TRAIN_BATCH_TOKENS ≤ 16384
- 32M model, grad_ckpt=0: micro-batch ≤ 1024 → TRAIN_BATCH_TOKENS ≤ 8192

**Recommendation:** Update run_2070.sh to use TRAIN_BATCH_TOKENS=16384 with GRADIENT_CHECKPOINT_ENABLED=1.

## Test Results

### Baseline (sp1024, no CaseOps)
- Config: TRAIN_BATCH_TOKENS=16384, GRADIENT_CHECKPOINT_ENABLED=1, NUM_LOOPS=0, TTT_ENABLED=0
- 10 steps: loss 6.94 → 6.09, 7227 tok/s, peak 3547 MiB
- No OOM, no errors

### CaseOps (sp8192)
- Config: TRAIN_BATCH_TOKENS=16384, CASEOPS_ENABLED=1, VOCAB_SIZE=8192
- 10 steps: loss 9.02 → 7.21, 7010 tok/s, peak 3547 MiB
- Validation passed: val_loss 6.96, val_bpb 3.24
- No OOM, no errors

## Environment Notes

- **GPTQ calibration:** Takes ~10 minutes after training. Use PREQUANT_ONLY=1 for quick tests.
- **TTT compilation:** Default TTT_ENABLED=1 causes slow torch.compile on sm_75. Disable for testing.
- **Layer loop warmup:** Default NUM_LOOPS=2 doubles cu_bucket warmup time. Disable for quick tests.
- **Stale processes:** SSH disconnect leaves GPU processes running. Always kill with `kill -9 PID` before new runs.
- **VAL_BATCH_TOKENS=65536:** Validation processes all 62M val tokens in 3784 batches (~19 min without compile). Use VAL_LOSS_EVERY=0 for smoke tests.
