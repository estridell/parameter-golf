# CaseOps vs Baseline Benchmark — 500 Steps

## Date: 2026-05-18
## Machine: RTX 2070 (8GB VRAM, sm_75)

## Configuration

| Parameter | Value |
|---|---|
| MODEL_DIM | 256 |
| NUM_LAYERS | 6 |
| TRAIN_BATCH_TOKENS | 16384 (65536 OOMs due to cu_seqlens attention) |
| ITERATIONS | 500 |
| SEED | 42 |
| WARMUP_STEPS | 5 |
| TRAIN_LOG_EVERY | 10 |
| GRADIENT_CHECKPOINT_ENABLED | 0 |
| TTT_ENABLED | 0 |
| EVAL_SEQ_LEN | 1024 |
| VAL_BATCH_TOKENS | 16384 |

## Model Sizes

- **Baseline** (VOCAB_SIZE=1024, sp1024 tokenizer): **4,596,820 params (4.6M)**
- **CaseOps** (VOCAB_SIZE=8192, sp8192 tokenizer): **6,431,841 params (6.4M)**

**NOTE:** The models are NOT the same size. Different tokenizers (sp1024 vs sp8192) produce different vocab sizes, which changes the embedding layer by ~3.5M params. This is NOT a pure apples-to-apples comparison of data — the model architecture differs due to tokenizer mismatch.

## Training Loss Comparison

| Step | Baseline | CaseOps |
|------|----------|---------|
| 10   | 5.7288   | 6.7870  |
| 50   | 4.6189   | 5.0133  |
| 100  | 4.4893   | 5.0324  |
| 200  | 3.9247   | 4.4167  |
| 300  | 3.5011   | 4.7581  |
| 400  | 3.7712   | 4.1557  |
| 500  | 3.3962   | 4.2588  |

## Validation (CaseOps only)

Baseline validation crashed due to a pre-existing bug: `val_data.val_bytes` is None when `caseops_enabled=False` because the sp1024 dataset directory has no `*_bytes_*.bin` sidecar files. The `eval_val` function unconditionally accesses `val_bytes` (line 2821 of train_gpt.py). This bug was partially fixed in commit 5af1cb2 (TTT path only), but the regular eval_val path remains broken for non-CaseOps runs.

| Step | val_loss | val_bpb |
|------|----------|---------|
| 0    | 9.0025   | 4.1900  |
| 100  | 4.8287   | 2.2474  |
| 200  | 4.5092   | 2.0987  |
| 300  | 4.5800   | 2.1317  |
| 400  | 4.1609   | 1.9366  |
| 500  | 3.9990   | 1.8612  |

## Performance

| Metric | Baseline | CaseOps |
|--------|----------|---------|
| Wallclock (500 steps) | 7.6 min | 8.1 min |
| Tok/s (pre-loop, step 300) | ~24,600 | ~23,100 |
| Tok/s (post-loop, step 500) | ~18,000 | ~16,800 |
| Step time avg (last 100 steps) | ~0.91 ms/step | ~0.97 ms/step |
| Peak VRAM (training) | ~3,091 MiB | 2,598 MiB |

Layer looping activated at step ~294-314 for both runs (enable_looping_at=0.35). Post-looping throughput drops ~25% due to parallel encoder-decoder path.

## Analysis

### Why CaseOps has HIGHER training loss

CaseOps training loss (4.26) is higher than baseline (3.40) at step 500. This is expected because:

1. **Different vocab sizes (8192 vs 1024):** The loss is cross-entropy over the vocabulary. With 8× more classes, the initial loss is log(8192) ≈ 9.01 vs log(1024) ≈ 6.93. The *relative* improvement is actually comparable:
   - Baseline: 6.93 → 3.40 = 51% reduction
   - CaseOps: 9.01 → 4.26 = 53% reduction

2. **Different model sizes (6.4M vs 4.6M):** CaseOps model is 40% larger due to bigger embedding table, which means more capacity but also more parameters to train.

3. **Training loss ≠ model quality:** The raw cross-entropy loss over different vocabulary sizes cannot be directly compared. The validation BPB (bits-per-byte) is the correct metric for comparing models with different tokenizers.

### The BPB question

CaseOps val_bpb = 1.86 at step 500. We have no baseline val_bpb due to the val_bytes bug. To properly compare, we need to either:
- Fix the eval_val bug for non-CaseOps runs
- Compute BPB manually for the baseline using the SentencePiece LUT approach (base_bytes_lut, has_leading_space_lut)

### OOM Issues Discovered

1. **TRAIN_BATCH_TOKENS=65536 OOMs** even with 4.6M model. The cu_seqlens packed attention code path uses `F.scaled_dot_product_attention(q, k, v, is_causal=True)` which creates a full causal attention matrix over the ENTIRE micro-batch (not per-document). With 8 heads × micro_batch² × 4 bytes per layer, and 6 layers saving for backward, the memory exceeds 8GB.

2. **Default VAL_BATCH_TOKENS=524288 causes post-training OOM.** The eval_val function uses `local_batch_tokens = val_batch_tokens / (world_size * grad_accum_steps)` = 65536 tokens per batch, creating a 65536² attention matrix.

3. **The attention code doesn't actually use cu_seqlens.** Lines 1036-1041 in train_gpt.py pass cu_seqlens to the attention forward but then use plain `F.scaled_dot_product_attention(q, k, v, is_causal=True)` which ignores document boundaries. This means all tokens in a micro-batch attend to each other (with causal masking), which is incorrect for packed multi-document sequences.

### Known Bug: eval_val val_bytes for non-CaseOps

The `eval_val` function (line 2821) accesses `val_data.val_bytes[...]` unconditionally. When `caseops_enabled=False`, `val_bytes` is None, causing a TypeError. The fix in commit 5af1cb2 only patched the TTT path, not the regular eval_val path. This prevents any non-CaseOps run from computing validation metrics.

## Recommendations

1. **Fix eval_val val_bytes bug:** Add `if val_data.val_bytes is not None:` guard around the sidecar access at line 2821.
2. **Investigate cu_seqlens attention:** The packed attention doesn't respect document boundaries. This affects both training quality (cross-document attention leak) and memory usage (full-batch attention matrix).
3. **Reduce VAL_BATCH_TOKENS default:** 524288 is way too large for 8GB GPUs. Consider defaulting to train_batch_tokens.
4. **For proper comparison:** Need same vocab_size for both runs, or use BPB as the comparison metric.
