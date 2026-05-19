# SOTA 30-Minute Baseline — RTX 2070

**Date:** 2026-05-19
**Machine:** RTX 2070 (sm_75), 8GB VRAM, Arch Linux
**Branch:** rtx2070
**Script:** train_gpt_sota.py (35.9M params, ported from #5 leaderboard entry)

## Configuration

```
FA3_ENABLED=0 TRITON_ENABLED=0 COMPRESSOR=brotli
TRAIN_BATCH_TOKENS=8192 VAL_BATCH_TOKENS=524288 (default)
GRADIENT_CHECKPOINT_ENABLED=1
CASEOPS_ENABLED=1 SMEAR_GATE_ENABLED=1 LQER_ENABLED=1
LQER_ASYM_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
MAX_WALLCLOCK_SECONDS=1800 GPTQ_RESERVE_SECONDS=10
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=42
```

## Training Results

| Metric | Value |
|---|---|
| Steps completed | 648 / 20000 |
| Training time | 29.9 min (1792s) |
| Speed | 3132 tok/s (at step 500) |
| Initial tok/s | 3899 tok/s (step 1) |
| Final train_loss | 4.2610 (step 500, last logged) |
| Layer loop | encoder:[0,1,2,3,4,5,3,4] decoder:[5,3,4,5,6,7,8,9,10] |
| Warmup time | ~9 min (cu_seqlens + loop warmup) |
| train_log_every | 500 (default) |

## Validation Results

| Phase | val_loss | val_bpb |
|---|---|---|
| Step 0 (untrained) | 9.0066 | 4.1929 |
| Step 648 (pre-EMA) | 3.7781 | 1.7588 |
| Step 648 (post-EMA) | 4.1124 | 1.9145 |
| Post-GPTQ quantized | 11.538 | 5.3713 |

**EMA impact:** EMA degrades val_bpb from 1.7588 to 1.9145 (+0.156). Known issue —
EMA decay=0.9965 may be too aggressive for only 648 steps.

**GPTQ impact:** Massive degradation from 1.9145 to 5.3713. At only 648 training steps,
the model hasn't converged enough for int6 quantization to preserve quality. The #5
leaderboard entry trains for much longer before quantizing.

## Artifacts

| File | Size |
|---|---|
| final_model.pt | 135,417,533 bytes (129 MB) |
| final_model.int6.ptz | 132,078 bytes (129 KB) |
| Code (uncompressed) | 162,912 bytes |
| Code (compressed, brotli) | 40,637 bytes |
| Total submission | 172,715 bytes |

## Memory

| Metric | Value |
|---|---|
| Peak allocated | 5,767 MiB |
| Peak reserved | 7,014 MiB |
| GPU total | 8,192 MiB |

## Issues Found

1. **OOM with default VAL_BATCH_TOKENS on first attempt.** After 500 training steps,
   memory fragmentation caused final validation to OOM (needed 2 GiB contiguous, only
   584 MiB free). Fixed by adding PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

2. **GPTQ degradation is severe at 648 steps.** The model needs more training before
   quantization. For a fair comparison with the #5 leaderboard entry, need to understand
   their training duration.

3. **TTT crash: "Invalid backend" in SDPA.** After GPTQ quantization, the test-time
   training phase crashes because SDPA backend configuration becomes invalid for the
   quantized model. This is post-submission and doesn't affect the artifact.

4. **Slow warmup.** ~9 min of warmup (cu_seqlens buckets + encoder/decoder loop warmup)
   eats into the 30-min budget. Effective training time is only ~21 min.

5. **train_log_every=500 too sparse.** Only logged at steps 1-5 and 500. For a 648-step
   run, we miss steps 6-499 and 501-648.

## Comparison with Smoke Test (300s)

| Metric | Smoke (300s) | Baseline (1800s) |
|---|---|---|
| Steps | 5 | 648 |
| Speed | 3565 tok/s | 3132 tok/s |
| train_loss | N/A | 4.261 |
| val_bpb (pre-EMA) | N/A | 1.7588 |
| GPTQ artifact | 150 KB | 132 KB |
| Total submission | N/A | 172 KB |

Speed dropped from 3565 to 3132 tok/s (-12%). The layer_loop overhead compounds
over longer runs.

## Next Steps

1. Need longer training (2-4 hours) for GPTQ to produce useful artifacts
2. Consider disabling EMA for short runs (or tuning decay)
3. Fix TTT SDPA backend issue for post-GPTQ eval
4. Reduce train_log_every for better monitoring
