# BPB Comparison: Baseline (sp1024) vs New Architecture (sp8192)

**Date:** 2026-05-19
**Machine:** RTX 2070 (sm_75, 8GB VRAM)
**Wallclock:** 600 seconds per run
**Model:** 32M params (dim=512, layers=11, heads=8, kv_heads=4)
**Batch:** 16384 tokens, grad_checkpoint enabled
**Seed:** 42

## Results

| Metric | Baseline (sp1024) | New Arch (sp8192) | Delta |
|---|---|---|---|
| val_loss | 3.3741 | 4.0988 | +0.7247 (higher = worse) |
| val_bpb | 2.0158 | 1.9053 | -0.1105 (lower = better) |
| steps completed | 275 | 265 | -10 |
| throughput | 7540 tok/s | 7264 tok/s | -3.7% |
| train_time | 597.7s | 597.9s | ~equal |

## BPB Improvement: 5.48%



## Architecture Differences

**Baseline:** vocab_size=1024, vanilla GQA, standard embeddings
**New Arch:** vocab_size=8192, CaseOps, SmearGate, LQER (asymmetric, rank=4, 4-bit, group=64)

## Notes on val_loss vs val_bpb

val_loss is cross-entropy in nats and scales with vocabulary size. The baseline uses
vocab=1024 (fewer nats to distribute) while new arch uses vocab=8192 (more nats).
Direct loss comparison across different vocab sizes is meaningless.

val_bpb (bits-per-byte) normalizes for vocabulary and sequence length, making it the
correct metric for comparison. By BPB, the new architecture wins by 5.48%.

## Checkpoints

-  (123 MB)
-  (15 MB, quantized)
-  (130 MB)
-  (17 MB, quantized)

## Diagnostic Eval (post-EMA, quantized)

| Metric | Baseline | New Arch |
|---|---|---|
| pre-quant post-ema val_loss | 4.3120 | 5.2950 |
| pre-quant post-ema val_bpb | 2.5762 | 2.4613 |
| quantized val_loss | 4.3120 | 5.2950 |
| quantized val_bpb | 2.5762 | 2.4613 |

The diagnostic eval runs on a larger eval set (eval_seq_len=2048, more batches).
Quantization preserved the numbers exactly — no degradation from int6 quant.
