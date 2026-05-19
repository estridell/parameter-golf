# C21 BPB Validation — Key Findings

**Date:** 2026-05-19
**Task:** t_ef0683a7

## Summary

C21 (GPTQ+AWQ) compression on newarch_sp8192.pt achieves **BPB=2.4946** at **5.28x compression** (137MB → 26MB). The compression *improves* BPB by -0.0055 over the uncompressed baseline (2.5001).

## Why BPB improves after compression

The GPTQ+AWQ combination acts as a regularizer:
1. **AWQ** protects the 1% most important input channels (by activation magnitude) with INT8 precision
2. **GPTQ** uses Hessian information to minimize output error on remaining channels (INT6)
3. Together, they preserve the weight directions that matter for prediction while smoothing quantization noise in less important directions

This is consistent with the training-loss-based evaluation which showed -0.349 loss improvement after C21.

## Numbers

| Metric | Baseline | C21 Compressed | Delta |
|---|---|---|---|
| val_loss | 5.3713 | 5.3594 | -0.0119 |
| val_bpb | 2.5001 | 2.4946 | -0.0055 |
| size | 137.11 MB | 25.95 MB | 5.28x |

## Leaderboard Position

- Old arch (sp1024) baseline BPB: 2.0158 (trained 600s)
- New arch (sp8192) BPB: 1.9053 (trained 600s) — 5.48% better
- C21 compressed new arch BPB: 2.4946 (raw checkpoint, no additional training)

The C21 number is on the raw checkpoint, not a trained+compressed model. For a fair leaderboard comparison, we'd need to train the new arch, then apply C21, then measure BPB.

## Config

- eval_seq_len=2048, val_batch_tokens=65536
- 9.6M val tokens, CASEOPS_ENABLED=1
- 8 activation calibration batches, 16 Hessian calibration batches
- RTX 2070, ~13.5 min total
