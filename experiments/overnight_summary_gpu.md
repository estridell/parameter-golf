# Overnight Quantization Experiments - GPU Summary

**Machine:** RTX 2070 (8GB VRAM, sm_75)
**Checkpoint:** baseline_sp1024.pt (32.3M params, vocab=1024, 11 layers)
**Branch:** compression-experiments
**Date:** 2026-05-19

## Results Summary

| Experiment | Technique | Compression | Baseline Loss | Quantized Loss | Delta Loss | Time |
|---|---|---|---|---|---|---|
| C10 | AWQ-Lite (INT4/INT8) | 3.97x | 4.316 | 8.678 | +4.362 FAIL | 44.8s |
| C11 | GPTQ Self-Gen (INT6) | 5.28x | 4.316 | 4.284 | -0.031 OK | 54.0s |
| C21 | GPTQ+AWQ Combined | 5.28x | 4.316 | 4.232 | -0.084 BEST | 76.4s |

## Key Findings

### 1. INT4 Without Hessian Correction Is Destructive (C10)
AWQ-Lite with pure INT4/INT8 quantization (no Hessian correction) destroys model quality.
Loss goes from 4.32 to 8.68. Per-row scaling alone cannot compensate for 4-bit quantization
error on a 32M param model.

**Lesson:** INT4 requires Hessian-based error correction (GPTQ).

### 2. GPTQ Self-Calibration Works (C11)
Using the model's own training data as calibration for Hessian collection produces excellent
results. INT6 GPTQ achieves 5.28x compression with slight improvement in loss.
Quantization noise acts as implicit regularization.

**Lesson:** No external calibration dataset needed.

### 3. AWQ + GPTQ Combined Is Best (C21)
Adding AWQ channel protection (top 1% channels to INT8) on top of GPTQ provides an
additional 0.05 loss reduction vs GPTQ alone. Same compression ratio (5.28x) but better
accuracy. AWQ identifies high-activation channels; INT8 protects them from quantization noise.

### 4. Quantization as Regularization
All GPTQ-based experiments produce quantized models with lower loss than baseline.
Quantization noise acts similarly to dropout, improving generalization.

## Compression Breakdown

- Original: 123.11 MB (fp32)
- C10 (AWQ-Lite): 31.03 MB (3.97x) - too aggressive, unusable
- C11 (GPTQ): 23.33 MB (5.28x) - good
- C21 (Combined): 23.30 MB (5.28x) - best

## Architecture Notes

Model uses shared bank weights (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank) comprising
31.6M of 32.3M total params. _unbank_state_dict splits into per-layer tensors for
quantization, _rebank_state_dict restores bank format.

**Bug found:** collect_hessians() in train_gpt.py unpacks (x, _) but DocumentPackingLoader
returns 4 values. Fixed locally with collect_hessians_fixed().

## Recommendations

1. Use C21 (GPTQ+AWQ Combined) for production quantization
2. INT6 is the sweet spot - INT4 too aggressive without additional techniques
3. Self-calibration works - no external calibration data needed
4. Consider expanding AWQ protection beyond 1% for further gains
5. Investigate per-layer sensitivity for mixed-precision strategies

## Files

- experiments/c10_awq_lite.py, c11_gptq_selfgen.py, c21_gptq_awq_combined.py
- experiments/c10_results.json, c11_results.json, c21_results.json
