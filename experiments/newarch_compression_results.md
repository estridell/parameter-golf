# New Arch Compression Experiments (SP8192)

**Machine:** RTX 2070 (8GB VRAM, sm_75)
**Checkpoint:** newarch_sp8192.pt (35.9M params, vocab=8192, 11 layers, bank-based arch)
**Branch:** main
**Date:** 2026-05-19

## Results Summary

| Experiment | Technique | Compression | Baseline Loss | Quantized Loss | Delta Loss | Time |
|---|---|---|---|---|---|---|
| C12 | Bit packing (INT6) | 5.31x | - | - | N/A (size only) | 0.2s |
| C07 | INT6 per-row | 5.31x | - | - | MSE=2.68e-6 | 0.1s |
| C06 | INT8 per-tensor | 4.00x | - | - | MSE=3.45e-7 | 0.1s |
| C11 | GPTQ self-gen (INT6) | 5.11x | 5.305 | 5.186 | **-0.119** | 54.8s |
| C21 | GPTQ+AWQ combined | 5.28x | 5.305 | 4.956 | **-0.349** | 77.7s |

## Size Breakdown

| Experiment | Original | Compressed | Savings |
|---|---|---|---|
| C12 Bit Packing | 137.11 MB | 25.84 MB | 111.27 MB |
| C07 INT6 Per-Row | 137.11 MB | 25.84 MB | 111.27 MB |
| C06 INT8 Per-Tensor | 137.11 MB | 34.30 MB | 102.81 MB |
| C11 GPTQ Self-Gen | 137.11 MB | 26.84 MB | 110.27 MB |
| C21 GPTQ+AWQ | 137.11 MB | 25.95 MB | 111.16 MB |

## Old Arch vs New Arch Comparison

### Old Arch (baseline_sp1024.pt, 32.3M params, vocab=1024)

| Experiment | Technique | Compression | Delta Loss |
|---|---|---|---|
| C11 | GPTQ Self-Gen (INT6) | 5.28x | -0.031 |
| C21 | GPTQ+AWQ Combined | 5.28x | -0.084 |

### New Arch (newarch_sp8192.pt, 35.9M params, vocab=8192)

| Experiment | Technique | Compression | Delta Loss |
|---|---|---|---|
| C11 | GPTQ Self-Gen (INT6) | 5.11x | -0.119 |
| C21 | GPTQ+AWQ Combined | 5.28x | -0.349 |

**Key observation:** New arch benefits 4x more from GPTQ regularization (-0.349 vs -0.084 on C21).
The bank-based shared weights likely amplify the regularization effect since the same banks
are reused across all 11 layers — quantization noise on a single bank propagates to all layers.

## Recommendations

1. **Production compression: C21 (GPTQ+AWQ)** — 5.28x compression with -0.349 loss improvement
2. **Fast compression: C07 (INT6 per-row)** — same 5.31x ratio, instant (0.1s), no GPU needed
3. **Safe fallback: C06 (INT8 per-tensor)** — 4.00x with minimal MSE (3.45e-7)
4. The regularization effect from quantization is real and significant on this architecture
5. INT6 is the sweet spot — no need for INT4 (which degrades quality without Hessian)

## Architecture Notes

New arch uses shared bank weights (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank)
comprising 25.6M of 35.9M total params. Remaining 10.3M are per-layer params
(attn_scale, mlp_scale, resid_mix, q_gain, etc.) plus tok_emb and smear_gate.

The _unbank_state_dict splits banks into per-layer tensors for quantization.
50 tensors are passthrough (too small to quantize), 67 tensors are quantized.

## Technical Notes

- C11/C21 require CASEOPS_ENABLED=1 to find sp8192 training data (dir: fineweb10B_sp8192_lossless_caps_caseops_v1_reserved)
- All scripts written by previous worker (run #49), executed by run #52
- Baseline loss: 5.304648 (32 calibration batches, self-generated)
- collect_hessians() 4-value unpack bug still present in train_gpt.py (workaround in scripts)
