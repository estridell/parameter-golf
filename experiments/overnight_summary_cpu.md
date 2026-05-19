# Overnight Compression Experiments - CPU Summary

**Machine:** 1080 Ti (VM151), 11GB VRAM
**Branch:** compression-experiments
**Date:** 2026-05-19

## Results Summary

| Experiment | Technique | Compression | Size | Notes |
|---|---|---|---|---|
| C12 | Bit packing baseline | 4.80x | 23.5 MB | 12→2.5 bytes/param |
| C07 | INT6 per-row quantization | 5.77x | 22.2 MB | MSE=0.000002 — BEST RATIO |
| C06 | INT8 per-tensor | 5.00x | 24.5 MB | Zero accuracy loss — safe option |
| C02 | Magnitude pruning (50%) | 1.85x | — | No value without quantization |
| C04 | Unstructured pruning+quant | 4.66x | 25.8 MB | Worse than pure quant |
| C05 | Delta encoding | 0.73x | — | INCREASED size by 27.7%, dead end |
| C13 | LQER | — | — | No benefit at INT8 level |

## Key Findings

### 1. Quantization Is the Primary Compression Driver
All viable compression techniques rely on quantization. Pruning, delta encoding, and LQER
either add no value or actively hurt compression when used alone.

### 2. INT6 Per-Row Is the Sweet Spot (C07)
5.77x compression (82.7% savings) with MSE=0.000002 — essentially lossless. Per-row
scaling adapts to the distribution of each weight matrix, capturing more precision than
per-tensor approaches.

### 3. INT8 Per-Tensor Is the Safe Option (C06)
5.00x compression with zero accuracy loss. Conservative but reliable. Good baseline for
production where you want guaranteed fidelity.

### 4. Pruning Without Quantization Is Worthless (C02)
50% magnitude pruning achieves only 1.85x compression. The sparsity overhead (indices)
eats into savings. Pruning only helps when combined with quantization, and even then
(C04) it's worse than pure quantization.

### 5. Delta Encoding Was Counterproductive (C05)
Delta encoding increased model size by 27.7%. The residuals between layers are not
compressible enough to offset the encoding overhead. Dead end for this architecture.

### 6. LQER Provides No Benefit at INT8 (C13)
Low-rank error correction adds parameters that offset quantization savings at INT8
precision. May help at INT4 but needs further investigation.

### 7. zstd Levels 1-19 Had Negligible Impact on Quantized Data
Quantized integer data has high entropy — general-purpose compression adds almost nothing.
Don't waste CPU cycles on zstd for quantized weights.

## Recommendations

1. Use C07 (INT6 per-row) for best compression ratio
2. Use C06 (INT8 per-tensor) when zero accuracy loss is required
3. Combine with GPTQ from GPU experiments (C21) for maximum savings
4. Do not use pruning, delta encoding, or LQER for this model size

## Cross-Reference: GPU Experiments

See overnight_summary_gpu.md for the GPU-side quantization results (C10, C11, C21).
The GPU experiments focus on Hessian-corrected quantization (GPTQ) and channel protection
(AWQ). Combined best: C21 (GPTQ+AWQ) at 5.28x with loss improvement.

## Files

- Experiment scripts ran on VM151 (1080 Ti) — not present in this repo checkout
- Results captured via cron job output
