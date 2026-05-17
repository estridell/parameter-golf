# Delta Encoding Entropy Test — Actual Model Weights

**Date:** 2026-05-17
**Model:** 17M param GPT, 9 layers (blocks 0-8), d_model=512, GQA (8 heads, 4 KV)
**Checkpoint:** final_model.pt (65MB)
**Method:** Shannon entropy (256-bin histogram) + zlib level 9 + zstd level 22

## Verdict: NOT WORTH PURSUING

Delta encoding between adjacent layers provides negligible compression benefit (<0.5% savings). Adjacent layers in this model are not similar enough.

## Raw Entropy Per Layer

| Layer | resid_mix | c_q | c_k | c_v | proj | mlp.fc | mlp.proj |
|-------|-----------|-----|-----|-----|------|--------|----------|
| 0     | 6.52      | 7.25| 7.31| 7.31| 5.84 | 6.99   | 5.49     |
| 1     | 6.58      | 7.29| 7.22| 7.08| 6.21 | 6.84   | 5.16     |
| 2     | 6.55      | 7.12| 7.23| 7.03| 6.04 | 6.84   | 5.84     |
| 3     | 6.51      | 7.14| 7.09| 6.81| 6.00 | 6.71   | 5.84     |
| 4     | 6.30      | 7.19| 7.18| 6.87| 5.76 | 6.68   | 5.86     |
| 5     | 6.30      | 7.06| 7.16| 6.84| 5.99 | 6.64   | 6.01     |
| 6     | 6.30      | 7.26| 7.15| 6.81| 6.21 | 6.55   | 6.18     |
| 7     | 6.31      | 7.21| 7.20| 6.76| 6.48 | 6.45   | 6.11     |
| 8     | 6.27      | 7.13| 7.05| 6.84| 6.25 | 6.51   | 6.08     |

Observations:
- Entropy ranges 5.16–7.31 bits/value across all parameters
- c_q and c_k weights have highest entropy (~7.1-7.3 bits) — nearly uniform
- mlp.proj weights have lowest entropy in early layers (~5.2-5.9 bits)
- Entropy is remarkably consistent across layers for the same parameter type

## Delta Encoding Results (Adjacent Layer Pairs)

Average delta/original entropy ratio: **0.98x** (deltas are ~2% MORE entropic)
Min ratio: **0.89x** (resid_mix blocks 4→5, 7→8)
Max ratio: **1.02x** (c_v blocks 1→2, mlp.proj 1→2)

Key finding: **Deltas have EQUAL OR HIGHER entropy than originals in nearly all cases.**
This means adjacent layers are NOT similar — their differences are as random as the weights themselves.

Worst performers (delta makes things worse):
- resid_mix: 0.89-0.95x consistently — these tiny (2,512) tensors diverge significantly
- mlp.proj early→mid: 0.91x for blocks 0→1

Best performers (delta barely helps):
- c_q, c_k: ~1.00x — identical entropy, zero benefit
- c_v: 0.97-1.02x — negligible difference

## Actual Compression Comparison

### zlib (level 9)
| Metric | Original | Delta |
|--------|----------|-------|
| Raw bytes | 58,753,024 | 58,753,024 |
| Compressed | 54,575,937 | 54,436,269 |
| Ratio | 0.929 | 0.927 |
| **Savings** | — | **0.3%** |

### zstd (level 22)
| Metric | Original | Delta |
|--------|----------|-------|
| Raw bytes | 58,753,024 | 58,753,024 |
| Compressed | 54,490,627 | 54,295,608 |
| Ratio | 0.927 | 0.924 |
| **Savings** | — | **0.4%** |

## Why Delta Encoding Fails Here

1. **Small model (9 layers, 17M params):** Layer similarity is low. In larger models (100+ layers), adjacent layers may share more structure.
2. **GQA with random init residuals:** The resid_mix parameters (learned interpolation between attention and MLP outputs) diverge significantly between layers.
3. **Only 9 delta opportunities:** Even with perfect deltas, you can only delta 8 of 9 layers. The base layer is always full-size.
4. **Float entropy is already near-random:** 7+ bits/value out of 8 max means weights are close to uniform. No structure to exploit.

## Estimated Artifact Size with Delta + zstd

- Current: 54.5 MB (zlib) / 54.5 MB (zstd)
- With delta encoding: 54.3 MB (zstd) — **saves ~200KB**
- Not worth the added decode complexity

## Recommendation

**Do not pursue delta encoding for this model.** The 0.3-0.4% savings do not justify the added complexity.

Better compression targets (from quantization survey):
- INT8 quantization + zlib: ~3.9x compression (already demonstrated: 67MB → 5.4MB)
- NF4/INT4 quantization: 4-8x compression
- SVD low-rank decomposition: potential 2-3x on attention weights
- Weight sharing/tying: direct parameter reduction
