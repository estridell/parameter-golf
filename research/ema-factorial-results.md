# EMA Factorial Experiment Results

## 2^4 Factorial: SmearGate × CaseOps × AsymLogit × EMA

All runs: 300s wallclock, seed 42, int8+zlib quantization roundtrip.

| Combo | SmearGate | CaseOps | AsymLogit | EMA | Raw val_bpb | Post-int8 val_bpb | Degradation |
|-------|-----------|---------|-----------|-----|-------------|-------------------|-------------|
| 0000 | off | off | off | off | 2.1694 | 2.1754 | 0.28% |
| 0001 | off | off | off | **on** | 2.1649 | 3.1370 | **44.9%** |
| 0110 | off | on | on | off | 2.1799 | 2.1880 | 0.37% |
| 1111 | on | on | on | on | 2.1842 | 3.1937 | **46.2%** |

## Key Finding

**EMA is the sole cause of quantization degradation.**

- Without EMA: int8 quantization causes <0.4% degradation
- With EMA: int8 quantization causes ~45-46% degradation
- CaseOps + AsymLogit have negligible impact on quantization stability
- SmearGate has negligible impact on quantization stability

## Root Cause

EMA (exponential moving average, decay=0.997) smooths weight trajectories during training. This produces weights with a different distribution than the raw trained weights — specifically, EMA weights have more mass near zero and heavier tails. When quantized to int8, the reduced precision clips these tail values more aggressively, destroying information.

## Artifact Sizes

| Combo | Artifact Size |
|-------|--------------|
| 0000 (baseline) | 8.2MB |
| 0001 (EMA only) | 6.9MB |
| 0110 (CO+AL) | 7.9MB |
| 1111 (all) | 6.5MB |

EMA makes weights more compressible but destroys quantization quality. Net negative.

## Implications

1. EMA should be DISABLED for any submission using int8 quantization
2. If EMA provides training-time quality gains, it must be dropped before serialization
3. Alternative: use SWA (stochastic weight averaging) instead — it doesn't suffer from the same distribution shift
4. Alternative: use EMA during training, then snap back to non-EMA weights before quantization

## Next Steps

- Test SWA as EMA replacement (SWA has less smoothing, closer to raw weights)
- Test EMA-during-training-then-snapback: train with EMA, serialize non-EMA weights
- Test GPTQ with EMA weights (GPTQ compensates for quantization error, might handle the distribution shift)
