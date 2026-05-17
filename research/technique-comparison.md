# Technique Comparison: Baseline vs SmearGate + CaseOps + AsymLogit + EMA

**Date:** 2026-05-16
**Machine:** RTX 2070 (8GB VRAM, sm_75)
**Seed:** 42 for both runs
**Wallclock cap:** 600 seconds (10 minutes)

## Summary

| Metric | Baseline | Techniques | Delta |
|--------|----------|-----------|-------|
| Steps completed | 162 | 160 | -2 |
| Final val_loss | 4.2337 | 4.2003 | -0.0334 |
| Final val_bpb | 2.5074 | 2.4877 | -0.0197 |
| Step time (avg) | 3710ms | 3761ms | +51ms (+1.4%) |
| Peak VRAM allocated | 6616 MiB | 6680 MiB | +64 MiB |
| Peak VRAM reserved | 6870 MiB | 7098 MiB | +228 MiB |
| Model params | 17,059,912 | 17,061,975 | +2,063 |

## Raw Training Results (pre-quantization)

### Baseline (all techniques disabled)
- Steps: 162 / 20000
- val_loss: 4.2337
- val_bpb: 2.5074
- Peak memory allocated: 6616 MiB, reserved: 6870 MiB
- Submission size (int8+zlib): 6,950,913 bytes

### Techniques (SmearGate + CaseOps + AsymLogit + EMA)
- Steps: 160 / 20000
- val_loss: 4.2003
- val_bpb: 2.4877
- Peak memory allocated: 6680 MiB, reserved: 7098 MiB
- Submission size (int8+zlib): 5,317,933 bytes

## Post-Quantization (int8+zlib roundtrip)

| Metric | Baseline | Techniques | Delta |
|--------|----------|-----------|-------|
| val_loss | 4.2471 | 5.8180 | +1.5709 |
| val_bpb | 2.5154 | 3.4457 | +0.9303 |
| Degradation | +0.3% | +38.5% | CRITICAL |

## Analysis

### Training Performance: Techniques Win
- BPB improved by 0.0197 (0.78% relative improvement)
- Loss converged faster in early steps (5.86 vs 6.56 at step 10)
- 2 fewer steps completed due to slightly slower step time

### Quantization: EMA Causes Catastrophic Degradation
The EMA weight averaging creates weights that are extremely sensitive to int8 quantization:
- Baseline: 0.3% BPB degradation after quantization
- Techniques: 38.5% BPB degradation after quantization

This is likely because EMA averages weights over time, creating distributions that are poorly suited to per-tensor int8 quantization. The quantization error gets amplified.

### VRAM: Modest Overhead
- +64 MiB allocated (+1.0%)
- +228 MiB reserved (+3.3%)
- Still well within 8GB budget

### Step Time: Negligible Overhead
- +51ms per step (+1.4%)
- 2 fewer steps in same wallclock time

## Conclusion

**The techniques improve raw training performance but break quantization.** The EMA weight averaging in particular causes catastrophic degradation when the model is quantized to int8 for submission.

**Recommendation:** Either:
1. Disable EMA and use only SmearGate + CaseOps + AsymLogit
2. Apply quantization-aware training or use a different quantization strategy for EMA models
3. Use float16 instead of int8 for submission

The SmearGate, CaseOps, and AsymLogit techniques show promise but need testing without EMA to isolate their individual contributions and confirm they don't also cause quantization issues.
