# Apples-to-Apples: Baseline vs New Arch, 600s Wallclock

**Date:** 2026-05-18
**Machine:** RTX 2070 (8GB VRAM, sm_75)
**Model:** 32M params (512 dim, 11 layers, 8 heads, 4 KV heads)
**Config:** SEED=42, TRAIN_BATCH_TOKENS=16384, GRADIENT_CHECKPOINT_ENABLED=1, WARMUP_STEPS=0

## Run 1: Baseline (sp1024, no techniques)
- **Tokenizer:** fineweb_1024_bpe (vocab_size=1024)
- **Techniques:** None (SmearGate=0, CaseOps=0, LQER_ASYM=0)
- **Steps completed:** 275 (in 596s effective wallclock)
- **Final train_loss:** 3.4102 (step 275)
- **tok/s:** 7544
- **Peak VRAM:** 3498 MiB allocated / 3706 MiB reserved

## Run 2: New Arch (sp8192, CaseOps + techniques)
- **Tokenizer:** fineweb_8192_bpe_lossless_caps_caseops_v1_reserved (vocab_size=8192)
- **Techniques:** SmearGate=1, CaseOps=1, LQER_ASYM=1
- **Steps completed:** 264 (in 596s effective wallclock)
- **Final train_loss:** 4.1514 (step 260, last logged 5-step interval)
- **tok/s:** 7240
- **Peak VRAM:** 3549 MiB allocated / 3680 MiB reserved

## Comparison

| Metric | Baseline (sp1024) | New Arch (sp8192) | Delta |
|--------|-------------------|-------------------|-------|
| Steps in 600s | 275 | 264 | -4.0% |
| tok/s | 7544 | 7240 | -4.0% |
| Final loss (raw) | 3.4102 | 4.1514 | +21.7% |
| Loss at step 260 | 3.4228 | 4.1514 | +21.3% |
| Peak VRAM (alloc) | 3498 MiB | 3549 MiB | +51 MiB |

### Loss Normalized by Vocab Size

Raw cross-entropy loss scales with vocab size. A random model would have:
- Baseline (1024 vocab): ln(1024) = 6.93 nats
- New arch (8192 vocab): ln(8192) = 9.01 nats

Normalized loss (fraction of random baseline) at step 260:
- **Baseline:** 3.4228 / 6.93 = 0.494 (49.4% of random)
- **New arch:** 4.1514 / 9.01 = 0.461 (46.1% of random)
- **New arch is 6.7% better** relative to random baseline

This means the new arch (CaseOps + SmearGate + LQER_ASYM) learns more efficiently
per step when accounting for the 8x larger vocabulary.

### Loss at Matching Steps

| Step | Baseline | New Arch | Ratio | Expected (vocab only) |
|------|----------|----------|-------|-----------------------|
| 100  | 4.3529   | 4.9758   | 1.143 | 1.30 |
| 150  | 3.8161   | 4.5349   | 1.188 | 1.30 |
| 200  | 3.6495   | 4.2233   | 1.157 | 1.30 |
| 250  | 3.4925   | 4.2685   | 1.222 | 1.30 |
| 260  | 3.4228   | 4.1514   | 1.213 | 1.30 |

The loss ratio (new_arch/baseline) is consistently ~1.15-1.22, well below the
1.30 expected from vocab size alone. The techniques are helping.

### Speed Overhead

CaseOps + SmearGate + LQER_ASYM add ~4% overhead (7240 vs 7544 tok/s).
This means 11 fewer steps in the same wallclock time (264 vs 275).

### VRAM

Minimal difference: +51 MiB for the new arch (3549 vs 3498 MiB).
Both fit comfortably in 8GB.

## Conclusions

1. **The new arch is more sample-efficient** — when normalized for vocab size, it
   achieves 6.7% lower loss per step than baseline.
2. **Speed penalty is modest** — 4% slower, losing ~11 steps per 600s run.
3. **VRAM is comparable** — +51 MiB is negligible.
4. **The big question is BPB** — bits-per-byte on validation data would give a
   definitive answer. The vocab-normalized analysis suggests the new arch wins,
   but we need val_bpb to confirm.

## Notes
- `train_time: 0ms` in stopping_early is a bug (training_time_ms not propagated
  correctly after patch to skip final-step validation). Does not affect results.
- Validation was skipped (VAL_LOSS_EVERY=0, PREQUANT_ONLY=1) for speed.
  val_bpb not available.
- Code patches on rtx2070 branch: skip final-step validation when VAL_LOSS_EVERY=0,
  skip diagnostic eval when PREQUANT_ONLY=1.
