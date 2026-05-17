# EMA Weight Averaging — Implementation Log

**Date:** 2026-05-16
**Task:** t_6d1bd2d8 — Implement EMA weight averaging in train_gpt.py
**Branch:** rtx2070

## What was implemented

Exponential Moving Average (EMA) weight averaging in train_gpt.py. Maintains a running
average of model weights during training and swaps to averaged weights for final model
serialization.

## How it works

1. **Initialization** (after model creation): If `EMA_ENABLED=1`, create a deep copy of
   all model parameters as the EMA shadow weights.

2. **Update** (after each optimizer step): For each parameter:
   `ema_shadow = decay * ema_shadow + (1 - decay) * param`
   Default decay: 0.997 (effective window ~333 steps).

3. **Final swap** (before serialization): Load EMA shadow weights into the model, then
   serialize. This means the submitted model uses averaged weights.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMA_ENABLED` | `0` | Enable EMA (1=on, 0=off) |
| `EMA_DECAY` | `0.997` | EMA decay rate |

## Code changes (~25 lines added)

- **Hyperparameters class**: Added `ema_enabled` and `ema_decay` fields
- **Model setup**: Initialize EMA shadow dict after model creation
- **Training loop**: Update EMA after each optimizer step
- **Serialization**: Swap to EMA weights before saving final model

## Smoke test results (RTX 2070, 60s run)

**EMA_ENABLED=1:**
- 16 steps completed, ~3.75s/step (no measurable overhead)
- Peak VRAM: 5633 MiB (same as without EMA)
- "EMA: enabled with decay=0.997" logged at startup
- "EMA: swapping to averaged weights for final model" logged at end
- Serialized: 67MB raw → 5MB int8+zlib

**EMA_ENABLED=0:**
- Normal training, no EMA code path executed
- No regression in speed or memory

## Notes

- EMA with depth recurrence is a known disaster combo (per task rules) — do not combine.
- EMA shadow weights are stored in a dict, not as model parameters, so they don't affect
  the serialized model size or parameter count.
- The shadow weights live in GPU memory, adding ~68MB for the 17M param model (negligible
  on 8GB VRAM).
- Commit `423998b` included the EMA code along with CaseOps+AsymLogit changes.
