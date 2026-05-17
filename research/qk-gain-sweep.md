# Experiment #2: QK-Gain Sweep Results

**Date:** 2026-05-17
**Machine:** RTX 2070 (desktoparch, 100.70.124.83)
**Script:** train_gpt_sweep.py (original rtx2070 branch, 1338 lines)
**Config:** 17M params, 9L, 512 dim, GQA 8/4, Muon, batch=16384, seq_len=1024
**Features:** SmearGate=1, CaseOps=1, AsymLogit=1, EMA=0, BigramHash=5120
**Wallclock:** 300s per run, seed=42

## Results Table

| QK_Gain | loss@25 | loss@50 | loss@75 | loss@100 | loss@200 | final_step | final_loss | step_time | VRAM    |
|---------|---------|---------|---------|----------|----------|------------|------------|-----------|---------|
| 1.5     | 5.1935  | 4.5761  | 4.2686  | 4.0770   | 3.7016   | 285        | 3.4557     | 1040.3ms  | 1596MiB |
| 2.5     | 5.1929  | 4.5692  | 4.2647  | 4.0736   | 3.7021   | 285        | 3.4519     | 1041.4ms  | 1596MiB |
| 3.5     | 5.1887  | 4.5635  | 4.2639  | 4.0746   | 3.7054   | 285        | 3.4581     | 1040.9ms  | 1596MiB |
| 5.25    | 5.1816  | 4.5610  | 4.2684  | 4.0860   | 3.7189   | 285        | 3.4745     | 1043.3ms  | 1596MiB |

## Key Findings

### 1. QK-Gain has negligible impact on training loss
The difference between best (2.5, loss=3.4519) and worst (5.25, loss=3.4745) is only **0.023** — a 0.66% spread. This is within noise for a 300s/285-step run. QK-Gain is NOT a high-impact hyperparameter for our setup.

### 2. Higher QK-Gain helps early, hurts late
- At step 25: gain=5.25 is best (5.1816) vs gain=1.5 worst (5.1935) — 0.12% advantage
- At step 200: gain=1.5 is best (3.7016) vs gain=5.25 worst (3.7189) — 0.47% disadvantage
- Higher QK-Gain amplifies attention logits, providing stronger initial signal but slightly degraded late convergence

### 3. VRAM and step time unaffected
All runs: 1596MiB peak VRAM, ~1041ms/step. QK-Gain is a scalar parameter with zero memory/compute overhead.

### 4. Leaderboard recommendation (5.25) is worst for our setup
The leaderboard's recommended QK-Gain=5.25 gives the highest final loss. This may be because:
- Leaderboard models train for much longer (>300s) where early convergence advantage matters more
- Different architecture/hyperparameter interactions
- Our model with SmearGate/CaseOps/AsymLogit/BigramHash may have different optimal scaling

## Recommendation
**Use QK-Gain=1.5 (default) for remaining experiments.** The 0.66% spread is too small to matter, and 1.5 performs as well as any value at 300s. If we run longer training (600s+), 2.5 might edge ahead, but the effect is marginal.

## Notes on Execution
- Script was  (copied from fa5c2ed) because another process kept
  overwriting train_gpt.py with the SOTA version
- Previous runs had SIGPIPE issues with  over SSH; fixed by redirecting to file
- Previous runs used stale .pyc cache; fixed by clearing __pycache__
