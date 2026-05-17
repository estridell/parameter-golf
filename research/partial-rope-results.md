# Partial RoPE Experiment — May 17, 2026

## Config
- 9L + BigramHash 5120 + SmearGate + CaseOps + AsymLogit
- PARTIAL_ROPE_FRACTION=0.5 (first half of head dims get RoPE)
- Same seed, batch, wallclock as other experiments

## Results

| Step | No Partial RoPE | Partial RoPE 0.5 | Delta  |
|------|-----------------|-------------------|--------|
| 25   | 5.19            | 5.19              | 0.0%   |
| 50   | 4.58            | 4.59              | +0.2%  |
| 75   | 4.27            | 4.28              | +0.2%  |
| 100  | 4.08            | 4.08              | 0.0%   |
| 200  | 3.70            | 3.72              | +0.5%  |
| 275  | 3.55            | 3.56              | +0.3%  |

Speed: 1036ms/step (vs 1038ms) — 0.2% faster
Steps in 300s: 290 (vs 289)

## Conclusion
Partial RoPE at fraction=0.5 shows no meaningful improvement. The slightly faster
step time (0.2%) may help in very long runs, but the loss is marginally worse at
every checkpoint. Not worth adding to the default stack.

May try fraction=0.75 or lower values, but unlikely to be a game-changer.
