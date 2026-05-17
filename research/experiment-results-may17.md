# Experiment Results — May 17, 2026

## Setup
- Machine: RTX 2070 (desktoparch), Python 3.14.5, PyTorch 2.12.0+cu130
- Batch: 16384 tokens, seq_len=1024, grad_accum_steps=8
- Base config: CaseOps + AsymLogit + SmearGate ON, Muon optimizer, GQA 8/4
- Seed: 42, wallclock: 300s, VAL_LOSS_EVERY=0

## Results Comparison

### Training Loss at Key Steps

| Step | 9L Baseline | 9L BH4096 | 9L BH5120 | 12L BH4096 | 12L Check | 12L MLP3 |
|------|-------------|-----------|-----------|------------|-----------|----------|
| 10   | 6.00        | 5.87      | 5.84      | 5.78       | 5.78      | 5.79     |
| 25   | 5.69        | 5.24      | 5.19      | 5.29       | 5.29      | 5.30     |
| 50   | 4.91        | 4.57      | 4.58      | 4.67       | 4.67      | 4.70     |
| 75   | 4.63        | 4.28      | 4.27      | 4.34       | 4.34      | 4.35     |
| 100  | -           | 4.09      | 4.08      | 4.13       | 4.13      | 4.15     |
| 150  | -           | 3.77      | 3.78      | 3.81       | 3.81      | 3.83     |
| 200  | -           | 3.73      | 3.70      | 3.77       | 3.77      | -        |

### Speed and Efficiency

| Config              | Params  | Step Time | Steps/300s | Loss@75 | Loss@200 | VRAM  |
|---------------------|---------|-----------|------------|---------|----------|-------|
| 9L Baseline         | 17.1M   | 1039ms    | ~288       | 4.63    | -        | 1.8GB |
| 9L BigramHash 4096  | 19.2M   | 1040ms    | ~288       | 4.28    | 3.73     | 1.8GB |
| 9L BigramHash 5120  | 19.7M   | 1038ms    | ~289       | 4.27    | 3.70     | 1.8GB |
| 12L BigramHash 4096 | 24.7M   | 1372ms    | ~219       | 4.34    | 3.77     | 2.1GB |
| 12L + Checkpoint    | 24.7M   | 1376ms    | ~218       | 4.34    | 3.77     | 2.1GB |
| 12L MLP3 + Check    | 27.2M   | 1556ms    | ~193       | 4.35    | -        | 2.2GB |

### Key Findings

1. **BigramHash is the clear winner.** ~7-8% lower loss at equivalent steps, zero speed overhead.
   - BH5120 slightly better than BH4096 but marginal.
   - BigramHash should be ON by default.

2. **12 layers is NOT better than 9 layers within 300s budget.**
   - 12L gets 24% fewer steps (219 vs 288) due to 32% slower step time.
   - At step 75, 12L loss (4.34) is WORSE than 9L+BigramHash (4.28).
   - At step 200, 12L loss (3.77) is WORSE than 9L+BigramHash (3.70).
   - More layers don't compensate for fewer training steps.

3. **Gradient checkpointing doesn't help at 16384 batch.**
   - Step time identical (1372 vs 1376ms) because VRAM isn't the bottleneck.
   - Would help at larger batch sizes that OOM without it.

4. **MLP 3x is slower with no loss benefit.**
   - 13% slower step time (1556 vs 1372ms), 12% fewer steps.
   - Loss is comparable or slightly worse at each step.

5. **The 16MB budget matters.**
   - 9L+BigramHash: ~11MB compressed (69% of budget)
   - 12L+BigramHash: ~11MB compressed (69% of budget)
   - Plenty of headroom for more techniques.

### Recommendations
1. Default stack: 9 layers + BigramHash (size 5120) + SmearGate + CaseOps + AsymLogit
2. Don't add more layers unless we can also speed up step time
3. Focus next on techniques that improve BPB without increasing step time:
   - Partial RoPE (~5 lines, no step time increase)
   - SwiGLU activation (same params, potentially better loss)
   - Better quantization (int5/int6 mixed precision)
4. For H100 extrapolation: 9L at 1038ms/2070 = ~12ms/H100 (85x ratio)
   - ~49,500 steps in 10 min — way above the ~4,950 step budget
   - This means we can afford more expensive techniques on H100

### Compression Notes
- 12L model: 93.5MB raw -> 11MB int8+zlib (3.76x compression)
- Well under 16MB limit
