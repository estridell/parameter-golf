# Reference Step Counts from OpenAI Parameter Golf

## Default Configuration

| Parameter | Default | Source |
|-----------|---------|--------|
| `ITERATIONS` | 20,000 | `train_gpt.py:54` |
| `MAX_WALLCLOCK_SECONDS` | 600 (10 min) | `train_gpt.py:59` |
| `TRAIN_BATCH_TOKENS` | 524,288 | `train_gpt.py:57` |
| Tokens per step | 524,288 | 8 grad accum × 65,536 micro-batch |

The code runs until either `ITERATIONS` or `MAX_WALLCLOCK_SECONDS` is hit. In practice, all 10-minute runs hit the wallclock cap, not the iteration cap.

## Step Counts from Official Runs (8xH100, 10 min)

| Submission | Date | Steps | Step Time | BPB | Complexity |
|------------|------|------:|----------:|----:|------------|
| Naive Baseline | 2026-03-17 | 13,780 | 43.5ms | 1.2244 | Simple 9L, 512d |
| Sliding Window Eval | 2026-03-19 | 13,450 | ~44ms | 1.1925 | + sliding eval |
| 11L EMA GPTQ-lite | 2026-03-22 | 7,096 | ~84ms | 1.1228 | 11L + EMA + GPTQ |
| CaseOps + SparseGate + PhasedTTT | 2026-04-23 | 4,961 | ~121ms | 1.0634 | Full stack |
| BOS-Fixed SmearGate + LQER | 2026-04-27 | 4,945 | ~121ms | 1.0611 | Full stack |
| SmearGate + 3Seed (best) | 2026-04-29 | 4,900-4,960 | ~121ms | 1.0614 | Full stack |

**Pattern:** As submissions get more complex (more layers, TTT, GPTQ, quantization), step time increases ~3x and step count drops ~3x.

## Cross-Reference: RTX 2070 Timing

- **2070 step time:** ~3,700ms/step (3.7s) without compile
- **8xH100 baseline step time:** ~43.5ms/step
- **Speed ratio:** ~85x slower

### Estimated 2070 Wallclock to Match H100 Step Counts

| Match Target | Steps | 2070 Wallclock | Feasible? |
|--------------|------:|---------------:|-----------|
| Naive baseline | 13,780 | 14.2 hours | Overnight run |
| Mid-tier (EMA+GPTQ) | 7,096 | 7.3 hours | Long run |
| Top submissions | ~4,950 | 5.1 hours | Long run |
| 2070 smoke test (65 steps) | 65 | 4 min | Quick test |

### With Our Batch Size (65,536 tokens)

Our 2070 uses `TRAIN_BATCH_TOKENS=65536` (8x smaller than default 524,288). This means:
- Each step processes 8x fewer tokens
- To see the same total tokens as 13,780 H100 steps: need 13,780 × 8 = 110,240 steps
- At 3.7s/step: 110,240 × 3.7s = 45.6 hours (not practical)

## Recommendations

### For Fair Step-Cap Comparisons

1. **Match total tokens, not steps.** The H100 baseline sees 13,780 × 524,288 = 7.2B tokens in 10 min. To match this on 2070 with 65,536 tokens/step: 7.2B / 65,536 = 110,240 steps. Not practical.

2. **Match steps for architecture comparison.** If comparing architectures (same step count, different hardware), use:
   - **4,950 steps** for comparison against top submissions (~5.1 hours on 2070)
   - **7,096 steps** for comparison against mid-tier (~7.3 hours on 2070)

3. **Recommended step cap for 2070 experiments:**
   - **Quick iteration:** 100-200 steps (6-12 min) — enough to see loss trajectory
   - **Standard run:** 500-1000 steps (30-60 min) — meaningful comparison
   - **Full comparison:** 4,950 steps (5.1 hours) — matches top H100 submissions

4. **For parameter golf scoring:** The challenge evaluates BPB after 10 min on 8xH100. On 2070, we can't match this. Instead:
   - Run for a fixed step count (e.g., 4,950)
   - Compare loss curves at same step count
   - Use relative improvement (ΔBPB) rather than absolute BPB

### Key Insight

The 2070 is ~85x slower per step but processes 8x fewer tokens per step. Net effect: each 2070 step covers 8x less ground than an H100 step. For the same total tokens, you need 8x more steps on 2070, each taking 85x longer = 680x more wallclock time. The 10-minute H100 challenge translates to ~113 hours on 2070 for equivalent token coverage.

This makes the 2070 unsuitable for the actual challenge, but excellent for:
- Architecture prototyping (test ideas quickly at small scale)
- Loss curve analysis (same step count, compare trajectories)
- Hyperparameter tuning (fast iteration on small runs)
