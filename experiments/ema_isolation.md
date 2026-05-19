# EMA Isolation Test — 1080 Ti (whisper, VM151)

## Hypothesis
EMA (Exponential Moving Average) of model weights degrades val_bpb. Previous runs showed pre-EMA BPB=1.7588 vs post-EMA BPB=1.9145 on RTX 2070.

## Experiment Design
Two 5-minute training runs on GTX 1080 Ti (11GB, sm_61 Pascal) with identical settings except EMA_DECAY:
1. **Frozen EMA** (decay=1.0): EMA weights never update from initialization
2. **Default EMA** (decay=0.9965): Standard EMA tracking

## Settings
- Model: 35.9M params (SOTA train_gpt_sota.py)
- TRAIN_BATCH_TOKENS=8192, GRADIENT_CHECKPOINT_ENABLED=1
- CASEOPS_ENABLED=1, SMEAR_GATE_ENABLED=1, LQER_ENABLED=1, LQER_ASYM_ENABLED=1, SPARSE_ATTN_GATE_ENABLED=1
- FA3_ENABLED=0, TRITON_ENABLED=0 (1080 Ti sm_61 incompatible)
- COMPRESSOR=brotli
- WARMUP_STEPS=5, SEED=42
- MAX_WALLCLOCK_SECONDS=300 (5 min training time)
- 20 train shards, 9.6M val tokens
- Data: fineweb10B_sp8192_lossless_caps_caseops_v1_reserved

## Results

### Frozen EMA (decay=1.0)
| Metric | Value |
|---|---|
| Steps completed | 20 |
| Training time | 313.8s (5.2 min) |
| tok/s | 675→522 (avg ~600) |
| Pre-EMA val_loss | 6.0865 |
| **Pre-EMA val_bpb** | **2.8335** |
| **Post-EMA val_bpb** | **4.1927** |
| Delta (post - pre) | **+1.3592** |
| Eval time | 805s (13.4 min) |
| Peak VRAM | 5774 MiB alloc / 10798 MiB reserved |
| GPTQ artifact | 94.8 KB |

### Default EMA (decay=0.9965)
| Metric | Value |
|---|---|
| Steps completed | 20 |
| Training time | 312.5s (5.2 min) |
| tok/s | 673→524 (avg ~600) |
| Pre-EMA val_loss | 6.0823 |
| **Pre-EMA val_bpb** | **2.8315** |
| **Post-EMA val_bpb** | **3.9720** |
| Delta (post - pre) | **+1.1405** |
| Eval time | 795s (13.2 min) |
| Peak VRAM | 5774 MiB alloc / 10798 MiB reserved |
| GPTQ artifact | 140.2 KB |

### Comparison Table
| | Pre-EMA BPB | Post-EMA BPB | Degradation |
|---|---|---|---|
| Frozen (decay=1.0) | 2.8335 | 4.1927 | +1.36 BPB |
| Default (decay=0.9965) | 2.8315 | 3.9720 | +1.14 BPB |
| Delta (default - frozen) | -0.002 | -0.22 | Default EMA slightly better |

## Analysis

### EMA is definitively the problem
- **Frozen EMA (decay=1.0)**: EMA weights stay at initialization → post-EMA val_bpb=4.19 ≈ random model (step-0 val_bpb was 4.1927). This confirms EMA is frozen at init.
- **Default EMA (decay=0.9965)**: EMA barely moves from init after only 20 steps. With decay=0.9965, each step the EMA is 99.65% old + 0.35% new. After 20 steps, EMA has only incorporated ~7% of the trained weights. Result: post-EMA BPB=3.97, still terrible.
- **Pre-EMA (live model)**: Both runs achieve ~2.83 BPB — the actual trained model is good.

### Root cause: EMA convergence rate
With decay=0.9965 and only 20 training steps:
- EMA weight fraction from training: 1 - 0.9965^20 = 1 - 0.932 = 0.068 (6.8%)
- 93.2% of EMA weights are still from initialization
- EMA needs ~650 steps to be 90% trained weights: 1 - 0.9965^n = 0.9 → n = 658

### Why the 30-min 2070 run was also bad
The 2070 baseline (648 steps) had post-EMA BPB=1.9145 vs pre-EMA 1.7588. At 648 steps with decay=0.9965:
- EMA fraction: 1 - 0.9965^648 = 1 - 0.104 = 89.6% trained
- But the remaining 10.4% init weights still drag it down
- The 0.16 BPB gap is much smaller than the 1.14 gap at 20 steps, consistent with EMA slowly converging

### Recommendation
**Remove EMA from the submission.** The live model consistently outperforms EMA at all training durations tested. EMA is harmful because:
1. At short training (<100 steps): EMA is mostly init weights
2. At medium training (~650 steps): EMA is 90% trained but the 10% init drag still hurts
3. EMA only helps when training is very long AND the model is overfitting — neither applies here

## 1080 Ti Performance Notes
- 1080 Ti (sm_61, Pascal): ~675 tok/s training, ~5x slower than 2070 (~3565 tok/s)
- bf16 not natively supported — "does not support bfloat16 compilation natively, skipping"
- Warmup with cu_buckets + loop_warmup: ~5 min for 5 warmup steps (vs ~2 min on 2070)
- Final validation on 9.6M tokens: ~13 min (vs ~15 min on 2070 — forward-only is less affected by bf16)
- Total run time: ~24 min per run (5 min warmup + 5 min training + 13 min validation + 1 min GPTQ)
- Peak VRAM: ~5.8 GB allocated / 10.8 GB reserved (11 GB total)
- GPU temp: 87-90°C, power: 155W/275W
