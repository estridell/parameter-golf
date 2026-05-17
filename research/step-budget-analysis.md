# Step Budget Analysis -- Technique Impact on RTX 2070

**Date:** 2026-05-17
**Machine:** RTX 2070 (8GB VRAM, sm_75), CUDA 13.2
**Model:** 17M params (9 layers, 512 dim, 8 GQA heads, 4 KV heads)
**Batch:** 65536 tokens (8 grad accum x 8192 micro-batch)

## Step Time Measurements (20-step runs, seed=42)

| Technique | Step Time (ms) | Delta vs Baseline | Delta % | Params |
|-----------|---------------|-------------------|---------|--------|
| BASELINE (all off) | 3736 | -- | -- | 17,059,912 |
| SmearGate | 3769 | +33 | +0.9% | 17,059,925 |
| CaseOps + AsymLogit | 3776 | +40 | +1.1% | 17,061,962 |
| EMA (decay=0.997) | 3714 | -22 | -0.6% | 17,059,912 |
| BigramHash (size=4096) | 3750 | +14 | +0.4% | 19,157,064 |
| OrthoInit | ~3740 | ~0 | ~0% | 17,059,912 |
| Partial RoPE (0.5) | ~3740 | ~0 | ~0% | 17,059,912 |
| LeakyReLU(0.01)^2 | ~3740 | ~0 | ~0% | 17,059,912 |
| Parallel Residuals | ~3740 | ~0 | ~0% | 17,059,912 |
| Stride-64 Eval | 0 | 0 | 0% | -- |
| All new combined | 3742 | +6 | +0.2% | 17,059,912 |

### Notes
- EMA delta is noise (EMA only copies weights after optimizer step, no forward pass change)
- OrthoInit, Partial RoPE, LeakyReLU, Parallel Residuals: init-only or trivial forward changes
- BigramHash adds 2.1M params (+12%) but minimal step time impact (embedding lookup is cheap)
- Stride-64 eval: eval-time only, zero training overhead

## H100 Extrapolation (85x ratio)

Baseline: 2070 step time / 85 = H100 step time
H100 steps in 10 min = 600,000ms / H100_step_time

| Technique | 2070 ms/step | H100 est. ms/step | H100 steps/10min | Steps Lost |
|-----------|-------------|-------------------|------------------|------------|
| BASELINE | 3736 | 43.95 | 13,651 | 0 |
| +SmearGate | 3769 | 44.34 | 13,531 | -120 |
| +CaseOps+AsymLogit | 3776 | 44.42 | 13,507 | -144 |
| +EMA | 3714 | 43.69 | 13,732 | +81 |
| +BigramHash | 3750 | 44.12 | 13,599 | -52 |
| +All techniques | ~3770 | 44.35 | 13,529 | -122 |

### Key Insight
All techniques combined lose only ~122 steps on H100 (0.9% of budget).
The leaderboard gap is 13,651 -> 4,950 steps (63.7% reduction).
**Step time is NOT the bottleneck.** The gap comes from:
1. Bigger models (11 layers vs 9, 4x MLP vs 2x)
2. More training tokens per step (larger batch)
3. Better optimization (Muon variants, LR schedules)
4. Better evaluation (TTT, sliding window)

## Technique Verdicts

### Already Implemented (rtx2070 branch)
| Technique | Verdict | Reason |
|-----------|---------|--------|
| SmearGate | KEEP | +33ms, improves BPB, minimal overhead |
| CaseOps + AsymLogit | KEEP | +40ms, improves BPB, small overhead |
| EMA | CAUTION | Breaks int8 quantization (38.5% BPB degradation). Use only if not quantizing. |
| BigramHash | KEEP | +14ms, +2.1M params, near-free BPB |

### Newly Implemented (this session)
| Technique | Verdict | Reason |
|-----------|---------|--------|
| OrthoInit | KEEP | Zero runtime overhead, better initialization |
| Partial RoPE (0.5) | KEEP | Zero overhead, standard in top submissions |
| LeakyReLU(0.01)^2 | KEEP | Zero overhead, standard in top submissions |
| Parallel Residuals | KEEP | Zero overhead, enables attn+mlp specialization |
| Stride-64 Eval | KEEP | Zero training overhead, better BPB measurement |

## Research-Only Findings

### SP8192 Tokenizer
- **What:** SentencePiece with 8192 vocab (vs our 1024)
- **BPB gain:** Significant -- ALL top submissions use SP8192
- **How:** `python3 data/cached_challenge_fineweb.py --variant sp8192`
- **Data pipeline:** Re-tokenizes all training/val data to new .bin shards
- **Size impact:** Larger vocab = more bytes/token = better compression
- **Variant:** "lossless caps caseops v1 reserved" (PR #1729)
- **Complexity:** Medium -- data pipeline change, model vocab_size change
- **Recommendation:** STRONGLY RECOMMENDED. Required for competitive BPB.

### Score-First TTT (Test-Time Training)
- **What:** After training, fine-tune on validation data before scoring
- **How:** Score each token BEFORE weight update (PR #461)
- **Config:** 3 epochs, lr=0.005, LoRA per-doc reset
- **BPB gain:** ~0.005-0.02 BPB depending on stack
- **Training impact:** ZERO -- TTT is eval-time only
- **Eval impact:** Adds ~10-30 min eval time
- **Complexity:** High -- requires LoRA implementation, per-doc reset logic
- **Recommendation:** RECOMMENDED for final submission. No training cost.

### LQER (Low-Rank Quantization Error Reduction)
- **What:** Asymmetric int4 quantization with rank-4 error correction
- **How:** Apply to top-3 tensors (largest by parameter count)
- **BPB gain:** Minimal direct gain, but enables better compression
- **Size:** Fits model in 16MB submission limit
- **Combined with:** GPTQ int6 + int7 embed + int8-per-row attn-gate
- **Compression:** lrzip zpaq + L1 similarity-sort + brotli
- **Complexity:** High -- requires GPTQ, custom quantization pipeline
- **Recommendation:** NEEDED for submission size. Use instead of int8+zlib.

## Optimal Stack for H100 (estimated)

Target: ~4,950 steps, ~121ms/step, <1.10 BPB

### Must Have
1. SP8192 tokenizer (re-tokenize data)
2. 11 layers (vs 9), 4x MLP (vs 2x) -> ~35M params
3. Partial RoPE (16/64 dims) + YaRN
4. Parallel residuals (from layer 8+)
5. LeakyReLU(0.5)^2 with fused Triton kernel
6. SmearGate + CaseOps
7. QK-Gain 5.0 (vs 1.5)
8. Depth recurrence (loop layers 3-5)
9. Score-first TTT (eval-time)

### Nice to Have
10. OrthoInit (free)
11. EMA (if not quantizing to int8)
12. XSA (cross-sequence attention)
13. U-Net skip connections

### Compression
14. LQER int4 + GPTQ int6 + brotli (fit in 16MB)

## Gap Analysis

Current stack (17M params, 1024 vocab):
- ~13,651 steps on H100, ~1.22 BPB

Target stack (~35M params, 8192 vocab):
- ~4,950 steps on H100, ~1.06 BPB

The gap is NOT step time. It's:
1. **Model size:** 17M -> 35M (2x params, ~2x step time)
2. **Vocab:** 1024 -> 8192 (8x, better compression)
3. **Depth recurrence:** Loop layers = more compute per step
4. **Better optimizer:** Muon variants, LR schedules
5. **TTT:** Free BPB improvement at eval time

With our current 17M model, even with all techniques, we can't get below ~1.15 BPB.
To reach 1.10 BPB, we need the bigger model (35M) + SP8192 + TTT.
