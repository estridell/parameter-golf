# Overnight Research Session — May 17, 2026

## Session Start: 02:28 CEST

## Research Summary: Parameter Golf Leaderboard Analysis

### Current State of the Art (as of May 2026)
- **Confirmed SOTA:** 1.1147 BPB (abaybektursun — Self-Gen GPTQ + XSA-all)
- **Pending PRs:** Claims as low as 0.8265 BPB (SLOT-24 + Pre-Quant AdamW TTT)
- **Our baseline:** ~1.22 BPB
- **Gap to confirmed SOTA:** ~0.11 BPB (9%)
- **Gap to pending SOTA:** ~0.39 BPB

### Top Techniques by Submission Count
| Technique | Submissions | Impact |
|-----------|-------------|--------|
| BigramHash | 583 | Top 5 ALL use it — essentially free lunch |
| SmearGate | 396 | We already have it |
| XSA (Cross-Sparse Attention) | 392 | Top 3 use it |
| U-Net skip connections | 275 | We already have it |
| Partial RoPE | 270 | Modest savings |
| LN Scale | 226 | Minor |
| GQA | 209 | We already have it |
| Depth recurrence | 116 | More layers without more params |

### Confirmed Leaderboard Top 10
1. Self-Gen GPTQ + XSA-all: 1.1147 BPB (abaybektursun)
2. LeakyReLU-squared + TTT + Muon: 1.1194 BPB (abaybektursun)
3. EMA + GPTQ-lite: 1.1228 BPB (signalrush)
4. Partial RoPE + LN Scale: 1.1248 BPB (jfprincz)
5. XSA4 + EMA + Int6: 1.1271 BPB (jfprincz)
6. Efficient Partial XSA: 1.1307 BPB (unnir)
7. Int5-MLP + BigramHash: 1.1428 BPB (thwu1)
8. SmearGate + BigramHash: 1.1458 BPB (Raahil Shah)
9. MLP3x + Int6 QAT: 1.1502 BPB (aruniyer)
10. Ternary U-Net 73.7M: 1.1570 BPB (CiprianFlorin-Ifrim)

### Key Architectural Patterns
- **BigramHash** (free features from character pairs): Hash adjacent tokens into a small embedding table, add to token embeddings. Zero overhead, meaningful BPB improvement.
- **Cross-Sparse Attention (XSA)**: Share Q/K/V projections across heads/positions. Saves attention parameters for more MLP width/layers.
- **Depth Recurrence**: 6 unique layers applied 2x = 12 effective layers with 6 layers' params. Good for parameter-constrained models.
- **SwiGLU activation**: Better than relu-squared. Used by ChrisGoesGolfing autoresearch agent.
- **3x MLP expansion**: Standard in top submissions. More capacity in MLP.
- **Parallel Residuals**: Compute attention and MLP in parallel, sum into residual. Saves one LayerNorm per layer.
- **LeakyReLU-squared**: Element-wise square of LeakyReLU. Second-order expressiveness without params.
- **TTT (Test-Time Training)**: Model keeps learning during evaluation. Key to #2 and pending top scores.

### What We Already Have (rtx2070 branch)
- SmearGate (learned left-neighbor blending)
- CaseOps (4-case embedding)
- AsymLogit (asymmetric pos/neg softcap)
- EMA weight averaging
- U-Net skip connections (encoder-decoder with learned skip weights)
- GQA (8 heads, 4 KV)
- RoPE
- Muon optimizer
- resid_mix, attn_scale, mlp_scale per-block control

### What We Are Missing (high impact)
1. **BigramHash** — IMPLEMENTED THIS SESSION, testing now
2. **Cross-Sparse Attention (XSA)** — needs implementation
3. **Depth recurrence** — needs implementation
4. **SwiGLU activation** — needs implementation (replace relu-squared)
5. **3x MLP expansion** — needs MLP_MULT=3
6. **Partial RoPE** — needs implementation
7. **SP8192 tokenizer** — research needed
8. **TTT (test-time training)** — complex, needs research

## Environment Notes
- Python 3.14.5, PyTorch 2.12.0+cu130
- 65536 batch tokens OOMs on current PyTorch (previously worked)
- 32768 batch OOMs after ~10 steps
- 16384 batch runs stable at ~1.05s/step, ~1.8GB VRAM
- No screen/tmux available on desktoparch
- setsid works for detaching but stdout needs exec redirect in script

## Experiment Log

### Experiment 1: Baseline (16384 batch, no BigramHash)
- **Time:** 02:42 CEST
- **Config:** batch=16384, seed=42, 600s wallclock, CaseOps+AsymLogit+SmearGate on
- **Model:** 17,061,975 params
- **Speed:** ~1.03s/step
- **Steps:** ~582 in 600s
- **Loss at step 200:** 3.8183
- **VRAM:** ~1.8GB

### Experiment 2: BigramHash (16384 batch, hash_size=4096)
- **Time:** 02:58 CEST (running)
- **Config:** batch=16384, seed=42, 600s wallclock, CaseOps+AsymLogit+SmearGate+BigramHash on
- **Model:** 19,159,127 params (+2.1M from BigramHash 4096x512)
- **Speed:** ~1.05s/step
- **Status:** running, comparing to baseline

## Next Steps
1. Compare BigramHash results to baseline
2. Implement gradient checkpointing (patched, ready to test)
3. Test more layers (12, 13) with gradient checkpointing
4. Implement Cross-Sparse Attention
5. Try MLP_MULT=3 (3x expansion)
6. Research SP8192 tokenizer
