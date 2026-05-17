# Overnight Session Summary — May 17, 2026 (02:28 - 05:30 CEST)

## What Was Done

### Research
- Analyzed the full parameter-golf leaderboard (1,500+ submissions)
- Identified top techniques: BigramHash (583 submissions), XSA (392), U-Net (275), Partial RoPE (270)
- Documented confirmed SOTA (1.1147 BPB) and pending claims (0.8265 BPB)
- Researched SP8192 tokenizer (0.02 BPB gain, data not available in HF repo)
- Researched TTT (test-time training) — complex eval-time technique, key to top scores

### Implementations
1. **BigramHash** — hash adjacent token pairs into embedding table (~30 lines)
   - Toggled via BIGRAM_HASH_ENABLED env var
   - Configurable hash size (default 4096, tested 5120)
2. **Gradient checkpointing** — recompute activations in backward pass
   - Toggled via GRADIENT_CHECKPOINT_ENABLED env var
3. **Skip final validation** when VAL_LOSS_EVERY=0 (saves ~15 min/run)
4. **Skip roundtrip validation** when VAL_LOSS_EVERY=0

### Experiments (12 configurations tested)
- 9L baseline, 9L BigramHash 4096, 9L BigramHash 5120
- 12L BigramHash, 12L gradient checkpoint, 12L MLP3x
- Partial RoPE 0.5
- All at seed=42, 16384 batch, 300s wallclock

### Key Findings
1. BigramHash: 7-8% lower loss, zero speed overhead (CLEAR WINNER)
2. 12 layers is WORSE than 9L+BigramHash within 300s budget
3. Gradient checkpointing doesn't help at 16384 batch
4. MLP 3x is slower with no benefit
5. Partial RoPE 0.5 shows no meaningful improvement
6. PyTorch 2.12.0 broke 65536 batch size (now OOMs)

### Commits
- bb9c9fa: feat: add BigramHash + gradient checkpointing
- 7682dbb: fix: skip final validation when VAL_LOSS_EVERY=0
- d648944: fix: skip roundtrip validation when VAL_LOSS_EVERY=0
- 383fc52: fix: indentation error
- fe35f60: docs: comprehensive experiment results
- 9994268: docs: Partial RoPE experiment

### Recommended Default Stack (for future runs)
- 9 layers, BigramHash 5120, SmearGate, CaseOps, AsymLogit
- GQA 8/4, Muon optimizer, model_dim=512
- Estimated H100 step time: ~12ms (85x ratio from 2070)
- Estimated H100 steps in 10 min: ~49,500 (far above budget)
- Artifact: ~11MB int8+zlib (69% of 16MB budget)

### Next Steps for Future Sessions
1. Try SwiGLU activation (replace relu-squared)
2. Implement Cross-Sparse Attention (XSA) — 392 leaderboard submissions use it
3. Try depth recurrence (6 unique layers x2 = 12 effective layers)
4. Test on H100 via cloud GPU (RunPod/Lambda) for actual BPB measurement
5. Implement test-time training (TTT) for eval-time improvement
6. Generate SP8192 tokenizer from scratch (train SentencePiece on FineWeb)
