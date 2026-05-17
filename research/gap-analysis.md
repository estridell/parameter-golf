# Gap Analysis: train_gpt.py vs Winning Parameter Golf Stack

**Date:** 2026-05-16
**Source:** `train_gpt.py` (1127 lines, rtx2070 branch)
**SOTA reference:** ~1.061 BPB (8×H100, 10min/16MB track, April 2026)
**Records dir:** `~/projects/parameter-golf/records/track_10min_16mb/`

---

## Summary Table

| # | Technique | Status | Effort | Impact | Priority |
|---|-----------|--------|--------|--------|----------|
| 1 | Muon optimizer | ✅ IMPLEMENTED (L112-165) | — | — | DONE |
| 2 | Warmdown LR schedule | ✅ IMPLEMENTED (L830-837) | — | — | DONE |
| 3 | U-Net skip connections | ✅ IMPLEMENTED (L669-713) | — | — | DONE |
| 4 | QK-Gain (attention scaling) | ✅ IMPLEMENTED (L580, L600) | — | — | DONE |
| 5 | Logit softcap | ✅ IMPLEMENTED (L715-718) | — | — | DONE |
| 6 | ReLU² MLP | ✅ IMPLEMENTED (L544-547) | — | — | DONE |
| 7 | GQA (grouped query attn) | ✅ IMPLEMENTED (L566-571) | — | — | DONE |
| 8 | SmearGate | ❌ NOT IMPLEMENTED | MEDIUM | HIGH | **1** |
| 9 | SP8192 tokenizer | ❌ NOT IMPLEMENTED | HARD | HIGH | **2** |
| 10 | Score-first TTT | ❌ NOT IMPLEMENTED | HARD | HIGH | **3** |
| 11 | CaseOps | ❌ NOT IMPLEMENTED | EASY | MEDIUM | **4** |
| 12 | AsymLogit | ❌ NOT IMPLEMENTED | EASY | MEDIUM | **5** |
| 13 | LQER quantization | ❌ NOT IMPLEMENTED | HARD | MEDIUM | 6 |
| 14 | EMA/SWA weight averaging | ❌ NOT IMPLEMENTED | EASY | MEDIUM | 7 |
| 15 | Parallel residuals | ❌ NOT IMPLEMENTED | MEDIUM | MEDIUM | 8 |
| 16 | Partial RoPE | ⚠️ PARTIAL | EASY | LOW | 9 |
| 17 | Depth recurrence | ❌ NOT IMPLEMENTED | HARD | MEDIUM | 10 |
| 18 | OrthoInit | ❌ NOT IMPLEMENTED | EASY | LOW | 11 |
| 19 | BigramHash | ❌ NOT IMPLEMENTED | MEDIUM | LOW | 12 |
| 20 | TrigramHash | ❌ NOT IMPLEMENTED | MEDIUM | LOW | 13 |
| 21 | GPTQ quantization | ❌ NOT IMPLEMENTED | HARD | MEDIUM | 14 |
| 22 | AWQ-lite quantization | ❌ NOT IMPLEMENTED | HARD | LOW | 15 |
| 23 | zstd-22 compression | ❌ NOT IMPLEMENTED | EASY | LOW | 16 |
| 24 | Stride-64 eval | ❌ NOT IMPLEMENTED | EASY | LOW | 17 |
| 25 | N-gram mixing (eval) | ❌ NOT IMPLEMENTED | MEDIUM | LOW | 18 |
| 26 | LeakyReLU² activation | ❌ NOT IMPLEMENTED | EASY | LOW | 19 |
| 27 | XSA (cross-sparse attn) | ❌ NOT IMPLEMENTED | HARD | LOW | 20 |

---

## Detailed Notes

### ✅ ALREADY IMPLEMENTED

**1. Muon optimizer** (L112-165)
Full Newton-Schulz orthogonalization with Nesterov momentum. Used for all 2D matrix params in transformer blocks. Momentum warmup from 0.85→0.95 over 500 steps.

**2. Warmdown LR schedule** (L830-837)
`lr_mul()` supports both step-based and wallclock-based warmdown. Linear decay over `WARMDOWN_ITERS` (default 1200).

**3. U-Net skip connections** (L669-713)
Encoder-decoder split: first half of layers stores activations, second half consumes them in reverse with learnable `skip_weights`. This is a significant architectural feature already in place.

**4. QK-Gain** (L580, L600)
Per-head learnable `q_gain` parameter (init=1.5) applied to queries after RoPE. Controlled via `QK_GAIN_INIT` env var.

**5. Logit softcap** (L715-718)
`logit_softcap=30.0` — tanh-based logit capping before cross-entropy.

**6. ReLU² MLP** (L544-547)
`torch.relu(fc(x)).square()` — squared ReLU activation, which is the modded-nanogpt standard.

**7. GQA** (L566-571)
8 query heads, 4 KV heads. Standard grouped query attention with `enable_gqa=True` in SDPA.

---

### ❌ NOT IMPLEMENTED (prioritized)

**8. SmearGate** — Priority 1, MEDIUM effort, HIGH impact
Adjacent token blending: before attention, blend each token with its left neighbor via a learned gate. This is the single most impactful missing technique — every competitive submission since mid-March uses it. Typical implementation: ~30-50 lines. Add a `SmearGate` module that does `x_blend = gate * x + (1 - gate) * shift_left(x)` before Q/K projection.

**9. SP8192 tokenizer** — Priority 2, HARD effort, HIGH impact
Currently using SP1024 (vocab_size=1024). Winners use SP8192. This requires:
- Training a new SentencePiece model with 8192 vocab
- Regenerating all training/validation shards with the new tokenizer
- Updating VOCAB_SIZE=8192 and all related hyperparams
- Re-tuning embedding LR, tied_embed_init_std, etc.
The larger vocab significantly improves BPB because it reduces the bits-per-byte overhead. Impact: likely 0.02-0.05 BPB improvement.

**10. Score-first TTT** — Priority 3, HARD effort, HIGH impact
Test-time training: during eval, do a few gradient steps on the eval data before scoring. "Score-first" means score, then train, then score again — the first score is the reported metric. This requires modifying eval_val() to support optional gradient steps. Impact: 0.005-0.02 BPB based on records.

**11. CaseOps** — Priority 4, EASY effort, MEDIUM impact
Capitalization operator: encode case information (all caps, title case, lowercase, mixed) as a separate feature channel. Typically adds a small per-token scalar or one-hot feature. ~15 lines. Records show consistent improvement.

**12. AsymLogit** — Priority 5, EASY effort, MEDIUM impact
Asymmetric logit scaling: apply different scaling to positive vs negative logits before softmax. ~10-15 lines. Used in PR #1787 base and subsequent submissions.

**13. LQER (Low-Quality Error Recovery)** — Priority 6, HARD effort, MEDIUM impact
Post-training quantization technique that recovers quality lost during int8 quantization. Requires implementing a second pass that identifies and compensates for high-error weights. Used in the 1.061 SOTA submission.

**14. EMA/SWA weight averaging** — Priority 7, EASY effort, MEDIUM impact
Exponential or stochastic weight averaging of model checkpoints during training. ~20 lines: maintain a running average of weights, use the averaged weights for final eval. Typical: EMA with decay 0.99 or SWA over last N checkpoints.

**15. Parallel residuals** — Priority 8, MEDIUM effort, MEDIUM impact
Run attention and MLP in parallel instead of sequentially: `x = x + attn(norm(x)) + mlp(norm(x))`. Requires restructuring Block.forward() and potentially re-tuning scales. Used in April 4+ records.

**16. Partial RoPE** — Priority 9, EASY effort, LOW impact
Currently RoPE is applied to the full head_dim. Partial RoPE applies rotary embeddings to only the first half of head dimensions, leaving the other half as position-independent. Change in `apply_rotary_emb`: only rotate `x[..., :dim//4]` and `x[..., dim//4:dim//2]`. ~5 lines.

**17. Depth recurrence** — Priority 10, HARD effort, MEDIUM impact
Share weights across layers (e.g., 3 unique layers repeated 3x = 9 effective layers). Reduces parameter count significantly, allowing wider or deeper models within the budget. Requires architectural changes to GPT.__init__.

**18. OrthoInit** — Priority 11, EASY effort, LOW impact
Initialize weight matrices as orthogonal matrices. ~10 lines: replace `_init_weights` with orthogonal init for 2D params. Minor improvement over default init.

**19. BigramHash** — Priority 12, MEDIUM effort, LOW impact
Add bigram feature hash to embeddings: for each position, hash (prev_token, curr_token) into a small lookup table and add to the embedding. ~30 lines. Low standalone impact.

**20. TrigramHash** — Priority 13, MEDIUM effort, LOW impact
Same as BigramHash but with trigrams. Even lower standalone impact, but stacks with BigramHash.

**21. GPTQ quantization** — Priority 14, HARD effort, MEDIUM impact
Higher-quality post-training quantization than current per-row int8. Requires implementing GPTQ algorithm (Hessian-based weight rounding). Significant code, but the current int8 scheme is already decent.

**22. AWQ-lite quantization** — Priority 15, HARD effort, LOW impact
Activation-aware weight quantization. Similar complexity to GPTQ, different approach. Marginal over GPTQ for this model size.

**23. zstd-22 compression** — Priority 16, EASY effort, LOW impact
Replace `zlib.compress(quant_raw, level=9)` with `zstd.compress(quant_raw, level=22)`. ~3 lines + `pip install zstandard`. Saves ~5% on artifact size (tested: 6.21MB vs 6.52MB). Low BPB impact since it's artifact-only.

**24. Stride-64 eval** — Priority 17, EASY effort, LOW impact
Use overlapping sequences with stride=64 during validation instead of non-overlapping chunks. Better utilization of eval data. ~20 lines in eval_val().

**25. N-gram mixing at eval** — Priority 18, MEDIUM effort, LOW impact
Blend model predictions with n-gram statistics during eval. Requires precomputing n-gram counts from training data. Low impact for this model size.

**26. LeakyReLU² activation** — Priority 19, EASY effort, LOW impact
Replace `torch.relu(x).square()` with `F.leaky_relu(x, 0.01).square()`. ~1 line change. Very marginal improvement.

**27. XSA (Cross-Sparse Attention)** — Priority 20, HARD effort, LOW impact
Sparse attention pattern where each head attends to a different subset of positions. Requires custom attention kernel or masking logic. Complex to implement, marginal benefit over dense attention at this scale.

---

## Recommended Implementation Order (Top 5)

### 1. SmearGate (EASY-MEDIUM, HIGH impact)
The single most impactful missing piece. Every competitive submission uses it. Implement first.
```
Effort: ~40 lines
Impact: ~0.01-0.03 BPB
Dependencies: None
```

### 2. CaseOps + AsymLogit (EASY, MEDIUM impact)
Both are small, independent changes that stack well. Implement together.
```
Effort: ~30 lines total
Impact: ~0.005-0.01 BPB combined
Dependencies: None (can be done in parallel)
```

### 3. EMA/SWA weight averaging (EASY, MEDIUM impact)
Simple to implement, consistent improvement across submissions.
```
Effort: ~25 lines
Impact: ~0.002-0.005 BPB
Dependencies: None
```

### 4. SP8192 tokenizer (HARD, HIGH impact)
Highest single-impact change but requires data pipeline work. Do after the quick wins.
```
Effort: Data pipeline + retuning
Impact: ~0.02-0.05 BPB
Dependencies: None (but everything needs retuning after)
```

### 5. Score-first TTT (HARD, HIGH impact)
Significant eval-time improvement. Implement after tokenizer upgrade since it affects eval.
```
Effort: ~80 lines
Impact: ~0.005-0.02 BPB
Dependencies: Benefits from SP8192 but not required
```

---

## Combinability Notes

**Can be combined (independent, no interaction):**
- SmearGate + CaseOps + AsymLogit — all modify different parts of the forward pass
- OrthoInit + EMA/SWA — init and averaging are orthogonal
- zstd-22 + any quantization change — compression is independent of quantization scheme

**Must be sequential:**
- SP8192 → everything else (retokenizing invalidates all hyperparams, need to retune)
- Parallel residuals → depth recurrence (both restructure the Block, can't do both at once)
- LQER/GPTQ → zstd-22 (quantization scheme affects compression ratio)

**Stacking order for maximum BPB:**
1. SmearGate + CaseOps + AsymLogit (quick wins, ~1 hour)
2. EMA/SWA + OrthoInit (init + averaging, ~30 min)
3. SP8192 (tokenizer upgrade, ~2-4 hours including data prep)
4. Score-first TTT (eval enhancement, ~1-2 hours)
5. LQER or GPTQ (quantization upgrade, ~2-3 hours)

**Note on RTX 2070 constraints:**
- SP8192 will increase embedding size significantly (1024→8192 vocab). With tied embeddings and dim=512, embedding goes from 0.5M to 4M params. Still fits in 8GB but leaves less room for model width/depth.
- TTT requires gradient computation during eval — increases eval memory usage. May need smaller eval batch on 8GB.
- Depth recurrence reduces param count, which could allow wider model to fit in VRAM — synergistic with SP8192.
