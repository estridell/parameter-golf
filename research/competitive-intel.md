# Parameter Golf Competitive Intel Report

**Compiled:** 2026-05-16
**Sources:** 1,614+ PRs analyzed (Field Guide), OpenAI official recap, CodeSOTA, participant writeups, GitHub discussions
**Competition:** March 18 - April 30, 2026 | 2,000+ submissions | 1,000+ participants | $1M RunPod credits

---

## Leaderboard Summary

### Confirmed Records (16MB Track, Neural Models)

| Rank | BPB | Author | Key Innovation | PR |
|------|-----|--------|---------------|-----|
| 1 | 1.0565 | codemath3000 | Calib32 Token-Only N-gram + AsymLogit Stack | latest |
| 2 | 1.0576 | simonbissonnette | Progressive Context Growth + Short-Doc Score-First TTT | #2014 |
| 3 | 1.0586 | andrewbaggio1 | Long-Context No-Q/V TTT + QK-Gain 5.25 | - |
| 4 | 1.0594 | alertcat | AWQ-Lite GPTQ + AsymLogit | - |
| 5 | 1.0611 | codemath3000 | BOS-Fixed SmearGate + LQER + SparseAttnGate | #1855 |
| 6 | 1.0634 | nprime06 | PR1736 + PolarNS + MIN_LR + SparseAttnGate | - |
| baseline | 1.2244 | OpenAI | 9-layer 512dim 1024vocab TiedEmbeddings 4 KV heads | - |

### Pending Claims (Open PRs, Unverified)

| BPB | Author | Technique |
|-----|--------|-----------|
| 0.8265 | ndokutovich | SLOT-24 + Pre-Quant AdamW TTT |
| 1.0600 | ndokutovich | Recur345 + Par7 + Pre-Quant TTT |
| 1.0736 | joshkmartinez | Pre-quant TTT + Parallel Residuals |

### Non-Record Track (Unlimited Compute)

| BPB | Author | Approach |
|-----|--------|----------|
| 1.1239 | CiprianFlorin-Ifrim | 1-Bit Quantization (106M params) |
| 1.1465 | agalimova | MDLM Text Diffusion |
| 1.1473 | mradassaad | Mamba-3 Hybrid SSM + SP8192 + Legal TTT |

### N-gram Approaches (Separate Category)

N-gram submissions achieved near-zero BPB by essentially memorizing the training corpus. These were recognized as technically valid but operated in a different league:

| BPB | Author | Technique | PR |
|-----|--------|-----------|-----|
| ~0 | hypery11 | Middle-Out Compression: Shannon Limit Broken | #721 |
| 0.00000035 | himanalot | Nacrith Log-Bias + Full-Rescore N-gram | #959 |
| 0.0109 | sofiabod | Packed Causal N-gram + Dirichlet Backoff | #1076 |

**Note:** Custom tokenizer submissions (e.g., Scylla) achieved ~0.9485 BPB, effectively creating a separate competition.

---

## 1. Architecture

### What Works Best at 16MB

**Dominant shape:** 9-11 layers, d_model=512, GQA 8/4 heads, MLP 3x expansion
- 899 of 1,162 parsed entries use vanilla Transformer backbone
- Sweet spot for 8xH100 training: enough depth for good representations, enough width for throughput

### Top Architecture Tricks (by adoption)

| Technique | Submissions | Impact | Description |
|-----------|-------------|--------|-------------|
| **BigramHash** | 583 | -0.001 to -0.008 BPB | Hash-based bigram features added to embeddings. Nearly free lunch. Top 5 neural submissions all use it. Hash trick maps 65K possible bigrams to 8K-16K entries. |
| **SmearGate** | 396 | Meaningful | Learned gate blending adjacent token representations. Cheap local operation before attention. `output[t] = gate[t] * input[t-1] + (1-gate[t]) * input[t]` |
| **XSA (Cross-Sparse Attention)** | 392 | Significant | Dominates 3 of top 6 leaderboard slots. Variants: all-layer, last-4, deepest-3. Reduces attention cost by skipping full context in some layers. |
| **U-Net Skip Connections** | 275 | Significant | Encoder-decoder skip connections. Late layers get direct access to early features (char-level + syntax + semantics simultaneously). |
| **Partial RoPE** | 270 | Moderate | Rotary embeddings on only first half of head dimensions. |
| **LN Scale** | 226 | Moderate | Per-layer learned scaling of layer norm outputs. |
| **Depth Recurrence** | ~50+ | Mixed (see below) | Repeat same layers multiple times. 5 unique layers × 2 = 10 effective layers. |

### Depth Recurrence: The Nuanced Picture

**What works:** 5×2 or 4×2 configs with FiLM conditioning (tiny per-iteration scale/shift). One participant got 1.1634 BPB with 8 unique blocks × 2 loops + FiLM + BigramHash + TrigramHash.

**What fails:**
- **Quantization amplification** — the killer nobody predicted. Shared blocks amplify quantization error through repeats: baseline gap 0.031 → 3×3 gap 0.039-0.044 → 2×6 gap 0.097. At 6 repeats, catastrophic.
- **More effective layers = fewer training steps.** 4×3@d=768 gets only 126 steps vs 334 for baseline (2.7x slower per step).
- **TTT conflicts with recurrence.** Updating shared block weights via SGD changes behavior in ALL loop iterations. Gradients compound through recurrence in ways they don't in standard architectures.
- **EMA disaster with short-run recurrence.** EMA averages weights from early poorly-converged states. With only ~4,000 steps, shadow weights contaminated by garbage from step 1.

**Key finding:** Unique capacity matters more than loop depth. 5×2 (10 effective layers) beat 4×3 (12 effective layers) despite fewer total block applications.

**Fix:** QAT is essential for recurrence. QAT_FRACTION=0.15 might reduce the 0.044 gap to ~0.01. 3×3@d=768 + QAT is the most promising config.

### Other Notable Architecture Ideas

| Technique | PR | Description |
|-----------|-----|-------------|
| **CaseOps** | #1729 | Lossless bijective case transform. Capitalization operator tokens with original-byte BPB sidecar accounting. |
| **SmearGate + BigramHash** | #65 | Learned previous-token embedding blend + adjacent-token-pair hash features. |
| **Mini Depth Recurrence** | #1204 | Repeat layers 4&5, delay recurrence until mid-training, partially untie repeated MLPs. First accepted submission making recurrent layers work. |
| **Parallel Residuals** | Various | Two-lane residual routing. |
| **QK-Gain** | Various | Attention scaling factor (5.25 optimal). |
| **LeakyReLU²** | Various | LeakyReLU squared activation — adopted by multiple top submissions. |

---

## 2. Compression / Quantization

### The Quantization Stack (Ranked by Impact)

| Method | Bits/Param | Best For | Usage |
|--------|-----------|----------|-------|
| **GPTQ** | 4-6 | Post-training, strong compression | Widespread in top entries |
| **GPTQ-lite** | 6 | Simpler GPTQ variant | PR #414 first successful submission |
| **AWQ-Lite** | 5-6 | Activation-aware weight quantization | Combined with GPTQ in top entries |
| **Int6 + zstd** | 6 | Default sweet spot | Most common combination |
| **Int5/Int6 mixed** | 5-6 | Aggressive compression | MLP int5, attention int6 |
| **Int4** | 4 | Maximum compression | Risky — quality drops fast |
| **QAT (STE int6)** | 6 | Quantization-aware training | 52 submissions, helps with depth recurrence |
| **Ternary {-1,0,1}** | ~1.5 | Extreme compression | 106M params in 16MB. CiprianFlorin-Ifrim: 1.1239 BPB |
| **LQER** | Various | Low-quality error recovery | Top-1 correction tensor to reduce artifact size |

### Self-Generated GPTQ Calibration (Breakthrough)

PR #1019 (abaybektursun): Generate calibration text FROM the trained model, build Hessians from those activations. This was the key innovation that put abaybektursun at #1 (1.1147 BPB). Creative calibration that doesn't need external data.

### Compression Algorithms (Post-Quantization)

| Algorithm | Submissions | Ratio | Speed | Notes |
|-----------|-------------|-------|-------|-------|
| **zstd-22** | 501 | 1.2-1.5x | Very fast | **Winner.** 331+ use max level 22. |
| zlib | 411 | 1.15-1.4x | Fast | Zero dependencies |
| lzma | 248 | 1.3-1.6x | Slow | Best ratio but risky for eval time limits |
| brotli | 11 | 1.2-1.5x | Fast | External dependency |

**Key insight:** Quantized weights compress well because neural network weights follow bell-shaped distributions centered near zero. ~80% of weights fall in range [-30, 30] out of [-128, 127], giving ~6 bits effective entropy (~25% compressible).

**Practical impact:** 10M params in int6 = 7.5 MB raw → ~5.5 MB with zstd-22. Savings of 2 MB = room for ~2.7M more int6 parameters → 0.01-0.03 BPB improvement for free.

### Weight Ordering for Compression

Sort weight rows by L2 norm before serialization. Similar rows become adjacent, creating longer runs of similar byte patterns. Store permutation index to restore at load time.

### Pruning + Compression Synergy

52 submissions use magnitude pruning. Setting small weights to zero dramatically improves compressibility. 5% pruning → ~10% better compression while losing only ~0.001 BPB.

---

## 3. Training

### Optimizer: Muon Dominates

**84.6% of submissions use Muon** (623 of 736 reporting optimizer).

Muon = Matrix Updates via Orthogonalization for Neural networks. Core insight: Adam treats weight matrices as "bags of numbers" (element-wise scaling distorts spectral structure). Muon treats them as matrices with spectral structure.

**Algorithm:**
1. Compute gradient G for weight matrix W
2. Apply momentum → combined direction M
3. Orthogonalize M via Newton-Schulz iterations (5 steps, O(n²) only)
4. Use orthogonalized direction as update

**Standard setup:** Muon for weight matrices, Adam for embeddings/biases/layer norm (Parallel Muon, 82 submissions).

### Learning Rate Schedule: Warmdown Wins

**411 submissions use warmdown.** Magic number: **3,500 warmdown steps.**

- Training runs are ~5,000-7,000 steps total
- Warmdown = hold peak LR for first half, linear decay to zero over final 3,500 steps
- Cosine decay (70 submissions) starts reducing immediately — disadvantage for short runs
- **Warmdown + weight averaging** = complementary pair. 8 of top 10 neural submissions use both.

**Peak LR:** 0.025 for matrices, 0.035 for tied embeddings.

### Weight Averaging: Free Lunch

| Method | Submissions | Decay/Interval | Notes |
|--------|-------------|---------------|-------|
| **EMA** | 353 | decay=0.997 (87% consensus) | Effective window ~333 steps |
| **SWA** | 313 | every 50 steps | Equal-weight average of final-phase checkpoints |
| **Both** | 181 | - | Combination used by many top entries |

**Warning:** EMA + depth recurrence = disaster (short training runs contaminate shadow weights with early garbage).

### Initialization: OrthoInit

171 submissions use OrthoInit. QR decomposition to get orthogonal matrix, scaled by `1/sqrt(num_layers)` for output projections. Prevents directional bias from wasting precious early training steps.

Alternative: Spectral initialization (18), Resid Mix (15, starts model as identity function).

### Data & Curriculum

- FineWeb dataset, SP8192 tokenizer most common
- Batch tokens: 65536 max practical on 8xH100
- No curriculum learning reported as major differentiator
- TrigramHash: big win when combined with dedicated embedding table. PR #571 failed on 1xH100 but succeeded on 8xH100 — suggests trigram features interact favorably with higher-step training regimes.

---

## 4. Tokenization

### SP8192 is Standard

- SP1024 (1,024 vocab) used by baseline
- SP8192 (8,192 vocab) adopted by most competitive submissions
- SP4096 → SP8192 transition revealed: architectural choices validated at smaller vocab **flip sign** at larger vocab due to embedding-table parameter cost dominating allocation choices

### CaseOps (PR #1729)

Lossless bijective capitalization operator. Instead of treating "Hello" and "hello" as separate tokens, CaseOps encodes the capitalization pattern as a separate operator token. Preserves all case information while reducing effective vocabulary pressure.

### Custom Tokenizers

Custom tokenizers (e.g., Scylla) achieved ~0.9485 BPB, effectively creating a separate competition category. Two frontiers emerged: standard-tokenizer submissions topped out around 1.0979, while custom tokenizers dominated overall.

### BigramHash / TrigramHash

Not tokenizers per se, but lexical features added to embeddings:
- BigramHash(20,480): standard, top 5 neural submissions all use it
- TrigramHash(8,192): 9.4x better return-per-MB than bigram scaling. Requires dedicated embedding table (shared table preserves only ~25% of gain).

---

## 5. Evaluation Tricks

### Sliding Window (Stride-Based) Eval

**77% of submissions use stride=64.** Only score tokens with full context (~960 tokens of prior context). Throws away predictions from tokens that lacked sufficient context.

| Stride | Overlap | Compute | Quality |
|--------|---------|---------|---------|
| 1024 (naive) | 0% | 1x | Worst |
| 256 | 75% | ~4x | Good |
| **64** | **94%** | **~16x** | **Excellent** |
| 32 | 97% | ~32x | Marginal gain |

### N-gram Mixing (PR #524)

Blend neural model predictions with bigram frequencies at eval time:
```python
final_probs = 0.93 * neural_probs + 0.07 * bigram_probs
```
Zero artifact bytes — frequencies computed from evaluation data. Temperature calibration (T=0.93) sharpens distribution.

### PPM (Prediction by Partial Matching)

Classical compression algorithm used as eval-time ensemble member. Builds Markov model of character sequences at test time. Blends across multiple context lengths (k=15 max order, k_values=[16, 12, 8, 6]).

### Score-First TTT (Test-Time Training)

The dominant eval innovation. Score chunk → record loss → train on same chunk → move to next. Legal because you predict before seeing the answer.

**LoRA TTT:** Only train low-rank adapters (rank 4-8, ~50-100K params). Reset per document. 3-5 epochs per chunk, lr=0.01. Less forgetting risk than full TTT.

**Full TTT:** Train ~81% of model params. 30 epochs per chunk. Freeze early layers (2 blocks). Higher compute cost but larger adaptation.

### AsymLogit

Asymmetric logit scaling — appears in top 4 submissions. Needs more investigation but seems to handle tail distributions better.

---

## 6. What Failed

### Depth Recurrence Without QAT
Quantization amplification kills gains. 2×6 recurrence: quant gap 0.097 (vs baseline 0.031). DO NOT use without QAT_FRACTION=0.15+.

### EMA + Short Training Runs + Recurrence
EMA averages weights from early poorly-converged states. With ~4,000 steps, shadow weights contaminated. "Short-run recurrence + EMA = disaster."

### TTT + Depth Recurrence
Updating shared block weights via SGD changes behavior in ALL loop iterations. Gradients compound through recurrence. Fundamental architectural conflict.

### Narrow-Deep SSMs (d=384, 12-16 layers)
Mamba-3 kernels have ~2-3ms fixed per-call overhead that doesn't scale with dim. Narrow-deep is worse despite similar params.

### Aggressive Quantization on SSMs
MLP INT5 (+8 mBPB regression), base INT5 (+17.5 mBPB). SSM weight matrices have less quantization headroom than larger-scale literature suggests.

### Low-Rank Factorization of SSM in_proj
At rank=128 with random init: +26 mBPB regression. Mamba-3 initializes `dd_A` rows as log decay rates for multi-scale temporal tracking. Random factored product carries none of that structure.

### Window TTT on SSMs
Gradient signal too weak per window.

### Truncated BPTT with Persistent SSM State
Document-stream-specific state unrecoverable at eval.

### GPTQ-lite on Already-Optimized Distributions
Standard max-based clipping was already optimal for some weight distributions. Marginal or negative improvement.

### Wider Models (d=640/768 for SSMs)
MLP cost scales with dim², erasing SSM throughput advantage. Bigger model = fewer training steps in the 10-minute window.

---

## 7. Gaps / Opportunities

### Untested or Underexplored

1. **QAT + Depth Recurrence** — The golfsummit.dev analysis shows this is the most promising combination but nobody has shipped it in a record submission yet. 3×3@d=768 + QAT_FRACTION=0.15.

2. **SVD-initialized low-rank factorization** — Random init failed for SSM in_proj, but SVD initialization of upstream dense weight remains untested.

3. **Megakernels** — OpenAI explicitly asked for 4-hour megakernels. No submissions in this category.

4. **E2E TTT** — End-to-end test-time training (not just score-first). OpenAI flagged this as wanted.

5. **Super long context for eval/training** — SP8192 is standard but nobody has pushed much beyond that for training context length.

6. **Learning adapters on random linear maps** — OpenAI requested this. No submissions.

7. **Byte-level models** — H-Net (DariusFeher, nonrecord #3) is the only byte-level submission. Could be pushed further.

8. **JEPA** — CiprianFlorin-Ifrim's nonrecord #1 combined SSM + JEPA. Interesting but unoptimized.

9. **Text Diffusion** — agalimova's MDLM (1.1465 BPB) shows diffusion models can compete at this scale. Underexplored.

10. **State-space models** — mradassaad's Mamba-3 hybrid achieved 1.1473 BPB (best SSM). Structural gap of ~30 mBPB appears paradigm-level. But: LZMA compression penalty on SSM weights is a tooling issue, not fundamental. Fixing AR GPTQ fallback could close ~5.5 mBPB.

### Technique Combinations Not Yet Tried

- Ternary quantization + TTT (ternary models haven't been paired with test-time training)
- CaseOps + aggressive quantization (custom tokenizer + GPTQ)
- MoE at 16MB scale (PR #1451 showed 1.1179 BPB with MoE + BigramHash4096 — competitive but not dominant)
- Cross-sparse attention + depth recurrence with QAT
- TrigramHash + LoRA-TTT (both are proven individually)

### Known Engineering Debt

- SSM submissions: AR GPTQ fallback gap of ~5.5 mBPB
- Custom tokenizer submissions create a separate competition — rules may need updating
- N-gram approaches are essentially compression, not language modeling — already separated

---

## 8. Meta-Observations

### AI Agent Usage

- Majority of submissions used AI coding agents (Claude, Codex, etc.)
- One participant with zero ML knowledge achieved 1.1634 BPB using multi-agent workflow
- Agents lowered barrier to entry but also generated noise (hundreds of small variations of top scorers)
- OpenAI couldn't manually inspect every submission at peak (hundreds per day)

### Technique Evolution Timeline

1. **Weeks 1-2:** Baseline optimization, Muon adoption, basic quantization (int8)
2. **Weeks 3-4:** GPTQ variants, BigramHash, SmearGate, EMA/SWA
3. **Weeks 5-6:** TTT (score-first, LoRA), XSA, U-Net skips, depth recurrence
4. **Weeks 7-8:** Self-Gen GPTQ calibration, CaseOps, n-gram approaches, custom tokenizers
5. **Final:** Stacking all techniques (the record holder uses SmearGate + XSA + Partial RoPE + depth recurrence + GPTQ + LQER + AWQ-lite + LoRA TTT + EMA + Muon)

### Key Takeaway

The competition evolved from "train a good small model" to "stack 15+ micro-optimizations." The winners didn't have one big idea — they systematically combined every proven technique. The final record (1.0565 BPB) uses: Muon optimizer, OrthoInit, warmdown LR, EMA/SWA, BigramHash, SmearGate, XSA, Partial RoPE, GPTQ + LQER + AWQ-lite quantization, zstd-22 compression, stride-64 eval, n-gram mixing, score-first TTT with LoRA, and CaseOps tokenization.

---

## Sources

- [OpenAI Official Recap](https://openai.com/index/what-parameter-golf-taught-us/)
- [Parameter Golf Field Guide](https://sameersegal.github.io/learn-parameter-golf/) (1,614 PRs analyzed)
- [CodeSOTA Leaderboard](https://www.codesota.com/parameter-golf)
- [GitHub Repository](https://github.com/openai/parameter-golf)
- [Participant Writeup: Depth Recurrence + FiLM](https://namspdr.substack.com/p/i-entered-openais-parameter-golf)
- [Why SSMs Struggle](https://mradassaad.github.io/posts/why-ssms-struggle-in-parameter-golf/)
- [Depth Recurrence Testing](https://golfsummit.dev/post/54d70bbf-92b3-4fba-9c48-5beef7c3378d)
- [Research Garden](https://golf.agustif.com/)
- [Field Guide: Test-Time Training](https://sameersegal.github.io/learn-parameter-golf/learn/test-time-training)
- [Field Guide: Quantization](https://sameersegal.github.io/learn-parameter-golf/learn/quantization-fundamentals)
- [Field Guide: Optimizers](https://sameersegal.github.io/learn-parameter-golf/learn/optimizers)
- [Field Guide: Architecture Tricks](https://sameersegal.github.io/learn-parameter-golf/learn/architecture-tricks)
- [Field Guide: Compression](https://sameersegal.github.io/learn-parameter-golf/learn/compression)
- [Field Guide: Weight Averaging](https://sameersegal.github.io/learn-parameter-golf/learn/weight-averaging)
- [Field Guide: Evaluation Strategies](https://sameersegal.github.io/learn-parameter-golf/learn/evaluation-strategies)
- [Field Guide: Learning Rate Schedules](https://sameersegal.github.io/learn-parameter-golf/learn/learning-rate-schedules)
- [Field Guide: Initialization](https://sameersegal.github.io/learn-parameter-golf/learn/initialization)
