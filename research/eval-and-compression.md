# Eval Pipeline & Compression Techniques Research

**Date:** 2026-05-16
**Purpose:** Catalog eval-time optimizations and compression techniques from Parameter Golf winners. These are "free" BPB improvements that don't require model changes.

---

## 1. Sliding Window Eval (stride=64)

### What It Does
Instead of chopping the validation set into non-overlapping 1024-token chunks (where early tokens get zero context), use overlapping windows advanced by 64 tokens. Only score the last 64 tokens per window — the ones with ~960 tokens of prior context.

### Why It Works
The naive eval averages over all 1024 tokens per chunk. Token 0 has zero context (blind guess), token 512 has 512 tokens, token 1023 has 1023. The poorly-contextualized early tokens drag the average BPB up. With stride=64, every scored token gets near-maximum context.

### Impact
- **~0.034 BPB improvement** with zero artifact cost
- 77% of submissions use this (333/430 specifying stride)
- Eval time increases ~16x (from ~16s to ~70s on 8xH100)

### Pseudocode
```python
def sliding_window_eval(model, tokens, seq_len=1024, stride=64):
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(tokens) - 1, stride):
        end = min(start + seq_len, len(tokens))
        input_ids = tokens[start:end].unsqueeze(0)
        logits = model(input_ids)
        # Only score the last 'stride' tokens (first window scores all)
        score_start = max(0, seq_len - stride) if start > 0 else 0
        score_logits = logits[0, score_start:-1]
        score_targets = input_ids[0, score_start + 1:]
        loss = F.cross_entropy(score_logits, score_targets, reduction='sum')
        total_loss += loss.item()
        total_tokens += score_targets.numel()
    return total_loss / total_tokens
```

### Integration Point
Replace the current `eval_val()` function (line 219 in train_gpt.py). Current code uses non-overlapping chunks (`local_batch_seqs * train_seq_len` stride). Need to add `EVAL_STRIDE` env var and modify the loop to advance by `stride` instead of `seq_len`.

### Stride Trade-offs
| Stride | Overlap | Compute | Quality |
|--------|---------|---------|---------|
| 1024 | 0% | 1x | Worst |
| 256 | 75% | ~4x | Good |
| **64** | **94%** | **~16x** | **Excellent** |
| 32 | 97% | ~32x | Marginal gain over 64 |

### Dependencies
None. Pure eval-time change. Can be done first, independently.

### Effort
**Low.** ~30 lines of code changes. Add `EVAL_STRIDE` env var, modify eval loop.

---

## 2. N-gram Mixing at Eval Time

### What It Does
Blend neural model predictions with bigram frequencies computed from eval data:
```python
final_probs = 0.93 * neural_probs + 0.07 * bigram_probs
```

### Why It Works
Bigram acts as a smoothing prior — catches cases where neural model assigns near-zero probability to a common bigram, preventing catastrophic log-loss spikes. Zero artifact bytes because frequencies are computed from eval data itself.

### Impact
- Small but consistent improvement (~0.005-0.01 BPB)
- Zero artifact cost
- Used in competitive submissions as a free boost

### Pseudocode
```python
# Precompute bigram frequencies from eval data (before scoring)
bigram_counts = defaultdict(lambda: defaultdict(int))
for i in range(1, len(eval_tokens)):
    bigram_counts[eval_tokens[i-1]][eval_tokens[i]] += 1

# At eval time, blend predictions
def blend_with_bigram(neural_logits, prev_token, alpha=0.93):
    neural_probs = softmax(neural_logits / alpha)  # temperature calibration
    bigram_probs = get_bigram_distribution(prev_token, bigram_counts)
    return alpha * neural_probs + (1 - alpha) * bigram_probs
```

### Temperature Calibration
The alpha parameter (0.93) serves dual duty: it's both the blend weight and the temperature. Dividing logits by 0.93 before softmax calibrates the neural distribution.

### Integration Point
In `eval_val()`, after computing logits, blend with bigram distribution before computing cross-entropy. Need to build bigram table from eval tokens at eval start.

### Legality
**Track A (Fixed Predictor):** This is borderline. The bigram stats come from eval data, but they're computed before any scoring happens. The Field Guide notes this is "zero artifact bytes (frequencies computed from eval data)" — seems accepted in practice.

**Track B (Adaptive):** Clearly legal — can build bigram stats from already-scored tokens.

### Dependencies
None. Independent of sliding window and other techniques.

### Effort
**Low.** ~20 lines of code. Build bigram table, modify softmax blending.

---

## 3. Score-First TTT (Test-Time Training)

### What It Does
Adapt the model during evaluation using backward-looking context:
1. Score a chunk of tokens (record loss)
2. Train on that same chunk (update model weights)
3. Move to next chunk

### Why It Works
The model adapts to domain-specific patterns in the eval data. Since you score BEFORE training (score-first), it's legal — you predict before seeing the answer.

### Two Variants

#### LoRA TTT (Lightweight)
- Only train low-rank adapters (rank 4-8, ~50-100K params)
- ~0.002-0.003 BPB improvement
- Faster, less risk of overfitting
- Used in PR #549 (1.1194 BPB, rank 2 SOTA)

#### Full TTT (Heavy)
- Train ~81% of model params, 30 epochs per chunk
- Larger improvement but much slower
- Used in some top submissions with careful regularization

### Pseudocode (Score-First LoRA TTT)
```python
def score_first_ttt_eval(model, tokens, lora_rank=8, lr=0.002, epochs=3):
    lora = LoRAAdapter(model, rank=lora_rank)
    optimizer = SGD(lora.parameters(), lr=lr, momentum=0.9)

    for chunk_start in range(0, len(tokens), chunk_size):
        chunk = tokens[chunk_start:chunk_start + chunk_size]

        # Step 1: Score (no gradient) — this is your BPB contribution
        with torch.no_grad():
            logits = model(chunk[:-1])
            score = cross_entropy(logits, chunk[1:])

        # Step 2: Train on same chunk (with gradient)
        model.train()
        for _ in range(epochs):
            logits = model(chunk[:-1])
            loss = cross_entropy(logits, chunk[1:])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
```

### Impact
- LoRA TTT: ~0.002-0.003 BPB (PR #549, #181)
- Full TTT: larger but riskier, neutral on some stacks (PR #1019 dropped it)
- On advanced stacks with XSA-all, TTT can be neutral or negative

### Integration Point
After model loading, before eval loop. Add LoRA adapter, modify eval loop to score-then-train per chunk.

### Legality
**Legal under Track B** (Adaptive Compression). Must follow score-before-update discipline:
1. Compute score at position t from p_t(x_t)
2. Only then update state using x_t

**Illegal** if: same-symbol adaptation, multi-pass rescoring, or pre-eval adaptation.

### Dependencies
None. Independent of sliding window and n-gram mixing.

### Effort
**Medium-High.** Need LoRA adapter implementation, careful score-before-train ordering, hyperparameter tuning. ~100-200 lines.

---

## 4. AsymLogit (Asymmetric Logit Rescale)

### What It Does
Replaces the single symmetric `logit_softcap` with two learnable scalar parameters — one for positive logits, one for negative logits:

```python
# Before (symmetric):
logits = softcap * tanh(logits / softcap)

# After (asymmetric):
logits = where(logits > 0,
    softcap_pos * tanh(logits / softcap_pos),
    softcap_neg * tanh(logits / softcap_neg))
```

### Why It Works
During per-document LoRA TTT, the model learns asymmetric logit distributions that a symmetric softcap cannot capture. Positive and negative logit ranges may need different scaling.

### Impact
- ~0.001-0.002 BPB improvement (PR #1923: -0.00137 BPB)
- Context-dependent: works best with AWQ-lite quantization + TTT
- ~8 bytes artifact cost (two fp16 scalars)
- Appears in top 4 submissions

### Pseudocode
```python
class AsymLogitRescale(nn.Module):
    def __init__(self, init_val=30.0):
        super().__init__()
        self.softcap_pos = nn.Parameter(torch.tensor(init_val))
        self.softcap_neg = nn.Parameter(torch.tensor(init_val))

    def forward(self, logits):
        return torch.where(
            logits > 0,
            self.softcap_pos * torch.tanh(logits / self.softcap_pos),
            self.softcap_neg * torch.tanh(logits / self.softcap_neg)
        )
```

### Integration Point
In `GPT.__init__()` (line ~659), add `softcap_pos` and `softcap_neg` as nn.Parameters. In `forward_logits()` (line ~723), replace the symmetric softcap with the asymmetric version. Gated by `ASYM_LOGIT_RESCALE` env var.

### Current Code (line 723)
```python
logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
```

### Dependencies
Most effective when combined with TTT (the scalars adapt during eval). Can be used standalone but improvement is smaller.

### Effort
**Very Low.** ~10 lines of code. Two new parameters, one conditional in forward.

---

## 5. PPM (Prediction by Partial Matching)

### What It Does
Classical compression algorithm used as eval-time ensemble member. Builds a Markov model of character sequences at test time, tracking how often each character follows a given context of length k.

### Why It Works
PPM captures repeated patterns in the eval data that the neural model may miss. It's especially good at catching exact or near-exact repetitions from earlier in the document.

### Key Design: Delayed PPM
PR #511 uses "delayed" PPM where the PPM bank only contains tokens from positions `<= i - 2048`. This ensures it cannot reuse anything still visible inside the transformer's sliding-window context — the two models are complementary.

### Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| k (max order) | 15 | Maximum context length for Markov model |
| delay | 2048 | Minimum distance between PPM source and scored token |
| min_confidence | 0.95-1.0 | Minimum count ratio to use PPM prediction |
| backoff orders | [16, 12, 8, 6] | Context lengths to try, longest first |

### Impact
- ~0.00126 BPB improvement (PR #511, 3-seed mean)
- **Not a valid submission** due to partition function inflation issues (PR #1147)
- Small but consistent improvement

### Pseudocode
```python
class PPMModel:
    def __init__(self, max_order=15, delay=2048):
        self.max_order = max_order
        self.delay = delay
        self.counts = defaultdict(lambda: defaultdict(int))

    def update(self, position, context, target):
        """Add token to PPM model (only if old enough)."""
        if position < self.delay:
            return
        for order in range(1, self.max_order + 1):
            ctx = tuple(context[-order:])
            self.counts[ctx][target] += 1

    def predict(self, context):
        """Get PPM distribution, backing off from longest to shortest."""
        for order in range(self.max_order, 0, -1):
            ctx = tuple(context[-order:])
            if ctx in self.counts:
                total = sum(self.counts[ctx].values())
                if total >= self.min_confidence:
                    return normalize(self.counts[ctx])
        return None  # no confident prediction
```

### Integration Point
After model loading, create PPM instance. During eval, update PPM with scored tokens (with delay), blend PPM predictions with neural predictions.

### Legality
**Legal under Track B** (Adaptive Compression). The 2048-token delay ensures no information from the current context window leaks in. However, PR #1147 identified partition function inflation issues with hashed n-gram caches.

### Dependencies
None. Independent of other techniques.

### Effort
**Medium.** ~80-100 lines. PPM model class, integration into eval loop, backoff logic.

---

## 6. GPTQ + Self-Generated Calibration

### What It Does
Instead of using validation data for GPTQ calibration (illegal within the 600s training window), the model autoregressively generates its own calibration tokens after training. Hessians H = X^T X are collected from self-generated sequences.

### Why It Works
Standard GPTQ uses calibration data to compute the Hessian for weight quantization. Using val data gives best results but is time-constrained. Self-generated text from the model produces Hessians nearly identical to val data — the model's own distribution is close enough.

### Impact (PR #1019, SOTA at 1.1147 BPB)
| Calibration Source | Tokens | Time | Sliding BPB | vs Val-calib |
|-------------------|--------|------|-------------|-------------|
| Val data (reference) | ~50M | ~5s | 1.1145 | — |
| **AR self-generation** | 131K | 186s | **1.1148** | +0.0003 |
| Random tokens | 131K | 3.4s | 1.1165 | +0.0020 |
| Random tokens (large) | 25M | 35s | 1.1165 | +0.0020 |

AR self-gen closes **84% of the val-vs-random gap** (0.0017 of 0.0020 BPB).

### Pseudocode
```python
def self_gen_gptq_calibration(model, n_seqs=64, seq_len=2048, temp=0.8, seed=42):
    """Generate calibration data from the trained model itself."""
    model.eval()
    torch.manual_seed(seed)

    # Generate sequences autoregressively
    calib_seqs = []
    for _ in range(n_seqs):
        prompt = torch.randint(0, vocab_size, (1, 1))  # random start token
        seq = model.generate(prompt, max_new_tokens=seq_len, temperature=temp)
        calib_seqs.append(seq)

    # Compute Hessians from self-generated data
    activations = []
    def hook_fn(module, input, output):
        activations.append(input[0].detach())

    # Register hooks on linear layers, run forward on calib data
    # H = X^T X for each layer
    # Run GPTQ with these Hessians
```

### Integration Point
After training completes, before artifact serialization. Generate calibration data, run full Hessian GPTQ with Cholesky error compensation.

### Key Insight from PR #1019
Upgraded from GPTQ-lite (diagonal Hessian) to **Full Hessian GPTQ** with Cholesky error compensation and column reordering. The full Hessian captures cross-weight correlations that the diagonal approximation misses.

### Dependencies
None. Independent of eval techniques. This is a quantization improvement.

### Effort
**High.** Full GPTQ implementation with Cholesky decomposition, autoregressive generation, hook-based activation collection. ~200-400 lines. May use existing GPTQ library.

---

## 7. Compression: zstd-22 vs zlib

### What It Does
Replace zlib-9 compression with zstd-22 (Zstandard level 22) for the artifact.

### Why It Works
zstd combines LZ77 matching with fast finite-state entropy coding. At maximum compression (level 22), it squeezes quantized weights tighter than zlib-9.

### Impact
- **We already confirmed:** zstd-22 saves 310KB (4.8%) over zlib-9 on our int8 artifact (6.21 MB vs 6.52 MB)
- 501 submissions use zstd vs 411 using zlib
- Decompression speed: very fast (~1 GB/s)

### Algorithm Comparison
| Algorithm | Submissions | Typical Ratio | Decompress Speed | Dependencies |
|-----------|-------------|---------------|------------------|--------------|
| **zstd-22** | 501 | 1.2–1.5x | Very fast (~1 GB/s) | External lib |
| zlib | 411 | 1.15–1.4x | Fast | Standard lib |
| lzma | 248 | 1.3–1.6x | Slow | Standard lib |
| brotli | 11 | 1.2–1.5x | Fast | External lib |

### Pseudocode
```python
import zstandard

def compress_artifact_zstd(data: bytes, level: int = 22) -> bytes:
    compressor = zstandard.ZstdCompressor(level=level)
    return compressor.compress(data)
```

### Advanced: Weight Ordering
Reorder weight rows by L2 norm before serialization. Similar rows end up adjacent → longer runs of similar byte patterns → better compression. Store a permutation index to undo at load time.

**Note:** We already confirmed L2 norm ordering gives no benefit for int8. May help with int6/int5 where value distributions are more structured.

### Integration Point
Replace `zlib.compress()` calls in artifact serialization with `zstandard.ZstdCompressor(level=22).compress()`. Add `zstandard` to requirements.

### Dependencies
None. Pure compression change.

### Effort
**Very Low.** ~5 lines of code. Install zstandard, swap compress call.

---

## Technique Independence & Ordering

### Can Be Done in Parallel
All 7 techniques are largely independent. They touch different parts of the pipeline:

| Technique | Pipeline Stage | Artifact Cost | Eval Time Cost |
|-----------|---------------|---------------|----------------|
| Sliding Window | Eval | 0 bytes | ~16x |
| N-gram Mixing | Eval | 0 bytes | ~1.1x |
| Score-First TTT | Eval | 0 (LoRA is ephemeral) | ~3-5x |
| AsymLogit | Eval | 8 bytes | ~1x |
| PPM | Eval | 0 bytes | ~1.5x |
| GPTQ Self-Gen | Quantization | 0 bytes | 0 (done in training time) |
| zstd-22 | Compression | 0 (saves space) | 0 |

### Recommended Implementation Order
1. **Sliding Window** — biggest bang (0.034 BPB), simplest to implement
2. **zstd-22** — already confirmed 4.8% space savings, trivial change
3. **AsymLogit** — tiny change, ~0.001 BPB
4. **N-gram Mixing** — free improvement, ~0.005-0.01 BPB
5. **Score-First TTT** — larger effort, ~0.002-0.003 BPB
6. **PPM** — medium effort, ~0.001 BPB, validity concerns
7. **GPTQ Self-Gen** — highest effort, requires full GPTQ implementation

### Sequential Dependencies
- AsymLogit works best WITH TTT (scalars adapt during eval)
- GPTQ Self-Gen is a quantization improvement, not eval — but it improves the base that eval operates on
- PPM + N-gram Mixing both blend with neural predictions — could conflict if both try to blend

---

## Estimated BPB Improvement Summary

| Technique | Estimated BPB Gain | Confidence | Source |
|-----------|-------------------|------------|--------|
| Sliding Window (stride=64) | -0.034 | High | Multiple ablations, PR #50, #65 |
| N-gram Mixing | -0.005 to -0.01 | Medium | PR #524, competitive intel |
| Score-First LoRA TTT | -0.002 to -0.003 | High | PR #549, #181, #1019 |
| AsymLogit | -0.001 to -0.002 | Medium | PR #1923, #2130 |
| PPM | -0.001 | Low | PR #511 (validity concerns) |
| GPTQ Self-Gen | -0.002 | High | PR #1019 calibration study |
| zstd-22 | Space savings only | High | Already confirmed 4.8% |

**Total potential (conservative):** ~0.045 BPB improvement from eval pipeline alone
**Total potential (optimistic):** ~0.055 BPB

---

## Field Guide References

- Eval strategies: https://sameersegal.github.io/learn-parameter-golf/learn/evaluation-strategies
- Quantization: https://sameersegal.github.io/learn-parameter-golf/learn/quantization-fundamentals
- Compression: https://sameersegal.github.io/learn-parameter-golf/learn/compression
- Field Guide to Valid Submissions: https://github.com/openai/parameter-golf/issues/1017
- Live Analysis (Issue #140): https://github.com/openai/parameter-golf/issues/140

## Key PRs

| PR | Technique | BPB | Author |
|----|-----------|-----|--------|
| #50 | Sliding Window Eval (first) | — | mattqlf |
| #524 | N-gram Mixing | — | — |
| #511 | PPM | — | AnirudhRahul |
| #549 | Score-First TTT + Parallel Muon | 1.1194 | sanjeevmadhav |
| #1019 | AR Self-Gen GPTQ + XSA-all | 1.1147 | abaybektursun |
| #1923 | AsymLogit Rescale | 1.0597 | jorge-asenjo |
| #2130 | N-gram Tilt + AsymLogit | 1.0567 | TanishGudise |
ENDOFDOC