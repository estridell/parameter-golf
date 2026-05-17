# Architecture Tricks from Parameter Golf Winners

Research compiled from Field Guide, CodeSOTA, GitHub PRs, and participant writeups.
Sources: sameersegal.github.io/learn-parameter-golf, codesota.com/parameter-golf,
openai/parameter-golf PRs, pragnyanramtha.xyz blog, PR #108, #826, #1903, #1056.

---

## 1. BigramHash

**Prevalence:** 583 submissions. Top 5 neural entries ALL use it.
**Best submission:** PR #1903 (0.9418 BPB), PR #826 (0.295 BPB with n-grams)

### What It Does

Hashes adjacent token pairs into a fixed-size embedding table, added to the token
embedding BEFORE the Transformer. Gives the model local context "for free" before
the first attention layer fires.

### Why It Works

Bigram statistics ("q" followed by "u") are extremely predictable. A Transformer
needs at least one attention layer to discover these. BigramHash provides this
upfront, letting the Transformer spend capacity on harder, longer-range patterns.

### Hash Trick

Naive bigram table for byte-level tokens: 256x256 = 65,536 entries. At 512 dims,
that is 128 MB -- wildly over budget. Hashing maps to 8K-16K entries. Collisions
are rare enough to be harmless.

### Implementation (from PR #108)

```python
# Environment: USE_BIGRAM_HASH=1
# Vocab: 4096, Embed dim: 32, Projection: 512
# ~524K parameters total

class BigramHash(nn.Module):
    def __init__(self, vocab_size=4096, num_buckets=4096, embed_dim=32, model_dim=512):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)
        self.proj = nn.Linear(embed_dim, model_dim, bias=False)
        # Init: std=0.01
        nn.init.normal_(self.embed.weight, std=0.01)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, tokens):
        # tokens: (batch, seq_len) of token IDs
        # Hash adjacent pairs: hash(tok[t-1], tok[t]) -> bucket index
        prev_tokens = F.pad(tokens[:, :-1], (1, 0), value=0)  # shift right
        # Simple hash: multiply-add with prime, mod num_buckets
        hash_idx = (prev_tokens * 1000003 + tokens) % self.num_buckets
        bigram_emb = self.embed(hash_idx)         # (B, T, 32)
        return self.proj(bigram_emb)              # (B, T, model_dim)

# Usage: added to token embedding before first transformer block
# x = tok_emb(tokens) + bigram_hash(tokens)
```

### Key Details
- **fp16 passthrough** (immune to int6 quantization damage)
- PR #108 init: std=0.01
- PR #1903 uses vocab=4096, dim=32, direct residual injection
- PR #2021 tested "uppercase enrichment" variant
- Some submissions removed BigramHash when vocab increased to 4096+ (PR #1580)
  because larger vocabs already capture bigram patterns

### Parameter Cost
- Embedding: num_buckets x embed_dim (4096 x 32 = 131K)
- Projection: embed_dim x model_dim (32 x 512 = 16K)
- Total: ~147K params (~0.6 MB at fp16)

### Gotchas
- Less useful with large vocab sizes (4096+) -- the tokenizer already captures bigrams
- Must be immune to quantization (fp16 passthrough) or it degrades badly
- Hash collisions are harmless at 4K buckets; do not over-engineer

---

## 2. SmearGate

**Prevalence:** 396 submissions
**Best submission:** PR #1056 (0.018 BPB, n-gram), PR #826 (0.295 BPB), PR #108 (1.1458 BPB)

### What It Does

Computes a learned gate (0 to 1) at each position that controls how much of the
previous token's representation gets "smeared" into the current position.

### The Formula

```
output[t] = gate[t] * input[t-1] + (1 - gate[t]) * input[t]
```

- gate=0: position keeps its own representation
- gate=1: position adopts neighbor's representation
- gate=intermediate: learned blend

### Why Not Just Attention?

Attention is O(n^2) and operates globally. SmearGate is O(1) per position and handles
the most common case: adjacent token information flow. It is "whispering to the
person next to you" vs "a conference call."

### Where It Lives

Applied right after token embedding, BEFORE the first Transformer block. Combined
with BigramHash, the Transformer receives rich representations before any attention.

### Implementation (from PR #108)

```python
# Environment: USE_SMEAR_GATE=1
# ~512 parameters (one gate value per model dimension)

class SmearGate(nn.Module):
    def __init__(self, model_dim=512):
        super().__init__()
        # Per-dimension gate bias (learned)
        self.gate_bias = nn.Parameter(torch.zeros(model_dim))
        # Init: +3.0 so sigmoid(3) approx 0.95, near-identity at start
        nn.init.constant_(self.gate_bias, 3.0)

    def forward(self, x):
        # x: (batch, seq_len, model_dim)
        gate = torch.sigmoid(self.gate_bias)  # (model_dim,), static per-dim
        # Shift x right by 1 position (previous token)
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0)
        # Blend: gate controls how much of neighbor to mix in
        return gate * x_prev + (1 - gate) * x

# Usage: applied after embedding + BigramHash, before first block
# x = smear_gate(tok_emb(tokens) + bigram_hash(tokens))
```

### Key Details
- PR #108: gate_bias init at +3.0 (sigmoid approx 0.95, near-identity start)
  - Previous init at 0 -> sigmoid=0.5 -> too aggressive blending at start
- PR #108: used "LeakyReLU(0.5)^2" on SmearGate variant (1.1444 BPB)
- Gate is a static per-dimension bias -- NOT input-dependent
  (cheaper than a learned gate network, and works just as well)

### Parameter Cost
- Gate bias: model_dim = 512 params (negligible)
- No projection matrices needed

### Gotchas
- Init matters a LOT. +3.0 (near-identity) >> 0.0 (50/50 blend)
- Static gate (per-dim bias) works as well as input-dependent gate at this scale
- Must be placed BEFORE the transformer stack, not between layers
- If vocab is large (4096+), SmearGate may be redundant with BigramHash

---

## 3. XSA (Cross-Sparse Attention)

**Prevalence:** 392 submissions
**Best submissions:** #1 (1.1147 BPB, XSA-all), #5 (1.1271 BPB, XSA4), #6 (1.1307 BPB, partial)

### What It Does

Replaces dense Q/K/V projections with structured sparse ones. Instead of every
layer having full multi-head attention, some layers skip attention entirely
(become pure MLPs) or use shared/sparse projections.

### Variants

| Variant | Description | Used By |
|---------|-------------|---------|
| XSA-all | Cross-sparse attention on ALL layers | #1 (1.1147 BPB) |
| XSA4 | Full attention only on last 4 layers | #5 (1.1271 BPB) |
| Partial XSA (3-deep) | Full attention on 3 deepest layers only | #6 (1.1307 BPB) |

### Why It Works

At 70M params, most attention heads are redundant. XSA enforces sparsity from the
start as a strong inductive bias. The saved parameters go to wider MLPs, more
layers, or better embeddings -- which matter more than full attention at this scale.

Early layers mostly learn local patterns (character-level, syntax) that do not
need global attention. Deeper layers do composition, which tolerates sparsity.

### Implementation Approach

```python
# XSA4: only last 4 layers get full attention
# Early layers: MLP-only (no Q/K/V projections)

class XSAConfig:
    def __init__(self, num_layers=11, xsa_layers=4):
        self.full_attn_layers = set(
            range(num_layers - xsa_layers, num_layers)
        )

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads)
        self.mlp = MLP(dim)
        self.resid_mix = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, x0):
        mix = self.resid_mix
        x = mix[0] * x + mix[1] * x0
        if self.use_attention:
            x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

# Usage in GPT:
# for i, block in enumerate(self.blocks):
#     x = block(x, x0)  # block internally decides whether to attend
```

### XSA-All Variant (more advanced)

Instead of skipping attention entirely, XSA-all uses shared/sparse projections:
- Share K/V projections across multiple heads (extreme GQA)
- Use structured sparsity in Q/K matrices
- The #1 submission combines this with self-generated GPTQ calibration

### Parameter Savings

For 11 layers with d=512, 8 heads:
- Full attention per layer: 4 x d x d = 4 x 512 x 512 = 1.05M params
- Skipping 7 layers of attention: saves ~7.3M params (can go to MLP width)

### Gotchas
- XSA4 vs XSA-all is a real tradeoff: XSA-all is better BPB but harder to implement
- Early layers MUST still process tokens (MLP-only is fine)
- Do not skip too many layers -- model needs some attention to propagate information
- The #1 submission pairs XSA-all with aggressive GPTQ quantization to recover precision

---

## 4. Partial RoPE

**Prevalence:** 270 submissions
**Best submission:** #4 (1.1248 BPB, Partial RoPE + LN Scale)

### What It Does

Applies Rotary Position Embeddings to only the FIRST HALF of head dimensions.
The remaining dimensions are position-invariant (no rotation).

Standard RoPE: rotates ALL dimensions of Q and K
Partial RoPE: rotates first N dims, passes rest through unchanged

### Why It Works

Creates a natural specialization:
- Rotated dims: learn position-sensitive features (word order, syntax)
- Non-rotated dims: learn position-agnostic features (semantic content, topic)

This split helps small models by making each head more interpretable and easier
to train. The model does not waste capacity learning to "undo" positional encoding
in dimensions where position is irrelevant.

### Implementation (from train_gpt.py)

```python
# Environment: ROPE_DIMS=16 (default in parameter-golf)
# Typical values: 16, 32, 64 (out of 64 head dims with d=512, 8 heads)

class Rotary(nn.Module):
    def __init__(self, dim, rope_dims=16, base=10000.0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rope_dims, 2) / self.rope_dims)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k, cos, sin):
        return (
            apply_rotary_emb(q, cos, sin, self.rope_dims),
            apply_rotary_emb(k, cos, sin, self.rope_dims),
        )

def apply_rotary_emb(x, cos, sin, rope_dims):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        # Standard RoPE rotation on x_rope only
        x1, x2 = x_rope[..., ::2], x_rope[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        x_rope = torch.stack([out1, out2], dim=-1).flatten(-2)
        return torch.cat([x_rope, x_pass], dim=-1)
    else:
        # Full RoPE (fallback)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)

# CLI: python train_gpt.py --rope-dims 32
# Env:  ROPE_DIMS=32 python train_gpt.py
```

### Key Details
- Default in parameter-golf: rope_dims=16 (out of 64 head dimensions)
- Submission #4 uses Partial RoPE + LN Scale (single scalar per LayerNorm)
- rope_dims=0 or > head_dim -> falls back to full RoPE
- Cannot change rope_dims after training (structural hyperparameter)

### Parameter Cost
- ZERO extra parameters (RoPE is a fixed transform, not learned)
- But saves compute by not rotating half the dimensions

### Gotchas
- rope_dims is fixed at init -- cannot change for inference
- Common values: 16 (quarter of 64), 32 (half of 64)
- Works best paired with LN Scale (single scalar per layer, saves 2d params/layer)
- The 0.0023 BPB gain from Partial RoPE + LN Scale comes entirely from the
  combination (Issue #140)

---

## 5. OrthoInit

**Prevalence:** 171 submissions
**Used by:** Majority of top 50 neural submissions

### What It Does

Replaces standard Kaiming/Xavier initialization with orthogonal initialization.
QR decomposition produces a Q matrix where every singular value is exactly 1,
so every feature direction is preserved equally through initialization.

### Why It Works

Parameter Golf models train for only 5,000-7,000 steps. Random initialization has
non-uniform singular values -- some directions get amplified, others suppressed.
The optimizer wastes hundreds of steps fixing this directional bias.

Orthogonal init preserves all feature directions equally from step 0:
- Stacked 11 orthogonal matrices preserve signal magnitude
- No "crushed features" contributing tiny gradients
- Every dimension starts on equal footing

### Implementation

```python
import torch.nn as nn

def orthogonal_init(module: nn.Module, scale: float = 1.0):
    """Apply orthogonal init to all linear layers, zero init to biases."""
    for name, param in module.named_parameters():
        if param.ndim >= 2:
            nn.init.orthogonal_(param, gain=scale)
        elif "bias" in name:
            nn.init.zeros_(param)

def orthogonal_init_scaled(module: nn.Module, num_layers: int = 11):
    """OrthoInit with per-layer scaling for residual networks.

    In a residual network, each layer ADDS to the residual stream.
    Without scaling, residual grows as sqrt(num_layers).
    Scaling output projections by 1/sqrt(num_layers) keeps it stable.
    """
    for name, param in module.named_parameters():
        if param.ndim >= 2:
            if "out_proj" in name or "o_proj" in name:
                nn.init.orthogonal_(param, gain=1.0 / (num_layers ** 0.5))
            else:
                nn.init.orthogonal_(param, gain=1.0)
        elif "bias" in name:
            nn.init.zeros_(param)

# Usage after model construction:
model = TransformerLM(num_layers=11, d_model=512)
orthogonal_init_scaled(model, num_layers=11)
```

### Key Details
- Pure upside -- costs nothing at inference time
- Most pronounced in small models with limited training steps
- The scaled version (1/sqrt(num_layers) for output projections) prevents
  residual stream growth in deep networks
- Related: Spectral init (18 submissions) adjusts based on computed spectral
  norm of entire layer stack -- more expensive but better for very deep nets
- Related: ResidMix (15 submissions) -- sets residual scaling near zero at init
  so model starts as identity function (perfect gradient flow)

### Parameter Cost
- ZERO extra parameters (just different initialization values)

### Gotchas
- Must be applied BEFORE training starts (not after)
- The gain parameter matters: 1.0 for most layers, 1/sqrt(n_layers) for output projections
- Some submissions use different gains for different layer types (Q vs K vs V vs out_proj)
- Overtone init (6 submissions) initializes embeddings using sine patterns -- another variant

---

## 6. Depth Recurrence

**Prevalence:** 116 submissions
**Best submission:** PR #1903 (0.9418 BPB, 3L-Progressive Recurrence)

### What It Does

Runs the same Transformer layers multiple times in sequence. 5 unique layers x 2
passes = 10 effective layers with only 5 layers' worth of parameters.

### The Trade-off

- Benefit: 2x depth for free
- Benefit: More parameter-efficient
- Benefit: Pairs with TTT naturally
- Risk: Shared layers may not specialize
- Risk: Loses representational diversity
- Risk: Quantization amplification problem

### Implementation (from train_gpt.py)

The repo uses an encoder-decoder split with U-Net-like skip connections:

```python
class GPT(nn.Module):
    def __init__(self, num_layers=9, ...):
        # Split layers into encoder/decoder
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        # Learned skip weights (zeros at init = disabled)
        self.skip_weights = nn.Parameter(
            torch.zeros(self.num_decoder_layers, model_dim)
        )

    def forward(self, x, x0):
        skips = []
        # Encoder: process and store intermediate states
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        # Decoder: consume skips in reverse order (U-Net pattern)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return x
```

### Progressive Recurrence (PR #1903)

Instead of simple 2x recurrence, uses a progressive pattern:

```
Pattern: [0, 1, 2, 3, 4, 5, 3, 4, 5]
-> 9 layers of depth with only 6 layers' worth of parameters
-> Layers 3-5 get applied twice (recurrence)
-> Layers 0-2 applied once (encoder only)
```

### FiLM Conditioning (Feature-wise Linear Modulation)

To restore specialization in shared layers, FiLM feeds a pass index into each layer:

```python
class FiLMConditioner(nn.Module):
    """Tiny per-iteration scale/shift conditioned on recurrence pass index."""
    def __init__(self, model_dim, max_passes=2):
        super().__init__()
        # One scale+shift per pass, per dimension
        self.scale = nn.Parameter(torch.ones(max_passes, model_dim))
        self.shift = nn.Parameter(torch.zeros(max_passes, model_dim))

    def forward(self, x, pass_idx):
        return self.scale[pass_idx] * x + self.shift[pass_idx]

# In block:
# x = film_conditioner(x, pass_idx=current_pass)
```

### The Quantization Amplification Problem

When you quantize a depth-recurrent model, errors compound across repeated passes.
A weight that is off by 0.01 in int6 gets applied twice -> 0.02 error. This makes
depth recurrence + quantization a dangerous combo.

**QAT Fix:** Use Quantization-Aware Training (STE-QAT from PR #108):

```python
# Straight-Through Estimator for QAT
# Forward: quantize weights
# Backward: gradients pass through as if weights were float
w_hat = quantize(w)
w_out = w + (w_hat - w).detach()  # STE: forward=quantized, backward=float
```

Activate QAT after 25% of training (STE_QAT_START_FRAC=0.25).

### Key Details
- PR #1903: 3L-progressive recurrence, layers [0,1,2,3,4,5,3,4,5]
  -> 9 effective layers, 6 unique, 3 recurrent -- 0.9418 BPB
- 5x2 config: 5 unique layers applied 2x = 10 effective layers
- 4x2 config: 4 unique layers applied 2x = 8 effective layers
- FiLM conditioning partially restores specialization
- resid_mix parameter blends hidden state with original embeddings at each layer:
  ```python
  mix = self.resid_mix  # shape [2, dim], init [1, 0]
  x = mix[0] * x + mix[1] * x0
  ```

### Parameter Cost
- Skip weights: num_decoder_layers x model_dim (e.g., 5 x 512 = 2,560 params)
- FiLM: max_passes x model_dim x 2 (e.g., 2 x 512 x 2 = 2,048 params)
- resid_mix: 2 x model_dim per layer (e.g., 2 x 512 x 9 = 9,216 params)
- Total overhead: ~14K params (negligible)

### Gotchas
- Quantization amplification is REAL -- must use QAT or careful mixed-precision
- Skip weights initialized to zeros (disabled at start, learned gradually)
- The encoder/decoder split ratio matters: 9 layers -> 4 encoder + 5 decoder
- Changing num_layers changes the U-Net topology, not just depth
- Depth recurrence pairs naturally with TTT (test-time training)

---

## Stacking These Tricks

The best submissions combine all six. PR #826 is the canonical example:

**Config:** 11L, 512d, GQA 8/4, MLP 3.0x
**Stack:** BigramHash(4096) + SmearGate + XSA-4 + Partial RoPE + LN Scale + U-Net skips
**Result:** 0.295 BPB (with n-gram eval boost)

### Compatibility Matrix

| Trick | Pairs With | Conflicts With |
|-------|-----------|----------------|
| BigramHash | Everything | Large vocab (4096+) reduces value |
| SmearGate | BigramHash | Large vocab reduces value |
| XSA | Depth recurrence | Full attention (by definition) |
| Partial RoPE | LN Scale | Nothing known |
| OrthoInit | Everything | Nothing (pure upside) |
| Depth Recurrence | XSA, TTT | Quantization (needs QAT fix) |

### Implementation Order (recommended)

1. OrthoInit (zero cost, immediate win)
2. Partial RoPE + LN Scale (zero param cost)
3. BigramHash (cheap, high impact)
4. SmearGate (cheap, pairs with BigramHash)
5. XSA (saves params for MLP width)
6. Depth Recurrence (most complex, needs QAT)

---

## What Does NOT Work (from competitive intel)

- **BigramHash with large vocab (4096+):** Redundant. PR #1580 removed it after
  increasing vocab size.
- **SmearGate init at 0.0:** Too aggressive blending from start. Use +3.0.
- **Depth recurrence without QAT:** Quantization errors compound through shared layers.
- **Full attention on all layers at 16MB:** Too many params in Q/K/V, not enough for MLP.
- **FlashAttention on Turing (sm_75):** Not supported. Must use math SDP backend.
- **torch.compile on first run (sm_75):** 10+ min compilation. Not worth for quick iteration.

---

## Quick Reference: Parameter Costs

| Technique | Extra Params | Relative Cost |
|-----------|-------------|---------------|
| BigramHash | ~147K | 0.6 MB (fp16) |
| SmearGate | ~512 | negligible |
| XSA | -7.3M (savings) | frees MLP capacity |
| Partial RoPE | 0 | free |
| OrthoInit | 0 | free |
| Depth Recurrence | ~14K | negligible |
| LN Scale | ~512/layer | negligible |
