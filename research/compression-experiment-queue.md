# Compression Experiment Queue

**Date:** 2026-05-17
**Purpose:** Prioritized queue of ALL compression and quantization experiments for parameter-golf's 16MB artifact constraint.
**Based on:** quantization-compression-survey.md, quantization-compression-survey-deep.md, llm-as-compressor-brainstorm.md, competitive-intel.md

---

## Current Best

| Metric | Value |
|--------|-------|
| Model | 17M params, GQA (8 heads, 4 KV), 1024 seq len, 9 layers |
| Best BPB | ~2.51 (int8 roundtrip, local) |
| Best artifact | int8 per-row + zlib-9 = ~8.2MB |
| Budget remaining | ~7.8MB (16MB - 8.2MB) |
| Compression pipeline | fp32 train -> int8 per-row quant -> zlib-9 |
| Top competitive BPB | 1.0565 (codemath3000) / 1.0576 (simonbissonnette) |

## Default Config

```
# Quantization
INT8_CLIP_PERCENTILE=99.99984
INT8_KEEP_FLOAT_MAX_NUMEL=65536
INT8_PER_ROW_SCALE_DTYPE=float16
INT8_KEEP_FLOAT_STORE_DTYPE=float16

# Compression
ALGORITHM=zlib (level=9)

# Control tensors (passthrough as fp16/fp32):
attn_scale, attn_scales, mlp_scale, mlp_scales, resid_mix, resid_mixes,
q_gain, skip_weight, skip_weights, smear_gate, smear_lambda
```

## Rules

1. **One variable at a time.** Change one thing, measure, compare to baseline.
2. **Always report:** artifact size (bytes), val_bpb (if applicable), decompression time, VRAM.
3. **Always test roundtrip:** quantize -> compress -> decompress -> dequantize -> compare weights.
4. **Use zstd-22 as compression baseline** unless testing compression algorithm itself.
5. **Commit results** to `research/` on rtx2070 branch with clear experiment name.
6. **Don't skip measurement.** Even "obvious" improvements need numbers.

---

## TIER 1 -- Quick Wins (< 10 min each, minimal code)

### C01. zstd-22 Compression (Already Validated)

- [ ] **Status:** CONFIRMED -- 4.8% improvement over zlib-9

**Question:** Does zstd-22 beat zlib-9 on our int8 weights?
**Background:** 501 parameter-golf submissions use zstd. Our prior test confirmed 20.9MB -> 6.2MB with zstd-22 vs 6.5MB with zlib-9.
**Method:**
```python
import zstd
quant_raw = torch.save(quant_obj, io.BytesIO()).getvalue()
compressed = zstd.compress(quant_raw, 22)
```
**Record:** compressed size, compression time, decompression time.
**Expected runtime:** 1 min.
**Impact:** HIGH (confirmed baseline for all future experiments).
**Dependencies:** None.

---

### C02. 5% Magnitude Pruning

- [ ] **Status:** NOT TESTED

**Question:** Does zeroing out the bottom 5% of weights by magnitude improve compression with negligible BPB loss?
**Background:** 52 parameter-golf submissions use magnitude pruning. Survey estimates ~10% better compression, ~0.001 BPB loss. Zeros compress extremely well (run-length encoding in zstd).
**Method:**
```python
# After training, before quantization:
for name, param in model.named_parameters():
    if param.ndim >= 2:
        threshold = torch.quantile(param.abs().flatten(), 0.05)
        param.data[param.abs() < threshold] = 0.0
```
Then quantize + zstd-22 as normal.
**Record:** artifact size, % zero weights, val_bpb delta vs baseline.
**Expected runtime:** 2 min.
**Impact:** MEDIUM (frees ~0.5-1MB, nearly free quality-wise).
**Dependencies:** None.

---

### C03. Pruning Sweep (1%, 3%, 5%, 7%, 10%)

- [ ] **Status:** NOT TESTED

**Question:** What's the optimal pruning percentage for our 17M model?
**Background:** C02 tests 5%. But the optimal % depends on our specific weight distribution. More pruning = smaller artifact but more quality loss.
**Method:** Run C02 at 1%, 3%, 5%, 7%, 10%. Plot size vs BPB.
**Record:** artifact size, BPB, % zero weights for each level.
**Expected runtime:** 10 min (5 runs x 2 min).
**Impact:** MEDIUM (finds optimal pruning level).
**Dependencies:** None.

---

### C04. zstd Strategy Comparison (btultra2 vs default)

- [ ] **Status:** NOT TESTED

**Question:** Does zstd's btultra2 strategy compress better than the default strategy at level 22?
**Background:** zstd supports multiple search strategies. btultra2 is the most aggressive. Compression happens once (training time), decompression once (eval time), so speed doesn't matter.
**Method:**
```python
import zstd
# Default strategy (level 22)
c_default = zstd.compress(quant_raw, 22)
# btultra2 strategy
cctx = zstd.ZSTD_CCtx()
cctx.set_parameter(zstd.CParameter.compressionLevel, 22)
cctx.set_parameter(zstd.CParameter.strategy, zstd.Strategy.btultra2)
result = cctx.compress(quant_raw)
```
**Record:** compressed size for each strategy, compression time.
**Expected runtime:** 2 min.
**Impact:** LOW (probably <1% difference at level 22).
**Dependencies:** None.

---

### C05. Delta Encoding Between Adjacent Layers

- [ ] **Status:** NOT TESTED

**Question:** Do adjacent transformer layers have enough similarity that delta encoding improves compression?
**Background:** EMNLP 2020 shows 85-94% neuron redundancy within same/adjacent layers. Delta-DNN paper shows 2-10x improvement on float weights. Our int8 weights may or may not benefit.
**Method:**
```python
# After quantization, before compression:
# For each weight name in blocks.1.*, compute delta from blocks.0.*
# Store: block.0 (full) + delta(1-0) + delta(2-1) + ... + delta(8-7)
# Measure entropy of deltas vs originals
import torch, math
def byte_entropy(t):
    bytes_flat = t.cpu().numpy().tobytes()
    from collections import Counter
    counts = Counter(bytes_flat)
    total = len(bytes_flat)
    return -sum(c/total * math.log2(c/total) for c in counts.values())
```
**Record:** byte entropy of deltas vs originals, compressed size with delta encoding vs without.
**Expected runtime:** 5 min.
**Impact:** MEDIUM (could be significant if layers are similar; zero if they're not).
**Dependencies:** None.

---

### C06. Per-Group Quantization Group Size Sweep

- [ ] **Status:** NOT TESTED

**Question:** What group size (32, 64, 128, 256, row) gives the best quality-size tradeoff?
**Background:** GGUF uses group size 32-256. Our current per-row quantization uses group size = row width (512). Smaller groups = more scales = more overhead but better quality. Top submissions use per-group quantization.
**Method:** Modify `quantize_float_tensor()` to use group size N instead of per-row. Sweep N in {32, 64, 128, 256, 512(row)}.
```python
def quantize_float_tensor_grouped(t, group_size=64):
    t32 = t.float()
    # Reshape to (num_groups, group_size)
    # Quantize each group independently
    # Store: quantized values + per-group scales
```
**Record:** artifact size, BPB (if quick val available), scale overhead bytes.
**Expected runtime:** 10 min (5 configs x 2 min).
**Impact:** MEDIUM (may improve quality at same size, or reduce size at same quality).
**Dependencies:** None.

---

## TIER 2 -- High Impact, Moderate Effort (30-60 min each)

### C07. INT6 Per-Row Quantization

- [ ] **Status:** NOT TESTED

**Question:** Does INT6 quantization (6 bits/param) preserve quality while reducing artifact size by ~25%?
**Background:** INT6 is the most common choice in competitive submissions. Expected: ~5.5MB for 17M params with zstd-22 (vs 8.2MB at int8). Frees ~2.7MB for more params.
**Method:** Change clamp range from [-127,127] to [-31,31], scale by 31/127.
```python
def quantize_float_tensor_int6(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 31.0).clamp_min(1.0 / 31.0)  # 31 not 127
        q = torch.clamp(torch.round(clipped / scale[:, None]), -31, 31).to(torch.int8)
        return q, scale.to(torch.float16)
    # Per-tensor variant similar
```
Note: values fit in int8 dtype but only use [-31,31] range. Bit packing (C12) needed for true 6-bit storage.
**Record:** artifact size (byte-packed vs bit-packed), BPB, weight MSE vs original.
**Expected runtime:** 5 min.
**Impact:** CRITICAL (competitive default; frees space for more params).
**Dependencies:** Bit packing (C12) for true size savings.

---

### C08. INT5 Per-Row Quantization

- [ ] **Status:** NOT TESTED

**Question:** Does INT5 quantization (5 bits/param) work without GPTQ calibration?
**Background:** INT5 = 16 levels. More aggressive than INT6. Without GPTQ, quality degradation may be significant. With GPTQ, competitive.
**Method:** Same as C07 but clamp to [-15,15], scale by 15/127.
```python
scale = (clip_abs / 15.0).clamp_min(1.0 / 15.0)
q = torch.clamp(torch.round(clipped / scale[:, None]), -15, 15).to(torch.int8)
```
**Record:** artifact size, BPB, weight MSE, comparison to int6.
**Expected runtime:** 5 min.
**Impact:** HIGH (enables more aggressive space savings; quality depends on model sensitivity).
**Dependencies:** Bit packing (C12) for true size savings.

---

### C09. Mixed-Precision Per-Layer (INT6 Attention, INT5 MLP)

- [ ] **Status:** NOT TESTED

**Question:** Does allocating more bits to attention layers and fewer to MLP layers improve quality at the same total size?
**Background:** SliM-LLM (ICML 2025) shows dramatic improvement with salience-driven mixed-precision. Attention layers are generally more sensitive to quantization than MLP layers.
**Method:**
```python
# Per-layer sensitivity: measure weight MSE at int5 vs int6 for each layer
# Assign int6 to high-sensitivity layers, int5 to low-sensitivity
# Simple heuristic: attention (q,k,v,proj) = int6, MLP (fc,proj) = int5
for name, param in state_dict.items():
    if 'attn' in name:
        quantize_int6(param)
    elif 'mlp' in name:
        quantize_int5(param)
```
**Record:** artifact size, BPB, per-layer bit allocation.
**Expected runtime:** 10 min.
**Impact:** MEDIUM (better allocation at same total size).
**Dependencies:** C07 (int6), C08 (int5).

---

### C10. AWQ-Lite Channel Importance

- [ ] **Status:** NOT TESTED

**Question:** Does scaling important channels before quantization improve quality?
**Background:** AWQ identifies important channels via activation magnitudes. alertcat used AWQ-Lite + GPTQ for 1.0594 BPB (top-4 submission). The scaling preserves channels that would otherwise be clipped.
**Method:**
```python
# 1. Run calibration data through model, record activation magnitudes per channel
# 2. For each weight matrix, scale columns by sqrt(activation_magnitude)
# 3. Quantize the scaled weights
# 4. Store scale factors alongside quantized weights
# At dequantize: multiply by inverse scale
```
**Record:** artifact size, BPB, per-channel scale overhead.
**Expected runtime:** 15 min (including calibration forward pass).
**Impact:** HIGH (proven in top submissions, pairs with GPTQ).
**Dependencies:** None (standalone), but best combined with GPTQ (C11).

---

### C11. GPTQ with Self-Generated Calibration

- [ ] **Status:** NOT TESTED

**Question:** Does Hessian-based GPTQ quantization with self-generated calibration data enable int4/int5 with minimal quality loss?
**Background:** PR #1019 breakthrough: generate calibration text FROM the trained model autoregressively (131K tokens, 186s). Build Hessians from those activations. Closes 84% of gap between random and validation-data calibration. Dominant technique in top submissions.
**Method:**
```python
# 1. Generate calibration data from trained model (autoregressive, 131K tokens)
# 2. Forward pass through model with calibration data, collect activations
# 3. Compute Hessian H = X^T X for each layer
# 4. Quantize column-by-column, using Cholesky decomposition of H^-1
# 5. Error compensation: propagate quantization error to remaining columns
```
200-400 lines. Most complex experiment in this queue.
**Record:** artifact size, BPB at int4/int5/int6, calibration time, comparison to RTN.
**Expected runtime:** 30 min (including calibration generation).
**Impact:** CRITICAL (breakthrough technique, enables aggressive quantization).
**Dependencies:** None (standalone), but combine with AWQ-Lite (C10) for best results.

---

### C12. Bit Packing for INT5/INT6

- [ ] **Status:** NOT TESTED

**Question:** Does bit packing save 25-37% storage vs byte-aligned sub-int8 quantization?
**Background:** Without bit packing, int5 still uses 1 byte/param (50% waste). Bit-packed int5 = 0.625 bytes/param. Essential for sub-int8 to actually save space.
**Method:**
```python
def bit_pack(values, bits):
    """Pack array of values using `bits` bits each."""
    packed = []
    accumulator = 0
    bits_in_acc = 0
    for v in values:
        accumulator |= (v & ((1 << bits) - 1)) << bits_in_acc
        bits_in_acc += bits
        while bits_in_acc >= 8:
            packed.append(accumulator & 0xFF)
            accumulator >>= 8
            bits_in_acc -= 8
    if bits_in_acc > 0:
        packed.append(accumulator & 0xFF)
    return bytes(packed)

def bit_unpack(data, bits, count):
    """Unpack `count` values using `bits` bits each."""
    values = []
    accumulator = 0
    bits_in_acc = 0
    byte_idx = 0
    for _ in range(count):
        while bits_in_acc < bits:
            accumulator |= data[byte_idx] << bits_in_acc
            bits_in_acc += 8
            byte_idx += 1
        values.append(accumulator & ((1 << bits) - 1))
        accumulator >>= bits
        bits_in_acc -= bits
    return values
```
**Record:** packed size vs byte-aligned size for int4/5/6/7/8, packing/unpacking speed.
**Expected runtime:** 15 min (implementation + test).
**Impact:** CRITICAL (prerequisite for all sub-int8 experiments to actually save space).
**Dependencies:** None.

---

### C13. LQER (Low-Quality Error Recovery)

- [ ] **Status:** NOT TESTED

**Question:** Does storing a small correction tensor for the most-errored weights improve BPB at same artifact size?
**Background:** codemath3000 used LQER for 1.0611 BPB (top-5 submission). After quantization, identify weights with highest quantization error, store corrections in sparse format. Sparse tensor compresses very well.
**Method:**
```python
# 1. Quantize weights (int6 or int5)
# 2. Compute error: original - dequantized
# 3. Find top-K% of weights by error magnitude
# 4. Store correction as sparse tensor (indices + values)
# 5. At load: apply corrections after dequantization
```
**Record:** artifact size with/without LQER, BPB improvement, % weights corrected, sparse overhead.
**Expected runtime:** 15 min.
**Impact:** HIGH (proven in top submissions, recovers quality from aggressive quantization).
**Dependencies:** C07 (int6) or C08 (int5) as base quantization.

---

## TIER 3 -- Medium Impact, Higher Effort (1-2 hours each)

### C14. SpinQuant Rotations Before Quantization

- [ ] **Status:** NOT TESTED

**Question:** Do learned rotation matrices improve quantization quality by making weights more "incoherent"?
**Background:** SpinQuant (ICLR 2025): learned rotations via Cayley optimization. Up to 13 points difference between good and bad rotations. Surpasses LLM-QAT by 19.1 points on LLaMA-2 7B W4A. The rotation makes weight matrices have uniform importance across entries (no outliers).
**Method:**
```python
# 1. For each weight matrix W, learn rotation R via Cayley optimization
# 2. Transform: W' = W @ R
# 3. Quantize W' (which is now incoherent)
# 4. Store: quantized W' + rotation R (small overhead: one orthogonal matrix per layer)
# 5. At load: dequantize W', multiply by R^T
```
**Record:** artifact size, BPB with/without rotation, rotation learning time, rotation overhead size.
**Expected runtime:** 30-60 min.
**Impact:** MEDIUM (proven in literature, untested in parameter-golf).
**Dependencies:** None.

---

### C15. SqueezeLLM Dense-and-Sparse Decomposition

- [ ] **Status:** NOT TESTED

**Question:** Does keeping outlier weights in high precision while quantizing the rest coarsely improve quality at same size?
**Background:** SqueezeLLM (ICML 2024): identify weights with highest quantization error (outliers), store them in fp16 sparse format. Quantize remaining weights coarsely (int4/int5). Sparse matrix compresses extremely well (mostly zeros).
**Method:**
```python
# 1. Quantize all weights to int5
# 2. Compute quantization error per weight
# 3. Keep top 1-5% of weights as fp16 sparse corrections
# 4. Store: int5 quantized + sparse fp16 corrections
# 5. At load: int5 base + sparse overlay = high quality
```
**Record:** artifact size, BPB, outlier %, sparse matrix compressibility.
**Expected runtime:** 30 min.
**Impact:** MEDIUM (alternative to LQER, may be better for very aggressive quantization).
**Dependencies:** C08 (int5), C12 (bit packing).

---

### C16. Learned Quantization Codebooks

- [ ] **Status:** NOT TESTED

**Question:** Does k-means clustering on weight values produce better quantization levels than uniform int?
**Background:** Like GGUF's imatrix quantization but for our model. Non-uniform levels can place more quantization points near zero (where most weights cluster). IDKM paper shows SOTA results with differentiable k-means.
**Method:**
```python
from sklearn.cluster import KMeans
# For each weight tensor:
kmeans = KMeans(n_clusters=64)  # 6-bit codebook
kmeans.fit(W.flatten().reshape(-1, 1))
assignments = kmeans.labels_  # 6-bit indices
codebook = kmeans.cluster_centers_  # 64 float values
# Store: assignments (bit-packed) + codebook (tiny)
```
**Record:** artifact size, BPB, codebook overhead, assignment entropy.
**Expected runtime:** 20 min.
**Impact:** MEDIUM (better than uniform quantization in theory; compressibility of non-uniform assignments is the question).
**Dependencies:** None.

---

### C17. Neural Arithmetic Coding

- [ ] **Status:** NOT TESTED

**Question:** Can a tiny neural model predicting weight bytes beat zstd's generic compression?
**Background:** From llm-as-compressor-brainstorm.md. Train a 1-layer LSTM (100K params) to predict the next weight byte given previous bytes. Use as probability model for arithmetic coding. The model can exploit cross-layer, cross-tensor correlations that zstd cannot. Estimated 2-4x improvement over zstd.
**Method:**
```python
# 1. Flatten all quantized weights into a byte sequence (preserving tensor order)
# 2. Train 1-layer LSTM: context -> p(next_byte)
# 3. Arithmetic encode using learned probabilities
# 4. Store: compressed bytes + LSTM weights (~200KB)
# 5. At load: arithmetic decode with LSTM, reconstruct weights
```
**Record:** compressed size vs zstd, LSTM training time, decompression time, bits/byte achieved.
**Expected runtime:** 60 min (implementation + training + test).
**Impact:** HIGH (most promising meta-compression approach; 2-4x over zstd if it works).
**Dependencies:** None.

---

### C18. Cross-Layer Weight Sharing (ResidualTransformer Style)

- [ ] **Status:** NOT TESTED

**Question:** Can storing a shared base layer + per-layer low-rank residuals compress better than storing each layer independently?
**Background:** ResidualTransformer (arXiv:2310.02489): K=3 sharing with R=2 low-rank residuals achieves ~3x compression with 1.8% WER gap. EMNLP 2020: 85-94% neuron redundancy within same/adjacent layers. Our 9-layer model should have high cross-layer redundancy.
**Method:**
```python
# 1. Compute average layer: W_shared = mean(W_0, W_1, ..., W_8)
# 2. For each layer i: residual_i = W_i - W_shared
# 3. Low-rank decompose residuals: U_i, S_i, V_i = SVD(residual_i, rank=R)
# 4. Store: W_shared (int8) + {U_i, S_i, V_i} per layer (int8)
# 5. At load: W_i = W_shared + U_i @ diag(S_i) @ V_i
```
**Record:** artifact size at R={4,8,16,32}, BPB, reconstruction error per layer.
**Expected runtime:** 30 min.
**Impact:** MEDIUM (significant if layers are truly similar; marginal if not).
**Dependencies:** None.

---

### C19. zstd Dictionary Mode

- [ ] **Status:** NOT TESTED

**Question:** Does pre-training a zstd dictionary on representative weight data improve compression?
**Background:** zstd dictionary mode learns byte patterns from training data, then uses them as context for compressing new data. Gains: 1-3% (modest for large models). Quick to test.
**Method:**
```python
import zstd
# 1. Collect sample weight byte sequences from multiple training checkpoints
# 2. Train dictionary
samples = [checkpoint_bytes_1, checkpoint_bytes_2, ...]
dict_data = zstd.train_dictionary(dict_size=65536, samples=samples)
# 3. Compress with dictionary
cctx = zstd.ZSTD_CCtx()
cctx.set_parameter(zstd.CParameter.compressionLevel, 22)
cctx.load_dictionary(dict_data)
compressed = cctx.compress(quant_raw)
```
**Record:** compressed size with/without dictionary, dictionary size, training time.
**Expected runtime:** 10 min.
**Impact:** LOW (1-3% improvement at best).
**Dependencies:** Multiple training checkpoints for dictionary training.

---

## TIER 4 -- Advanced / Exotic (2+ hours, higher risk)

### C20. ParetoQ (90/10 QAT Split + SEQ)

- [ ] **Status:** NOT TESTED

**Question:** Does ParetoQ's training budget allocation (90% FP + 10% QAT) and Stretched Elastic Quant improve quality at int5/int6?
**Background:** ParetoQ (NeurIPS 2025): first unified framework for binary/ternary/2-4 bit QAT. Key findings: 90% FP training + 10% QAT is optimal. Level symmetry is vital. SEQ balances quantized levels. Sub-4-bit often outperforms 4-bit.
**Method:**
```python
# Modify training loop:
# - First 90% of steps: normal fp32 training
# - Last 10% of steps: QAT with fake quantization (STE)
# - Use SEQ: symmetric levels without zero (e.g., -1.5, -0.5, 0.5, 1.5 for 2-bit)
```
**Record:** BPB at int5/int6 with and without QAT, training time overhead.
**Expected runtime:** 2+ hours (need full training run).
**Impact:** HIGH (proven framework, untested in parameter-golf).
**Dependencies:** Full training pipeline modification.

---

### C21. GPTQ + AWQ-Lite Combined Pipeline

- [ ] **Status:** NOT TESTED

**Question:** Does combining AWQ-Lite channel importance with GPTQ rounding produce the best quantization quality?
**Background:** This is exactly what top-4 submission alertcat uses (1.0594 BPB). AWQ-Lite identifies important channels, GPTQ optimizes rounding with Hessian information. The combination is proven in competition.
**Method:**
```python
# 1. Run AWQ-Lite: profile activations, identify important channels
# 2. Scale important channels before quantization
# 3. Apply GPTQ with self-generated calibration on scaled weights
# 4. Store: GPTQ-quantized weights + AWQ scale factors
```
**Record:** BPB, artifact size, comparison to GPTQ alone and AWQ alone.
**Expected runtime:** 45 min.
**Impact:** CRITICAL (the proven winning combination).
**Dependencies:** C10 (AWQ-Lite), C11 (GPTQ).

---

### C22. FP8 Quantization (H100 Only)

- [ ] **Status:** NOT TESTED (requires H100)

**Question:** Does FP8 E4M3 quantization work on H100 and what's the artifact size?
**Background:** `torch.float8_e4m3fn` available natively on H100. NOT available on RTX 2070 (sm_75). FP8 is excellent for inference speed but worse for compression (float distributions have more entropy than int).
**Method:**
```python
# On H100 only:
for name, param in model.named_parameters():
    if param.ndim >= 2:
        param_fp8 = param.to(torch.float8_e4m3fn)
        # Store fp8 values + per-row scales
```
**Record:** artifact size, BPB, comparison to int8.
**Expected runtime:** 10 min (on H100).
**Impact:** LOW for artifact size (same as int8, worse compressibility), HIGH for inference speed.
**Dependencies:** H100 access.

---

### C23. GGUF imatrix Quantization

- [ ] **Status:** NOT TESTED

**Question:** Does llama.cpp's imatrix (importance-weighted) quantization produce better quality than our per-row int quantization?
**Background:** GGUF uses block-wise quantization with super-blocks. imatrix uses calibration data to weight importance of different blocks. Q4_K_M is community consensus "best tradeoff." Q6_K is near-lossless.
**Method:**
```python
# 1. Export model to GGUF format (need custom conversion from our state_dict)
# 2. Run llama.cpp quantize with imatrix
# 3. Import quantized GGUF back to our state_dict
# 4. Compare size and quality
```
**Record:** artifact size at Q4_K_M, Q5_K_M, Q6_K, Q8_0, BPB for each.
**Expected runtime:** 60 min (format conversion + quantization + comparison).
**Impact:** MEDIUM (good quantization quality, but format conversion overhead may negate gains).
**Dependencies:** llama.cpp installed, format conversion code.

---

### C24. Stochastic Rounding for QAT

- [ ] **Status:** NOT TESTED

**Question:** Does stochastic rounding (instead of round-to-nearest) during QAT improve convergence?
**Background:** Stochastic rounding rounds to nearest integer with probability proportional to distance. Unbiased estimator -- E[round_stochastic(x)] = x. Can improve QAT convergence, especially at low bit widths.
**Method:**
```python
def stochastic_round(x):
    """Round x to nearest integer, with probability proportional to fractional part."""
    floor_x = torch.floor(x)
    frac = x - floor_x
    rand = torch.rand_like(x)
    return torch.where(rand < frac, floor_x + 1, floor_x)
```
Integrate into QAT forward pass (C20).
**Record:** BPB with stochastic vs deterministic rounding, convergence speed.
**Expected runtime:** 2+ hours (full training run with QAT).
**Impact:** MEDIUM (improves QAT quality, but only relevant if QAT is adopted).
**Dependencies:** C20 (ParetoQ / QAT).

---

### C25. AutoRound (Signed Gradient Descent)

- [ ] **Status:** NOT TESTED

**Question:** Does Intel AutoRound's signed gradient descent produce better rounding than GPTQ for our model?
**Background:** AutoRound (EMNLP 2024): jointly optimizes weight rounding and clipping ranges via signed gradient descent. 2.1x higher accuracy than baselines at INT2. 128 calibration samples, 200 steps. Competitive with GPTQ at INT4, faster.
**Method:**
```python
# 1. Install auto-round: pip install auto-round
# 2. Prepare calibration data (can use self-generated like GPTQ)
# 3. Run AutoRound quantization to int6/int5
# 4. Export quantized weights
# 5. Compare to GPTQ at same bit width
```
**Record:** artifact size, BPB at int5/int6, quantization time, comparison to GPTQ.
**Expected runtime:** 30 min.
**Impact:** MEDIUM (potentially better than GPTQ, untested in parameter-golf).
**Dependencies:** auto-round pip package.

---

### C26. HQQ (Half-Quadratic Quantization)

- [ ] **Status:** NOT TESTED

**Question:** Does HQQ produce good quantization without any calibration data?
**Background:** HQQ (Badri & Shaji 2023): fast PTQ using half-quadratic optimization. No calibration data needed. Computes optimal quantization in seconds. Competitive with GPTQ at 4-bit, much faster.
**Method:**
```python
# pip install hqq
from hqq.core.quantize import HQQLinear
# Quantize each weight matrix
for name, param in model.named_parameters():
    if param.ndim >= 2:
        hqq_layer = HQQLinear(param, quant_config={'nbits': 6, 'group_size': 64})
```
**Record:** artifact size, BPB, quantization time, comparison to GPTQ and RTN.
**Expected runtime:** 15 min.
**Impact:** MEDIUM (easy to test, calibration-free, competitive quality).
**Dependencies:** hqq pip package.

---

### C27. NF4 (NormalFloat4) via bitsandbytes

- [ ] **Status:** NOT TESTED

**Question:** Does NF4 produce better 4-bit quantization than uniform INT4 for our model?
**Background:** NF4: quantile-based format optimized for normally-distributed weights. 16 quantile values from N(0,1). Best 4-bit format in theory. Used by QLoRA. Non-uniform levels may compress worse than uniform int.
**Method:**
```python
# pip install bitsandbytes
import bitsandbytes as bnb
# NF4 quantize each weight matrix
# Compare compressed size: NF4 vs INT4 vs INT5
```
**Record:** artifact size (NF4 vs int4/int5), BPB, byte entropy of NF4 quantized values.
**Expected runtime:** 15 min.
**Impact:** LOW (non-uniform levels compress worse; mainly useful for quality comparison at 4-bit).
**Dependencies:** bitsandbytes pip package.

---

### C28. SVD Low-Rank + Sparse Residual

- [ ] **Status:** NOT TESTED

**Question:** Does decomposing weight matrices into low-rank + sparse residual compress better than direct quantization?
**Background:** From llm-as-compressor-brainstorm.md. SVD captures most variance in top-R components. Residual is sparse after thresholding. Combined with entropy coding, estimated 1.3-1.8x over zstd.
**Method:**
```python
for name, W in state_dict.items():
    if W.ndim != 2: continue
    U, S, Vt = torch.linalg.svd(W.float())
    # Low-rank: R components
    R = 16
    W_low = (U[:, :R] * S[:R]) @ Vt[:R, :]
    residual = W - W_low
    # Threshold small residuals to zero
    threshold = torch.quantile(residual.abs(), 0.9)
    residual[residual.abs() < threshold] = 0
    # Store: low-rank (int8) + sparse residual (entropy-coded)
```
**Record:** artifact size, reconstruction error, sparsity %, BPB.
**Expected runtime:** 30 min.
**Impact:** MEDIUM (alternative compression paradigm; may or may not beat direct quantization).
**Dependencies:** None.

---

### C29. ZipNN Compression Drop-In

- [ ] **Status:** NOT TESTED

**Question:** Does ZipNN's data-type-aware preprocessing improve compression over plain zstd for our quantized weights?
**Background:** ZipNN (arxiv 2411.05239): lossless compression designed for AI models. Exploits neural network weight file structure. Claims 2-3x improvement over generic compression.
**Method:**
```python
# pip install zipnn
from zipnn import ZipNN
zpn = ZipNN()
compressed = zpn.compress(quant_raw)
# Compare to zstd-22
```
**Record:** compressed size ZipNN vs zstd-22, compression/decompression time.
**Expected runtime:** 10 min.
**Impact:** LOW (designed for float weights; our int8 may not benefit much).
**Dependencies:** zipnn pip package.

---

### C30. Transformer-Specific Per-Tensor-Type Compression

- [ ] **Status:** NOT TESTED

**Question:** Does using different compression strategies for different tensor types (embeddings, attention, MLP) improve overall compression?
**Background:** From llm-as-compressor-brainstorm.md. Different tensor types have different structures: embeddings have adjacent-vocab similarity, attention weights are low-rank, MLP weights are dense. Custom strategies per type could help.
**Method:**
```python
# Embeddings (tok_emb, bigram_emb): delta encoding (store embedding[0] + deltas)
# Attention (q,k,v,proj): SVD low-rank (rank-16) + sparse residual
# MLP (fc, proj): standard int8 + zstd
# Control params: passthrough (already handled)
```
**Record:** per-type compressed size, overall artifact size, BPB.
**Expected runtime:** 30 min.
**Impact:** MEDIUM (incremental per-type optimization).
**Dependencies:** None.

---

## TIER 5 -- Experimental / Research (uncertain payoff)

### C31. INT4 + GPTQ + AWQ

- [ ] **Status:** NOT TESTED

**Question:** Can INT4 quantization work for our 17M model with GPTQ + AWQ?
**Background:** INT4 = 16 levels. Very aggressive for 17M params. Risky -- quality may collapse. But if it works: 3.8MB artifact, room for 2x more params.
**Method:** Combine C08 (int4 clamp), C10 (AWQ-Lite), C11 (GPTQ).
**Record:** BPB, artifact size, quality degradation analysis.
**Expected runtime:** 45 min.
**Impact:** HIGH if quality is acceptable, LOW if it collapses.
**Dependencies:** C10, C11, C12.

---

### C32. Ternary Quantization {-1, 0, 1}

- [ ] **Status:** NOT TESTED

**Question:** Can ternary weights work for our 17M model?
**Background:** CiprianFlorin-Ifrim achieved 1.1239 BPB with 106M ternary params. At 17M params, ternary would be catastrophic. But: could we use ternary for some layers?
**Method:** Ternary quantize least-sensitive layers, keep others at int6.
**Record:** BPB at various ternary/int6 mixes, artifact size.
**Expected runtime:** 30 min.
**Impact:** LOW for 17M (too few params for ternary to work).
**Dependencies:** Per-layer sensitivity analysis.

---

### C33. Entropy Coding with Per-Layer Probability Models

- [ ] **Status:** NOT TESTED

**Question:** Does custom ANS/arithmetic coding with per-layer byte-frequency models beat zstd's generic entropy coder?
**Background:** zstd uses tANS internally with generic byte-level models. A custom entropy coder with per-layer probability tables could be more accurate. Expected gain: 1-3% over zstd.
**Method:**
```python
# For each layer's quantized weights:
# 1. Compute byte histogram
# 2. Use histogram as probability table for arithmetic coding
# 3. Encode layer bytes with learned probabilities
```
**Record:** compressed size custom vs zstd, bits/byte per layer.
**Expected runtime:** 60 min (implement arithmetic coder + test).
**Impact:** LOW (marginal gain over zstd, high implementation effort).
**Dependencies:** None.

---

### C34. Co-Optimized Quantization + Compression

- [ ] **Status:** NOT TESTED

**Question:** Can we optimize quantization levels to minimize COMPRESSED size rather than just MSE?
**Background:** The turboquant finding: quantization that minimizes MSE can produce weights that compress worse. The optimal quantization for artifact size minimizes compressed size, not reconstruction error.
**Method:**
```python
# For each candidate quantization config (clip percentile, scale dtype, group size):
# 1. Quantize weights
# 2. Compress with zstd-22
# 3. Measure compressed size
# 4. Pick config that minimizes compressed size (subject to BPB constraint)
```
**Record:** compressed size for each config, BPB, optimal config.
**Expected runtime:** 30 min.
**Impact:** MEDIUM (could find non-obvious optimal settings).
**Dependencies:** None.

---

## Dependency Graph

```
C12 (Bit Packing) ---+--> C07 (INT6) --> C09 (Mixed INT6/INT5)
                     |                  +--> C13 (LQER)
                     |
                     +--> C08 (INT5) --> C15 (SqueezeLLM)
                     |                  +--> C31 (INT4 + GPTQ)
                     |
                     +--> C20 (ParetoQ QAT) --> C24 (Stochastic Rounding)

C10 (AWQ-Lite) ---+--> C11 (GPTQ) --> C21 (GPTQ + AWQ Combined)
                   |
                   +--> C13 (LQER)

C07 (INT6) --> C09 (Mixed INT6/INT5)
```

## Execution Order (Recommended)

```
Phase 1 -- Baseline (30 min):
  C01 (zstd-22) [DONE] -> C02 (5% pruning) -> C04 (zstd strategy) -> C05 (delta encoding)

Phase 2 -- Quantization (1 hour):
  C12 (bit packing) -> C07 (INT6) -> C08 (INT5) -> C06 (group sweep) -> C09 (mixed)

Phase 3 -- Quality Recovery (1 hour):
  C10 (AWQ-Lite) -> C11 (GPTQ) -> C13 (LQER) -> C21 (GPTQ+AWQ combined)

Phase 4 -- Advanced (2+ hours):
  C16 (codebooks) -> C14 (SpinQuant) -> C15 (SqueezeLLM) -> C18 (cross-layer sharing)

Phase 5 -- Meta-Compression (2+ hours):
  C17 (neural arithmetic coding) -> C28 (SVD+sparse) -> C30 (per-type compression)

Phase 6 -- Exotic (as needed):
  C20 (ParetoQ) -> C25 (AutoRound) -> C26 (HQQ) -> C27 (NF4) -> C22 (FP8 on H100)
```

---

*End of Compression Experiment Queue. 34 experiments total, prioritized by quick wins -> high impact -> unlockers -> exotic.*
