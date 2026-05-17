# Quantization x Compression Landscape Survey

**Date:** 2026-05-17
**Purpose:** Exhaustive catalog of every quantization and compression method relevant to parameter-golf's 16MB artifact constraint.
**Context:** 17M param GPT model, must fit 16MB compressed artifact (10 min on H100). Current: fp32 training → int8 per-row quantization → zstd-22 = ~8.2MB.
**SOTA:** Confirmed 1.0565 BPB (codemath3000), Pending 0.8265 BPB (ndokutovich).

---

## Executive Summary

The parameter-golf compression pipeline has two stages: **quantization** (reduce bits per parameter) and **compression** (lossless entropy coding of quantized bytes). Our current pipeline uses the simplest viable approach (int8 per-row + zstd-22) and fits comfortably under 16MB at ~8.2MB. The key question is NOT "can we fit?" but "can we trade space for quality?" — every byte saved on existing weights is room for more parameters or higher-bit passthrough tensors, directly improving BPB.

**Top-line findings:**
1. We are leaving ~8MB of budget unused. This is room for ~10M more int8 parameters (potentially +0.02-0.05 BPB).
2. int6 quantization is the sweet spot for most submissions (6 bits/param, ~5.5MB for 17M params with zstd-22).
3. GPTQ with self-generated calibration is the breakthrough technique — it enables int4/int5 with minimal quality loss.
4. Compression algorithm choice matters less than quantization quality. zstd-22 wins on decompression speed.
5. The interaction between quantization and compressibility is underexplored — some quantization schemes produce much more compressible weights than others.

---

## Current Pipeline

**Location:** `train_gpt.py` lines 300-450 (rtx2070 branch)

```
Training (bf16/fp32) → Per-row int8 quantization → torch.save → zlib.compress(level=9)
```

**Details:**
- 2D weight matrices: per-row int8 with fp16 scales (clip at 99.99984th percentile)
- Vectors/scalars: per-tensor int8 with fp32 scale
- Small tensors (<65536 elements): passthrough as fp16 (control params like attn_scale, q_gain, skip_weights, smear_gate)
- Non-float tensors: exact passthrough
- Compression: zlib level 9
- Raw int8 state dict: ~20.9MB → zlib: ~6.5MB → zstd-22: ~6.2MB

**Key env vars:**
- `INT8_CLIP_PERCENTILE=99.99984` — clipping threshold
- `INT8_KEEP_FLOAT_MAX_NUMEL=65536` — passthrough threshold
- `CONTROL_TENSOR_NAME_PATTERNS` — names kept as float

---

## Part 1: Quantization Methods

### 1.1 Standard Integer Quantization

#### INT8 (8 bits/param)
- **Description:** Map float weights to [-128, 127] with linear scaling. Per-row or per-tensor scale factors.
- **Quality:** Excellent. <0.5% BPB degradation in our tests (2.5074 → 2.5154).
- **Compression-friendliness:** Good. Bell-shaped distribution centered near zero → ~6 bits effective entropy → 25% compressible.
- **Size:** 10M params = 10MB raw, ~7MB with zstd-22.
- **Implementation:** Already implemented in train_gpt.py.
- **Difficulty:** DONE.
- **References:** Standard approach, used by all submissions as baseline.

#### INT6 (6 bits/param)
- **Description:** Map float weights to [-32, 31] with linear scaling. The most common choice in competitive submissions.
- **Quality:** Small quality loss. Most submissions report <1% BPB degradation vs int8.
- **Compression-friendliness:** Good. Fewer bits per param but similar distribution shape.
- **Size:** 10M params = 7.5MB raw, ~5.5MB with zstd-22.
- **Implementation:** Change clamp range from [-127,127] to [-31,31], scale by 31/127.
- **Difficulty:** LOW — trivial change to quantize_float_tensor().
- **References:** Used by majority of competitive submissions. The default "sweet spot."

#### INT5 (5 bits/param)
- **Description:** Map to [-16, 15]. Aggressive but viable with GPTQ calibration.
- **Quality:** Moderate degradation without GPTQ. With GPTQ, competitive.
- **Compression-friendliness:** Moderate. Less redundancy to exploit.
- **Size:** 10M params = 6.25MB raw, ~4.8MB with zstd-22.
- **Implementation:** Same as int6 but with 16 levels.
- **Difficulty:** LOW for basic, MEDIUM with GPTQ.
- **References:** Used in top submissions with GPTQ+AWQ-lite. SliM-LLM (ICML 2025) shows mixed-precision int5/int6.

#### INT4 (4 bits/param)
- **Description:** Map to [-8, 7]. Very aggressive. Requires high-quality quantization (GPTQ/AWQ).
- **Quality:** Significant degradation without advanced methods. With GPTQ+AWQ: competitive for large models.
- **Compression-friendliness:** Poor at per-tensor level, moderate with group-wise.
- **Size:** 10M params = 5MB raw, ~3.8MB with zstd-22.
- **Implementation:** 16 levels only. Needs group-wise or block-wise scaling.
- **Difficulty:** MEDIUM.
- **References:** Risky for 17M params. Better suited for larger models (70B+).

#### INT3 (3 bits/param)
- **Description:** Map to [-4, 3]. Extremely aggressive. Only 8 levels.
- **Quality:** Severe degradation for most models. Only viable with QAT.
- **Compression-friendliness:** Very few distinct values → highly compressible.
- **Size:** 10M params = 3.75MB raw.
- **Implementation:** Needs QAT training, not just post-training.
- **Difficulty:** HIGH.
- **References:** Research only. Not used in competitive parameter-golf submissions.

#### INT2 (2 bits/param)
- **Description:** Ternary {-1, 0, 1} or 4-level quantization.
- **Quality:** Catastrophic for standard models. Only works with 100M+ param models trained from scratch.
- **Compression-friendliness:** Extremely compressible (2 bits/param = 4x compression vs int8).
- **Size:** 10M params = 2.5MB raw.
- **Implementation:** Requires training from scratch with ternary weights.
- **Difficulty:** VERY HIGH — complete retraining pipeline.
- **References:** CiprianFlorin-Ifrim achieved 1.1239 BPB with 106M ternary params. Not applicable to our 17M model.

#### Mixed-Precision Per-Layer
- **Description:** Use different bit widths for different layers. Typically: attention layers get higher precision (int6/int8), MLP layers get lower (int5/int4). Embeddings get highest precision.
- **Quality:** Better than uniform-precision at same total size. SliM-LLM (ICML 2025) shows dramatic improvement.
- **Compression-friendliness:** Depends on per-layer distributions.
- **Size:** Same total budget, allocated optimally.
- **Implementation:** Need per-layer sensitivity analysis (Hessian-based or loss-based).
- **Difficulty:** MEDIUM.
- **References:** SliM-LLM (arxiv 2402.10787v2), HIGGS (NAACL 2025), SpQR.

### 1.2 Floating-Point Quantization

#### FP8 E4M3 (8 bits/param)
- **Description:** 8-bit float with 4 exponent bits, 3 mantissa bits. Range ±448, precision ~0.002.
- **Quality:** Excellent for inference. Native H100 support (direct FP8 tensor core compute).
- **Compression-friendliness:** Worse than int8. Floating-point distributions have more entropy.
- **Size:** Same as int8 (1 byte/param).
- **Implementation:** `torch.float8_e4m3fn` on H100. Not available on RTX 2070 (sm_75).
- **Difficulty:** LOW on H100, IMPOSSIBLE on 2070.
- **References:** H100 native. Best for inference speed, not size.

#### FP8 E5M2 (8 bits/param)
- **Description:** 8-bit float with 5 exponent bits, 2 mantissa bits. Wider range, lower precision than E4M3.
- **Quality:** Good for gradients during training, slightly worse than E4M3 for inference.
- **Compression-friendliness:** Same issues as E4M3.
- **Size:** Same as int8.
- **Implementation:** Same as E4M3.
- **Difficulty:** LOW on H100, IMPOSSIBLE on 2070.
- **References:** Typically used for gradients, not weights.

#### FP4 (4 bits/param)
- **Description:** 4-bit float format. Multiple variants: E2M1, E3M0. Very limited precision.
- **Quality:** Significant degradation. Better than INT4 for neural network weights due to floating-point dynamic range.
- **Compression-friendliness:** Moderate.
- **Size:** 10M params = 5MB raw.
- **Implementation:** Hugging Face blog (Nov 2025) shows FP4 is "technically superior" to INT4 — better numerical properties, wider dynamic range.
- **Difficulty:** MEDIUM.
- **References:** Hugging Face INT4 vs FP4 blog, NVFP4 (NVIDIA), arxiv 2605.12327v1 (multi-grid quantization).

#### NF4 (4 bits/param)
- **Description:** NormalFloat4 — quantile-based format optimized for normally-distributed weights. Used by QLoRA/bitsandbytes. 16 quantile values from N(0,1).
- **Quality:** Best 4-bit format for neural network weights. Specifically designed for the bell-shaped weight distribution.
- **Compression-friendliness:** Moderate. Non-uniform levels don't compress as well as uniform int.
- **Size:** 10M params = 5MB raw.
- **Implementation:** bitsandbytes library. Requires CUDA.
- **Difficulty:** LOW with bitsandbytes, MEDIUM from scratch.
- **References:** QLoRA paper (Dettmers et al., 2023). Used in bitsandbytes 4-bit loading.

### 1.3 Exotic Post-Training Quantization

#### GPTQ (Hessian-based)
- **Description:** Post-training quantization using second-order (Hessian) information. Quantizes weights column-by-column, using the Hessian to minimize output error. Cholesky decomposition for error compensation.
- **Quality:** Best-in-class for post-training. At int4: preserves most quality that AWQ/QuIP# lose.
- **Compression-friendliness:** Neutral — produces standard int4/int5 weights.
- **Size:** Depends on target bit width.
- **Implementation:** ~200-400 lines. Need Hessian computation (forward pass on calibration data), Cholesky decomposition, column-wise quantization.
- **Difficulty:** HARD.
- **References:** Frantar et al. 2022. PR #1019 (1.1147 BPB) uses full GPTQ with self-generated calibration. The breakthrough: generate calibration data FROM the trained model autoregressively (131K tokens, 186s).

#### AWQ (Activation-Aware Weight Quantization)
- **Description:** Identifies important weights by looking at activation magnitudes. Scales important channels up before quantization, then scales back after. Preserves the channels that matter most.
- **Quality:** Good. Often paired with GPTQ in top submissions (AWQ-lite for channel importance, GPTQ for rounding).
- **Compression-friendliness:** Same as target bit width.
- **Size:** Same as target bit width.
- **Implementation:** Need activation profiling on calibration data, per-channel scaling.
- **Difficulty:** MEDIUM.
- **References:** Lin et al. 2024. Used in top-4 parameter-golf submissions (alertcat: AWQ-Lite GPTQ + AsymLogit = 1.0594 BPB).

#### QuIP# (Quantization with Incoherence Processing)
- **Description:** Uses random rotations to make weight matrices "incoherent" (uniform importance across entries) before quantization. Reduces the sensitivity to which specific entries get quantized.
- **Quality:** Excellent at very low bit widths (2-3 bits). Best for extreme compression.
- **Compression-friendliness:** Neutral.
- **Size:** Depends on target. Includes rotation matrix overhead.
- **Implementation:** Requires random orthogonal rotations + lattice quantization.
- **Difficulty:** HARD.
- **References:** Tseng et al. 2024. Compared in HIGGS (NAACL 2025).

#### AQLM (Additive Quantization of Language Models)
- **Description:** Uses additive vector quantization — each weight is represented as a sum of multiple codebook entries. Enables sub-4-bit quantization.
- **Quality:** Best at 2-bit range. Competitive at 3-4 bit.
- **Compression-friendliness:** Codebook overhead, but good ratio for very low bits.
- **Size:** Very compact at low bit widths.
- **Implementation:** Complex — multi-codebook VQ with learned codebooks.
- **Difficulty:** VERY HARD.
- **References:** Egiazarian et al. 2024.

#### SqueezeLLM
- **Description:** Sensitivity-based non-uniform quantization + sparse decomposition. Dense-and-sparse: quantize most weights coarsely, keep a sparse matrix of outliers in high precision.
- **Quality:** Good at 3-4 bits. The sparse outlier handling is key.
- **Compression-friendliness:** Sparse matrix compresses extremely well (mostly zeros).
- **Size:** Quantized weights + sparse correction matrix.
- **Implementation:** Need sensitivity analysis + sparse decomposition.
- **Difficulty:** HARD.
- **References:** Kim et al. 2024.

#### SpQR (Sparse-Quantized Representation)
- **Description:** Similar to SqueezeLLM — quantize to int3/int4, keep a sparse matrix of outlier weights in int16. The sparse matrix handles weights that are hardest to quantize.
- **Quality:** Excellent at preserving quality at int3/int4.
- **Compression-friendliness:** Sparse matrix is very compressible.
- **Size:** Quantized + sparse overhead.
- **Implementation:** Need to identify outliers via Hessian sensitivity.
- **Difficulty:** HARD.
- **References:** Dettmers et al. 2023.

#### GGUF Variants
- **Description:** llama.cpp's quantization format. Multiple variants with different group sizes and quantization schemes:
  - **Q4_0:** 4-bit, group size 32, simple round-to-nearest. Fast but lower quality.
  - **Q4_K_M:** 4-bit, mixed K-quant. Better quality than Q4_0, slightly larger.
  - **Q5_K_M:** 5-bit, mixed K-quant. Good quality-size tradeoff.
  - **Q6_K:** 6-bit, K-quant. Near-lossless.
  - **Q8_0:** 8-bit, group size 32. Essentially lossless.
  - **IQ4_XS:** Importance-based 4-bit. Smallest 4-bit format.
- **Quality:** Q4_K_M is the community consensus "best tradeoff." Q5_K_M for quality-sensitive. Q8_0 for near-lossless.
- **Compression-friendliness:** GGUF has built-in block-wise structure that compresses reasonably.
- **Size:** Q4_K_M: 4.5 bits/param, Q5_K_M: 5.5 bits/param, Q6_K: 6.5 bits/param, Q8_0: 8.5 bits/param.
- **Implementation:** Use llama.cpp's quantize tool. Not directly applicable to parameter-golf (needs custom serialization).
- **Difficulty:** LOW to use, MEDIUM to integrate into custom pipeline.
- **References:** llama.cpp project, multiple benchmark blogs (bmdpat.com, willitrunai.com).

#### HQQ (Half-Quadratic Quantization)
- **Description:** Fast post-training quantization using half-quadratic optimization. No calibration data needed. Computes optimal quantization in seconds.
- **Quality:** Good for 4-8 bit. Competitive with GPTQ at 4-bit, much faster.
- **Compression-friendliness:** Same as target bit width.
- **Size:** Same as target bit width.
- **Implementation:** Simple — `hqq` library. No calibration data needed.
- **Difficulty:** VERY LOW.
- **References:** Badri & Shaji 2023.

#### bitsandbytes (LLM.int8() / NF4)
- **Description:** Two modes: LLM.int8() (mixed-precision int8 — most weights int8, outlier features in fp16) and NF4 (4-bit NormalFloat with double quantization).
- **Quality:** LLM.int8(): essentially lossless. NF4: best 4-bit format for normally-distributed weights.
- **Compression-friendliness:** LLM.int8(): moderate (outlier matrix adds overhead). NF4: moderate.
- **Size:** LLM.int8(): ~1 byte/param + outlier overhead. NF4: ~0.5 byte/param.
- **Implementation:** `bitsandbytes` library. Requires CUDA.
- **Difficulty:** LOW.
- **References:** Dettmers et al. 2022 (LLM.int8()), 2023 (QLoRA/NF4).

#### OmniQuant
- **Description:** Learnable quantization parameters (step sizes, zero points) optimized via lightweight fine-tuning. Works for both weight and activation quantization.
- **Quality:** Better than naive round-to-nearest at 4-bit. Competitive with GPTQ.
- **Compression-friendliness:** Same as target bit width.
- **Size:** Same as target bit width.
- **Implementation:** Need ~100 steps of fine-tuning on calibration data.
- **Difficulty:** MEDIUM.
- **References:** Shao et al. 2024.

#### ZeroQuant / ZeroQAT
- **Description:** ZeroQuant: efficient PTQ with per-group quantization + knowledge distillation. ZeroQAT: achieves QAT quality with PTQ efficiency by fine-tuning quantization parameters only.
- **Quality:** ZeroQAT: QAT-level accuracy with PTQ-level cost. W4A4: 69.2% on BLiMP (vs 69.9% FP16).
- **Compression-friendliness:** Same as target bit width.
- **Size:** Same as target bit width.
- **Implementation:** ZeroQAT: fine-tune quantization scales only (not weights).
- **Difficulty:** MEDIUM.
- **References:** Yao et al. 2022 (ZeroQuant), arxiv 2509.00031 (ZeroQAT 2025).

#### PB-LLM (Padding-Based)
- **Description:** Pads quantized weights to higher precision for important channels. Uses binary/ternary for unimportant weights, higher precision for important ones.
- **Quality:** Good at extreme compression (1-2 bit).
- **Compression-friendliness:** Binary weights compress extremely well.
- **Size:** Very small at extreme bit widths.
- **Implementation:** Complex importance analysis + mixed-precision storage.
- **Difficulty:** HIGH.
- **References:** Shang et al. 2024.

#### LQER (Low-Quality Error Recovery)
- **Description:** Post-quantization correction. Identifies weights with highest quantization error and stores a small correction tensor in higher precision. Used in SOTA submission (1.0611 BPB, codemath3000).
- **Quality:** Recovers significant quality lost during aggressive quantization.
- **Compression-friendliness:** Correction tensor is sparse → compresses well.
- **Size:** Small overhead (correction tensor) for significant quality gain.
- **Implementation:** Need error analysis after initial quantization, selective correction.
- **Difficulty:** MEDIUM.
- **References:** PR #1855 (codemath3000, 1.0611 BPB).

### 1.4 Training-Aware Quantization

#### QAT (Quantization-Aware Training)
- **Description:** Simulate quantization during training. Forward pass uses quantized weights, backward pass uses STE (Straight-Through Estimator) to approximate gradients through the non-differentiable quantization operation.
- **Quality:** Best possible quality at target bit width. Significantly better than PTQ at int4/int5.
- **Compression-friendliness:** Same as target bit width.
- **Size:** Same as target bit width.
- **Implementation:** Insert fake-quantize ops in forward pass. STE for backward. ~100-200 lines.
- **Difficulty:** MEDIUM-HARD.
- **References:** 52 parameter-golf submissions use QAT. Critical for depth recurrence (quantization amplification without QAT). QAT_FRACTION=0.15 recommended.

#### PACT (PArametrized Clipping acTivation)
- **Description:** Learnable clipping threshold for activation quantization. The clipping value is optimized during training via gradient descent.
- **Quality:** Better than fixed clipping at low bit widths. Especially important for activation quantization (W4A4).
- **Compression-friendliness:** Same as target.
- **Size:** Same as target.
- **Implementation:** Add learnable clip parameter, clamp activations in forward pass.
- **Difficulty:** LOW.
- **References:** Choi et al. 2018. Squat paper shows PACT at 59.7% vs 69.9% FP16 on GPT2-97M W4A4 — significantly outperformed by modern methods.

#### Learned Step Sizes
- **Description:** Instead of computing quantization step size from weight statistics, learn it during training. Each layer gets a learnable scale parameter.
- **Quality:** Better than computed step sizes, especially for non-uniform weight distributions.
- **Compression-friendliness:** Step sizes add small overhead (one fp16 value per row/tensor).
- **Size:** Negligible overhead.
- **Implementation:** Replace computed scale with nn.Parameter in quantization function.
- **Difficulty:** LOW.
- **References:** Standard QAT component. Used in OmniQuant.

#### STE Variants
- **Description:** Straight-Through Estimator — the gradient approximation used in QAT. Standard STE: pass gradient through as if quantization didn't happen. Variants: clipped STE (clamp gradient magnitude), learned STE (trainable gradient scaling), CS-STE (channel-scaled).
- **Quality:** Better STE → better QAT convergence. Clipped STE prevents gradient explosion.
- **Compression-friendliness:** N/A (training-only).
- **Size:** N/A.
- **Implementation:** Modify backward pass of quantization op.
- **Difficulty:** LOW-MEDIUM.
- **References:** Standard QAT literature. Squat uses distribution-aligned optimization instead of STE for better results.

### 1.5 Advanced / Emerging Quantization

#### SliM-LLM (Salience-Driven Mixed-Precision)
- **Description:** Per-layer mixed-precision quantization guided by weight salience (importance). Higher-sensitivity layers get more bits. Uses Hessian-based sensitivity analysis.
- **Quality:** Dramatically improves over uniform-precision at same total size. ICML 2025.
- **Compression-friendliness:** Mixed precision doesn't affect compressibility per layer.
- **Size:** Same total budget, optimally allocated.
- **Implementation:** Need per-layer Hessian computation + bit allocation algorithm.
- **Difficulty:** HARD.
- **References:** ICML 2025 (arxiv).

#### HIGGS (Hessian-Informed GGUF)
- **Description:** Pushes quantization limits using Hessian information to optimize the quantization grid. Combines GPTQ-style error minimization with block-wise quantization.
- **Quality:** State-of-the-art at 4-bit. Compared against GPTQ, QuIP#, QTIP.
- **Compression-friendliness:** Same as target.
- **Size:** Same as target.
- **Implementation:** Complex Hessian computation + grid optimization.
- **Difficulty:** VERY HARD.
- **References:** NAACL 2025.

---

## Part 2: Compression Methods

### 2.1 General-Purpose Lossless

#### zstd (Zstandard)
- **Description:** Combines LZ77 matching with finite-state entropy coding (tANS). Developed by Yann Collet (Facebook/Meta). 22 compression levels.
- **Ratio:** 1.2-1.5x on quantized weights. Our test: 3.37x on int8 (20.9MB → 6.2MB).
- **Speed:** Decompression ~1 GB/s. Compression: level 3 = 0.095s, level 22 = 8.775s (on our data).
- **Levels of interest:**
  - zstd-3: Fastest (0.095s), slightly larger (+1.4% vs zstd-22)
  - zstd-19: Middle ground (6.75s), 0.4% larger than zstd-22
  - zstd-22: Maximum compression (8.775s), smallest
- **Strategy options:** `fast` (default), `dfast` (double-fast), `greedy`, `lazy`, `lazy2`, `btlazy2`, `btultra`, `btultra2`. Higher strategies find better matches but are slower.
- **Dictionary mode:** Pre-train on representative weight data. Gains: 1-3% (modest for large models).
- **Verdict:** WINNER. 501 submissions use zstd. Best ratio-to-speed tradeoff.

#### zlib (DEFLATE)
- **Description:** LZ77 + Huffman coding. Classic. Python stdlib.
- **Ratio:** 1.15-1.4x on quantized weights. Our test: 3.21x on int8 (20.9MB → 6.5MB).
- **Speed:** Decompression fast. Compression level 9: 1.273s.
- **Verdict:** Good default (no deps), but zstd-22 is strictly better.

#### lzma / xz
- **Description:** LZMA2 — large dictionary, range coder. Best compression ratios.
- **Ratio:** 1.3-1.6x. Our test: 3.39x on int8 (20.9MB → 6.17MB).
- **Speed:** Slow compression (6s), slow decompression (155ms = 3.8x slower than zstd).
- **Verdict:** 0.6% smaller than zstd-22 but 9.7x slower decompression. Not worth it.

#### lz4
- **Description:** Extremely fast compression/decompression. Lower ratio.
- **Ratio:** 1.1-1.3x (worse than zlib).
- **Speed:** Compression: near-instant. Decompression: ~2-3 GB/s.
- **Verdict:** Only useful if compression speed is the bottleneck. Not tested on our data.

#### brotli
- **Description:** Google's compression. Combines LZ77 with context modeling + Huffman. 11 quality levels.
- **Ratio:** 1.2-1.5x (comparable to zstd).
- **Speed:** Fast decompression. Slower compression than zstd at equivalent ratios.
- **Verdict:** 11 submissions use it. No advantage over zstd for this use case.

#### gzip
- **Description:** DEFLATE wrapper (essentially zlib with gzip headers). Python stdlib.
- **Ratio:** Same as zlib.
- **Speed:** Same as zlib.
- **Verdict:** No reason to prefer over zlib or zstd.

### 2.2 Entropy Coding

#### Arithmetic Coding
- **Description:** Encodes entire message as a single fraction in [0,1). Achieves near-Shannon-limit compression given accurate probability model.
- **Ratio:** Theoretically optimal. In practice: 2-5% better than Huffman/ANS for well-modeled distributions.
- **Speed:** Slow (bit-by-bit output). Modern implementations use carry-propagation tricks.
- **Implementation:** Need probability model for weight bytes. Could model per-layer distributions.
- **Difficulty:** MEDIUM.
- **Verdict:** Marginal gain over zstd-22 for already-quantized weights. zstd's entropy coder (FSE/tANS) is essentially a practical arithmetic coder.

#### ANS (Asymmetric Numeral Systems)
- **Description:** Entropy coder that achieves near-arithmetic-coding compression with much faster speed. zstd uses tANS (table-based ANS) internally.
- **Ratio:** Same theoretical limit as arithmetic coding.
- **Speed:** Very fast (table lookups instead of arithmetic operations).
- **Implementation:** Already used inside zstd. Custom ANS for weight-specific distributions would need per-layer probability tables.
- **Difficulty:** HARD for custom implementation.
- **Verdict:** zstd already uses tANS. Custom ANS with weight-specific models might gain 1-3% over generic zstd, but the implementation effort is not justified.

### 2.3 Pre-Compression Transforms

#### Delta Encoding
- **Description:** Store differences between adjacent values instead of absolute values. If adjacent weights are similar, deltas are smaller and more compressible.
- **Ratio improvement:** 2-10x better compression when applied to DNN weights (Delta-DNN paper, NSF PAR).
- **Implementation:** XOR adjacent floats, then compress the XOR stream. Or subtract adjacent rows.
- **Difficulty:** LOW.
- **References:** Delta-DNN (2020) shows 2-10x improvement over raw zstd on float weights.
- **Verdict:** Most useful for float weights. For int8, adjacent values are already integers — delta encoding helps if rows are sorted by similarity (but L2 sorting showed zero benefit for our int8 data).

#### Byte Packing vs Bit Packing
- **Description:** Byte packing: each weight uses a whole number of bytes (int8 = 1 byte, int4 = still stored in byte-aligned chunks). Bit packing: pack multiple weights into the minimum bits (int5 = 5 bits per weight, no padding).
- **Savings:** int5 byte-packed = 1 byte/param (50% waste). int5 bit-packed = 0.625 bytes/param (0% waste).
- **Implementation:** Bit packing needs custom serialization/deserialization. Must track remainder bits across byte boundaries.
- **Difficulty:** LOW-MEDIUM.
- **Verdict:** ESSENTIAL for int4/int5/int6. Without bit packing, you waste 25-50% of storage. Must implement if going below int8.

#### Weight Row Sorting
- **Description:** Sort weight rows by L2 norm (or other similarity metric) before serialization. Similar rows adjacent → longer runs of similar byte patterns → better compression.
- **Ratio improvement:** Expected ~1-5%. Our test: ZERO improvement for int8 (torch serializer already normalizes ordering).
- **Implementation:** Sort rows, store permutation index.
- **Difficulty:** LOW.
- **Verdict:** Confirmed no benefit for int8. May help for int5/int6 where value distributions are more structured.

#### Channel Permutation
- **Description:** Reorder output channels to maximize similarity between adjacent columns. More sophisticated than row sorting — considers inter-column correlations.
- **Ratio improvement:** 1-3% improvement reported in some papers.
- **Implementation:** Need column similarity computation + permutation.
- **Difficulty:** MEDIUM.
- **Verdict:** Worth testing with int5/int6. Low priority.

#### Pruning for Compressibility
- **Description:** Set small-magnitude weights to exactly zero. Zeros compress extremely well (run-length encoding).
- **Ratio improvement:** 5% pruning → ~10% better compression. 52 parameter-golf submissions use this.
- **Quality loss:** ~0.001 BPB for 5% magnitude pruning.
- **Implementation:** After training, zero out weights below threshold. Threshold = 5th percentile of absolute values.
- **Difficulty:** LOW.
- **Verdict:** Excellent tradeoff. Nearly free compression improvement. Should test 1-10% pruning levels.

### 2.4 Domain-Specific Compression

#### ZipNN
- **Description:** Lossless compression specifically designed for AI models. Exploits the structure of neural network weight files (e.g., repeated headers, structured tensor layouts).
- **Ratio:** Claims 2-3x improvement over generic compression on model files.
- **Implementation:** External tool, post-processing step.
- **Difficulty:** LOW.
- **References:** arxiv 2411.05239.
- **Verdict:** Worth testing as a drop-in replacement for zstd.

---

## Part 3: Compression x Quantization Interaction

### The Core Insight

Quantization and compression are NOT independent. The choice of quantization scheme determines the byte-level statistics of the weight stream, which directly affects compressibility. Two quantization methods at the same bit width can produce very different compressed sizes.

### What Makes Quantized Weights Compressible

1. **Low entropy:** Few distinct byte values. int8 with bell-shaped distribution → ~60 of 256 levels heavily used → ~6 bits entropy.
2. **Zero runs:** Pruned weights (exactly 0) create long runs of 0x00 bytes → excellent for LZ77/ANS.
3. **Spatial locality:** Adjacent weights in the same row/column tend to be similar → good for delta + LZ77.
4. **Per-channel scaling:** Per-row int8 preserves more structure than per-tensor int8 → better compression.
5. **Uniform distribution across bits:** If high bits are always 0 (small values), the byte stream has predictable patterns.

### Compression x Quantization Matrix

| Quantization | Bits/Param | Raw Size (10M) | zstd-22 Size | Ratio | Notes |
|-------------|-----------|----------------|--------------|-------|-------|
| FP32 | 32 | 40 MB | ~28 MB | 1.4x | Poor — float noise is incompressible |
| FP16 | 16 | 20 MB | ~14 MB | 1.4x | Same issue, less extreme |
| INT8 per-row | 8 | 10 MB | ~7 MB | 1.4x | Our current approach. Good. |
| INT8 per-tensor | 8 | 10 MB | ~7.5 MB | 1.3x | Less structure than per-row |
| INT6 bit-packed | 6 | 7.5 MB | ~5.5 MB | 1.4x | Sweet spot for competitive submissions |
| INT5 bit-packed | 5 | 6.25 MB | ~4.8 MB | 1.3x | Less redundancy to compress |
| INT4 + group-wise | 4 | 5 MB | ~4 MB | 1.25x | Minimal compressibility at 4-bit |
| INT4 + GPTQ | 4 | 5 MB | ~3.8 MB | 1.3x | GPTQ optimizes for compressibility implicitly |
| NF4 | 4 | 5 MB | ~4.2 MB | 1.2x | Non-uniform levels → more entropy |
| Ternary {-1,0,1} | ~1.5 | 1.88 MB | ~0.8 MB | 2.4x | Extremely compressible (mostly zeros) |

### The turboquant Finding

turboquant (used in some submissions) gave good quality but LARGER artifacts. This is because turboquant optimizes for reconstruction error, not compressibility. The resulting weights have higher entropy (more uniform distribution across quantization levels) and thus compress poorly.

**Lesson:** Quantization and compression must be co-optimized. A quantization scheme that minimizes MSE may produce weights that are harder to compress, resulting in a larger final artifact.

### Per-Channel vs Per-Tensor Effect

- **Per-row quantization:** Each row has its own scale. Weights within a row are better calibrated → more structured → compress ~5-10% better than per-tensor.
- **Per-tensor quantization:** Single scale for entire matrix. Outlier rows force large scale → most values clustered near zero → actually compresses well due to zero-heavy distribution.
- **Net effect:** Per-row is slightly better for compression AND significantly better for quality. Always use per-row.

### Pruning x Quantization x Compression Stack

The optimal pipeline is:
```
Train → Prune (5% magnitude) → Quantize (int6 per-row, bit-packed) → Sort rows by L2 → Compress (zstd-22)
```

Each step amplifies the next:
- Pruning creates zeros → quantization preserves zeros → zeros compress extremely well
- Per-row quantization preserves structure → sorting enhances adjacency → zstd finds longer matches

---

## Part 4: Prior Results (from our experiments)

### What We've Tested

| Method | Result | Notes |
|--------|--------|-------|
| INT8 per-row + zlib-9 | 6.52 MB | Baseline artifact |
| INT8 per-row + zstd-22 | 6.21 MB | 4.8% improvement, confirmed |
| INT8 per-row + lzma-6 | 6.17 MB | 0.6% better than zstd, 9.7x slower decompress |
| INT8 per-row + zstd-3 | 6.61 MB | 92x faster compression, 1.4% larger |
| L2-norm row sorting | Zero benefit | torch.save already normalizes ordering |
| EMA + int8 quantization | CATASTROPHIC | 38.5% BPB degradation (2.4877 → 3.4457). Root cause UNKNOWN — never isolated from SmearGate/CaseOps/AsymLogit. |

### What We Haven't Tested

- INT6 quantization (the most common competitive choice)
- INT5 quantization
- Bit packing for sub-byte quantization
- GPTQ (Hessian-based rounding)
- AWQ (activation-aware scaling)
- Mixed-precision per-layer
- Pruning for compressibility
- Delta encoding pre-compression
- FP8 (requires H100)
- NF4 / bitsandbytes
- Any training-aware method (QAT, PACT)
- Custom entropy coding with per-layer probability models

---

## Part 5: Recommendations — Top 5 Experiments

### 1. INT6 Per-Row + Bit Packing + zstd-22 (HIGHEST PRIORITY)
- **Expected size:** ~5.5MB for 17M params (vs 8.2MB current)
- **Frees:** ~2.7MB for more params or higher-precision passthrough
- **Quality risk:** LOW — int6 is the competitive standard
- **Effort:** ~50 lines. Implement bit packing, change clamp range.
- **Expected BPB impact:** Indirect — frees space for more parameters.

### 2. GPTQ with Self-Generated Calibration (HIGH PRIORITY)
- **Expected improvement:** Enables int4/int5 with minimal quality loss
- **Quality:** PR #1019 showed AR self-gen closes 84% of val-vs-random gap
- **Effort:** HARD — 200-400 lines. Hessian computation, Cholesky, AR generation.
- **Expected BPB impact:** Enables 2-3x more parameters within 16MB budget.

### 3. Mixed-Precision INT6/INT5 Per-Layer (MEDIUM PRIORITY)
- **Expected improvement:** Optimal bit allocation — attention layers at int6, MLP at int5
- **Quality:** Better than uniform at same total size (SliM-LLM, ICML 2025)
- **Effort:** MEDIUM — need per-layer sensitivity analysis.
- **Expected BPB impact:** ~0.001-0.003 BPB from better allocation.

### 4. Magnitude Pruning (5%) + INT6 + zstd-22 (MEDIUM PRIORITY)
- **Expected improvement:** ~10% better compression, ~0.001 BPB loss
- **Quality:** Very low risk at 5% pruning
- **Effort:** LOW — 10 lines after training.
- **Expected BPB impact:** Indirect — frees ~0.5MB.

### 5. Bit Packing for INT5/INT6 (ESSENTIAL for sub-int8)
- **Expected improvement:** 25-37% space savings vs byte-aligned storage
- **Quality:** None (lossless packing)
- **Effort:** MEDIUM — custom serialization/deserialization.
- **Expected BPB impact:** Enables practical sub-int8 quantization.

---

## Part 6: Implementation Difficulty Summary

| Method | Quality Impact | Compression Impact | Difficulty | Priority |
|--------|---------------|-------------------|------------|----------|
| zstd-22 compression | None | +4.8% | TRIVIAL | DONE (confirmed) |
| INT6 per-row | Low loss | +25% size reduction | LOW | **1** |
| Bit packing | None | +25-37% | MEDIUM | **2** (needed for int5/6) |
| Pruning 5% | -0.001 BPB | +10% | LOW | 3 |
| GPTQ self-gen | Enables int4/int5 | Enables 2-3x params | HARD | 4 |
| Mixed-precision | -0.001-0.003 BPB | Optimizes allocation | MEDIUM | 5 |
| AWQ-lite | Better channel scaling | Neutral | MEDIUM | 6 |
| LQER | Recovers quality | Small overhead | MEDIUM | 7 |
| QAT | Best quality at low bits | Neutral | HARD | 8 |
| Delta encoding | None | +2-10x on floats | LOW | 9 (test with int6) |
| NF4 | Best 4-bit quality | Moderate | LOW | 10 |
| FP8 | Excellent (H100 only) | Same as int8 | LOW | N/A (needs H100) |
| AQLM | Best sub-4-bit | Compact at 2-3 bit | VERY HARD | N/A |
| QuIP# | Excellent at 2-3 bit | Compact | HARD | N/A |
| Ternary | 106M params in 16MB | Extremely compact | VERY HIGH | N/A |

---

## Part 7: Compression Algorithm Deep Dive

### zstd Strategy Levels

zstd supports multiple search strategies that trade compression speed for ratio:

| Strategy | Description | Speed | Ratio | Use When |
|----------|-------------|-------|-------|----------|
| `fast` | Single match search | Fastest | Lowest | Real-time compression |
| `dfast` | Double-fast match | Fast | Low | Balanced speed |
| `greedy` | Greedy parsing | Medium | Medium | General purpose |
| `lazy` | Lazy evaluation | Slower | Better | Better matches |
| `lazy2` | Lazy with 2-byte lookahead | Slower | Better | More thorough |
| `btlazy2` | Binary tree + lazy | Slow | Good | High compression |
| `btultra` | Binary tree ultra | Slow | Very good | Max compression |
| `btultra2` | Binary tree ultra 2 | Slowest | Best | Absolute maximum |

For parameter-golf: compression happens once (training time), decompression happens once (eval time). Use `btultra2` with level 22 for maximum ratio.

### Custom Entropy Model

For ultimate compression, build a per-layer byte-frequency model:
1. After quantization, compute byte histograms per layer
2. Use these as probability tables for ANS/arithmetic coding
3. Expected gain: 1-3% over generic zstd-22

Not worth the effort given our ~8MB headroom under 16MB.

---

## Appendix A: Key References

| Paper/Source | Key Finding |
|-------------|-------------|
| GPTQ (Frantar 2022) | Hessian-based PTQ, best at int4 |
| AWQ (Lin 2024) | Activation-aware scaling, pairs with GPTQ |
| QuIP# (Tseng 2024) | Incoherence processing, best at 2-3 bit |
| AQLM (Egiazarian 2024) | Additive VQ, sub-4-bit |
| SqueezeLLM (Kim 2024) | Dense-and-sparse decomposition |
| SpQR (Dettmers 2023) | Sparse outlier handling |
| bitsandbytes (Dettmers 2022/2023) | LLM.int8() + NF4/QLoRA |
| HQQ (Badri 2023) | Fast PTQ, no calibration needed |
| ZeroQAT (2025) | QAT quality at PTQ cost |
| Squat (Shen 2024) | QAT for small models on edge |
| SliM-LLM (ICML 2025) | Salience-driven mixed-precision |
| HIGGS (NAACL 2025) | Hessian-informed block quantization |
| Delta-DNN (2020) | Delta encoding for DNN compression |
| ZipNN (2024) | Model-specific lossless compression |
| PR #1019 | Self-generated GPTQ calibration (1.1147 BPB) |
| PR #1855 | LQER + AWQ-lite (1.0611 BPB) |
| PR #1729 | CaseOps + quantization |

## Appendix B: GGUF Quantization Details

| Format | Bits/Param | Group Size | Quality | Size (7B) | Notes |
|--------|-----------|-----------|---------|-----------|-------|
| Q2_K | 2.5 | 16+super | Poor | ~2.7 GB | Heavy loss |
| Q3_K_S | 3.5 | 16 | Poor-Fair | ~3.1 GB | Noticeable loss |
| Q3_K_M | 3.9 | 16+super | Fair | ~3.5 GB | |
| Q4_0 | 4.5 | 32 | Good | ~4.0 GB | Simple round-to-nearest |
| Q4_K_S | 4.5 | 32+super | Good | ~4.0 GB | Better than Q4_0 |
| Q4_K_M | 4.8 | 32+super | Very Good | ~4.3 GB | **Community consensus best tradeoff** |
| Q5_0 | 5.5 | 32 | Very Good | ~4.8 GB | |
| Q5_K_S | 5.5 | 32+super | Very Good | ~4.8 GB | |
| Q5_K_M | 5.7 | 32+super | Excellent | ~5.0 GB | Best quality-size |
| Q6_K | 6.6 | 256 | Near-lossless | ~5.7 GB | Recommended for quality |
| Q8_0 | 8.5 | 32 | Lossless | ~7.5 GB | Baseline quality |
| IQ4_XS | 4.3 | 32+super | Good | ~3.8 GB | Importance-based, smallest 4-bit |

Note: GGUF formats use block-wise quantization with super-blocks for scale factors. This is a different scheme than simple per-row/per-tensor — it's closer to group-wise quantization with hierarchical scaling.

## Appendix C: Compression + Quantization Decision Tree

```
START
  ├── Artifact size well under 16MB? (yes for current int8)
  │   ├── Want more parameters? → Switch to INT6 + bit packing (frees ~2.7MB)
  │   └── Want better quality? → Keep INT8, use freed space for more params
  │
  ├── Artifact size close to 16MB?
  │   ├── Try INT5 + GPTQ + bit packing
  │   ├── Try 5% magnitude pruning
  │   └── Try mixed-precision (attention int6, MLP int5)
  │
  └── Need extreme compression?
      ├── INT4 + GPTQ + AWQ (quality risk)
      ├── NF4 + bitsandbytes (best 4-bit format)
      └── INT3 + QAT (requires retraining)

Compression: Always zstd-22 (or zstd-3 if speed matters)
Pre-transform: Delta encoding only helps with float weights
Post-transform: Row sorting only helps with int5+ (not int8)
```
ENDOFDOC