# LFM2.5-350M Evaluation for Parameter-Golf

**Date:** 2026-05-18
**Source:** https://huggingface.co/LiquidAI/LFM2.5-350M
**Paper:** arXiv:2511.23404 (LFM2 Technical Report)

---

## Model Overview

| Spec | LFM2.5-350M | Our Model (32M) |
|------|-------------|-----------------|
| Parameters | 350M | 32M |
| Layers | 16 (10 conv + 6 GQA attn) | 11 (all GQA attn) |
| Hidden dim | 1024 | 512 |
| Attn heads | 16 | 8 |
| KV heads | 8 | 4 |
| GQA ratio | 2:1 | 2:1 |
| FFN dim | 6656 (SwiGLU) | ~2048 (LeakyReLU-squared) |
| Vocab | 65,536 | 8,192 |
| Context | 32,768 | 1,024 |
| Tied embeddings | Yes | Yes |
| RoPE theta | 1,000,000 | 10,000 |
| Training tokens | 28T | 10B |
| Optimizer | AdamW | Muon |
| Architecture | Hybrid conv+GQA | Pure GQA |
| Precision | BF16 | BF16 |

## Key Architecture: Gated Short Convolution

LFM2's signature innovation is a **gated short convolution block** that replaces most attention layers:

```
(B, C, h_tilde) = Linear(h)
y = B * h_tilde           (element-wise gate)
z = Conv_k(y)             (depthwise 1D, kernel size k=3)
o = Linear(C * z)         (output gate)
```

10 of 16 layers use this conv block; only 6 use full GQA attention.
Hardware-in-the-loop NAS found this mix optimal for edge latency+quality.

## Liquid AI Background

- **Legit research lab.** MIT CSAIL spinoff (Daniela Rus lab). Co-founders: Ramin Hasani, Mathias Lechner, Alexander Amini.
- **Funded.** $297M total ($46.6M seed Dec 2023, $250M Series A at ~$2B valuation). Investors: Samsung, Shopify, OSS Capital.
- **Published.** LFM2 paper on arXiv, multiple model releases, benchmarks are real and independently verifiable.
- **Not a hype shop.** Serious architecture research with hardware-aware NAS, knowledge distillation, and multi-stage RL post-training.

---

## Techniques We Already Have

| Technique | LFM2.5 | Ours | Notes |
|-----------|--------|------|-------|
| GQA (2:1) | Yes (16:8) | Yes (8:4) | Same ratio, different scale |
| Tied embeddings | Yes | Yes | Saves params, we default ON |
| RoPE | Yes | Yes | They use theta=1M (long context), we use 10K |
| SwiGLU-style FFN | SwiGLU | LeakyReLU-squared | Ours is a custom fused Triton kernel, competitive |
| Depthwise conv | N/A | N/A | Not applicable to our pure-attention arch |

## Techniques That Could Help Us

### 1. Nothing directly applicable.

This is the honest answer. Here's why each promising technique doesn't fit:

**Gated Short Convolutions** -- The conv block is their key innovation, but:
- Designed for CPU edge inference (cache-friendly, no attention overhead)
- We're optimizing for GPU H100 training in 10 minutes, not CPU inference
- At 17-36M params, we can't afford to waste layers on local-only mixing
- Their own NAS found conv+attn beats pure conv -- but we're already pure attn
- Would require kernel work for our sm_75 target too

**SwiGLU** -- Standard choice, but our LeakyReLU-squared is a fused Triton kernel specifically optimized for our pipeline. SwiGLU would need its own kernel or use eager PyTorch, likely slower. Not worth the engineering for marginal quality difference.

**65K vocab** -- Way too expensive for 16MB compression. Our 8192 vocab with SentencePiece is the right call. Embedding table at 65K x 1024 = 262MB alone -- larger than our entire budget.

**Knowledge Distillation** -- They distill from LFM1-7B teacher. We have 10 minutes total. Can't train a teacher first.

**28T tokens** -- We get 10B tokens. 2800x more data isn't a technique, it's a budget.

**RoPE theta=1M** -- Only matters for long context. We're at 1024 tokens.

### 2. One minor idea worth noting:

**Conv for positional encoding** -- Their conv layers implicitly learn local position via the causal kernel. This is a cheaper positional signal than RoPE for very small models. But RoPE is already cheap (16 dims), and we have it working. Not worth switching.

## Techniques Irrelevant to Our Scale

- **Hardware-in-the-loop NAS:** We have 1 architecture, hand-tuned. NAS is for production model families.
- **Multi-stage RL post-training:** We're pre-training only. No SFT/RLHF phase.
- **12-model ensemble curriculum learning:** We don't have 12 models.
- **Model merging (TIES, DARE, DELLA):** Post-training technique, not applicable.
- **Fill-in-the-middle (50% of code data):** We're training on FineWeb, not code.
- **MoE (8B-A1B variant):** Way beyond our scale.
- **Quantization-aware training:** We quantize post-training. QAT during training would slow our 10-min budget.

---

## Bottom Line

**Nothing to adopt.** LFM2.5-350M is a well-engineered model from a serious lab, but every technique that makes it good is either (a) already in our stack (GQA, tied embeds, RoPE), (b) designed for a different target (CPU edge inference vs GPU training), or (c) requires 100-1000x more compute than we have.

The hybrid conv+attention architecture is genuinely interesting research -- it's a real innovation that outperforms pure attention at their scale on edge hardware. But it's the wrong tool for parameter-golf on H100. We're better off doubling down on what we have: BigramHash, SmearGate, CaseOps, AsymLogit, Muon, depth recurrence. These are the techniques that win at our scale and constraints.

**Recommendation:** No code changes. File as reference.
