# Evaluation: FareedKhan-dev/train-llm-from-scratch

**Repo:** https://github.com/FareedKhan-dev/train-llm-from-scratch
**Date:** 2026-05-18
**Evaluator:** trainer

## What the Repo Implements

A tutorial-level vanilla GPT-2 transformer from scratch in PyTorch. The repo is structured as an educational resource (blog post companion) for building a ~13M parameter LLM. It includes:

- **Architecture:** Standard transformer with learned positional embeddings, vanilla multi-head attention (no GQA), MLP with ReLU activation (no SwiGLU/GELU), LayerNorm (no RMSNorm)
- **Training:** AdamW optimizer with step LR decay at fixed step, no warmup schedule, no gradient clipping
- **Data:** Pile dataset, HDF5 format, basic shuffled batch iterator
- **Configs:** 13M param (128-dim, 1 block) and 3B param (2048-dim, 64 blocks) presets
- **Extras:** SFT/RLHF notebook (DPO + PPO tutorial), text generation script
- **No:** Distributed training, mixed precision, gradient checkpointing, custom kernels, quantization, compression

## Techniques We Already Have (No Action Needed)

| Technique | Their Version | Our Version |
|-----------|--------------|-------------|
| Multi-head attention | Vanilla MHA, 8 heads | GQA (8 heads, 4 KV heads) |
| Positional encoding | Learned embeddings | RoPE with YaRN scaling |
| Normalization | LayerNorm | RMSNorm |
| MLP activation | ReLU | SwiGLU (up/down/gate) |
| Optimizer | AdamW (fixed step decay) | Muon (learned Newton-Schulz coefficients) |
| Sequence length | 128-512 | 1024 (with RoPE extrapolation) |
| Softcap | None | Logit softcap=30 (fused Triton CE kernel) |
| Token embedding | Standard | Tied embeddings |

## Techniques That Could Help Us

**None.** Every technique in this repo is strictly inferior to what we already have. The repo implements the bare minimum Attention is All You Need architecture with no modern improvements.

Specific gaps in their code that we've already solved:
- No grouped-query attention (we use GQA for KV cache efficiency)
- No rotary position embeddings (we use RoPE + YaRN)
- No SwiGLU/GEGLU activation (we use SwiGLU)
- No RMSNorm (we use RMSNorm)
- No warmup + cosine schedule (we use warmup + cosine)
- No gradient checkpointing (we support it via env var)
- No fused kernels (we have fused softcapped CE via Triton)
- No quantization-aware training or post-training compression (we have int8+zlib)
- No depth recurrence / layer sharing (we have looping layers)
- No SmearGate, CaseOps, AsymLogit, or any parameter-golf specific techniques

## Techniques That Are Irrelevant or Inferior

- **Learned positional embeddings:** Strictly inferior to RoPE for length generalization. We already use RoPE with YaRN.
- **ReLU activation:** Outdated. SwiGLU/GELU are strictly better for transformer LMs.
- **Standard MHA:** We use GQA which reduces KV parameters and improves throughput.
- **Step LR decay:** We use warmup + cosine decay which is more stable.
- **HDF5 data format:** We use memory-mapped binary shards which are faster for distributed training.
- **No gradient accumulation:** Their code does single-batch updates. We use gradient accumulation for effective larger batch sizes.

## Bottom Line

**Not worth adopting anything.** This is a beginner tutorial repo that implements the 2017 Attention is All You Need paper with zero modern improvements. Every technique it uses, we have already surpassed with better alternatives. The repo has no novel research, no custom kernels, no compression techniques, and no training optimizations relevant to the parameter-golf challenge.

The only mildly interesting aspect is that it includes SFT/RLHF code (DPO + PPO), but this is also tutorial-level and irrelevant to our 10-minute training constraint.

**Classification:** Tutorial/Educational — not a source of techniques for competitive parameter-golf.
