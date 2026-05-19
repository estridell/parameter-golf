# SOTA train_gpt.py Port to RTX 2070

Date: 2026-05-19
Source: `records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/train_gpt.py` (3753 lines)

## Changes Made

### 1. FA3_ENABLED gate (import + 4 call sites)
- Import gated behind `FA3_ENABLED=1` (default on)
- `flash_attn_varlen_func` (cu_seqlens path): per-document causal SDPA loop
- `flash_attn_3_func` (3 causal paths): SDPA with `is_causal=True`
- GQA expansion: SDPA doesn't handle GQA natively, so K/V heads are `repeat_interleave`-d to match Q heads

### 2. TRITON_ENABLED gate (import + 3 kernel blocks)
- Triton import gated behind `TRITON_ENABLED=1` (default on)
- Softcapped CE kernels + ops + autograd + `softcapped_cross_entropy()` all gated
- `linear_leaky_relu_square` kernel + `FusedLinearLeakyReLUSquareFunction` + `FusedLeakyReLUSquareMLP` gated
- `MLP.use_fused = TRITON_ENABLED` (falls back to `F.leaky_relu().square()` when off)
- `fused_ce_enabled` forced to 0 when Triton off (uses eager softcap + F.cross_entropy)

### 3. lrzip fallback
- If `COMPRESSOR=pergroup` and `lrzip` not found, auto-fallback to brotli

### 4. torch.compile conditional
- Disabled when `FA3_ENABLED=0` (varlen fallback creates dynamic shapes incompatible with `fullgraph=True`)
- Affects: train model, eval model, forward_logits, TTT compile

### 5. SDP backend config
- `sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)` around SDPA calls
- Required because sm_75 doesn't support flash SDP

## Key Findings

### Varlen Attention Fallback
- **Quadratic mask OOM**: A (total_len, total_len) float mask for 65536 tokens = 16GB. OOMs on 8GB VRAM.
- **Solution**: Per-document SDPA loop. Process each document separately with `is_causal=True`. Slower but O(max_doc_len^2) memory.
- **Performance**: ~3565 tok/s with per-document loop vs ~6563 tok/s with FA3/varlen (both without torch.compile)

### GQA + SDPA
- SDPA doesn't support grouped query attention natively (8 query heads, 4 KV heads)
- Must expand K/V heads via `repeat_interleave(reps, dim=1)` before SDPA

### torch.compile + Dynamic Shapes
- `torch.compile(fullgraph=True)` fails when varlen fallback creates dynamic-shaped masks
- `torch.compiler.disable` creates graph breaks incompatible with `fullgraph=True`
- Solution: skip compile entirely when FA3 is off

## Smoke Test Results (2070, 120s, NUM_LOOPS=0)

| Metric | Value |
|--------|-------|
| Model params | 35,942,599 |
| Throughput | 3565 tok/s |
| Train VRAM | 2706 MiB (warmup), 5771 MiB (peak alloc) |
| Steps | 51 |
| Train loss | 9.01 → 5.36 |
| Val loss | 5.30 |
| Val BPB | 2.47 |
| GPTQ int6 + brotli | 150,753 bytes |

## Usage

```bash
FA3_ENABLED=0 TRITON_ENABLED=0 COMPRESSOR=brotli \
TRAIN_BATCH_TOKENS=8192 GRADIENT_CHECKPOINT_ENABLED=1 \
CASEOPS_ENABLED=1 SMEAR_GATE_ENABLED=1 LQER_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
python3 -u train_gpt_sota.py
```

On H100s (default): just run `python3 train_gpt_sota.py` — FA3 and Triton are on by default.
