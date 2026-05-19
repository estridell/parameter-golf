# SOTA 2500-Step Baseline — RTX 2070

**Date:** 2026-05-19
**Machine:** RTX 2070 (sm_75), 8GB VRAM, Arch Linux
**Branch:** main
**Script:** train_gpt_sota.py (35.9M params, ported from #5 leaderboard entry)

## Configuration

```
FA3_ENABLED=0 TRITON_ENABLED=0 COMPRESSOR=brotli
TRAIN_BATCH_TOKENS=8192 VAL_BATCH_TOKENS=65536
GRADIENT_CHECKPOINT_ENABLED=1
CASEOPS_ENABLED=1 SMEAR_GATE_ENABLED=1 LQER_ENABLED=1
LQER_ASYM_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
ITERATIONS=2500 MAX_WALLCLOCK_SECONDS=0 GPTQ_RESERVE_SECONDS=10
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=42
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

## Training Results

| Step | train_loss | train_time | tok/s |
|---|---|---|---|
| 1 | 9.0084 | 0.0m | 3996 |
| 500 | 4.4320 | 17.9m | 3813 |
| 1000 | 3.6889 | 38.0m | 3592 |
| 1500 | 3.6783 | 65.0m | 3152 |
| 2000 | 3.5023 | 91.9m | 2971 |
| 2500 | 3.2991 | 118.4m | 2883 |

**Speed degradation:** 3996 -> 2883 tok/s (-28%) over 2500 steps. Layer loop overhead compounds.

## Validation Results

| Phase | val_loss | val_bpb |
|---|---|---|
| Step 0 (untrained) | 9.0066 | 4.1922 |
| Step 2500 (pre-EMA) | 3.2519 | 1.5136 |
| Step 2500 (post-EMA) | 3.2422 | 1.5091 |
| Post-GPTQ quantized | 16.8195 | 7.8289 |

**EMA impact:** EMA now HELPS (1.5136->1.5091, -0.005 BPB). At 648 steps EMA hurt (+0.156 BPB).
EMA needs sufficient training steps to be beneficial.

**GPTQ impact:** CATASTROPHIC. 1.5091->7.8289 (+6.32 BPB). Worse than the 30-min baseline (5.37 BPB).
More training made GPTQ WORSE, not better. This is NOT a training-length issue.

## Artifacts

| File | Size |
|---|---|
| final_model.pt | 135,417,533 bytes (129 MB) |
| final_model.int6.ptz | 135,320 bytes (132 KB) |
| Code (uncompressed) | 162,912 bytes |
| Code (compressed, brotli) | 40,637 bytes |
| Total submission | 175,957 bytes (172 KB) |

## Memory

| Metric | Value |
|---|---|
| Peak allocated | 2,319 MiB |
| Peak reserved | 2,678 MiB |
| GPU total | 8,192 MiB |

## Run History

- **Run 1:** OOM during post-training validation (cross_entropy needed 2 GiB, only 584 MiB free).
  Fragmentation issue (3.01 GiB reserved but unallocated). No expandable_segments.
- **Run 2:** Succeeded with expandable_segments + VAL_BATCH_TOKENS=65536. GPTQ ran but produced
  catastrophic quantization. TTT crashed with "Invalid backend" on SDPA.

## Critical Finding: GPTQ Degradation Worsens With Training

| Run | Steps | Pre-EMA BPB | Post-EMA BPB | Post-GPTQ BPB | GPTQ Delta |
|---|---|---|---|---|---|
| 30-min | 648 | 1.7588 | 1.9145 | 5.3713 | +3.46 |
| 2500-step | 2500 | 1.5136 | 1.5091 | 7.8289 | +6.32 |

GPTQ degradation DOUBLED with 4x more training. Hypothesis: GPTQ calibration is broken,
not undertrained. Possible causes:
1. Hessians collected too fast (67 in 0.6s) -- calibration data insufficient
2. LQER asymmetric quantization producing degenerate weights
3. int6 quantization too aggressive for this model architecture
4. Brotli compression destroying quantized weight precision

## Next Steps

1. **Investigate GPTQ calibration quality** -- check if Hessians are degenerate
2. **Try int8-only quantization** (skip int6) to see if precision is the issue
3. **Disable LQER_ASYM** and test GPTQ alone
4. **Check actual quantized weight distribution** -- are they all near-zero?
5. **Compare with non-SOTA model's GPTQ** -- does it also degrade?
