# Model Scaling Report: Filling the 16MB Artifact Budget

**Date:** 2026-05-16
**Machine:** RTX 2070 (8GB VRAM, sm_75)
**Branch:** rtx2070
**Protocol:** 5-min smoke tests (300s wallclock), SEED=42, TRAIN_BATCH_TOKENS=65536

## Summary

All layer counts (10-12) compress well under the 15MB artifact budget.
**VRAM (8GB), not artifact size, is the binding constraint.** 13 layers OOMs.
12 layers is the maximum for the 2070, leaving 8.95MB of artifact budget unused.

For the actual submission (8xH100, 80GB VRAM), the model can be scaled much
further. Consider 16-20+ layers if VRAM permits.

## Results

| Layers | Params     | Steps | Step Time | VRAM Alloc | VRAM Rsvd | Compressed | Total Sub  | Loss @stop |
|--------|------------|-------|-----------|------------|-----------|------------|------------|------------|
| 9*     | ~17,060K   | ~65   | 3.72s     | 5,561 MiB  | 5,966 MiB | ~6.90 MB   | ~6.95 MB   | ~5.55      |
| 10     | 18,897,488 | 73    | 4.13s     | 6,098 MiB  | 6,384 MiB | 6.04 MB    | 6.09 MB    | 5.45       |
| 11     | 20,734,552 | 67    | 4.52s     | 6,636 MiB  | 6,916 MiB | 6.52 MB    | 6.57 MB    | 5.48       |
| 12     | 22,572,128 | 61    | 4.94s     | 7,174 MiB  | 7,404 MiB | 7.00 MB    | 7.05 MB    | 5.57       |
| 13     | OOM        | -     | -         | -          | -         | -          | -          | -          |

*9-layer baseline from prior runs (240s test).

## Per-Layer Scaling

- **Param delta:** ~1,837K params/layer (consistent across 10-12)
- **Compressed size delta:** ~470 KB/layer (int8+zlib level 9)
- **Step time delta:** ~400ms/step
- **VRAM delta:** ~540 MiB/layer
- **Compression ratio:** 3.92-3.93x (int8 payload → zlib)

## Artifact Budget Analysis

- **Budget:** 16,000,000 bytes (16 MB)
- **Code size:** ~48 KB
- **Headroom for weights:** ~15.95 MB
- **12 layers uses:** 7.05 MB (44% of budget)
- **Remaining:** 8.95 MB → room for ~19 more layers (by artifact size alone)

## VRAM Constraint

| Config  | VRAM Used | Headroom | Status |
|---------|-----------|----------|--------|
| 10 layers | 6.4 GB | 1.2 GB | OK |
| 11 layers | 6.9 GB | 0.7 GB | OK |
| 12 layers | 7.4 GB | 0.2 GB | OK (tight) |
| 13 layers | >7.6 GB | - | OOM |

## BPB Estimates (from training loss at 5-min mark)

Training loss at ~5 min is a proxy for BPB. The actual submission trains for
10 min on 8xH100s, so these are directional only.

| Layers | Loss @ stop | Est. BPB (loss/ln2/bytes_per_tok) |
|--------|-------------|-----------------------------------|
| 10     | 5.45        | ~2.24 (very rough)                |
| 11     | 5.48        | ~2.25                             |
| 12     | 5.57        | ~2.29                             |

NOTE: These losses are at the 5-min mark. Larger models have higher loss at
equal wallclock time because they take longer per step (fewer steps). The
advantage of more layers shows up with longer training — at 10 min on H100s,
11-12 layers should outperform 9 layers.

## Recommendations

1. **For 2070 local iteration:** 12 layers is the max. Use it for architecture
   experiments but expect ~20% fewer steps than 9 layers in the same time.

2. **For H100 submission:** Scale to 16-20+ layers. The artifact budget allows
   it (7 MB headroom = ~15 more layers). H100 VRAM (80GB) won't be the
   bottleneck. Flash attention will make each step fast.

3. **Leaderboard context:** Current leaders use 11 layers (1.11-1.12 BPB).
   With 12 layers and proper training, matching or beating that should be
   achievable.

4. **Dim scaling:** model_dim could also be increased (currently 512), but
   this changes head_dim (512/8=64). Dim=576 would give head_dim=72, dim=640
   gives head_dim=80. Worth testing on H100 if VRAM permits.

## Methodology Notes

- All runs used math SDP backend (no FlashAttention on sm_75)
- torch.compile disabled (10+ min compilation for marginal gains)
- Final roundtrip validation skipped for speed (adds ~15 min per run)
- Training loss logged every step (TRAIN_LOG_EVERY=1)
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True used
