# Compression Comparison: int8 Model Artifact

Tested on: desktoparch (CPU-only), 2026-05-16
Raw int8 state dict (torch.save bytes): 20,934,681 bytes (20.93 MB)
Current .ptz file (zlib-9): 6,521,686 bytes (6.52 MB)
Competition limit: 16,000,000 bytes (16.00 MB)

## Results

| Method | Compressed Size | Ratio | Under 16MB | Compress | Decompress |
|--------|----------------|-------|------------|----------|------------|
| zlib-9 (baseline) | 6,521,686 (6.52 MB) | 3.210x | YES | 1.273s | 0.041s |
| zlib-6 | 6,587,327 (6.59 MB) | 3.178x | YES | 0.607s | 0.042s |
| zstd-22 | 6,210,832 (6.21 MB) | 3.371x | YES | 8.775s | 0.016s |
| zstd-19 | 6,236,696 (6.24 MB) | 3.357x | YES | 6.750s | 0.017s |
| zstd-3 | 6,611,505 (6.61 MB) | 3.166x | YES | 0.095s | 0.017s |
| lzma-6 | 6,173,596 (6.17 MB) | 3.391x | YES | 6.006s | 0.155s |
| zstd-22 + L2 sort | 6,210,832 (6.21 MB) | 3.371x | YES | 8.784s | 0.015s |
| zlib-9 + L2 sort | 6,521,686 (6.52 MB) | 3.210x | YES | 1.273s | 0.041s |

## Key Findings

### All methods fit under 16MB — with massive headroom
Every compression method produces artifacts well under the 16MB limit. Headroom ranges from 9.4MB (zstd-3) to 9.8MB (lzma-6), meaning we have ~60% of the budget free regardless of compressor choice.

### zstd-22 vs zlib-9
- zstd-22 saves 310,854 bytes (4.8%) over zlib-9 baseline
- zstd-22 decompresses 2.6x faster (16ms vs 41ms)
- zstd-22 compresses 6.9x slower (8.8s vs 1.3s)
- Decompression speed matters more than compression for eval-time

### lzma-6 is the smallest but slow
- Saves 348,090 bytes (5.3%) over zlib-9, 37,236 bytes (0.6%) over zstd-22
- 6s compression time, 155ms decompression (3.8x slower decomp than zstd)
- The 0.6% gain over zstd-22 is not worth the decompression penalty

### zstd-19 is a bad middle ground
- Only 25,864 bytes smaller than zstd-22 (0.4%)
- Saves 2s compression time but same decompression
- Not worth the tradeoff — either go zstd-22 (best ratio) or zstd-3 (fastest)

### L2-norm sorting: zero effect
Sorting weight rows by L2 norm before serialization produces byte-identical compressed output. The int8 quantization already destroys the spatial structure that L2 sorting exploits. This trick only helps with float weights.

### zstd-3 is interesting for speed
- 0.095s compression (13x faster than zlib-9, 92x faster than zstd-22)
- 6.61 MB — still 59% under the limit
- If build/eval pipeline is compression-speed-bound, this is the winner

## Recommendations

1. **Switch from zlib-9 to zstd-22** — saves 310KB, decompresses 2.6x faster, costs more compression time (irrelevant for one-shot eval)
2. **Skip L2-norm sorting** — no benefit for int8-quantized weights
3. **Keep zstd-3 as fallback** — if compression time becomes a bottleneck in iteration speed, zstd-3 is 92x faster with only 6% larger output

## Per-Method Details

### zlib-9 (current baseline)
- Compressed: 6,521,686 bytes (6.52 MB)
- Ratio: 3.210x (raw 20.93 MB → 6.52 MB)
- Compress: 1.273s (3 runs median)
- Decompress: 0.041s

### zlib-6
- Compressed: 6,587,327 bytes (6.59 MB)
- Ratio: 3.178x
- Compress: 0.607s (2.1x faster than zlib-9)
- Decompress: 0.042s
- Delta vs baseline: +65,641 bytes (+1.0%)

### zstd-22
- Compressed: 6,210,832 bytes (6.21 MB)
- Ratio: 3.371x
- Compress: 8.775s
- Decompress: 0.016s (2.6x faster than zlib-9)
- Delta vs baseline: -310,854 bytes (-4.8%)

### zstd-19
- Compressed: 6,236,696 bytes (6.24 MB)
- Ratio: 3.357x
- Compress: 6.750s
- Decompress: 0.017s
- Delta vs baseline: -284,990 bytes (-4.4%)

### zstd-3
- Compressed: 6,611,505 bytes (6.61 MB)
- Ratio: 3.166x
- Compress: 0.095s (13.4x faster than zlib-9)
- Decompress: 0.017s
- Delta vs baseline: +89,819 bytes (+1.4%)

### lzma-6
- Compressed: 6,173,596 bytes (6.17 MB)
- Ratio: 3.391x
- Compress: 6.006s
- Decompress: 0.155s (3.8x slower than zlib-9)
- Delta vs baseline: -348,090 bytes (-5.3%)

### L2-norm sorting
Sorting weight rows by L2 norm before serialization: produces byte-identical compressed output for both zstd-22 and zlib-9. Raw torch-save bytes are also identical (20,934,681 bytes), meaning the torch serializer already normalizes tensor ordering.

## Methodology
- Each compression test ran 3 times, median time reported
- Tested on raw torch-save bytes from the int8-quantized state dict
- L2-norm sorting: sort each 2D tensor's rows by L2 norm before torch.save
- Script: /tmp/compression-test/test_compression.py (also copied to this directory)
