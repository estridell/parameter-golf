#!/usr/bin/env python3
"""Compare compression algorithms on int8-quantized model artifact."""
import os, sys, time, zlib, struct, statistics
import torch
import io

WORKDIR = "/tmp/compression-test"
INT8_PATH = os.path.join(WORKDIR, "final_model.int8.ptz")
RAW_PATH = os.path.join(WORKDIR, "final_model.pt")
RESULTS_PATH = os.path.join(WORKDIR, "results.md")
SCRIPT_COPY = os.path.join(WORKDIR, "test_compression.py")
LIMIT = 16_000_000  # competition limit in bytes

def load_raw_int8_bytes():
    """Get raw bytes of the int8 state dict.
    
    .ptz format: zlib.compress(torch.save(quantized_state_dict))
    We decompress to get the torch-save bytes, then use those as the
    raw payload to recompress with different algorithms.
    """
    with open(INT8_PATH, "rb") as f:
        compressed = f.read()
    torch_bytes = zlib.decompress(compressed)
    # Also load the state dict for L2-norm sorting
    state_dict = torch.load(io.BytesIO(torch_bytes), map_location="cpu", weights_only=False)
    return torch_bytes, state_dict

def compress_zlib(data, level):
    return zlib.compress(data, level)

def compress_zstd(data, level):
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)

def compress_lzma(data):
    import lzma
    return lzma.compress(data, preset=6)

def decompress_zlib(data):
    return zlib.decompress(data)

def decompress_zstd(data):
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)

def decompress_lzma(data):
    import lzma
    return lzma.decompress(data)

def bench(fn, *args, runs=3):
    """Run fn 3 times, return (result, median_time)."""
    results = []
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = fn(*args)
        t1 = time.perf_counter()
        results.append(r)
        times.append(t1 - t0)
    return results[0], statistics.median(times)

def sort_by_l2_norm(state_dict):
    """Sort weight tensor rows by L2 norm before serialization."""
    sorted_sd = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            v_float = v.float()
            norms = v_float.flatten(1).norm(dim=1)
            sorted_idx = norms.argsort()
            sorted_sd[k] = v[sorted_idx]
        else:
            sorted_sd[k] = v
    buf = io.BytesIO()
    torch.save(sorted_sd, buf)
    return buf.getvalue()

def fmt_size(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} MB"
    return f"{n / 1_000:.1f} KB"

def main():
    print("Loading int8 model...")
    raw_bytes, state_dict = load_raw_int8_bytes()
    raw_size = len(raw_bytes)
    print(f"Raw int8 state dict: {raw_size:,} bytes ({fmt_size(raw_size)})")

    # Ensure zstandard is available
    try:
        import zstandard
    except ImportError:
        print("Installing zstandard...")
        os.system(f"{sys.executable} -m pip install zstandard -q --break-system-packages")
        import zstandard

    tests = [
        ("zlib-9 (baseline)", lambda d: compress_zlib(d, 9), decompress_zlib),
        ("zlib-6",            lambda d: compress_zlib(d, 6), decompress_zlib),
        ("zstd-22",           lambda d: compress_zstd(d, 22), decompress_zstd),
        ("zstd-19",           lambda d: compress_zstd(d, 19), decompress_zstd),
        ("zstd-3",            lambda d: compress_zstd(d, 3),  decompress_zstd),
        ("lzma-6",            compress_lzma, decompress_lzma),
    ]

    results = []
    for name, compress_fn, decompress_fn in tests:
        print(f"\nTesting {name}...")
        compressed, comp_time = bench(compress_fn, raw_bytes)
        _, decomp_time = bench(decompress_fn, compressed)
        ratio = raw_size / len(compressed)
        under = len(compressed) <= LIMIT
        results.append({
            "name": name,
            "compressed_size": len(compressed),
            "ratio": ratio,
            "comp_time": comp_time,
            "decomp_time": decomp_time,
            "under_limit": under,
        })
        status = "OK" if under else "OVER LIMIT"
        print(f"  Size: {len(compressed):,} bytes ({fmt_size(len(compressed))}) [{status}]")
        print(f"  Ratio: {ratio:.3f}x")
        print(f"  Compress: {comp_time:.3f}s, Decompress: {decomp_time:.3f}s")

    # L2-norm sorted test
    print("\nSorting weight rows by L2 norm...")
    sorted_bytes = sort_by_l2_norm(state_dict)
    sorted_size = len(sorted_bytes)
    print(f"Sorted int8 state dict: {sorted_size:,} bytes ({fmt_size(sorted_size)})")
    print(f"Size change from sorting: {sorted_size - raw_size:+,} bytes")

    print("\nTesting zstd-22 on L2-sorted data...")
    sorted_compressed, sorted_comp_time = bench(lambda d: compress_zstd(d, 22), sorted_bytes)
    _, sorted_decomp_time = bench(decompress_zstd, sorted_compressed)
    sorted_ratio = sorted_size / len(sorted_compressed)
    sorted_under = len(sorted_compressed) <= LIMIT
    results.append({
        "name": "zstd-22 + L2 sort",
        "compressed_size": len(sorted_compressed),
        "ratio": sorted_ratio,
        "comp_time": sorted_comp_time,
        "decomp_time": sorted_decomp_time,
        "under_limit": sorted_under,
    })
    status = "OK" if sorted_under else "OVER LIMIT"
    print(f"  Size: {len(sorted_compressed):,} bytes ({fmt_size(len(sorted_compressed))}) [{status}]")
    print(f"  Ratio: {sorted_ratio:.3f}x")
    print(f"  Compress: {sorted_comp_time:.3f}s, Decompress: {sorted_decomp_time:.3f}s")

    # Also test sorted with zlib-9
    print("\nTesting zlib-9 on L2-sorted data...")
    sorted_zlib, sorted_zlib_comp = bench(lambda d: compress_zlib(d, 9), sorted_bytes)
    _, sorted_zlib_decomp = bench(decompress_zlib, sorted_zlib)
    sorted_zlib_ratio = sorted_size / len(sorted_zlib)
    sorted_zlib_under = len(sorted_zlib) <= LIMIT
    results.append({
        "name": "zlib-9 + L2 sort",
        "compressed_size": len(sorted_zlib),
        "ratio": sorted_zlib_ratio,
        "comp_time": sorted_zlib_comp,
        "decomp_time": sorted_zlib_decomp,
        "under_limit": sorted_zlib_under,
    })
    status = "OK" if sorted_zlib_under else "OVER LIMIT"
    print(f"  Size: {len(sorted_zlib):,} bytes ({fmt_size(len(sorted_zlib))}) [{status}]")
    print(f"  Ratio: {sorted_zlib_ratio:.3f}x")
    print(f"  Compress: {sorted_zlib_comp:.3f}s, Decompress: {sorted_zlib_decomp:.3f}s")

    # Write markdown report
    with open(RESULTS_PATH, "w") as f:
        f.write("# Compression Comparison: int8 Model Artifact\n\n")
        f.write(f"**Raw int8 state dict:** {raw_size:,} bytes ({fmt_size(raw_size)})\n")
        f.write(f"**Competition limit:** {LIMIT:,} bytes ({fmt_size(LIMIT)})\n")
        f.write(f"**L2-sorted raw:** {sorted_size:,} bytes ({fmt_size(sorted_size)}) ({sorted_size - raw_size:+,} bytes vs unsorted)\n\n")

        f.write("## Results\n\n")
        f.write("| Method | Compressed Size | Ratio | Under 16MB? | Compress Time | Decompress Time |\n")
        f.write("|--------|----------------|-------|-------------|---------------|------------------|\n")
        for r in results:
            check = "Y" if r["under_limit"] else "N"
            f.write(f"| {r['name']} | {r['compressed_size']:,} ({fmt_size(r['compressed_size'])}) | {r['ratio']:.3f}x | {check} | {r['comp_time']:.3f}s | {r['decomp_time']:.3f}s |\n")

        f.write("\n## Analysis\n\n")
        best = min(results, key=lambda r: r["compressed_size"])
        fastest_comp = min(results, key=lambda r: r["comp_time"])
        fastest_decomp = min(results, key=lambda r: r["decomp_time"])
        under_limit = [r for r in results if r["under_limit"]]

        f.write(f"- **Smallest:** {best['name']} at {best['compressed_size']:,} bytes ({best['ratio']:.3f}x)\n")
        f.write(f"- **Fastest compression:** {fastest_comp['name']} at {fastest_comp['comp_time']:.3f}s\n")
        f.write(f"- **Fastest decompression:** {fastest_decomp['name']} at {fastest_decomp['decomp_time']:.3f}s\n")
        f.write(f"- **Under 16MB limit:** {len(under_limit)}/{len(results)} methods\n\n")

        unsorted_best = min((r for r in results if "L2" not in r["name"]), key=lambda r: r["compressed_size"])
        sorted_zstd22 = [r for r in results if r["name"] == "zstd-22 + L2 sort"][0]
        if sorted_zstd22["compressed_size"] < [r for r in results if r["name"] == "zstd-22"][0]["compressed_size"]:
            f.write("- **L2-norm sorting helps!** Sorting weight rows by L2 norm before compression yields smaller files.\n\n")
        else:
            f.write("- **L2-norm sorting does not help** for this model/quantization.\n\n")

        f.write("## Recommendations\n\n")
        f.write("Based on these results:\n\n")
        zstd22 = [r for r in results if r["name"] == "zstd-22"][0]
        zstd22L2 = [r for r in results if r["name"] == "zstd-22 + L2 sort"][0]
        if zstd22L2["compressed_size"] <= zstd22["compressed_size"]:
            savings = zstd22["compressed_size"] - zstd22L2["compressed_size"]
            f.write(f"1. Use **zstd-22 + L2 norm sorting** — saves {savings:,} bytes vs plain zstd-22\n")
        else:
            f.write(f"1. Use **zstd-22** (plain, no L2 sorting needed)\n")

        headroom = LIMIT - best["compressed_size"]
        f.write(f"\nHeadroom to 16MB limit: {headroom:,} bytes ({headroom / LIMIT * 100:.1f}%)\n")

    print(f"\nResults written to {RESULTS_PATH}")

    # Copy script to workdir for reproducibility
    import shutil
    shutil.copy2(__file__, SCRIPT_COPY)
    print(f"Script copied to {SCRIPT_COPY}")

if __name__ == "__main__":
    main()
