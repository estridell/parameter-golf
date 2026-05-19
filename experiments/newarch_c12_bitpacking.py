#!/usr/bin/env python3
"""
Newarch Bit Packing Baseline (C12 adapted)
============================================
Pack INT6 quantized weights using bit packing.
4.80x was the baseline on old arch.
"""
import os, sys, time, json, torch

PG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PG_DIR)
os.chdir(PG_DIR)

from train_gpt import _unbank_state_dict

def log(msg):
    print(f"[NEWARCH-C12] {msg}", flush=True)

def pack_int6(tensor):
    """Pack INT6 values (stored as int8, range [-31,31]) into 6 bits each."""
    flat = tensor.contiguous().view(-1)
    # Shift to unsigned [0, 63]
    unsigned = (flat.int() + 31).to(torch.uint8)
    n = unsigned.numel()
    # Pack 4 values into 3 bytes (4 * 6 bits = 24 bits = 3 bytes)
    padded_n = ((n + 3) // 4) * 4
    padded = torch.zeros(padded_n, dtype=torch.uint8)
    padded[:n] = unsigned
    padded = padded.view(-1, 4)
    # Pack: [a,b,c,d] -> [a<<2|b>>4, (b&0xF)<<4|c>>2, (c&3)<<6|d]
    b0 = (padded[:, 0] << 2) | (padded[:, 1] >> 4)
    b1 = ((padded[:, 1] & 0xF) << 4) | (padded[:, 2] >> 2)
    b2 = ((padded[:, 2] & 3) << 6) | padded[:, 3]
    packed = torch.stack([b0, b1, b2], dim=1).view(-1)
    return packed, n  # return original count for unpacking

def main():
    t0 = time.time()
    log("=" * 60)
    log("Newarch Bit Packing Baseline (INT6)")
    log("=" * 60)

    ckpt_path = "checkpoints/newarch_sp8192.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_layers = 11

    flat_sd = _unbank_state_dict(state, num_layers)
    log(f"Unbanked: {len(flat_sd)} tensors")

    clip_range = 31
    clip_sigmas = 3.0
    packed_total = 0
    metadata_total = 0
    orig_float_bytes = 0
    n_packed = 0

    for name, tensor in flat_sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            metadata_total += t.numel() * 2  # fp16 passthrough
            continue

        orig_float_bytes += t.numel() * 4
        n_packed += 1

        # Quantize to INT6
        row_std = t.float().std(dim=1)
        scale = (clip_sigmas * row_std / clip_range).clamp_min(1e-10)
        Q = torch.clamp(torch.round(t.float() / scale.float().view(-1, 1)), -clip_range, clip_range).to(torch.int8)

        # Pack
        packed, orig_count = pack_int6(Q)
        packed_total += packed.numel()
        metadata_total += t.shape[0] * 2  # per-row scale (fp16)

    total_bytes = packed_total + metadata_total
    orig_bytes = orig_float_bytes + sum(t.numel() * 2 for t in flat_sd.values() if not t.is_floating_point() and t.is_floating_point())
    orig_bytes = sum(t.numel() * 4 for t in flat_sd.values() if t.is_floating_point())

    ratio = orig_bytes / max(total_bytes, 1)

    log(f"Packed tensors: {n_packed}")
    log(f"Original size:  {orig_bytes / 1024 / 1024:.2f} MB")
    log(f"Packed size:    {total_bytes / 1024 / 1024:.2f} MB")
    log(f"  Packed data:  {packed_total / 1024 / 1024:.2f} MB")
    log(f"  Metadata:     {metadata_total / 1024 / 1024:.2f} MB")
    log(f"Compression:    {ratio:.2f}x")
    log(f"Time:           {time.time() - t0:.1f}s")
    log("=" * 60)

    results = {
        "experiment": "NEWARCH_C12_BitPacking",
        "checkpoint": ckpt_path,
        "original_bytes": orig_bytes,
        "packed_bytes": total_bytes,
        "packed_data_bytes": packed_total,
        "metadata_bytes": metadata_total,
        "compression_ratio": ratio,
        "n_packed_tensors": n_packed,
        "time_seconds": time.time() - t0,
    }
    with open("experiments/newarch_c12_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("Results saved to experiments/newarch_c12_results.json")

if __name__ == "__main__":
    main()
