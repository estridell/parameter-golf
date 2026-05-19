#!/usr/bin/env python3
"""
Newarch INT6 Per-Row Quantization (C07 adapted)
=================================================
Quantize bank weights to INT6 with per-row scaling.
Best compression ratio from overnight experiments.
"""
import os, sys, time, json, torch

PG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PG_DIR)
os.chdir(PG_DIR)

from train_gpt import _unbank_state_dict, _rebank_state_dict

def log(msg):
    print(f"[NEWARCH-C07] {msg}", flush=True)

def main():
    t0 = time.time()
    log("=" * 60)
    log("Newarch INT6 Per-Row Quantization")
    log("=" * 60)

    ckpt_path = "checkpoints/newarch_sp8192.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_layers = 11

    flat_sd = _unbank_state_dict(state, num_layers)
    log(f"Unbanked: {len(flat_sd)} tensors")

    clip_range = 31  # 2^5 - 1 for INT6
    q_sd = {}
    meta = {}
    for name, tensor in flat_sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            q_sd[name] = t.to(torch.float16)
            meta[name] = "passthrough"
            continue

        # INT6 per-row: clip sigmas = 3.0 (conservative)
        clip_sigmas = 3.0
        row_std = t.float().std(dim=1)
        scale = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
        Q = torch.clamp(torch.round(t.float() / scale.float().view(-1, 1)), -clip_range, clip_range).to(torch.int8)
        q_sd[name + ".q"] = Q
        q_sd[name + ".scale"] = scale
        meta[name] = "int6_perrow"

    # Compute sizes
    orig_bytes = sum(t.numel() * 4 for t in flat_sd.values() if t.is_floating_point())
    quant_bytes = 0
    for name, tensor in flat_sd.items():
        if not tensor.is_floating_point():
            continue
        info = meta.get(name, "")
        if info == "passthrough":
            quant_bytes += tensor.numel() * 2
        elif info == "int6_perrow":
            quant_bytes += int(tensor.numel() * 0.75) + tensor.shape[0] * 2

    ratio = orig_bytes / max(quant_bytes, 1)

    # Dequantize and compute MSE
    deq_sd = {}
    for name, orig in flat_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        if info == "passthrough":
            deq_sd[name] = q_sd[name].to(orig.dtype)
        elif info == "int6_perrow":
            Q = q_sd[name + ".q"]
            s = q_sd[name + ".scale"]
            deq_sd[name] = (Q.float() * s.float().view(-1, 1)).to(orig.dtype)

    mse_total = 0.0
    mse_count = 0
    for name, orig in flat_sd.items():
        if meta.get(name) == "int6_perrow":
            diff = (deq_sd[name].float() - orig.float())
            mse_total += (diff ** 2).sum().item()
            mse_count += orig.numel()
    mse = mse_total / max(mse_count, 1)

    log(f"Original size:  {orig_bytes / 1024 / 1024:.2f} MB")
    log(f"Quantized size: {quant_bytes / 1024 / 1024:.2f} MB")
    log(f"Compression:    {ratio:.2f}x")
    log(f"MSE:            {mse:.8f}")
    log(f"Time:           {time.time() - t0:.1f}s")
    log("=" * 60)

    results = {
        "experiment": "NEWARCH_C07_INT6_PerRow",
        "checkpoint": ckpt_path,
        "original_bytes": orig_bytes,
        "quantized_bytes": quant_bytes,
        "compression_ratio": ratio,
        "mse": mse,
        "clip_sigmas": 3.0,
        "tensor_categories": {k: sum(1 for v in meta.values() if v == k) for k in set(meta.values())},
        "time_seconds": time.time() - t0,
    }
    with open("experiments/newarch_c07_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("Results saved to experiments/newarch_c07_results.json")

if __name__ == "__main__":
    main()
