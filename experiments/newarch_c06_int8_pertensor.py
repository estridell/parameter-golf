#!/usr/bin/env python3
"""
Newarch INT8 Per-Tensor Quantization (C06 adapted)
====================================================
Quantize all bank weights to INT8 with per-tensor scaling.
Pass-through small params (smear_gate, skip_weights, etc.)
"""
import os, sys, time, json, torch

PG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PG_DIR)
os.chdir(PG_DIR)

from train_gpt import _unbank_state_dict, _rebank_state_dict

def log(msg):
    print(f"[NEWARCH-C06] {msg}", flush=True)

def main():
    t0 = time.time()
    log("=" * 60)
    log("Newarch INT8 Per-Tensor Quantization")
    log("=" * 60)

    ckpt_path = "checkpoints/newarch_sp8192.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_layers = 11
    model_dim = 512
    kv_dim = 4 * (512 // 8)  # num_kv_heads * head_dim
    hidden_dim = int(4.0 * 512)

    flat_sd = _unbank_state_dict(state, num_layers)
    log(f"Unbanked: {len(flat_sd)} tensors")

    # Quantize: INT8 per-tensor for large float tensors, passthrough for small/other
    q_sd = {}
    meta = {}
    for name, tensor in flat_sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            q_sd[name] = t.to(torch.float16)
            meta[name] = "passthrough"
            continue

        # INT8 per-tensor
        absmax = t.float().abs().max()
        scale = (absmax / 127.0).clamp_min(1e-10)
        Q = torch.clamp(torch.round(t.float() / scale), -127, 127).to(torch.int8)
        q_sd[name + ".q"] = Q
        q_sd[name + ".scale"] = scale
        meta[name] = "int8_pertensor"

    # Compute sizes
    orig_bytes = sum(t.numel() * 4 for t in flat_sd.values() if t.is_floating_point())
    quant_bytes = 0
    for name, tensor in flat_sd.items():
        if not tensor.is_floating_point():
            continue
        info = meta.get(name, "")
        if info == "passthrough":
            quant_bytes += tensor.numel() * 2
        elif info == "int8_pertensor":
            quant_bytes += tensor.numel() * 1 + 4  # int8 + scalar float

    ratio = orig_bytes / max(quant_bytes, 1)

    # Dequantize and compute MSE
    deq_sd = {}
    for name, orig in flat_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        if info == "passthrough":
            deq_sd[name] = q_sd[name].to(orig.dtype)
        elif info == "int8_pertensor":
            Q = q_sd[name + ".q"]
            s = q_sd[name + ".scale"]
            deq_sd[name] = (Q.float() * s).to(orig.dtype)

    # MSE across all quantized tensors
    mse_total = 0.0
    mse_count = 0
    for name, orig in flat_sd.items():
        if meta.get(name) == "int8_pertensor":
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
        "experiment": "NEWARCH_C06_INT8_Pertensor",
        "checkpoint": ckpt_path,
        "original_bytes": orig_bytes,
        "quantized_bytes": quant_bytes,
        "compression_ratio": ratio,
        "mse": mse,
        "tensor_categories": {k: sum(1 for v in meta.values() if v == k) for k in set(meta.values())},
        "time_seconds": time.time() - t0,
    }
    with open("experiments/newarch_c06_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("Results saved to experiments/newarch_c06_results.json")

if __name__ == "__main__":
    main()
