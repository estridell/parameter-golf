#!/usr/bin/env python3
"""
C10: AWQ-Lite — Activation-aware Weight Quantization
=====================================================
Compute channel importance using activations from a calibration dataset.
Protect top 1% channels from quantization error.
Mix of INT4/INT8 per-channel based on sensitivity.

Uses train_gpt.py infrastructure: _unbank_state_dict, collect_hessians, gptq_quantize_weight.
"""
import os, sys, time, math, json, collections

# CRITICAL: Set env vars BEFORE importing train_gpt (Hyperparameters reads them at class definition time)
os.environ["VOCAB_SIZE"] = "1024"
os.environ["NUM_LOOPS"] = "0"

import torch
import torch.nn.functional as F

# Setup path
PG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PG_DIR)
os.chdir(PG_DIR)

from train_gpt import (
    GPT, Hyperparameters, DocumentPackingLoader,
    _unbank_state_dict, restore_fp32_params,
    gptq_quantize_weight, _build_cu_seqlens,
)

BOS_ID = 1


def log(msg):
    print(f"[C10-AWQ] {msg}", flush=True)


def load_model_and_data():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(False)
    enable_mem_efficient_sdp(True)
    enable_math_sdp(True)

    h = Hyperparameters()
    h.train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", "16384"))
    h.grad_accum_steps = 1

    model = GPT(h).to(device).bfloat16()
    restore_fp32_params(model)

    ckpt_path = os.environ.get("CKPT_PATH", "checkpoints/baseline_sp1024.pt")
    log(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    train_loader = DocumentPackingLoader(h, device)
    return model, h, device, train_loader


def collect_activations(model, train_loader, h, device, n_batches=32):
    activations = {}
    hooks = []

    def make_hook(name, layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            key = f"{name}_L{layer_idx}"
            if key not in activations:
                activations[key] = x.T @ x
            else:
                activations[key].addmm_(x.T, x)
        return hook_fn

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_hook("attn", i)))
        hooks.append(block.mlp.register_forward_hook(make_hook("mlp", i)))

    model.eval()
    with torch.no_grad():
        for batch_idx in range(n_batches):
            x, y, cu_seqlens, max_seqlen = train_loader.next_batch(
                h.train_batch_tokens, h.grad_accum_steps
            )
            model(x, y, cu_seqlens=cu_seqlens, max_seqlen=h.train_seq_len)
            if (batch_idx + 1) % 8 == 0:
                log(f"  Calibration batch {batch_idx + 1}/{n_batches}")

    for hook in hooks:
        hook.remove()

    for key in activations:
        activations[key] = activations[key].cpu() / n_batches

    return activations


def compute_channel_importance(activations, n_layers):
    importance = {}
    for i in range(n_layers):
        attn_key = f"attn_L{i}"
        if attn_key in activations:
            diag = activations[attn_key].diag()
            importance[f"blocks.{i}.attn.c_q.weight"] = diag.clone()
            importance[f"blocks.{i}.attn.c_k.weight"] = diag.clone()
            importance[f"blocks.{i}.attn.c_v.weight"] = diag.clone()
            importance[f"blocks.{i}.attn.proj.weight"] = diag.clone()
            importance[f"blocks.{i}.mlp.fc.weight"] = diag.clone()
            importance[f"blocks.{i}.mlp.proj.weight"] = diag.clone()
    return importance


def awq_quantize_weight(w, channel_importance, clip_sigmas=3.0, bits=4, protect_fraction=0.01):
    W = w.float().contiguous()
    rows, cols = W.shape
    row_std = W.std(dim=1)

    if channel_importance is not None and channel_importance.numel() >= cols:
        imp = channel_importance[:cols]
    else:
        imp = W.abs().sum(dim=0)

    n_protect = max(1, int(cols * protect_fraction))
    _, protect_indices = torch.topk(imp, n_protect)
    protect_mask = torch.zeros(cols, dtype=torch.bool)
    protect_mask[protect_indices] = True

    int8_range = 127
    int4_range = 7

    s4 = (clip_sigmas * row_std / int4_range).clamp_min(1e-10).to(torch.float16)
    s8 = (clip_sigmas * row_std / int8_range).clamp_min(1e-10).to(torch.float16)

    Q = torch.zeros(rows, cols, dtype=torch.int8)
    for j in range(cols):
        if protect_mask[j]:
            q_col = torch.clamp(torch.round(W[:, j] / s8.float()), -int8_range, int8_range)
            Q[:, j] = q_col.to(torch.int8)
        else:
            q_col = torch.clamp(torch.round(W[:, j] / s4.float()), -int4_range, int4_range)
            Q[:, j] = q_col.to(torch.int8)

    return Q, s4, s8, protect_mask


def awq_dequantize(Q, s4, s8, protect_mask):
    rows, cols = Q.shape
    W = torch.zeros(rows, cols, dtype=torch.float32)
    for j in range(cols):
        if protect_mask[j]:
            W[:, j] = Q[:, j].float() * s8.float()
        else:
            W[:, j] = Q[:, j].float() * s4.float()
    return W


def compute_loss(model, train_loader, h, device, n_batches=16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, cu_seqlens, max_seqlen = train_loader.next_batch(
                h.train_batch_tokens, h.grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    return total_loss / total_tokens


def main():
    t0 = time.time()
    log("=" * 60)
    log("C10: AWQ-Lite — Activation-aware Weight Quantization")
    log("=" * 60)

    model, h, device, train_loader = load_model_and_data()

    log("Computing baseline loss...")
    baseline_loss = compute_loss(model, train_loader, h, device, n_batches=16)
    log(f"Baseline loss: {baseline_loss:.6f}")

    log("Collecting activations (32 calibration batches)...")
    activations = collect_activations(model, train_loader, h, device, n_batches=32)
    log(f"Collected activations for {len(activations)} layer keys")

    importance = compute_channel_importance(activations, h.num_layers)
    log(f"Computed channel importance for {len(importance)} weight matrices")

    flat_sd = _unbank_state_dict(model.state_dict(), h.num_layers)

    log("Applying AWQ-Lite quantization (INT4/INT8 mixed, top 1% protected)...")
    quantized = {}
    meta = {}
    n_protected_total = 0
    n_total_cols = 0

    for name, tensor in flat_sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            quantized[name] = t.to(torch.float16)
            meta[name] = "passthrough"
            continue

        if "tok_emb" in name:
            cs = 20.0
            bits = 8
        elif ".mlp." in name:
            cs = 10.0
            bits = 4
        elif ".attn." in name:
            cs = 13.0
            bits = 4
        else:
            cs = 12.85
            bits = 4

        ch_imp = importance.get(name, None)
        Q, s4, s8, protect_mask = awq_quantize_weight(
            t, ch_imp, clip_sigmas=cs, bits=bits, protect_fraction=0.01
        )
        quantized[name + ".awq_q"] = Q
        quantized[name + ".awq_s4"] = s4
        quantized[name + ".awq_s8"] = s8
        quantized[name + ".awq_protect"] = protect_mask
        meta[name] = "awq_mixed"
        n_protected_total += protect_mask.sum().item()
        n_total_cols += protect_mask.numel()

    protect_pct = 100.0 * n_protected_total / max(n_total_cols, 1)
    log(f"Protected {n_protected_total} / {n_total_cols} channels ({protect_pct:.2f}%)")

    log("Dequantizing and computing quantized loss...")
    from train_gpt import _rebank_state_dict
    deq_sd = {}
    for name, orig in flat_sd.items():
        if meta.get(name) == "passthrough":
            deq_sd[name] = quantized[name].to(orig.dtype)
        elif meta.get(name) == "awq_mixed":
            Q = quantized[name + ".awq_q"]
            s4 = quantized[name + ".awq_s4"]
            s8 = quantized[name + ".awq_s8"]
            pm = quantized[name + ".awq_protect"]
            W = awq_dequantize(Q, s4, s8, pm)
            deq_sd[name] = W.to(orig.dtype)

    head_dim = h.model_dim // h.num_heads
    kv_dim = h.num_kv_heads * head_dim
    hidden_dim = int(h.mlp_mult * h.model_dim)
    rebanked = _rebank_state_dict(deq_sd, h.num_layers, h.model_dim, kv_dim, hidden_dim)

    model.load_state_dict(rebanked, strict=True)
    model.to(device).bfloat16()

    quantized_loss = compute_loss(model, train_loader, h, device, n_batches=16)
    log(f"Quantized loss: {quantized_loss:.6f}")

    orig_bytes = sum(t.numel() * 4 for t in flat_sd.values() if t.is_floating_point())
    quant_bytes = 0
    for name, orig in flat_sd.items():
        if not orig.is_floating_point():
            continue
        if meta.get(name) == "passthrough":
            quant_bytes += orig.numel() * 2
        elif meta.get(name) == "awq_mixed":
            quant_bytes += orig.numel() * 1
            quant_bytes += orig.shape[0] * 2 * 2
            quant_bytes += orig.shape[1] * 1

    compression_ratio = orig_bytes / max(quant_bytes, 1)
    accuracy_loss = quantized_loss - baseline_loss

    log("=" * 60)
    log(f"RESULTS:")
    log(f"  Baseline loss:    {baseline_loss:.6f}")
    log(f"  Quantized loss:   {quantized_loss:.6f}")
    log(f"  Accuracy loss:    {accuracy_loss:+.6f}")
    log(f"  Original size:    {orig_bytes / 1024 / 1024:.2f} MB")
    log(f"  Quantized size:   {quant_bytes / 1024 / 1024:.2f} MB")
    log(f"  Compression:      {compression_ratio:.2f}x")
    log(f"  Protected:        {protect_pct:.2f}% of channels (INT8)")
    log(f"  Time:             {time.time() - t0:.1f}s")
    log("=" * 60)

    results = {
        "experiment": "C10_AWQ_Lite",
        "baseline_loss": baseline_loss,
        "quantized_loss": quantized_loss,
        "accuracy_loss": accuracy_loss,
        "original_bytes": orig_bytes,
        "quantized_bytes": quant_bytes,
        "compression_ratio": compression_ratio,
        "protected_channels_pct": protect_pct,
        "n_calibration_batches": 32,
        "time_seconds": time.time() - t0,
    }
    with open("experiments/c10_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("Results saved to experiments/c10_results.json")


if __name__ == "__main__":
    main()
