#!/usr/bin/env python3
"""
C21 BPB Validation
==================
Loads newarch_sp8192.pt, runs BPB eval (baseline),
applies C21 (GPTQ+AWQ) quantization, runs BPB eval (quantized).
Reports BPB numbers for leaderboard.
"""
import os, sys, time, math

os.environ["VOCAB_SIZE"] = "8192"
os.environ["NUM_LOOPS"] = "0"
os.environ["SMEAR_GATE_ENABLED"] = "1"
os.environ["CASEOPS_ENABLED"] = "1"

import torch
import torch.nn.functional as F

PG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PG_DIR)
os.chdir(PG_DIR)

from train_gpt import (
    GPT, Hyperparameters, ValidationData, eval_val, set_logging_hparams, log,
    _unbank_state_dict, _rebank_state_dict, restore_fp32_params,
    gptq_quantize_weight,
)

def setup_model(device):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(False)
    enable_mem_efficient_sdp(True)
    enable_math_sdp(True)

    h = Hyperparameters()
    set_logging_hparams(h)

    model = GPT(h).to(device).bfloat16()
    restore_fp32_params(model)

    ckpt_path = os.environ.get("CKPT_PATH", "checkpoints/newarch_sp8192.pt")
    log(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, h

def collect_activations(model, h, device, val_data, n_batches=8):
    """Collect activation gram matrices for channel importance (AWQ)."""
    gram_matrices = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            for prefix in ["attn", "mlp"]:
                key = f"{prefix}_L{layer_idx}"
                if key not in gram_matrices:
                    gram_matrices[key] = x.T @ x
                else:
                    gram_matrices[key].addmm_(x.T, x)
        return hook_fn

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_hook(i)))

    model.eval()
    seq_len = h.eval_seq_len
    val_tokens = val_data.val_tokens
    total_seqs = (val_tokens.numel() - 1) // seq_len
    n_seqs = min(n_batches, total_seqs)

    with torch.no_grad():
        for i in range(n_seqs):
            raw_start = i * seq_len
            raw_end = (i + 1) * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].unsqueeze(0)
            from train_gpt import _build_cu_seqlens, BOS_ID
            bos_pos = (x.squeeze(0) == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = _build_cu_seqlens(bos_pos, x.numel(), x.device, seq_len, 64)
            model.forward_logits(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            if (i + 1) % 4 == 0:
                log(f"  Activation batch {i + 1}/{n_seqs}")

    for hook in hooks:
        hook.remove()
    for key in gram_matrices:
        gram_matrices[key] = gram_matrices[key].cpu() / n_seqs
    return gram_matrices

def compute_channel_importance(activations, n_layers):
    importance = {}
    for i in range(n_layers):
        attn_key = f"attn_L{i}"
        if attn_key in activations:
            diag = activations[attn_key].diag()
            for suffix in ["c_q", "c_k", "c_v", "proj"]:
                importance[f"blocks.{i}.attn.{suffix}.weight"] = diag.clone()
            importance[f"blocks.{i}.mlp.fc.weight"] = diag.clone()
            importance[f"blocks.{i}.mlp.proj.weight"] = diag.clone()
    return importance

def collect_hessians(model, h, device, val_data, n_batches=16):
    """Collect Gauss-Newton Hessians for GPTQ."""
    hessians = {}
    hooks = []

    for i, block in enumerate(model.blocks):
        block.attn._calib = True
        block.mlp._calib = True
        block.mlp.use_fused = False

    def make_attn_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            for suffix in ["c_q", "c_k", "c_v"]:
                name = f"blocks.{layer_idx}.attn.{suffix}.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            y = module._last_proj_input
            if y is not None:
                y = y.float()
                if y.ndim == 3:
                    y = y.reshape(-1, y.shape[-1])
                name = f"blocks.{layer_idx}.attn.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(y.shape[1], y.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(y.T, y)
        return hook_fn

    def make_mlp_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            name = f"blocks.{layer_idx}.mlp.fc.weight"
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)
            y = module._last_down_input
            if y is not None:
                y = y.float()
                if y.ndim == 3:
                    y = y.reshape(-1, y.shape[-1])
                name = f"blocks.{layer_idx}.mlp.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(y.shape[1], y.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(y.T, y)
        return hook_fn

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(block.mlp.register_forward_hook(make_mlp_hook(i)))

    if model.tie_embeddings:
        hook_module = model.final_norm
        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            return hook_fn
        hooks.append(hook_module.register_forward_hook(make_output_hook("tok_emb.weight")))

    model.eval()
    seq_len = h.eval_seq_len
    val_tokens = val_data.val_tokens
    total_seqs = (val_tokens.numel() - 1) // seq_len
    n_seqs = min(n_batches, total_seqs)

    with torch.no_grad():
        for i in range(n_seqs):
            raw_start = i * seq_len
            raw_end = (i + 1) * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].unsqueeze(0)
            from train_gpt import _build_cu_seqlens, BOS_ID
            bos_pos = (x.squeeze(0) == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = _build_cu_seqlens(bos_pos, x.numel(), x.device, seq_len, 64)
            model.forward_logits(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            if (i + 1) % 4 == 0:
                log(f"  Hessian batch {i + 1}/{n_seqs}")

    for hook in hooks:
        hook.remove()
    for i, block in enumerate(model.blocks):
        block.attn._calib = False
        block.mlp._calib = False
        block.mlp.use_fused = True
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_seqs
    return hessians

def combined_quantize(flat_sd, hessians, importance, h, protect_fraction=0.01):
    result = {}
    meta = {}
    n_protected_total = 0
    n_total_cols = 0

    for name, tensor in flat_sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16)
            meta[name] = "passthrough"
            continue

        rows, cols = t.shape
        if "tok_emb" in name:
            base_bits = 8; cs = 20.0
        elif ".mlp." in name:
            base_bits = 6; cs = 10.0
        elif ".attn." in name:
            base_bits = 6; cs = 13.0
        else:
            base_bits = 6; cs = 12.85

        clip_range = 2 ** (base_bits - 1) - 1
        W = t.float()

        ch_imp = importance.get(name, None)
        if ch_imp is not None and ch_imp.numel() >= cols:
            imp = ch_imp[:cols]
        else:
            imp = W.abs().sum(dim=0)

        n_protect = max(1, int(cols * protect_fraction))
        _, protect_indices = torch.topk(imp, n_protect)
        protect_mask = torch.zeros(cols, dtype=torch.bool)
        protect_mask[protect_indices] = True

        row_std = W.std(dim=1)
        s8 = (cs * row_std / 127).clamp_min(1e-10).to(torch.float16)

        if name in hessians:
            H = hessians[name]
            q_gptq, s_gptq = gptq_quantize_weight(t, H, clip_sigmas=cs, clip_range=clip_range)
        else:
            s_gptq = (cs * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
            q_gptq = torch.clamp(torch.round(W / s_gptq.float().view(-1, 1)), -clip_range, clip_range).to(torch.int8)

        Q_combined = q_gptq.clone() if isinstance(q_gptq, torch.Tensor) else q_gptq
        for j in range(cols):
            if protect_mask[j]:
                q_int8 = torch.clamp(torch.round(W[:, j] / s8.float()), -127, 127)
                Q_combined[:, j] = q_int8.to(torch.int8)

        result[name + ".q"] = Q_combined
        result[name + ".scale"] = s_gptq
        result[name + ".s8"] = s8
        result[name + ".protect"] = protect_mask
        meta[name] = "combined_awq_gptq"

        n_protected_total += protect_mask.sum().item()
        n_total_cols += cols

    return result, meta, n_protected_total, n_total_cols

def dequantize_combined(result, meta, flat_sd):
    out = {}
    for name, orig in flat_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        if info == "passthrough":
            out[name] = result[name].to(orig.dtype)
            continue
        Q = result[name + ".q"]
        s_gptq = result[name + ".scale"]
        s8 = result[name + ".s8"]
        pm = result[name + ".protect"]
        rows, cols = Q.shape
        W = torch.zeros(rows, cols, dtype=torch.float32)
        for j in range(cols):
            if pm[j]:
                W[:, j] = Q[:, j].float() * s8.float()
            else:
                if s_gptq.ndim > 0:
                    W[:, j] = Q[:, j].float() * s_gptq.float()
                else:
                    W[:, j] = Q[:, j].float() * float(s_gptq.item())
        out[name] = W.to(orig.dtype)
    return out

def main():
    t0 = time.time()
    device = torch.device("cuda")
    log("=" * 60)
    log("C21 BPB Validation: GPTQ+AWQ on newarch_sp8192")
    log("=" * 60)

    model, h = setup_model(device)
    val_data = ValidationData(h, device)
    log(f"Val tokens: {val_data.val_tokens.numel()-1}")

    # --- Baseline BPB ---
    log("Computing baseline BPB...")
    baseline_loss, baseline_bpb = eval_val(h, device, val_data, model)
    log(f"Baseline: loss={baseline_loss:.4f}, bpb={baseline_bpb:.4f}")

    # --- Collect activations + Hessians ---
    n_calib_act = 8
    n_calib_hess = 16
    log(f"Collecting activations ({n_calib_act} batches)...")
    activations = collect_activations(model, h, device, val_data, n_batches=n_calib_act)
    log(f"Collected activations for {len(activations)} layer keys")

    log(f"Collecting Hessians ({n_calib_hess} batches)...")
    hessians = collect_hessians(model, h, device, val_data, n_batches=n_calib_hess)
    log(f"Collected Hessians for {len(hessians)} weight matrices")

    importance = compute_channel_importance(activations, h.num_layers)

    # --- C21 Quantization ---
    flat_sd = _unbank_state_dict(model.state_dict(), h.num_layers)
    log(f"Unbanked state dict: {len(flat_sd)} tensors")

    log("Applying AWQ+GPTQ combined quantization...")
    result, meta, n_prot, n_total = combined_quantize(
        flat_sd, hessians, importance, h, protect_fraction=0.01
    )
    protect_pct = 100.0 * n_prot / max(n_total, 1)
    log(f"Protected {n_prot} / {n_total} channels ({protect_pct:.2f}%)")

    cats = {}
    for name, cat in meta.items():
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items()):
        log(f"  {cat}: {count} tensors")

    # --- Dequantize and compute quantized BPB ---
    log("Dequantizing...")
    deq_sd = dequantize_combined(result, meta, flat_sd)

    head_dim = h.model_dim // h.num_heads
    kv_dim = h.num_kv_heads * head_dim
    hidden_dim = int(h.mlp_mult * h.model_dim)
    rebanked = _rebank_state_dict(deq_sd, h.num_layers, h.model_dim, kv_dim, hidden_dim)

    model.load_state_dict(rebanked, strict=True)
    model.to(device).bfloat16()

    log("Computing quantized BPB...")
    quantized_loss, quantized_bpb = eval_val(h, device, val_data, model)
    log(f"Quantized: loss={quantized_loss:.4f}, bpb={quantized_bpb:.4f}")

    # --- Size ---
    orig_bytes = sum(t.numel() * 4 for t in flat_sd.values() if t.is_floating_point())
    quant_bytes = 0
    for name, orig in flat_sd.items():
        if not orig.is_floating_point():
            continue
        info = meta.get(name, "")
        if info == "passthrough":
            quant_bytes += orig.numel() * 2
        elif info == "combined_awq_gptq":
            quant_bytes += int(orig.numel() * 0.75)
            quant_bytes += orig.shape[0] * 2 * 2
            quant_bytes += orig.shape[1] // 8

    compression_ratio = orig_bytes / max(quant_bytes, 1)

    elapsed = time.time() - t0
    log("=" * 60)
    log("RESULTS:")
    log(f"  Baseline loss:    {baseline_loss:.6f}")
    log(f"  Baseline BPB:     {baseline_bpb:.6f}")
    log(f"  Quantized loss:   {quantized_loss:.6f}")
    log(f"  Quantized BPB:    {quantized_bpb:.6f}")
    log(f"  BPB delta:        {quantized_bpb - baseline_bpb:+.6f}")
    log(f"  Loss delta:       {quantized_loss - baseline_loss:+.6f}")
    log(f"  Original size:    {orig_bytes / 1024 / 1024:.2f} MB")
    log(f"  Quantized size:   {quant_bytes / 1024 / 1024:.2f} MB")
    log(f"  Compression:      {compression_ratio:.2f}x")
    log(f"  Time:             {elapsed:.1f}s")
    log("=" * 60)

    import json
    results = {
        "experiment": "C21_BPB_Validation",
        "checkpoint": "checkpoints/newarch_sp8192.pt",
        "baseline_loss": baseline_loss,
        "baseline_bpb": baseline_bpb,
        "quantized_loss": quantized_loss,
        "quantized_bpb": quantized_bpb,
        "bpb_delta": quantized_bpb - baseline_bpb,
        "loss_delta": quantized_loss - baseline_loss,
        "original_bytes": orig_bytes,
        "quantized_bytes": quant_bytes,
        "compression_ratio": compression_ratio,
        "protected_channels_pct": protect_pct,
        "n_calibration_act_batches": n_calib_act,
        "n_calibration_hess_batches": n_calib_hess,
        "tensor_categories": cats,
        "time_seconds": elapsed,
    }
    with open("experiments/c21_bpb_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("Results saved to experiments/c21_bpb_results.json")

if __name__ == "__main__":
    main()
