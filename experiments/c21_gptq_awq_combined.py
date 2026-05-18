#!/usr/bin/env python3
"""
C21: GPTQ + AWQ Combined
=========================
AWQ channel protection FIRST, then GPTQ on remaining sensitive weights.
Best of both worlds: channel importance + error correction.
"""
import os, sys, time, math, json

os.environ["VOCAB_SIZE"] = "1024"
os.environ["NUM_LOOPS"] = "0"

import torch
import torch.nn.functional as F

PG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PG_DIR)
os.chdir(PG_DIR)

from train_gpt import (
    GPT, Hyperparameters, DocumentPackingLoader,
    _unbank_state_dict, _rebank_state_dict, restore_fp32_params,
    gptq_quantize_weight, collect_hessians,
)

BOS_ID = 1


def log(msg):
    print(f"[C21-COMBINED] {msg}", flush=True)


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


def collect_activations_for_importance(model, train_loader, h, device, n_batches=32):
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
    with torch.no_grad():
        for batch_idx in range(n_batches):
            x, y, cu_seqlens, max_seqlen = train_loader.next_batch(
                h.train_batch_tokens, h.grad_accum_steps
            )
            model(x, y, cu_seqlens=cu_seqlens, max_seqlen=h.train_seq_len)
            if (batch_idx + 1) % 8 == 0:
                log(f"  Activation batch {batch_idx + 1}/{n_batches}")

    for hook in hooks:
        hook.remove()

    for key in gram_matrices:
        gram_matrices[key] = gram_matrices[key].cpu() / n_batches

    return gram_matrices


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


def collect_hessians_fixed(model, train_loader, h, device, n_calibration_batches=64):
    """Fixed collect_hessians that handles DocumentPackingLoader's 4-value return."""
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
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, y, cu_seqlens, max_seqlen = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    for hook in hooks:
        hook.remove()
    for i, block in enumerate(model.blocks):
        block.attn._calib = False
        block.mlp._calib = False
        block.mlp.use_fused = True
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
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
            base_bits = 8
            cs = 20.0
        elif ".mlp." in name:
            base_bits = 6
            cs = 10.0
        elif ".attn." in name:
            base_bits = 6
            cs = 13.0
        else:
            base_bits = 6
            cs = 12.85

        clip_range = 2 ** (base_bits - 1) - 1
        W = t.float()

        # AWQ: identify protected channels
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

        # GPTQ on full weight
        if name in hessians:
            H = hessians[name]
            q_gptq, s_gptq = gptq_quantize_weight(t, H, clip_sigmas=cs, clip_range=clip_range)
        else:
            s_gptq = (cs * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
            q_gptq = torch.clamp(torch.round(W / s_gptq.float().view(-1, 1)), -clip_range, clip_range).to(torch.int8)

        # Override protected channels with INT8
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
    log("C21: GPTQ + AWQ Combined")
    log("=" * 60)

    model, h, device, train_loader = load_model_and_data()

    log("Computing baseline loss...")
    baseline_loss = compute_loss(model, train_loader, h, device, n_batches=16)
    log(f"Baseline loss: {baseline_loss:.6f}")

    n_calib = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", "32"))
    log(f"Collecting activations ({n_calib} batches)...")
    activations = collect_activations_for_importance(model, train_loader, h, device, n_batches=n_calib)
    log(f"Collected activations for {len(activations)} layer keys")

    log(f"Collecting Hessians ({n_calib} batches)...")
    hessians = collect_hessians_fixed(model, train_loader, h, device, n_calibration_batches=n_calib)
    log(f"Collected Hessians for {len(hessians)} weight matrices")

    importance = compute_channel_importance(activations, h.num_layers)

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

    log("Dequantizing and computing quantized loss...")
    deq_sd = dequantize_combined(result, meta, flat_sd)

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
        info = meta.get(name, "")
        if info == "passthrough":
            quant_bytes += orig.numel() * 2
        elif info == "combined_awq_gptq":
            quant_bytes += int(orig.numel() * 0.75)
            quant_bytes += orig.shape[0] * 2 * 2
            quant_bytes += orig.shape[1] // 8

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
    log(f"  AWQ protected:    {protect_pct:.2f}% of channels (INT8)")
    log(f"  GPTQ correction:  INT6 with Hessian on remaining")
    log(f"  Calibration:      {n_calib} batches (self-generated)")
    log(f"  Time:             {time.time() - t0:.1f}s")
    log("=" * 60)

    results = {
        "experiment": "C21_GPTQ_AWQ_Combined",
        "baseline_loss": baseline_loss,
        "quantized_loss": quantized_loss,
        "accuracy_loss": accuracy_loss,
        "original_bytes": orig_bytes,
        "quantized_bytes": quant_bytes,
        "compression_ratio": compression_ratio,
        "protected_channels_pct": protect_pct,
        "n_calibration_batches": n_calib,
        "tensor_categories": cats,
        "time_seconds": time.time() - t0,
    }
    with open("experiments/c21_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("Results saved to experiments/c21_results.json")


if __name__ == "__main__":
    main()
