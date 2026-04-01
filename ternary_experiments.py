from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
from torch import Tensor, nn

from ternary_utils import (
    compress_bytes,
    dequantize_ternary_grouped_tensor,
    dequantize_ternary_tensor,
    quantize_ternary_grouped_tensor,
    quantize_ternary_tensor,
    serialize_custom_quantized_entries,
)

QuantMode = Literal["int8", "ternary", "fp16"]
LayoutMode = Literal["alphabetical", "by_type", "by_magnitude", "by_sparsity"]
TernaryScaleFn = Literal["absmean", "median"]

BLOCK_FAMILY_RE = re.compile(r"^blocks\.(\d+)\.(attn|mlp)\.")
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
FAMILIES = ("embeddings", "attn", "mlp", "other")
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


@lru_cache(maxsize=1)
def train_gpt_api() -> dict[str, object]:
    from train_gpt import (
        GPT,
        CastedLinear,
        build_sentencepiece_luts,
        load_validation_tokens,
        restore_low_dim_params_to_fp32,
    )

    return {
        "CastedLinear": CastedLinear,
        "GPT": GPT,
        "build_sentencepiece_luts": build_sentencepiece_luts,
        "load_validation_tokens": load_validation_tokens,
        "restore_low_dim_params_to_fp32": restore_low_dim_params_to_fp32,
    }


def default_value(name: str, fallback, cast):
    raw = os.environ.get(name, str(fallback))
    if cast is bool:
        return bool(int(raw))
    return cast(raw)


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(
    name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]
) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(
            torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None]
        )
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = (
            torch.clamp(torch.round(clipped / scale[:, None]), -127, 127)
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = (
        float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
        if t32.numel()
        else 0.0
    )
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


def parse_map_arg(
    raw: str, value_parser, allowed_keys: set[str] | None = None
) -> dict[str, object]:
    out: dict[str, object] = {}
    if not raw.strip():
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE item, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if allowed_keys is not None and key not in allowed_keys:
            raise ValueError(
                f"Unsupported key '{key}', expected one of {sorted(allowed_keys)}"
            )
        out[key] = value_parser(value.strip())
    return out


def parse_block_map_arg(
    raw: str, value_parser, allowed_families: set[str] | None = None
) -> dict[tuple[str, int], object]:
    out: dict[tuple[str, int], object] = {}
    if not raw.strip():
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(f"Expected FAMILY:BLOCK=VALUE item, got: {item}")
        key, value = item.split("=", 1)
        family, block = key.split(":", 1)
        family = family.strip()
        if allowed_families is not None and family not in allowed_families:
            raise ValueError(
                f"Unsupported block family '{family}', expected one of {sorted(allowed_families)}"
            )
        out[(family, int(block.strip()))] = value_parser(value.strip())
    return out


def parse_scale_fn(value: str) -> TernaryScaleFn:
    if value not in {"absmean", "median"}:
        raise ValueError(f"Unsupported scale_fn '{value}', expected absmean or median")
    return value  # type: ignore[return-value]


def quant_spec(
    mode: QuantMode,
    *,
    zero_threshold: float | None = None,
    scale_fn: TernaryScaleFn | None = None,
    group_size: int | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {"mode": mode}
    if mode == "ternary":
        spec["zero_threshold"] = (
            0.5 if zero_threshold is None else float(zero_threshold)
        )
        spec["scale_fn"] = "absmean" if scale_fn is None else scale_fn
        spec["group_size"] = 0 if group_size is None else int(group_size)
    return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-training ternary compression experiments."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("final_model.pt"))
    parser.add_argument(
        "--output", type=Path, default=Path("ternary_experiments_results.json")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip val_bpb evaluation and report size/MSE only.",
    )
    parser.add_argument(
        "--prefer-zstd",
        action="store_true",
        help="Use zstd-22 when zstandard is installed.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Write compressed artifacts to the output directory.",
    )
    parser.add_argument("--smart-delta-threshold", type=float, default=0.01)
    parser.add_argument(
        "--layout-config", choices=["conservative", "aggressive"], default="aggressive"
    )
    parser.add_argument("--zero-thresholds", default="0.4,0.5,0.6,0.7")
    parser.add_argument(
        "--ternary-zero-threshold",
        type=float,
        default=default_value("TERNARY_ZERO_THRESHOLD", 0.5, float),
    )
    parser.add_argument(
        "--ternary-scale-fn",
        choices=["absmean", "median"],
        default=os.environ.get("TERNARY_SCALE_FN", "absmean"),
    )
    parser.add_argument(
        "--ternary-group-size",
        type=int,
        default=default_value("TERNARY_GROUP_SIZE", 0, int),
        help="0 disables grouped ternary; positive values use per-group scales.",
    )
    parser.add_argument("--ternary-family-zero-thresholds", default="")
    parser.add_argument("--ternary-block-zero-thresholds", default="")
    parser.add_argument("--ternary-family-scale-fns", default="")
    parser.add_argument("--ternary-block-scale-fns", default="")
    parser.add_argument("--ternary-family-group-sizes", default="")
    parser.add_argument("--ternary-block-group-sizes", default="")
    parser.add_argument("--layer-sensitivity", action="store_true")
    parser.add_argument("--hybrid-configs", action="store_true")
    parser.add_argument("--layout-reordering", action="store_true")
    parser.add_argument("--zero-pressure", action="store_true")
    parser.add_argument(
        "--vocab-size", type=int, default=default_value("VOCAB_SIZE", 1024, int)
    )
    parser.add_argument(
        "--num-layers", type=int, default=default_value("NUM_LAYERS", 9, int)
    )
    parser.add_argument(
        "--model-dim", type=int, default=default_value("MODEL_DIM", 512, int)
    )
    parser.add_argument(
        "--num-heads", type=int, default=default_value("NUM_HEADS", 8, int)
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=default_value("NUM_KV_HEADS", 4, int)
    )
    parser.add_argument(
        "--mlp-mult", type=int, default=default_value("MLP_MULT", 2, int)
    )
    parser.add_argument(
        "--tie-embeddings",
        type=int,
        choices=[0, 1],
        default=int(default_value("TIE_EMBEDDINGS", 1, bool)),
    )
    parser.add_argument(
        "--tied-embed-init-std",
        type=float,
        default=default_value("TIED_EMBED_INIT_STD", 0.005, float),
    )
    parser.add_argument(
        "--logit-softcap",
        type=float,
        default=default_value("LOGIT_SOFTCAP", 30.0, float),
    )
    parser.add_argument(
        "--rope-base", type=float, default=default_value("ROPE_BASE", 10000.0, float)
    )
    parser.add_argument(
        "--qk-gain-init",
        type=float,
        default=default_value("QK_GAIN_INIT", 1.5, float),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.environ.get(
            "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
        ),
    )
    parser.add_argument(
        "--data-path",
        default=os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"),
    )
    parser.add_argument(
        "--val-files",
        default=os.path.join(
            os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"),
            "fineweb_val_*.bin",
        ),
    )
    parser.add_argument(
        "--train-seq-len", type=int, default=default_value("TRAIN_SEQ_LEN", 1024, int)
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=default_value("VAL_BATCH_SIZE", 524_288, int),
    )
    args = parser.parse_args()
    if not any(
        (
            args.layer_sensitivity,
            args.hybrid_configs,
            args.layout_reordering,
            args.zero_pressure,
        )
    ):
        args.layer_sensitivity = True
        args.hybrid_configs = True
        args.layout_reordering = True
        args.zero_pressure = True
    args.ternary_family_zero_thresholds_map = parse_map_arg(
        args.ternary_family_zero_thresholds, float, allowed_keys=set(FAMILIES)
    )
    args.ternary_block_zero_thresholds_map = parse_block_map_arg(
        args.ternary_block_zero_thresholds, float, allowed_families={"attn", "mlp"}
    )
    args.ternary_family_scale_fns_map = parse_map_arg(
        args.ternary_family_scale_fns, parse_scale_fn, allowed_keys=set(FAMILIES)
    )
    args.ternary_block_scale_fns_map = parse_block_map_arg(
        args.ternary_block_scale_fns, parse_scale_fn, allowed_families={"attn", "mlp"}
    )
    args.ternary_family_group_sizes_map = parse_map_arg(
        args.ternary_family_group_sizes, int, allowed_keys=set(FAMILIES)
    )
    args.ternary_block_group_sizes_map = parse_block_map_arg(
        args.ternary_block_group_sizes,
        int,
        allowed_families={"attn", "mlp"},
    )
    return args


def tensor_family(name: str) -> tuple[str, int | None]:
    if name.startswith("tok_emb.") or name.startswith("lm_head."):
        return "embeddings", None
    match = BLOCK_FAMILY_RE.match(name)
    if match:
        return match.group(2), int(match.group(1))
    return "other", None


def is_large_float_tensor(tensor: Tensor) -> bool:
    return tensor.is_floating_point() and tensor.numel() > INT8_KEEP_FLOAT_MAX_NUMEL


def resolve_ternary_settings(
    args: argparse.Namespace, family: str, block_index: int | None
) -> dict[str, object]:
    settings = {
        "zero_threshold": float(args.ternary_zero_threshold),
        "scale_fn": args.ternary_scale_fn,
        "group_size": int(args.ternary_group_size),
    }
    if family in args.ternary_family_zero_thresholds_map:
        settings["zero_threshold"] = float(
            args.ternary_family_zero_thresholds_map[family]
        )
    if family in args.ternary_family_scale_fns_map:
        settings["scale_fn"] = args.ternary_family_scale_fns_map[family]
    if family in args.ternary_family_group_sizes_map:
        settings["group_size"] = int(args.ternary_family_group_sizes_map[family])
    if block_index is not None:
        block_key = (family, block_index)
        if block_key in args.ternary_block_zero_thresholds_map:
            settings["zero_threshold"] = float(
                args.ternary_block_zero_thresholds_map[block_key]
            )
        if block_key in args.ternary_block_scale_fns_map:
            settings["scale_fn"] = args.ternary_block_scale_fns_map[block_key]
        if block_key in args.ternary_block_group_sizes_map:
            settings["group_size"] = int(args.ternary_block_group_sizes_map[block_key])
    return settings


def summarize_quant_plan(plan: dict[str, dict[str, object]]) -> dict[str, object]:
    mode_counts: dict[str, int] = {}
    ternary_scale_fns: dict[str, int] = {}
    ternary_group_sizes: dict[str, int] = {}
    ternary_zero_thresholds: dict[str, int] = {}
    for spec in plan.values():
        mode = str(spec["mode"])
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if mode != "ternary":
            continue
        scale_fn = str(spec["scale_fn"])
        group_size = int(spec["group_size"])
        zero_threshold = float(spec["zero_threshold"])
        ternary_scale_fns[scale_fn] = ternary_scale_fns.get(scale_fn, 0) + 1
        ternary_group_sizes[str(group_size)] = (
            ternary_group_sizes.get(str(group_size), 0) + 1
        )
        threshold_key = f"{zero_threshold:.4f}".rstrip("0").rstrip(".")
        ternary_zero_thresholds[threshold_key] = (
            ternary_zero_thresholds.get(threshold_key, 0) + 1
        )
    return {
        "mode_counts": mode_counts,
        "ternary_scale_fns": ternary_scale_fns,
        "ternary_group_sizes": ternary_group_sizes,
        "ternary_zero_thresholds": ternary_zero_thresholds,
    }


def build_quant_plan(
    state_dict: dict[str, Tensor],
    *,
    args: argparse.Namespace,
    embed_mode: QuantMode,
    attn_mode: QuantMode,
    mlp_mode: QuantMode,
    other_mode: QuantMode = "int8",
    overrides: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    plan: dict[str, dict[str, object]] = {}
    overrides = overrides or {}
    for name, tensor in state_dict.items():
        if not is_large_float_tensor(tensor):
            continue
        if name in overrides:
            plan[name] = overrides[name]
            continue
        family, block_index = tensor_family(name)
        ternary_settings = resolve_ternary_settings(args, family, block_index)
        if family == "embeddings":
            plan[name] = quant_spec(embed_mode, **ternary_settings)
        elif family == "attn":
            plan[name] = quant_spec(attn_mode, **ternary_settings)
        elif family == "mlp":
            plan[name] = quant_spec(mlp_mode, **ternary_settings)
        else:
            plan[name] = quant_spec(other_mode, **ternary_settings)
    return plan


def quantize_mixed_state_dict(
    state_dict: dict[str, Tensor],
    quant_plan: dict[str, dict[str, object]],
) -> tuple[dict[str, object], dict[str, float]]:
    entries: dict[str, dict[str, object]] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    total_sq_error = 0.0
    total_numel = 0
    total_zero_count = 0.0
    total_quantized_count = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        family, block_idx = tensor_family(name)
        metadata = {
            "family": family,
            "block_index": block_idx,
            "numel": int(t.numel()),
            "abs_mean": float(t.detach().float().abs().mean().item())
            if t.is_floating_point() and t.numel()
            else 0.0,
        }

        if not t.is_floating_point():
            entry = {"type": "raw", "data": t, **metadata}
            entries[name] = entry
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            entry = {"type": "fp16", "data": kept, **metadata}
            if name in passthrough_orig_dtypes:
                entry["orig_dtype"] = passthrough_orig_dtypes[name]
            entries[name] = entry
            continue

        spec = quant_plan.get(name, quant_spec("int8"))
        mode = spec["mode"]
        if mode == "int8":
            q, scale = quantize_float_tensor(t)
            entry = {
                "type": "int8",
                "data": q,
                "scale": scale,
                "dtype": str(t.dtype).removeprefix("torch."),
                "scheme": "per_row" if scale.ndim > 0 else "per_tensor",
                "zero_fraction": float((q == 0).float().mean().item()),
                **metadata,
            }
            reconstructed = dequantize_entry(entry)
        elif mode == "ternary":
            scale_fn = spec["scale_fn"]
            zero_threshold = float(spec["zero_threshold"])
            group_size = int(spec["group_size"])
            if group_size > 0 and t.ndim >= 2:
                entry = {
                    **quantize_ternary_grouped_tensor(
                        t,
                        group_size=group_size,
                        scale_fn=scale_fn,
                        zero_threshold=zero_threshold,
                    ),
                    **metadata,
                }
            else:
                entry = {
                    **quantize_ternary_tensor(
                        t, scale_fn=scale_fn, zero_threshold=zero_threshold
                    ),
                    **metadata,
                }
            reconstructed = dequantize_entry(entry)
        elif mode == "fp16":
            entry = {
                "type": "fp16",
                "data": t.to(torch.float16).contiguous(),
                "orig_dtype": str(t.dtype).removeprefix("torch."),
                **metadata,
            }
            reconstructed = dequantize_entry(entry)
        else:
            raise ValueError(f"Unsupported quantization mode: {mode}")

        entries[name] = entry
        total_sq_error += float(
            torch.sum((t.float() - reconstructed.float()) ** 2).item()
        )
        total_numel += int(t.numel())
        if "zero_fraction" in entry:
            total_zero_count += entry["zero_fraction"] * t.numel()
            total_quantized_count += int(t.numel())

    obj: dict[str, object] = {
        "__quant_format__": "mixed_precision_v1",
        "entries": entries,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    stats = {
        "roundtrip_mse": total_sq_error / max(total_numel, 1),
        "quantized_zero_fraction": total_zero_count / max(total_quantized_count, 1),
    }
    return obj, stats


def dequantize_entry(entry: dict[str, object]) -> Tensor:
    entry_type = entry["type"]
    if entry_type == "raw":
        return entry["data"].detach().cpu().contiguous()
    if entry_type == "fp16":
        out = entry["data"].detach().cpu().contiguous()
        if "orig_dtype" in entry:
            out = out.to(dtype=getattr(torch, entry["orig_dtype"]))
        return out
    if entry_type == "int8":
        q = entry["data"]
        scale = entry["scale"]
        dtype = getattr(torch, entry["dtype"])
        if entry["scheme"] == "per_row" or scale.ndim > 0:
            scale = scale.to(dtype=torch.float32)
            return (
                (q.float() * scale.view(q.shape[0], *([1] * (q.ndim - 1))))
                .to(dtype=dtype)
                .contiguous()
            )
        return (q.float() * float(scale.item())).to(dtype=dtype).contiguous()
    if entry_type == "ternary":
        return dequantize_ternary_tensor(entry)
    if entry_type == "ternary_grouped":
        return dequantize_ternary_grouped_tensor(entry)
    raise ValueError(f"Unsupported entry type: {entry_type}")


def dequantize_mixed_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
    return {name: dequantize_entry(entry) for name, entry in obj["entries"].items()}


def entry_payload_bytes(entry: dict[str, object]) -> int:
    entry_type = entry["type"]
    if entry_type in {"raw", "fp16"}:
        return tensor_nbytes(entry["data"])
    if entry_type == "int8":
        return tensor_nbytes(entry["data"]) + tensor_nbytes(entry["scale"])
    if entry_type in {"ternary", "ternary_grouped"}:
        return len(entry["packed"]) + tensor_nbytes(entry["scale"])
    raise ValueError(f"Unsupported entry type: {entry_type}")


def ordered_entries(
    entries: dict[str, dict[str, object]], layout_mode: LayoutMode
) -> OrderedDict[str, dict[str, object]]:
    items = list(entries.items())

    def type_rank(entry: dict[str, object]) -> int:
        return {"raw": 0, "fp16": 1, "int8": 2, "ternary": 3, "ternary_grouped": 4}.get(
            entry["type"], 9
        )

    if layout_mode == "alphabetical":
        items.sort(key=lambda item: item[0])
    elif layout_mode == "by_type":
        items.sort(
            key=lambda item: (
                type_rank(item[1]),
                item[1].get("family", ""),
                item[1].get("block_index", -1),
                item[0],
            )
        )
    elif layout_mode == "by_magnitude":
        items.sort(
            key=lambda item: (
                -float(item[1].get("abs_mean", 0.0)),
                type_rank(item[1]),
                item[0],
            )
        )
    elif layout_mode == "by_sparsity":
        items.sort(
            key=lambda item: (
                -float(item[1].get("zero_fraction", 0.0)),
                type_rank(item[1]),
                item[0],
            )
        )
    else:
        raise ValueError(f"Unsupported layout_mode: {layout_mode}")
    return OrderedDict(items)


def serialize_quantized_object(
    obj: dict[str, object],
    *,
    layout_mode: LayoutMode,
    prefer_zstd: bool,
) -> tuple[dict[str, bytes | str], bytes]:
    ordered = {
        "__quant_format__": obj["__quant_format__"],
        "entries": ordered_entries(obj["entries"], layout_mode),
    }
    if "passthrough_orig_dtypes" in obj:
        ordered["passthrough_orig_dtypes"] = obj["passthrough_orig_dtypes"]
    torch_buffer = io.BytesIO()
    torch.save(ordered, torch_buffer)
    torch_raw = torch_buffer.getvalue()
    torch_compressed, codec = compress_bytes(torch_raw, prefer_zstd=prefer_zstd)
    custom_raw = serialize_custom_quantized_entries(ordered["entries"])
    custom_compressed, _ = compress_bytes(custom_raw, prefer_zstd=prefer_zstd)
    return (
        {
            "torch_raw": torch_raw,
            "torch_compressed": torch_compressed,
            "custom_raw": custom_raw,
            "custom_compressed": custom_compressed,
            "codec": codec,
        },
        custom_compressed,
    )


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    api = train_gpt_api()
    GPT = api["GPT"]
    CastedLinear = api["CastedLinear"]
    restore_low_dim_params_to_fp32 = api["restore_low_dim_params_to_fp32"]
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=bool(args.tie_embeddings),
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device)
    if device.type == "cuda":
        model = model.bfloat16()
    else:
        model = model.float()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    return model


def eval_val_single(
    model: nn.Module,
    *,
    device: torch.device,
    val_tokens: Tensor,
    args: SimpleNamespace,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    batch_tokens = args.val_batch_size
    if batch_tokens < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE must cover at least one sequence")
    batch_seqs = batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(0, total_seqs, batch_seqs):
            batch_seq_end = min(batch_seq_start + batch_seqs, total_seqs)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=(device.type == "cuda")
            )
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


@dataclass
class EvalContext:
    args: argparse.Namespace
    device: torch.device
    model: nn.Module | None
    val_tokens: Tensor | None
    base_bytes_lut: Tensor | None
    has_leading_space_lut: Tensor | None
    is_boundary_token_lut: Tensor | None

    def evaluate(self, state_dict: dict[str, Tensor]) -> tuple[float, float] | None:
        if self.model is None or self.val_tokens is None:
            return None
        self.model.load_state_dict(state_dict, strict=True)
        return eval_val_single(
            self.model,
            device=self.device,
            val_tokens=self.val_tokens,
            args=SimpleNamespace(
                train_seq_len=self.args.train_seq_len,
                val_batch_size=self.args.val_batch_size,
            ),
            base_bytes_lut=self.base_bytes_lut,
            has_leading_space_lut=self.has_leading_space_lut,
            is_boundary_token_lut=self.is_boundary_token_lut,
        )


def build_eval_context(args: argparse.Namespace) -> EvalContext:
    device = torch.device(args.device)
    if not args.skip_val:
        import sentencepiece as spm

        api = train_gpt_api()
        load_validation_tokens = api["load_validation_tokens"]
        build_sentencepiece_luts = api["build_sentencepiece_luts"]
        if not args.tokenizer_path.endswith(".model"):
            raise ValueError(
                f"Expected SentencePiece .model tokenizer, got {args.tokenizer_path}"
            )
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        if int(sp.vocab_size()) != args.vocab_size:
            raise ValueError(
                f"--vocab-size={args.vocab_size} does not match tokenizer vocab size={int(sp.vocab_size())}"
            )
        val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
            build_sentencepiece_luts(sp, args.vocab_size, device)
        )
        model = build_model(args, device)
        return EvalContext(
            args,
            device,
            model,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
    return EvalContext(args, device, None, None, None, None, None)


def measure_configuration(
    state_dict: dict[str, Tensor],
    *,
    quant_plan: dict[str, dict[str, object]],
    eval_ctx: EvalContext,
    prefer_zstd: bool,
    layout_mode: LayoutMode = "alphabetical",
) -> tuple[dict[str, object], bytes]:
    quant_obj, quant_stats = quantize_mixed_state_dict(state_dict, quant_plan)
    serialized, compressed_blob = serialize_quantized_object(
        quant_obj, layout_mode=layout_mode, prefer_zstd=prefer_zstd
    )
    reconstructed_state = dequantize_mixed_state_dict(quant_obj)
    eval_result = eval_ctx.evaluate(reconstructed_state)
    payload_bytes = sum(
        entry_payload_bytes(entry) for entry in quant_obj["entries"].values()
    )
    ternary_entries = [
        entry
        for entry in quant_obj["entries"].values()
        if entry["type"] in {"ternary", "ternary_grouped"}
    ]
    grouped_ternary_entries = [
        entry
        for entry in quant_obj["entries"].values()
        if entry["type"] == "ternary_grouped"
    ]
    int8_entries = [
        entry for entry in quant_obj["entries"].values() if entry["type"] == "int8"
    ]
    fp16_entries = [
        entry for entry in quant_obj["entries"].values() if entry["type"] == "fp16"
    ]

    result = {
        "layout_mode": layout_mode,
        "codec": serialized["codec"],
        "payload_bytes": payload_bytes,
        "raw_serialized_bytes": len(serialized["custom_raw"]),
        "compressed_bytes": len(serialized["custom_compressed"]),
        "custom_raw_serialized_bytes": len(serialized["custom_raw"]),
        "custom_compressed_bytes": len(serialized["custom_compressed"]),
        "torch_raw_serialized_bytes": len(serialized["torch_raw"]),
        "torch_compressed_bytes": len(serialized["torch_compressed"]),
        "roundtrip_mse": quant_stats["roundtrip_mse"],
        "quantized_zero_fraction": quant_stats["quantized_zero_fraction"],
        "ternary_tensor_count": len(ternary_entries),
        "ternary_grouped_tensor_count": len(grouped_ternary_entries),
        "int8_tensor_count": len(int8_entries),
        "fp16_tensor_count": len(fp16_entries),
        "plan_summary": summarize_quant_plan(quant_plan),
    }
    if eval_result is not None:
        result["val_loss"], result["val_bpb"] = eval_result
    return result, compressed_blob


def load_state_dict(checkpoint_path: Path) -> dict[str, Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state_dict-like checkpoint, got {type(state)}")
    return state


def baseline_and_hybrids(
    state_dict: dict[str, Tensor], eval_ctx: EvalContext, args: argparse.Namespace
) -> tuple[dict[str, object], dict[str, bytes]]:
    configs = {
        "baseline_int8": build_quant_plan(
            state_dict, args=args, embed_mode="int8", attn_mode="int8", mlp_mode="int8"
        ),
        "conservative": build_quant_plan(
            state_dict,
            args=args,
            embed_mode="fp16",
            attn_mode="int8",
            mlp_mode="ternary",
        ),
        "aggressive": build_quant_plan(
            state_dict,
            args=args,
            embed_mode="fp16",
            attn_mode="ternary",
            mlp_mode="ternary",
        ),
    }
    results: dict[str, object] = {}
    blobs: dict[str, bytes] = {}
    for name, plan in configs.items():
        results[name], blobs[name] = measure_configuration(
            state_dict, quant_plan=plan, eval_ctx=eval_ctx, prefer_zstd=args.prefer_zstd
        )
    return results, blobs


def run_layer_sensitivity(
    state_dict: dict[str, Tensor],
    baseline_val_bpb: float | None,
    eval_ctx: EvalContext,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    targets: list[tuple[str, dict[str, dict[str, object]]]] = [
        (
            "embeddings",
            build_quant_plan(
                state_dict,
                args=args,
                embed_mode="ternary",
                attn_mode="int8",
                mlp_mode="int8",
            ),
        )
    ]

    for name in state_dict:
        family, block_index = tensor_family(name)
        if (
            family not in {"attn", "mlp"}
            or block_index is None
            or not is_large_float_tensor(state_dict[name])
        ):
            continue
        target_name = f"block_{block_index:02d}_{family}"
        overrides = {
            tensor_name: quant_spec(
                "ternary",
                **resolve_ternary_settings(args, family, block_index),
            )
            for tensor_name, tensor in state_dict.items()
            if is_large_float_tensor(tensor)
            and tensor_family(tensor_name) == (family, block_index)
        }
        targets.append(
            (
                target_name,
                build_quant_plan(
                    state_dict,
                    args=args,
                    embed_mode="int8",
                    attn_mode="int8",
                    mlp_mode="int8",
                    overrides=overrides,
                ),
            )
        )

    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for name, plan in targets:
        if name in seen:
            continue
        seen.add(name)
        result, _ = measure_configuration(
            state_dict, quant_plan=plan, eval_ctx=eval_ctx, prefer_zstd=args.prefer_zstd
        )
        result["name"] = name
        if baseline_val_bpb is not None and "val_bpb" in result:
            result["delta_val_bpb"] = result["val_bpb"] - baseline_val_bpb
        results.append(result)
    results.sort(key=lambda item: item["name"])
    return results


def build_smart_hybrid_plan(
    state_dict: dict[str, Tensor],
    sensitivity_results: list[dict[str, object]],
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    plan = build_quant_plan(
        state_dict, args=args, embed_mode="fp16", attn_mode="int8", mlp_mode="int8"
    )
    for result in sensitivity_results:
        delta = result.get("delta_val_bpb")
        if delta is None or delta > threshold:
            continue
        name = result["name"]
        if name == "embeddings":
            continue
        match = re.match(r"block_(\d+)_(attn|mlp)$", name)
        if not match:
            continue
        block_idx = int(match.group(1))
        family = match.group(2)
        for tensor_name, tensor in state_dict.items():
            if is_large_float_tensor(tensor) and tensor_family(tensor_name) == (
                family,
                block_idx,
            ):
                plan[tensor_name] = quant_spec(
                    "ternary",
                    **resolve_ternary_settings(args, family, block_idx),
                )
    return plan


def run_layout_reordering(
    state_dict: dict[str, Tensor],
    *,
    eval_ctx: EvalContext,
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    config_name = args.layout_config
    if config_name == "conservative":
        plan = build_quant_plan(
            state_dict,
            args=args,
            embed_mode="fp16",
            attn_mode="int8",
            mlp_mode="ternary",
        )
    else:
        plan = build_quant_plan(
            state_dict,
            args=args,
            embed_mode="fp16",
            attn_mode="ternary",
            mlp_mode="ternary",
        )

    results: dict[str, dict[str, object]] = {}
    for layout in ("alphabetical", "by_type", "by_magnitude", "by_sparsity"):
        result, _ = measure_configuration(
            state_dict,
            quant_plan=plan,
            eval_ctx=eval_ctx,
            prefer_zstd=args.prefer_zstd,
            layout_mode=layout,
        )
        result["config"] = config_name
        results[layout] = result
    return results


def run_zero_pressure(
    state_dict: dict[str, Tensor],
    *,
    eval_ctx: EvalContext,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    thresholds = [
        float(value) for value in args.zero_thresholds.split(",") if value.strip()
    ]
    plan = build_quant_plan(
        state_dict,
        args=args,
        embed_mode="fp16",
        attn_mode="ternary",
        mlp_mode="ternary",
    )
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        sweep_plan = {
            name: (
                {**spec, "zero_threshold": threshold}
                if spec["mode"] == "ternary"
                else dict(spec)
            )
            for name, spec in plan.items()
        }
        result, _ = measure_configuration(
            state_dict,
            quant_plan=sweep_plan,
            eval_ctx=eval_ctx,
            prefer_zstd=args.prefer_zstd,
        )
        result["sweep_zero_threshold"] = threshold
        results.append(result)
    results.sort(key=lambda item: item["sweep_zero_threshold"])
    return results


def save_artifact(output_path: Path, name: str, blob: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path.parent / f"{output_path.stem}_{name}.bin"
    artifact_path.write_bytes(blob)


def main() -> None:
    args = parse_args()
    state_dict = load_state_dict(args.checkpoint)
    eval_ctx = build_eval_context(args)
    output: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "prefer_zstd": args.prefer_zstd,
        "skip_val": args.skip_val,
        "ternary_defaults": {
            "zero_threshold": args.ternary_zero_threshold,
            "scale_fn": args.ternary_scale_fn,
            "group_size": args.ternary_group_size,
        },
        "ternary_family_overrides": {
            "zero_thresholds": args.ternary_family_zero_thresholds_map,
            "scale_fns": args.ternary_family_scale_fns_map,
            "group_sizes": args.ternary_family_group_sizes_map,
        },
        "ternary_block_overrides": {
            "zero_thresholds": {
                f"{family}:{block}": value
                for (
                    family,
                    block,
                ), value in args.ternary_block_zero_thresholds_map.items()
            },
            "scale_fns": {
                f"{family}:{block}": value
                for (family, block), value in args.ternary_block_scale_fns_map.items()
            },
            "group_sizes": {
                f"{family}:{block}": value
                for (family, block), value in args.ternary_block_group_sizes_map.items()
            },
        },
    }

    hybrid_results, hybrid_blobs = baseline_and_hybrids(state_dict, eval_ctx, args)
    output["hybrid_configs"] = hybrid_results
    if args.save_artifacts:
        for name, blob in hybrid_blobs.items():
            save_artifact(args.output, name, blob)

    baseline_val_bpb = hybrid_results["baseline_int8"].get("val_bpb")

    if args.layer_sensitivity:
        sensitivity_results = run_layer_sensitivity(
            state_dict, baseline_val_bpb, eval_ctx, args
        )
        output["layer_sensitivity"] = sensitivity_results
    else:
        sensitivity_results = []

    if args.hybrid_configs and sensitivity_results:
        smart_plan = build_smart_hybrid_plan(
            state_dict, sensitivity_results, args.smart_delta_threshold, args
        )
        smart_result, smart_blob = measure_configuration(
            state_dict,
            quant_plan=smart_plan,
            eval_ctx=eval_ctx,
            prefer_zstd=args.prefer_zstd,
        )
        smart_result["smart_delta_threshold"] = args.smart_delta_threshold
        output["hybrid_configs"]["smart_hybrid"] = smart_result
        if args.save_artifacts:
            save_artifact(args.output, "smart_hybrid", smart_blob)

    if args.layout_reordering:
        output["layout_reordering"] = run_layout_reordering(
            state_dict, eval_ctx=eval_ctx, args=args
        )

    if args.zero_pressure:
        output["zero_pressure"] = run_zero_pressure(
            state_dict, eval_ctx=eval_ctx, args=args
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
