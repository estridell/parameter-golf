from __future__ import annotations

import struct
import zlib
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import zstandard as zstd
except ImportError:
    zstd = None


CompressionCodec = Literal["zstd", "zlib"]
ScaleFn = Literal["absmean", "median"]

DTYPE_TO_CODE = {
    torch.uint8: 1,
    torch.int8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.float16: 6,
    torch.float32: 7,
    torch.bfloat16: 8,
    torch.bool: 9,
}
ENTRY_TYPE_TO_CODE = {
    "raw": 1,
    "fp16": 2,
    "int8": 3,
    "ternary": 4,
    "ternary_grouped": 5,
}


def compress_bytes(
    payload: bytes, prefer_zstd: bool = True, zstd_level: int = 22
) -> tuple[bytes, CompressionCodec]:
    if prefer_zstd and zstd is not None:
        return zstd.ZstdCompressor(level=zstd_level).compress(payload), "zstd"
    return zlib.compress(payload, level=9), "zlib"


def decompress_bytes(blob: bytes, codec: CompressionCodec) -> bytes:
    if codec == "zstd":
        if zstd is None:
            raise RuntimeError(
                "zstandard is required to decompress zstd-compressed payloads"
            )
        return zstd.ZstdDecompressor().decompress(blob)
    return zlib.decompress(blob)


def pack_ternary_base3(q: Tensor) -> tuple[bytes, int]:
    flat = (q.reshape(-1).to(torch.int8).cpu().numpy().astype(np.int16) + 1).astype(
        np.uint8
    )
    count = int(flat.size)
    pad = (-count) % 5
    if pad:
        flat = np.pad(flat, (0, pad))
    grouped = flat.reshape(-1, 5).astype(np.uint16)
    packed = (
        grouped[:, 0]
        + 3 * grouped[:, 1]
        + 9 * grouped[:, 2]
        + 27 * grouped[:, 3]
        + 81 * grouped[:, 4]
    )
    return packed.astype(np.uint8).tobytes(), count


def unpack_ternary_base3(data: bytes, count: int) -> Tensor:
    values = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    trits = np.zeros((values.size, 5), dtype=np.int8)
    for idx in range(5):
        trits[:, idx] = values % 3
        values //= 3
    return torch.from_numpy(trits.reshape(-1)[:count].astype(np.int8) - 1)


def ternary_round_ste(w: Tensor, scale: Tensor, threshold: float = 0.5) -> Tensor:
    scaled = w / scale.clamp_min(1e-5)
    with torch.no_grad():
        q = torch.zeros_like(scaled)
        q[scaled > threshold] = 1.0
        q[scaled < -threshold] = -1.0
    return q.detach() + (scaled - scaled.detach())


def _row_scale(w: Tensor, scale_fn: ScaleFn) -> Tensor:
    w32 = w.float()
    if w32.ndim == 1:
        w32 = w32.unsqueeze(0)
    if scale_fn == "absmean":
        scale = w32.abs().mean(dim=-1, keepdim=True)
    elif scale_fn == "median":
        scale = w32.abs().median(dim=-1, keepdim=True).values
    else:
        raise ValueError(f"Unsupported scale_fn: {scale_fn}")
    return scale.clamp_min(1e-8)


def _encode_dims(shape: list[int] | tuple[int, ...]) -> bytes:
    if len(shape) > 255:
        raise ValueError(f"Too many dimensions to encode: {len(shape)}")
    header = bytearray(struct.pack("<B", len(shape)))
    for dim in shape:
        header.extend(struct.pack("<I", int(dim)))
    return bytes(header)


def _tensor_raw_bytes(tensor: Tensor) -> bytes:
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def _encode_tensor_blob(tensor: Tensor) -> bytes:
    t = tensor.detach().cpu().contiguous()
    dtype_code = DTYPE_TO_CODE.get(t.dtype)
    if dtype_code is None:
        raise ValueError(f"Unsupported tensor dtype for custom binary: {t.dtype}")
    raw = _tensor_raw_bytes(t)
    return (
        struct.pack("<B", dtype_code)
        + _encode_dims(list(t.shape))
        + struct.pack("<I", len(raw))
        + raw
    )


def serialize_custom_quantized_entries(entries: dict[str, dict[str, object]]) -> bytes:
    buf = bytearray(b"TQF1")
    buf.extend(struct.pack("<I", len(entries)))
    for name, entry in entries.items():
        name_bytes = name.encode("utf-8")
        buf.extend(struct.pack("<H", len(name_bytes)))
        buf.extend(name_bytes)
        entry_type = entry["type"]
        type_code = ENTRY_TYPE_TO_CODE.get(entry_type)
        if type_code is None:
            raise ValueError(f"Unsupported entry type for custom binary: {entry_type}")
        buf.extend(struct.pack("<B", type_code))

        if entry_type == "raw":
            buf.extend(_encode_tensor_blob(entry["data"]))
            continue

        if entry_type == "fp16":
            orig_dtype = entry.get("orig_dtype")
            orig_torch_dtype = (
                getattr(torch, orig_dtype)
                if isinstance(orig_dtype, str)
                else entry["data"].dtype
            )
            buf.extend(struct.pack("<B", DTYPE_TO_CODE[orig_torch_dtype]))
            buf.extend(_encode_tensor_blob(entry["data"]))
            continue

        if entry_type == "int8":
            buf.extend(struct.pack("<B", DTYPE_TO_CODE[getattr(torch, entry["dtype"])]))
            buf.extend(struct.pack("<B", 1 if entry["scheme"] == "per_row" else 0))
            buf.extend(_encode_tensor_blob(entry["data"]))
            buf.extend(_encode_tensor_blob(entry["scale"]))
            continue

        if entry_type == "ternary":
            buf.extend(struct.pack("<B", DTYPE_TO_CODE[getattr(torch, entry["dtype"])]))
            buf.extend(_encode_dims(entry["shape"]))
            buf.extend(_encode_tensor_blob(entry["scale"]))
            packed = entry["packed"]
            buf.extend(struct.pack("<I", entry["n_trits"]))
            buf.extend(struct.pack("<I", len(packed)))
            buf.extend(packed)
            continue

        if entry_type == "ternary_grouped":
            buf.extend(struct.pack("<B", DTYPE_TO_CODE[getattr(torch, entry["dtype"])]))
            buf.extend(_encode_dims(entry["shape"]))
            buf.extend(struct.pack("<I", int(entry["group_size"])))
            buf.extend(struct.pack("<I", int(entry["pad_cols"])))
            buf.extend(_encode_tensor_blob(entry["scale"]))
            packed = entry["packed"]
            buf.extend(struct.pack("<I", entry["n_trits"]))
            buf.extend(struct.pack("<I", len(packed)))
            buf.extend(packed)
            continue

    return bytes(buf)


def quantize_ternary_tensor(
    tensor: Tensor,
    *,
    scale_fn: ScaleFn = "absmean",
    zero_threshold: float = 0.5,
    scale_dtype: torch.dtype = torch.float16,
) -> dict[str, object]:
    t = tensor.detach().to("cpu").float().contiguous()
    original_shape = list(t.shape)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim != 2:
        raise ValueError(
            f"Ternary export expects 1D or 2D tensors, got shape {tuple(original_shape)}"
        )
    scale = _row_scale(t, scale_fn)
    scaled = t / scale
    q = torch.zeros_like(scaled, dtype=torch.int8)
    q[scaled > zero_threshold] = 1
    q[scaled < -zero_threshold] = -1
    packed, trit_count = pack_ternary_base3(q)
    return {
        "type": "ternary",
        "packed": packed,
        "n_trits": trit_count,
        "scale": scale.to(dtype=scale_dtype).squeeze(-1).contiguous(),
        "shape": original_shape,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "scale_fn": scale_fn,
        "zero_threshold": float(zero_threshold),
        "zero_fraction": float((q == 0).float().mean().item()),
    }


def quantize_ternary_grouped_tensor(
    tensor: Tensor,
    *,
    group_size: int,
    scale_fn: ScaleFn = "absmean",
    zero_threshold: float = 0.5,
    scale_dtype: torch.dtype = torch.float16,
) -> dict[str, object]:
    if group_size <= 0:
        raise ValueError(
            f"group_size must be positive for grouped ternary, got {group_size}"
        )
    t = tensor.detach().to("cpu").float().contiguous()
    original_shape = list(t.shape)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim != 2:
        raise ValueError(
            f"Grouped ternary export expects 1D or 2D tensors, got shape {tuple(original_shape)}"
        )
    pad_cols = (-t.shape[1]) % group_size
    padded = F.pad(t, (0, pad_cols)) if pad_cols else t
    grouped = padded.reshape(t.shape[0], -1, group_size)
    if scale_fn == "absmean":
        scale = grouped.abs().mean(dim=-1, keepdim=True)
    elif scale_fn == "median":
        scale = grouped.abs().median(dim=-1, keepdim=True).values
    else:
        raise ValueError(f"Unsupported scale_fn: {scale_fn}")
    scale = scale.clamp_min(1e-8)
    normalized = grouped / scale
    q = torch.zeros_like(grouped, dtype=torch.int8)
    q[normalized > zero_threshold] = 1
    q[normalized < -zero_threshold] = -1
    packed, trit_count = pack_ternary_base3(q.reshape(-1))
    return {
        "type": "ternary_grouped",
        "packed": packed,
        "n_trits": trit_count,
        "scale": scale.to(dtype=scale_dtype).reshape(t.shape[0], -1).contiguous(),
        "shape": original_shape,
        "group_size": int(group_size),
        "pad_cols": int(pad_cols),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "scale_fn": scale_fn,
        "zero_threshold": float(zero_threshold),
        "zero_fraction": float((q == 0).float().mean().item()),
    }


def dequantize_ternary_tensor(entry: dict[str, object]) -> Tensor:
    q = unpack_ternary_base3(entry["packed"], entry["n_trits"]).float()
    shape = list(entry["shape"])
    if len(shape) == 1:
        q = q.reshape(1, shape[0])
        scale = entry["scale"].float().reshape(1, 1)
        out = q * scale
    else:
        q = q.reshape(shape[0], shape[1])
        scale = entry["scale"].float().reshape(shape[0], 1)
        out = q * scale
    dtype = getattr(torch, entry["dtype"])
    return out.reshape(shape).to(dtype=dtype).contiguous()


def dequantize_ternary_grouped_tensor(entry: dict[str, object]) -> Tensor:
    q = unpack_ternary_base3(entry["packed"], entry["n_trits"]).float()
    shape = list(entry["shape"])
    rows = shape[0] if len(shape) > 1 else 1
    cols = shape[-1]
    group_size = int(entry["group_size"])
    pad_cols = int(entry.get("pad_cols", 0))
    padded_cols = cols + pad_cols
    q = q.reshape(rows, padded_cols // group_size, group_size)
    scale = entry["scale"].float().reshape(rows, -1, 1)
    out = (q * scale).reshape(rows, padded_cols)[..., :cols]
    dtype = getattr(torch, entry["dtype"])
    return out.reshape(shape).to(dtype=dtype).contiguous()


def weight_mse(reference: Tensor, reconstructed: Tensor) -> float:
    ref = reference.detach().float()
    rec = reconstructed.detach().float()
    return float(torch.mean((ref - rec) ** 2).item())


class BitLinear(nn.Module):
    """BitNet-style ternary linear with per-group scaling and STE."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        scale_fn: ScaleFn = "absmean",
        zero_threshold: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.scale_fn_name = scale_fn
        self.zero_threshold = zero_threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _group_scale(self, weight: Tensor) -> tuple[Tensor, int]:
        if self.group_size <= 0:
            return _row_scale(weight, self.scale_fn_name), 0
        pad = (-weight.shape[1]) % self.group_size
        if pad:
            weight = F.pad(weight, (0, pad))
        grouped = weight.reshape(weight.shape[0], -1, self.group_size)
        if self.scale_fn_name == "absmean":
            scale = grouped.abs().mean(dim=-1, keepdim=True)
        elif self.scale_fn_name == "median":
            scale = grouped.abs().median(dim=-1, keepdim=True).values
        else:
            raise ValueError(f"Unsupported scale_fn: {self.scale_fn_name}")
        return scale.clamp_min(1e-8), pad

    def _quantized_weight(self, weight: Tensor) -> Tensor:
        if self.group_size <= 0:
            scale = _row_scale(weight, self.scale_fn_name)
            q = ternary_round_ste(weight, scale, self.zero_threshold).clamp(-1, 1)
            return q * scale
        scale, pad = self._group_scale(weight)
        padded = F.pad(weight, (0, pad)) if pad else weight
        grouped = padded.reshape(weight.shape[0], -1, self.group_size)
        q = ternary_round_ste(grouped, scale, self.zero_threshold).clamp(-1, 1)
        out = (q * scale).reshape(weight.shape[0], -1)
        if pad:
            out = out[:, : weight.shape[1]]
        return out

    def forward(self, x: Tensor) -> Tensor:
        q_weight = self._quantized_weight(self.weight)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, q_weight.to(dtype=x.dtype), bias)

    @torch.no_grad()
    def ternary_export(
        self, scale_dtype: torch.dtype = torch.float16
    ) -> dict[str, object]:
        if self.group_size <= 0:
            return quantize_ternary_tensor(
                self.weight,
                scale_fn=self.scale_fn_name,
                zero_threshold=self.zero_threshold,
                scale_dtype=scale_dtype,
            )

        return quantize_ternary_grouped_tensor(
            self.weight,
            group_size=self.group_size,
            scale_fn=self.scale_fn_name,
            zero_threshold=self.zero_threshold,
            scale_dtype=scale_dtype,
        )
