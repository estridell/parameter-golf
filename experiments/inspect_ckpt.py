#!/usr/bin/env python3
"""Inspect newarch_sp8192 checkpoint structure."""
import torch
ckpt = torch.load("checkpoints/newarch_sp8192.pt", map_location="cpu", weights_only=False)
print("Top-level keys:", list(ckpt.keys()))
if "model" in ckpt:
    s = ckpt["model"]
elif "state_dict" in ckpt:
    s = ckpt["state_dict"]
else:
    s = ckpt
print(f"Total tensors: {len(s)}")
total_params = sum(v.numel() for v in s.values() if isinstance(v, torch.Tensor))
print(f"Total params: {total_params:,}")
print()
for k in sorted(s.keys()):
    v = s[k]
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} {v.dtype}")
