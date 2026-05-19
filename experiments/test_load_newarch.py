#!/usr/bin/env python3
"""Test that newarch checkpoint loads correctly."""
import os, sys, torch
os.environ["VOCAB_SIZE"] = "8192"
os.environ["NUM_LOOPS"] = "0"
os.environ["SMEAR_GATE_ENABLED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import GPT, Hyperparameters, restore_fp32_params
h = Hyperparameters()
print(f"Hyperparams: vocab={h.vocab_size}, layers={h.num_layers}, dim={h.model_dim}, heads={h.num_heads}, kv={h.num_kv_heads}")
print(f"smear_gate_enabled={h.smear_gate_enabled}, skip_gates_enabled={h.skip_gates_enabled}")

model = GPT(h).bfloat16()
restore_fp32_params(model)
state = torch.load("checkpoints/newarch_sp8192.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state, strict=True)
print(f"SUCCESS: loaded newarch checkpoint")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
