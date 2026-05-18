#!/usr/bin/env python3
"""Add gradient checkpointing support to SOTA train_gpt.py for RTX 2070."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'train_gpt.py'
with open(path, 'r') as f:
    lines = f.readlines()

out = []
i = 0
in_hyperparams = False
added_checkpoint_import = False
added_checkpoint_hparam = False

while i < len(lines):
    line = lines[i]

    # Add checkpoint import after existing imports
    if not added_checkpoint_import and line.strip().startswith('from torch import Tensor, nn'):
        out.append(line)
        out.append('from torch.utils.checkpoint import checkpoint as grad_checkpoint\n')
        added_checkpoint_import = True
        i += 1
        continue

    # Add gradient_checkpoint_enabled to Hyperparameters (after ema_decay)
    if not added_checkpoint_hparam and 'ema_decay' in line and 'float' in line and 'os.environ' in line:
        out.append(line)
        out.append('    gradient_checkpoint_enabled = bool(int(os.environ.get("GRADIENT_CHECKPOINT_ENABLED", "0")))\n')
        added_checkpoint_hparam = True
        i += 1
        continue

    # Find the Block.__init__ to store the flag
    if 'class Block(nn.Module):' in line:
        out.append(line)
        i += 1
        # Find the __init__ and add the flag
        while i < len(lines):
            if 'def __init__' in lines[i]:
                out.append(lines[i])
                i += 1
                # Find end of super().__init__()
                while i < len(lines):
                    out.append(lines[i])
                    if 'super().__init__' in lines[i]:
                        break
                    i += 1
                i += 1
                out.append('        self.use_checkpoint = False  # set by GPT.__init__\n')
                continue
            # If we hit forward without finding init, break
            if 'def forward' in lines[i]:
                break
            out.append(lines[i])
            i += 1
        continue

    # In GPT.__init__, after creating blocks, set checkpoint flag
    if 'self.blocks = nn.ModuleList(' in line and not any('use_checkpoint' in l for l in out):
        # Collect the entire ModuleList(...) block
        while i < len(lines):
            out.append(lines[i])
            if lines[i].strip().endswith(')'):
                break
            i += 1
        i += 1
        # Add checkpoint flag setting after blocks creation
        out.append('        # RTX 2070 patch: gradient checkpointing\n')
        out.append('        if h.gradient_checkpoint_enabled:\n')
        out.append('            for blk in self.blocks:\n')
        out.append('                blk.use_checkpoint = True\n')
        continue

    # Wrap block calls with grad_checkpoint in _forward_hidden encoder loop
    if 'x = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)' in line:
        indent = line[:len(line) - len(line.lstrip())]
        out.append(indent + 'if getattr(self.blocks[i], "use_checkpoint", False) and self.blocks[i].training:\n')
        out.append(indent + '    x = grad_checkpoint(self.blocks[i], x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens, max_seqlen, use_reentrant=False)\n')
        out.append(indent + 'else:\n')
        out.append(indent + '    x = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)\n')
        i += 1
        continue

    # Also wrap parallel block calls
    if 'lane0, lane1 = self._parallel_block(' in line:
        indent = line[:len(line) - len(line.lstrip())]
        # Collect the full call (may span multiple lines)
        call_lines = [line]
        while ')' not in lines[i]:
            i += 1
            call_lines.append(lines[i])
        full_call = ''.join(call_lines).strip()
        # Don't wrap parallel blocks for now (complex signature)
        for cl in call_lines:
            out.append(cl)
        i += 1
        continue

    out.append(line)
    i += 1

with open(path, 'w') as f:
    f.writelines(out)

print(f"Gradient checkpointing added: {len(out)} lines")
