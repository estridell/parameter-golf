# TST Proposal for Parameter-Golf

**Paper:** "Efficient Pre-Training with Token Superposition" (arXiv:2605.06546)
**Authors:** Bowen Peng, Théo Gigant, Jeffrey Quesnelle (Nous Research)
**Date:** 2026-05-16
**Status:** Research only — not implemented

---

## 1. What TST Does

Token Superposition Training (TST) is a two-phase pre-training method that improves data throughput per FLOP by processing "bags" of contiguous tokens instead of individual tokens during the early part of training.

### Phase 1: Superposition (first 20-40% of training)

**Input side:** Take contiguous bags of `s` tokens and replace them with the mean of their embeddings:

```python
# Before: input_ids shape (batch, seq_len)
# After:  averaged bags shape (batch, seq_len // s)
h = tok_embeddings(tokens[:, 0])
for i in range(1, s):
    h = h + tok_embeddings(tokens[:, i])
h = h / s
```

This compresses the sequence by `s×`, so you increase raw sequence length by `s×` to maintain equal FLOPs per step.

**Output side:** Multi-hot cross-entropy (MCE) loss — predict the next bag of `s` tokens, with loss averaged across all `s` targets:

```python
# For each bag, compute CE loss against each of the s target tokens
loss = sum(CE(pred, target[i]) for i in range(s)) / s
```

This is computationally equivalent to `s` ordinary CE terms fused together. No new kernels needed — reuse existing fused cross-entropy.

**Causality:** Labels are shifted left by `s-1` positions so bag `[t, t+s-1]` predicts bag `[t+s, t+2s-1]`.

### Phase 2: Recovery (remaining 60-80% of training)

Revert to standard next-token prediction (NTP). The model, optimizer, and all weights carry over. TST code is fully removed from the forward pass to avoid any inference contamination.

### Why It Works (Two Mechanisms)

1. **Output-side:** Related to multi-token prediction (MTP). Single head predicting mean distribution over next `s` tokens. Cheaper than full k-head MTP. Acts as a "bag-of-words topic model" over a short future window.

2. **Input-side:** The mean operator acts as a low-pass filter, dampening high-frequency variance in early training. Also forces the embedding space into an angularly dispersed layout. The paper frames this as "pre-pretraining" on a coarser distribution.

Both mechanisms have approximately additive effects (confirmed by ablation).

### Hyperparameters

| Parameter | Optimal Range | Notes |
|-----------|--------------|-------|
| Bag size `s` | 3-16 | Scales with model size. 270M: s=3-8, 600M: s=6-10, 10B: s=16 |
| Step ratio `r` | 0.2-0.4 | Fraction of total steps spent in superposition phase |

For `s >= 8`, the paper uses power-law weighting: i-th target position contributes `1/i` to the loss, motivated by the observation that mutual information between English tokens decays as power-law with distance.

### Results at Published Scales

| Model | Speedup | Notes |
|-------|---------|-------|
| 270M | ~1.5× | s=6, r=0.3. Loss 3.212 → 3.141 at matched steps |
| 600M | ~1.5× | s=6, r=0.3. Loss 3.019 → 2.943 at matched steps |
| 3B dense | ~2× | Matches 36k-step baseline at 20k steps |
| 10B-A1B MoE | ~2.5× | 12,311 → 4,768 B200-hours. Final loss 2.252 → 2.236 |

The 10B result is most impressive: TST at 2T tokens reaches lower loss than baseline at 1.05T tokens, in ~40% of wall-clock time.

---

## 2. Applicability to Parameter-Golf

### Our Setup

- **Model:** 17M params, GQA (8 heads, 4 KV), encoder-decoder with skip connections
- **Sequence length:** 1024 tokens
- **Batch:** 65536 tokens (8 grad accum × 8192 micro-batch)
- **Speed:** ~3.7s/step without compile
- **VRAM:** 5.6GB peak / 8GB total
- **Hardware:** RTX 2070 (sm_75, no FlashAttention, math SDP backend)
- **Data:** FineWeb 10B tokens

### Key Concerns

**1. Model size is far below paper's tested range.**

The paper tests 270M+ models. Our 17M model is **16× smaller** than the smallest tested scale. The optimal bag size at 270M is s=3-8; at 17M it might be s=2-3, or TST might not help at all. Small models have less redundancy in their representations, so the "superposition" compression may lose critical information.

**2. Sequence length tradeoff is complex.**

With bag size `s`, effective sequence length becomes `1024/s`. To maintain equal FLOPs, we'd increase raw seq_len to `1024*s`:

| Bag size | Raw seq_len | Effective seq_len | Memory impact |
|----------|-------------|-------------------|---------------|
| s=2 | 2048 | 1024 | ~2× attention memory |
| s=3 | 3072 | 1024 | ~3× attention memory |
| s=4 | 4096 | 1024 | ~4× attention memory |

Our 2070 has 2.4GB headroom (5.6GB used of 8GB). s=2 might fit; s=4 almost certainly won't without reducing batch size.

**3. RoPE extrapolation.**

Our model trains with RoPE at seq_len=1024. Jumping to 2048-4096 during superposition phase means RoPE encounters positions it's never seen. This could cause training instability. NTK-aware RoPE scaling or YaRN could mitigate this, but adds complexity.

**4. Wall-clock speedup mechanism differs from our bottleneck.**

TST's speedup comes from processing `s×` more data tokens per forward pass. But on our 2070:
- Attention is O(n²) — increasing seq_len by `s×` increases attention cost by `s²×`
- Our bottleneck is memory bandwidth (math SDP, no flash attention)
- The net effect might be *slower* steps, not faster

The paper's speedup assumes compute-bound setups (B200 GPUs with FSDP). Our 2070 is memory-bound.

**5. Data efficiency vs wall-clock.**

TST trades compute for data: it consumes more tokens from the loader per "equivalent" FLOP. With only 10B tokens in FineWeb, and our 17M model already training on a small fraction, data efficiency matters. TST could actually hurt us if we're data-bound rather than compute-bound.

---

## 3. What We'd Need to Change in train_gpt.py

### New Environment Variables

```python
superposition_bag_size = int(os.environ.get("SUPERPOSITION_BAG_SIZE", 0))  # 0 = disabled
superposition_ratio = float(os.environ.get("SUPERPOSITION_RATIO", 0.3))    # fraction of steps
```

### Change 1: GPT.forward() — Input Superposition

Modify the embedding step to optionally average bag embeddings:

```python
def forward(self, input_ids: Tensor, target_ids: Tensor, bag_size: int = 1) -> Tensor:
    x = self.tok_emb(input_ids)  # (batch, seq_len, dim)

    if bag_size > 1:
        # Reshape into bags and average
        bs, seq, dim = x.shape
        x = x.reshape(bs, seq // bag_size, bag_size, dim).mean(dim=2)  # (batch, seq//bag_size, dim)
        # Also reshape targets: use first token of each next bag as target
        # (or implement MCE loss separately)

    x = F.rms_norm(x, (x.size(-1),))
    # ... rest of forward pass unchanged
```

### Change 2: Loss Computation — Multi-hot Cross-Entropy

Replace the single-target CE with bag-of-targets CE:

```python
def forward(self, input_ids: Tensor, target_ids: Tensor, bag_size: int = 1) -> Tensor:
    # ... embedding and transformer blocks ...

    x = self.final_norm(x).reshape(-1, x.size(-1))

    if bag_size > 1:
        # MCE: average CE loss across all s targets in the bag
        bs, seq = target_ids.shape
        # Reshape targets into bags: (batch, seq//bag_size, bag_size)
        targets = target_ids.reshape(bs, seq // bag_size, bag_size)
        # For each bag position, compute CE against each of the s targets
        loss = 0.0
        for i in range(bag_size):
            t = targets[:, :, i].reshape(-1)
            logits_proj = self.lm_head(x) if not self.tie_embeddings else F.linear(x, self.tok_emb.weight)
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            loss += F.cross_entropy(logits.float(), t, reduction="mean")
        return loss / bag_size
    else:
        # Standard NTP loss (existing code)
        targets = target_ids.reshape(-1)
        logits_proj = self.lm_head(x) if not self.tie_embeddings else F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
```

### Change 3: Training Loop — Phase Switching

Add logic to switch between superposition and standard NTP at the right step:

```python
# In main training loop:
total_steps = args.iterations  # or estimate from wallclock
superposition_steps = int(total_steps * args.superposition_ratio)

while True:
    # ... existing loop structure ...

    bag_size = args.superposition_bag_size if step < superposition_steps else 1

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch(
            args.train_batch_tokens * bag_size,  # s× more tokens during superposition
            args.train_seq_len * bag_size,        # s× longer sequences
            grad_accum_steps
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(x, y, bag_size=bag_size)
        # ... backward, optimizer step, etc.
```

### Change 4: Sequence Length Adjustment

During superposition, increase raw sequence length by `s×` so effective sequence length stays at 1024. This requires adjusting the data loader's `next_batch` call and the RoPE cache.

### Summary of Changes

| Component | Complexity | Lines of Code |
|-----------|-----------|---------------|
| New env vars | Trivial | ~3 lines |
| Input superposition in forward() | Medium | ~10 lines |
| MCE loss computation | Medium | ~20 lines |
| Phase switching in training loop | Low | ~15 lines |
| Seq len adjustment + RoPE scaling | Medium-High | ~30 lines |
| Testing and validation | High | Unknown |

**Total estimate:** ~80-100 lines of code changes, plus significant testing.

---

## 4. Expected Speedup on RTX 2070

### Optimistic Scenario (s=2, everything works)

- Superposition phase: 2× more tokens per step, but attention cost increases ~4× (seq_len 1024→2048)
- Net step time: ~5-6s/step (up from 3.7s/step)
- Effective throughput: 2× tokens per step at ~1.5× step time → ~1.3× speedup
- Over a 600s run: ~110 effective steps equivalent vs 162 baseline steps
- **Net gain: ~1.3× if it works**

### Realistic Scenario (with overhead and issues)

- RoPE extrapolation at 2× seq_len may cause training instability
- VRAM pressure may force batch size reduction, negating gains
- Phase switching adds implementation complexity and potential bugs
- **Net gain: 0.8-1.2× (might actually be slower)**

### Pessimistic Scenario

- 17M model is too small for TST to provide meaningful gains
- Attention overhead dominates at longer seq_len on memory-bound 2070
- RoPE issues cause loss spikes
- **Net gain: <1× (slower than baseline)**

---

## 5. Risks and Unknowns

| Risk | Severity | Mitigation |
|------|----------|-----------|
| TST untested below 270M | High | Start with s=2, smallest possible bag |
| RoPE extrapolation at 2× seq_len | High | Use NTK-aware scaling or YaRN |
| VRAM OOM at longer seq_len | Medium | Reduce batch size or use gradient checkpointing |
| Training instability during phase switch | Medium | Smooth transition (gradually change bag size?) |
| Data efficiency loss (more tokens consumed) | Medium | Monitor token consumption vs loss curve |
| Implementation bugs in MCE loss | Medium | Unit test MCE against manual computation |
| Time investment vs uncertain gain | High | Cap implementation at 2 hours, abort if not working |

---

## 6. Recommended Next Steps

### Option A: Skip TST (Recommended)

**Rationale:** At 17M params on a memory-bound RTX 2070, the potential gain (~1.3× in optimistic case) doesn't justify the implementation risk and complexity. The paper's results are at 270M+ scales where the model has enough capacity to benefit from superposition. Our model is 16× smaller.

**Better alternatives:**
- Run longer training runs (current 10-min tests are short)
- Try different learning rate schedules
- Experiment with batch size / grad accum tradeoffs
- Wait for TST to be integrated into a framework (e.g., nanoGPT, litgpt) where we can just flip a flag

### Option B: Minimal Output-Only TST Experiment

If you want to test TST with minimal risk, try **output-only superposition** — skip the input averaging, just use MCE loss on the output side:

1. Add MCE loss option (s=2, predict next 2 tokens)
2. Run for 10 minutes, compare loss curve to baseline
3. If no improvement, abandon

**Time investment:** ~1 hour implementation + 10-min test run
**Expected gain:** 1.0-1.2× (output-only has weaker effect than full TST)

### Option C: Full TST Implementation

Implement both input and output superposition with phase switching:

1. Implement all changes described in Section 3
2. Test with s=2, r=0.3 on a 10-min run
3. Compare loss curve and wall-clock time to baseline

**Time investment:** ~3-4 hours implementation + multiple test runs
**Expected gain:** 0.8-1.3× (wide uncertainty)

---

## 7. Bottom Line

TST is a clever technique, but it's designed for large-scale pre-training where the model has enough capacity to benefit from coarse-to-fine learning. At 17M params on a memory-bound 2070, the gains are uncertain and the implementation is non-trivial. **Recommend skipping for now** — revisit if/when TST gets framework integration or if we scale up to a larger model.

If you want to experiment anyway, start with output-only TST (Option B) as the lowest-risk probe.

---

## Sources

- Paper: https://arxiv.org/abs/2605.06546
- Nous blog: https://nousresearch.com/token-superposition
- Tweet: https://x.com/nousresearch/status/2054610062836892054
