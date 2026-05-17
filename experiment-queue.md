# Experiment Queue — Parameter Golf

**Updated:** 2026-05-17
**Machine:** RTX 2070 (desktoparch 100.70.124.83)
**Branch:** rtx2070
**Model:** 17M params, 9L, 512 dim, GQA 8/4, Muon optimizer, batch=16384
**Current best:** ~1.22 BPB (our fork), Confirmed SOTA: 1.1147 BPB, Pending SOTA: 0.8265 BPB

## How to use this file

Trainer: work through experiments top to bottom. After each one:
1. Write results to `research/` directory
2. `cd ~/projects/parameter-golf && git add research/ && git commit -m "experiment: <short description>"`
3. Mark it done here: `- [x]` and add a one-line result summary
4. Check the clock — if time remains, move to next experiment

If you finish the queue before your deadline, pick the next unpicked item from `research/competitive-intel.md`.

### HARD RULES
- ALL work goes to the 2070 PC via SSH, NEVER on the Mac
- ALL findings must be committed to git before you finish
- One variable at a time. If you change two things, the result is garbage.
- Log step time, VRAM, loss at key steps (10, 25, 50, 75, 100, 200)
- Use seed=42 for reproducibility unless testing variance

---

## Tier 1: MUST DO (highest impact, lowest uncertainty)

### #1 — EMA Quantization Isolation [CRITICAL]

**Question:** Which technique(s) break int8 quantization?

**Background:** We tested SmearGate+CaseOps+AsymLogit+EMA all at once vs baseline. Raw training improved (-0.0197 BPB), but after int8 quantization the techniques run degraded 38.5% (2.4877 → 3.4457) vs baseline 0.3% (2.5074 → 2.5154). We blamed EMA but NEVER isolated the variable. Could be SmearGate, CaseOps, AsymLogit, or any combination.

**Method:** 2^4 factorial design. Run all 16 combinations of {SmearGate, CaseOps, AsymLogit, EMA} on/off. 300s wallclock each.

**For each run, record:**
- Raw val_bpb (before quantization)
- Post-int8+zlib val_bpb
- Degradation % = (quant_bpb - raw_bpb) / raw_bpb * 100

**Expected runtime:** ~80 min (16 runs × 300s)
**Impact:** CRITICAL — determines whether EMA is actually the problem
**Deps:** None

- [ ] Not started

---

### #2 — QK-Gain Sweep

**Question:** What's the optimal attention scaling factor?

**Background:** QK-Gain controls the scaling of attention logits before softmax. We use 1.5 (default). Top leaderboard submissions use 5.25. This is a single env var change that could meaningfully improve loss. Never tested.

**Method:** Run 4 values: 1.5, 2.5, 3.5, 5.25. 300s wallclock each. Same config otherwise.

**Record:** loss at step 25, 50, 75 + step time for each value.

**Expected runtime:** ~20 min (4 × 300s)
**Impact:** HIGH — cheap experiment, potentially big gain
**Deps:** None

- [ ] Not started

---

### #3 — Full Stack 600s + Validation [BPB Measurement]

**Question:** What's our actual BPB with the current best stack?

**Background:** We have NEVER run the full stack with validation enabled. All numbers are training loss proxies. The leaderboard compares val_bpb, not loss. We need one real number.

**Method:** Single 600s run with:
- Best config from #2 (QK-Gain winner)
- 9L, BigramHash 5120, SmearGate, CaseOps, AsymLogit
- VAL_LOSS_EVERY=50 (not 0)
- Enable final validation + roundtrip validation
- EVAL_STRIDE=0 first (baseline), then stride-64 in #9

**Record:** val_bpb, submission size int8+zlib, peak VRAM, total steps.

**Expected runtime:** ~10 min (one 600s run)
**Impact:** HIGH — gives us our first real BPB number
**Deps:** #2 (use best QK-Gain)

- [ ] Not started

---

### #4 — Partial RoPE 0.25 + LN Scale

**Question:** Does the leaderboard-standard RoPE config work?

**Background:** We tested Partial RoPE at fraction=0.5 and found "no meaningful improvement." But the leaderboard standard is rope_dims=16 out of 64 head dimensions (fraction=0.25, NOT 0.5). Also, Partial RoPE is always paired with LN Scale (per-layer learned scaling) in top submissions. We tested the wrong config without its partner.

**Method:** Implement rope_dims=16 (fraction=0.25) WITH LN Scale. 300s run vs matched baseline.

**Record:** loss comparison at step 25, 50, 75 vs baseline at same steps.

**Expected runtime:** ~5 min
**Impact:** MEDIUM — could be a free improvement
**Deps:** None

- [ ] Not started

---

### #5 — MLP 3x on 9 Layers

**Question:** Does wider MLP help when not confounded by too many layers?

**Background:** We tested MLP 3x on 12 layers and found it "slower with no benefit." But 12L is already worse than 9L within 300s (24% fewer steps). Adding MLP 3x on top of an already-penalized 12L compounds the step-time penalty. The test was confounded.

**Method:** Run 9L + MLP_MULT=3, 300s. Compare to 9L baseline at same wallclock time.

**Record:** loss at step 25, 50, 75, step time, VRAM usage.

**Expected runtime:** ~5 min
**Impact:** MEDIUM — could unlock wider MLPs on 9L
**Deps:** None

- [ ] Not started

---

## Tier 2: SHOULD DO (high impact, medium effort)

### #6 — SwiGLU Activation

**Question:** Is SwiGLU better than relu-squared?

**Background:** Multiple leaderboard submissions and competitive agents use SwiGLU. We still use relu-squared. architecture-tricks.md doesn't even mention SwiGLU. It's a simple MLP activation swap.

**Method:** Replace relu^2 with swish(x) * linear(x) in MLP. 300s run vs baseline.

**Record:** loss comparison, step time delta.

**Expected runtime:** ~5 min
**Impact:** MEDIUM — standard technique, easy to test
**Deps:** None

- [ ] Not started

---

### #7 — BigramHash Size Sweep

**Question:** What's the optimal hash table size?

**Background:** We tested 4096 and 5120. 5120 was marginally better. But we haven't tried 8192 or 16384. Bigger hash = more unique bigram features but more params. Where's the sweet spot?

**Method:** Run 4 sizes: 4096, 5120, 8192, 16384. 300s each. Same config otherwise.

**Record:** params, loss at step 75, step time, VRAM.

**Expected runtime:** ~20 min (4 × 300s)
**Impact:** MEDIUM — might squeeze a few more %
**Deps:** None

- [ ] Not started

---

### #8 — Batch Size Recovery (65536 OOM Investigation)

**Question:** Why does 65536 tokens/batch OOM on PyTorch 2.12?

**Background:** Earlier experiments used 65536 batch. After PyTorch 2.12.0+cu130 upgrade, it OOMs. All recent experiments are at 16384 (4x smaller). Nobody investigated why. If we can recover 65536, all experiments run 4x faster per step.

**Method:**
1. Run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. If still OOM, run with `torch.cuda.memory_summary()` and analyze
3. Check if it's fragmentation vs actual memory increase
4. Try `torch.cuda.empty_cache()` before run

**Record:** root cause, fix if possible, VRAM breakdown.

**Expected runtime:** ~30 min investigation
**Impact:** HIGH — 4x faster experiments if fixed
**Deps:** None

- [ ] Not started

---

### #9 — Stride-64 Eval Confirmation

**Question:** Does stride-64 eval actually improve measured BPB?

**Background:** 77% of leaderboard submissions use stride=64 eval. research/eval-and-compression.md estimates ~0.034 BPB improvement at zero artifact cost. We haven't confirmed this with our own model.

**Method:** Take the #3 run config, run twice: EVAL_STRIDE=0 vs EVAL_STRIDE=64. 600s each.

**Record:** val_bpb comparison.

**Expected runtime:** ~20 min (2 × 600s)
**Impact:** HIGH — free BPB improvement if confirmed
**Deps:** #3 (need same config for fair comparison)

- [ ] Not started

---

## Tier 3: IMPLEMENT NEW TECHNIQUES

### #10 — XSA-4 (Cross-Sparse Attention)

**Question:** Does sparse attention help?

**Background:** XSA appears in 392 submissions and DOMINATES the top 6 leaderboard slots: #1 XSA-all (1.1147 BPB), #5 XSA4 (1.1271 BPB), #6 Partial XSA (1.1307 BPB). We have zero implementation. XSA saves ~7.3M parameters by skipping attention on early layers — these saved params go to wider MLPs.

**Method:** Implement XSA-4 (attention only on layers 4+, skip layers 0-3). Restructure attention mask. Run 300s smoke test.

**Record:** loss comparison, step time, VRAM, param count.

**Expected runtime:** 2-3 hr (implementation + testing)
**Impact:** HIGH — most correlated technique with top scores
**Deps:** None

- [ ] Not started

---

### #11 — Depth Recurrence (5 Unique × 2)

**Question:** Can we get more effective layers without more params?

**Background:** 116 submissions use depth recurrence. 5 unique layers applied 2x = 10 effective layers with only 5 layers of params. Known risk: quantization error compounds through repeated layers (needs QAT). PR #1903 achieved 0.9418 BPB with this approach.

**Method:** Implement layer loop (5 unique layers, each applied twice). Run 300s smoke test.

**Record:** loss comparison vs 9L and 10L, step time, VRAM.

**Expected runtime:** 2-3 hr (implementation + testing)
**Impact:** HIGH — more depth for free
**Deps:** Needs QAT for quantization-safe submission

- [ ] Not started

---

### #12 — SP8192 Tokenizer

**Question:** Does larger vocab = better BPB?

**Background:** ALL competitive submissions use SP8192 (SentencePiece 8192 vocab). We use SP1024. The gap is estimated at 0.02-0.05 BPB — potentially the single largest improvement available. This is the one technique that blocks everything else: all hyperparams need retuning after tokenizer change.

**Method:**
1. Run `python3 data/cached_challenge_fineweb.py --variant sp8192` to retokenize data
2. Retrain with new vocab_size=8192
3. May need to adjust model_dim or other params for the larger vocab

**Record:** BPB comparison vs SP1024 baseline.

**Expected runtime:** 4-6 hr (data pipeline + training)
**Impact:** HIGHEST — required for competitive BPB
**Deps:** Data pipeline change. Do this when ready for a major config shift.

- [ ] Not started

---

### #13 — Score-First TTT (Test-Time Training)

**Question:** Free eval-time BPB improvement?

**Background:** TTT fine-tunes on validation data before scoring. Score each token BEFORE weight update (PR #461). 3 epochs, lr=0.005, LoRA per-doc reset. ~0.005-0.02 BPB improvement, zero training cost. Top 2 pending claims use TTT.

**Method:** Implement LoRA adapter, per-doc reset logic, score-first evaluation loop.

**Record:** BPB with vs without TTT on same model.

**Expected runtime:** 2-3 hr (implementation + testing)
**Impact:** HIGH — free BPB at eval time
**Deps:** LoRA implementation needed

- [ ] Not started

---

### #14 — TrigramHash

**Question:** Does it stack with BigramHash?

**Background:** BigramHash is a confirmed winner (7-8% loss reduction). TrigramHash extends the idea to adjacent triplets. Easy to test, low priority.

**Method:** Implement trigram hash on top of BigramHash. 300s run vs BigramHash-only baseline.

**Record:** loss comparison, param count, step time.

**Expected runtime:** 1 hr
**Impact:** LOW — incremental at best
**Deps:** BigramHash (already done)

- [ ] Not started

---

## Tier 4: NICE TO HAVE

### #15 — zstd-22 Compression

**Background:** Already confirmed 4.8% savings over zlib-9, decompresses 2.6x faster. 501 submissions use zstd. Just needs implementation in submission pipeline.

**Expected runtime:** 30 min
**Impact:** LOW — artifact size optimization

- [ ] Not started

---

### #16 — Verify Pending SOTA Claims

**Background:** ndokutovich claims 0.8265 BPB (SLOT-24 + Pre-Quant AdamW TTT). If valid, our SOTA gap is 0.29 BPB (not 0.11). We haven't analyzed their code.

**Expected runtime:** 2 hr
**Impact:** LOW — intel only

- [ ] Not started

---

### #17 — GPTQ Self-Gen Calibration

**Background:** Better quantization for submission artifact. Current int8+zlib leaves a lot of quality on the table.

**Expected runtime:** 3-4 hr
**Impact:** MEDIUM — better artifact quality

- [ ] Not started

---

### #18 — N-gram Mixing at Eval

**Background:** Character-level mixing during evaluation. Free eval boost with no training cost.

**Expected runtime:** 1 hr
**Impact:** LOW

- [ ] Not started

---

## Current Default Config

```bash
# Environment variables for training
export LAYERS=9
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export BATCH_TOKENS=16384
export SEQ_LEN=1024
export OPTIMIZER=muon
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_SIZE=5120
export SMEARGATE_ENABLED=1
export CASEOPS_ENABLED=1
export ASYMLOGIT_ENABLED=1
export EMA_ENABLED=0          # disabled until quantization sorted (#1)
export QK_GAIN=1.5            # sweep pending (#2)
export VAL_LOSS_EVERY=0       # 0=skip for speed, >0=enable for BPB measurement
export EVAL_STRIDE=0          # 0=standard, 64=stride eval (#9)
export SEED=42
export MAX_WALLCLOCK_SECONDS=300
```

## Known Issues

1. **EMA quantization disaster** — root cause unknown. 38.5% BPB degradation after int8. Needs isolation (#1).
2. **Partial RoPE** — tested at wrong fraction (0.5 not 0.25) without LN Scale. Invalid result.
3. **MLP 3x** — tested on 12L not 9L. Confounded by too-many-layers penalty.
4. **Batch regression** — 65536 OOMs on PyTorch 2.12, never investigated.
5. **No actual BPB** — all numbers are training loss proxies. Never ran full validation.
6. **H100 extrapolation** — 85x ratio may be wrong. Top submissions show ~30x ratio (121ms/step on H100 vs our 3.7s on 2070).

## Reference Files

- `research/audit-report.md` — full audit with detailed analysis
- `research/competitive-intel.md` — 1,500+ leaderboard submissions analyzed
- `research/architecture-tricks.md` — technique catalog with code examples
- `research/gap-analysis.md` — what we're missing vs leaderboard
- `research/step-budget-analysis.md` — technique step-time measurements
- `research/technique-comparison.md` — EMA quantization disaster (needs isolation)
