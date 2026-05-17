# Parameter-Golf Research Audit — Full Review

**Date:** 2026-05-17
**Auditor:** Trainer (automated)
**Scope:** ALL research files in ~/projects/parameter-golf/research/
**Branch:** rtx2070
**Current best:** ~1.22 BPB (our fork), Confirmed SOTA: 1.1147 BPB, Pending SOTA: 0.8265 BPB

---

## Executive Summary

We have built a solid foundation — SmearGate, CaseOps, AsymLogit, BigramHash, U-Net skips, GQA, Muon optimizer — but we are stuck at ~1.22 BPB while the confirmed SOTA is 1.1147 BPB. The 0.11 BPB gap comes from **five missing high-impact techniques** (XSA, depth recurrence, SP8192, TTT, SwiGLU) and one **unresolved disaster** (EMA + int8 quantization).

The research corpus is extensive (21 files, ~130KB) but has critical gaps: no isolation experiments, no EMA root-cause analysis, no BigramHash+full-stack validation, and zero experiments on the techniques that dominate the top 10 leaderboard positions.

---

## PART 1: THE GOOD — Solid, Well-Supported Findings

### 1.1 BigramHash: Confirmed Winner ★★★★★

**Evidence:** Three independent experiment files confirm 7-8% loss reduction with zero speed overhead.
- bigrampatch-results.md: BH4096 = -7.6% loss at step 75 (4.28 vs 4.63)
- experiment-results-may17.md: BH5120 = -7.8% at step 25, -7.6% at step 75
- overnight-session-2026-05-17.md: BH4096 = -7.8% at step 25
- All 583 BigramHash submissions in competitive intel confirm this
- Top 5 neural leaderboard entries ALL use BigramHash

**Verdict:** KEEP as default. Enable BIGRAM_HASH_ENABLED=1, size=5120.

**Caveat:** We only tested at 16384 batch (not 65536). The relative improvement should hold but hasn't been confirmed at the original batch size.

### 1.2 SmearGate: Confirmed Winner ★★★★

**Evidence:**
- 396 submissions use it, including top entries
- step-budget-analysis.md: +33ms/step (+0.9%), negligible overhead
- technique-comparison.md: part of the combo that achieved -0.0197 BPB (0.78% improvement)
- architecture-tricks.md documents init at +3.0 (near-identity) is critical

**Verdict:** KEEP. Already implemented with proper init.

### 1.3 CaseOps + AsymLogit: Confirmed Winners ★★★★

**Evidence:**
- technique-comparison.md: combo achieves -0.0197 BPB with +40ms/step
- caseops-asymlogit-notes.md: smoke-tested, no regression
- Both appear in top 4 leaderboard submissions (AsymLogit)
- CaseOps appears in top-6 submissions (#1729)

**Verdict:** KEEP. Both are cheap and stack well.

### 1.4 OrthoInit: Free Lunch ★★★★

**Evidence:**
- 171 submissions use it
- step-budget-analysis.md: zero runtime overhead (init-only)
- architecture-tricks.md: properly documented with scaled variant
- No downside risk — pure initialization improvement

**Verdict:** KEEP. Should be ON by default (currently defaults to OFF).

### 1.5 Stride-64 Eval: Confirmed Winner ★★★★★

**Evidence:**
- eval-and-compression.md: ~0.034 BPB improvement, zero artifact cost
- 77% of submissions (333/430) use stride=64
- step-budget-analysis.md: zero training overhead (eval-time only)
- Well-documented pseudocode in research

**Verdict:** KEEP. This is the single largest free BPB gain we have NOT been using by default. EVAL_STRIDE should default to 64, not 0.

### 1.6 zstd-22 Compression: Confirmed ★★★

**Evidence:**
- compression-comparison.md: 310KB savings (4.8%) over zlib-9, decompresses 2.6x faster
- 501 submissions use zstd vs 411 using zlib
- Straightforward implementation (~5 lines)

**Verdict:** Implement for submission artifact.

### 1.7 U-Net Skip Connections, GQA, Muon, Warmdown LR ★★★★

Already implemented and working. These are table-stakes for competitive submissions. No issues found.

---

## PART 2: THE BAD — Potentially Wrong or Misleading

### 2.1 THE EMA QUANTIZATION DISASTER — Root Cause Unknown ★★★★★ CRITICAL

**The problem:**
technique-comparison.md shows that SmearGate+CaseOps+AsymLogit+EMA achieves -0.0197 BPB in raw training, but after int8 quantization:
- Baseline: 0.3% BPB degradation (2.5074 → 2.5154)
- Techniques: 38.5% BPB degradation (2.4877 → 3.4457)

**The gap in analysis:** The report blames EMA but NEVER ISOLATED THE VARIABLE.

We tested: Baseline (no techniques) vs SmearGate+CaseOps+AsymLogit+EMA (all at once).

We NEVER tested:
- SmearGate+CaseOps+AsymLogit WITHOUT EMA
- EMA ALONE (without SmearGate/CaseOps/AsymLogit)
- SmearGate alone + quantization
- CaseOps alone + quantization
- AsymLogit alone + quantization

**Hypotheses (untested):**
1. EMA creates weight distributions hostile to per-tensor int8 quantization
2. SmearGate/CaseOps/AsymLogit each create quantization-sensitive weights
3. The combination has a multiplicative interaction with quantization
4. The quantization scheme itself is broken for any non-trivial model

**Impact:** This is the single most important unresolved question. If EMA is actually fine and the issue is SmearGate, we have a different problem entirely.

**Required experiment:** Run ALL 16 combinations of {SmearGate, CaseOps, AsymLogit, EMA} on/off, measure post-quantization BPB degradation for each. This is a 2^4 factorial design. At 300s per run = ~80 minutes total. CRITICAL.

### 2.2 Partial RoPE: Inconclusive at 0.5 Fraction ★★★

**The finding:** partial-rope-results.md reports "no meaningful improvement" at fraction=0.5.

**The problem:** The competitive intel (architecture-tricks.md) says the standard is rope_dims=16 out of 64 head dimensions (fraction=0.25, not 0.5). We tested the WRONG fraction. The leaderboard standard is 16/64 = 0.25, not 0.5.

Also, Partial RoPE is always paired with LN Scale (per-layer learned scaling) in top submissions. We tested Partial RoPE alone without LN Scale.

**Conclusion:** The "no improvement" finding is unreliable. We tested a non-standard configuration without its natural partner technique.

**Required experiment:** Test rope_dims=16 (fraction=0.25) WITH LN Scale.

### 2.3 MLP 3x: Negative Result, But Confounded ★★★

**The finding:** experiment-results-may17.md: "MLP 3x is slower with no loss benefit" (13% slower, loss comparable or worse at each step).

**The problem:** This was tested on 12 layers, not 9. 12 layers is ALREADY known to be worse than 9 within the 300s budget (24% fewer steps). Adding MLP 3x on top of an already-penalized configuration compounds the step-time penalty.

**Required experiment:** Test MLP 3x on 9 layers (not 12) to isolate its effect.

### 2.4 12-Layer Results: Fair Comparison? ★★★

**The finding:** 12L gets 24% fewer steps than 9L (219 vs 288) and worse loss at every checkpoint within 300s.

**The concern:** This comparison uses 12L without BigramHash default ON. If BigramHash gives 7-8% improvement, 12L+BigramHash at the same step count might beat 9L+BigramHash. The experiment DID test 12L+BigramHash but compared at different step counts.

At step 75: 9L+BH = 4.28, 12L+BH = 4.34. But 12L took longer to reach step 75. If we compare at the SAME WALLCLOCK TIME (step ~75 for 12L vs step ~100 for 9L), the comparison is fairer but still favors 9L.

**Conclusion:** The 9L > 12L finding within 300s is likely correct but needs a longer run (600s+) to confirm. At H100 speeds, 12L may win because step-time penalty is smaller.

### 2.5 Batch Size Regression — PyTorch 2.12 ★★★★

**The problem:** 65536 tokens per batch OOMs on PyTorch 2.12.0+cu130 but worked on earlier versions. This forced all overnight experiments to use 16384 batch (4x smaller).

**Impact:** All loss numbers from the overnight session are at 1/4 the batch size of earlier experiments. The relative improvements (BigramHash -7%, etc.) should still hold, but absolute loss values are NOT comparable to the baseline results in model-scaling-report.md (which used 65536 batch).

**The gap:** Nobody investigated WHY 65536 OOMs. Is it:
- A PyTorch 2.12 memory allocation change?
- A CUDA 13.0 regression?
- Fragmentation from multiple runs?
- A code change that increased memory usage?

**Required action:** Run `nvidia-smi` + `torch.cuda.memory_summary()` with 65536 batch to identify the OOM source. If it is fragmentation, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` might fix it.

### 2.6 BPB Estimates from Loss — Unreliable Proxy ★★

**The problem:** model-scaling-report.md uses "loss/ln2/bytes_per_tok" to estimate BPB from training loss. This is a rough approximation. The actual BPB depends on validation data distribution, quantization effects, eval strategy, and tokenizer.

**Impact:** Low — the estimates are clearly labeled as approximate. But don't use them to compare against leaderboard BPB numbers.

### 2.7 H100 Extrapolation: 85x Ratio May Be Wrong ★★★

**The claim:** step-budget-analysis.md uses an 85x speed ratio (2070 step time / 85 = H100).

**The concern:** This ratio was derived from comparing our 2070 baseline (3.7s/step) to the naive H100 baseline (43.5ms/step). But the H100 uses FlashAttention + torch.compile + 524K tokens/step. The ratio changes with model complexity (top submissions: 121ms/step, ratio = 30x).

**Impact:** Medium — affects H100 step-count estimates. Don't trust exact step counts.

---

## PART 3: THE UGLY — Critical Gaps and Missed Opportunities

### 3.1 XSA (Cross-Sparse Attention) — NOT IMPLEMENTED, NOT TESTED ★★★★★

**The gap:** XSA appears in 392 submissions and DOMINATES the top 6 leaderboard slots:
- #1: XSA-all (1.1147 BPB)
- #5: XSA4 (1.1271 BPB)
- #6: Partial XSA (1.1307 BPB)

We have zero implementation, zero experiments, and only research notes.

**Why it matters:** XSA saves ~7.3M parameters by skipping attention on early layers. These saved params go to wider MLPs.

**Complexity:** HARD — requires restructuring which layers get attention.

**Priority:** HIGHEST. This is the single technique most correlated with top scores.

### 3.2 Depth Recurrence — NOT IMPLEMENTED, NOT TESTED ★★★★★

**The gap:** 116 submissions, including PR #1903 at 0.9418 BPB. Zero implementation.

**Why it matters:** 5 unique layers x 2 = 10 effective layers with only 5 layers of parameters.

**Known risk:** Quantization amplification (errors compound through repeated layers). Requires QAT.

**Priority:** HIGH. Must be paired with QAT.

### 3.3 SP8192 Tokenizer — NOT IMPLEMENTED ★★★★★

**The gap:** ALL competitive submissions use SP8192. We use SP1024. The gap is estimated at 0.02-0.05 BPB — potentially the single largest improvement available.

**Priority:** HIGH but blocks everything else (all hyperparams need retuning after).

### 3.4 SwiGLU Activation — NOT IMPLEMENTED ★★★★

**The gap:** Competitive agents and multiple submissions use SwiGLU. We still use relu-squared. architecture-tricks.md doesn't even mention SwiGLU.

**Complexity:** LOW — replace relu^2 with swish(x) * linear(x) in MLP.

**Priority:** MEDIUM-HIGH. Quick win.

### 3.5 Score-First TTT — NOT IMPLEMENTED ★★★★★

**The gap:** TTT is the key eval-time innovation. Top 2 pending claims use TTT. We have extensive research but zero implementation.

**Why it matters:** 0.005-0.02 BPB improvement, zero training cost.

**Priority:** HIGH for final submission. Can be done last since it is eval-time only.

### 3.6 QK-Gain Tuning — NOT TESTED ★★★

**The gap:** QK-Gain is implemented (init=1.5) but never tuned. Top submissions use 5.25.

**Complexity:** TRIVIAL — change one env var.

**Priority:** HIGH (cheap experiment, potentially meaningful impact).

### 3.7 No Individual Technique Isolation ★★★★★

**The gap:** We have NEVER tested techniques individually with quantization. The only quantization test was SmearGate+CaseOps+AsymLogit+EMA all at once.

**What we need:**
- Each technique ON/OFF x int8 quantization = 8 runs minimum
- Measure: raw BPB, post-quant BPB, degradation %
- Identify which techniques are quantization-safe

**Cost:** ~40 minutes (8 runs x 300s each)

### 3.8 No Full Training Run with Current Stack ★★★★

**The gap:** All experiments are 300s smoke tests. We have NEVER run the full stack for 600s+ with final validation.

**Required:** One full 600s run with final validation enabled to get a real BPB number.

### 3.9 BigramHash + Full Stack: Untested at Scale ★★★

**The gap:** BigramHash was tested with SmearGate+CaseOps+AsymLogit baseline. Never tested with all techniques combined, or at larger hash sizes (8192, 16384), or with TrigramHash.

### 3.10 Pending SOTA Claims: Unverified ★★

**The gap:** Pending claims of 0.8265 BPB (ndokutovich) are UNVERIFIED. We haven't analyzed their code. If valid, the SOTA gap is 0.29 BPB (not 0.11 BPB).

---

## PART 4: Experiment Queue — Prioritized

### Tier 1: MUST DO (highest impact, lowest uncertainty)

| # | Experiment | Question | Runtime | Impact | Deps |
|---|-----------|----------|---------|--------|------|
| 1 | **EMA quantization factorial** | Which techniques break int8? | 80 min | CRITICAL | None |
| 2 | **QK-Gain sweep** (1.5, 2.5, 3.5, 5.25) | Optimal attention scaling? | 20 min | HIGH | None |
| 3 | **Full stack 600s + validation** | Actual BPB of current stack? | 600s | HIGH | None |
| 4 | **Partial RoPE 0.25 + LN Scale** | Does the standard combo work? | 300s | MEDIUM | None |
| 5 | **MLP 3x on 9L** | Does MLP 3x help on 9L? | 300s | MEDIUM | None |

### Tier 2: SHOULD DO (high impact, medium effort)

| # | Experiment | Question | Runtime | Impact | Deps |
|---|-----------|----------|---------|--------|------|
| 6 | **SwiGLU activation** | Better than relu^2? | 300s | MEDIUM | None |
| 7 | **BigramHash size sweep** (4096, 5120, 8192, 16384) | Optimal hash size? | 40 min | MEDIUM | None |
| 8 | **Batch size recovery** (investigate 65536 OOM) | Can we get 4x batch back? | 30 min | HIGH | None |
| 9 | **Stride-64 eval by default** | Confirm BPB improvement | 600s | HIGH | Tier 1 #3 |

### Tier 3: IMPLEMENT NEW TECHNIQUES

| # | Experiment | Question | Runtime | Impact | Deps |
|---|-----------|----------|---------|--------|------|
| 10 | **XSA-4 implementation** | Does sparse attention help? | 2-3 hr | HIGH | None |
| 11 | **Depth recurrence (5x2)** | More depth for free? | 2-3 hr | HIGH | QAT |
| 12 | **SP8192 tokenizer** | Larger vocab = better BPB? | 4-6 hr | HIGHEST | Data pipeline |
| 13 | **Score-first TTT** | Free eval-time BPB? | 2-3 hr | HIGH | LoRA impl |
| 14 | **TrigramHash** | Stacks with BigramHash? | 1 hr | LOW | BigramHash |

### Tier 4: NICE TO HAVE

| # | Experiment | Question | Runtime | Impact | Deps |
|---|-----------|----------|---------|--------|------|
| 15 | **zstd-22 compression** | Implement for submission | 30 min | LOW | None |
| 16 | **Verify pending SOTA claims** | Is 0.8265 real? | 2 hr | LOW | None |
| 17 | **GPTQ self-gen calibration** | Better quantization? | 3-4 hr | MEDIUM | GPTQ impl |
| 18 | **N-gram mixing at eval** | Free eval boost? | 1 hr | LOW | None |

---

## PART 5: Key Assumptions That Could Be Wrong

### 5.1 "9 layers is optimal for 2070"
**Why it might be wrong:** 12L was only tested at 300s. At 600s+ or on H100, 12L may win. With depth recurrence (5 unique x 2 = 10 effective), we get 10L with 5L params.

### 5.2 "BigramHash with large vocab is redundant"
**Why it might be wrong:** This refers to TOKENIZER vocab (SP8192), not hash table size. With SP1024, BigramHash is still valuable. We haven't tested BigramHash + SP8192.

### 5.3 "EMA is bad and should be disabled"
**Why it might be wrong:** The quantization disaster may not be EMA's fault. EMA is used by 353 submissions. Disabling it without proper isolation is premature.

### 5.4 "We can match leaderboard with our 17M model"
**Why it might be wrong:** step-budget-analysis.md notes: "With our current 17M model, even with all techniques, we can't get below ~1.15 BPB." To reach 1.10 BPB, we need ~35M params + SP8192 + TTT.

### 5.5 "RTX 2070 experiments transfer to H100"
**Why it might be wrong:** FlashAttention on H100 changes the attention memory pattern entirely. Techniques that help on memory-bound math SDP may not help on compute-bound FlashAttention.

---

## PART 6: Research Quality Assessment

### What Was Done Well
- Comprehensive competitive intel (competitive-intel.md is excellent, 1,500+ submissions analyzed)
- Good technique documentation (architecture-tricks.md, eval-and-compression.md)
- Systematic step-time measurement for all techniques
- Compression benchmark with multiple algorithms
- TST proposal analysis (correctly concluded "skip for now")

### What Was Done Poorly
- **No factorial/isolation experiments** — techniques always tested in bundles
- **No root-cause analysis** of EMA quantization disaster
- **Wrong Partial RoPE fraction** tested (0.5 instead of standard 0.25)
- **Confounded MLP 3x test** (tested on 12L, not 9L)
- **No actual BPB measurement** — all numbers are loss proxies
- **Batch size regression uninvestigated** — just accepted and worked around
- **No code review** of top leaderboard submissions

### Missing Documentation
- No consolidated "current best config" file
- No tracking of which env vars are ON/OFF in the default stack
- No list of what the rtx2070 branch changes vs upstream

---

## PART 7: Recommendations for Next Session

### Immediate (next 30 minutes)
1. Run EMA factorial experiment (8 runs, 300s each = 40 min)
2. Run QK-Gain sweep (4 values, 300s each = 20 min)

### Short-term (next 2 hours)
3. Run full stack 600s with validation to get real BPB
4. Test Partial RoPE 0.25 + LN Scale (the standard combo)
5. Implement SwiGLU activation (simple MLP swap)

### Medium-term (next 8 hours)
6. Implement XSA-4 (restructure attention layers)
7. Implement depth recurrence with QAT
8. Investigate batch size regression

### Long-term (requires data pipeline changes)
9. SP8192 tokenizer (retokenize all data)
10. Score-first TTT with LoRA
11. Cloud H100 run for actual competition BPB

---

## Appendix: File-by-File Summary

| File | Status | Key Finding |
|------|--------|-------------|
| architecture-tricks.md | SOLID | Comprehensive trick catalog, good code examples |
| competitive-intel.md | SOLID | Excellent leaderboard analysis, 1,500+ PRs |
| tst-proposal.md | SOLID | Correctly recommends skipping TST at 17M scale |
| gap-analysis.md | MOSTLY SOLID | Good prioritization, SmearGate priority was right |
| eval-and-compression.md | SOLID | Thorough eval technique catalog |
| model-scaling-report.md | MOSTLY SOLID | VRAM constraints well-documented, BPB estimates rough |
| optimization-report.md | SOLID | 2070 is maxed out — confirmed |
| compression-comparison.md | SOLID | zstd-22 confirmed, L2 sorting = no benefit |
| experiment-results-may17.md | NEEDS WORK | Confounded comparisons, missing isolation |
| technique-comparison.md | CRITICAL FLAW | EMA quantization disaster not isolated |
| caseops-asymlogit-notes.md | SOLID | Implementation well-documented |
| implementation-log.md | SOLID | EMA implementation correct |
| reference-step-counts.md | SOLID | Good H100 cross-reference |
| overnight-session-2026-05-17.md | NEEDS WORK | Wrong batch size, missing analysis |
| session-summary.md | MOSTLY SOLID | Good summary, some outdated conclusions |
| step-budget-analysis.md | MOSTLY SOLID | Good technique verdicts, 85x ratio approximate |
| bigrampatch-results.md | SOLID | BigramHash confirmed winner |
| partial-rope-results.md | NEEDS WORK | Wrong fraction tested |
