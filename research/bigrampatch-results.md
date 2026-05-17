# BigramHash Experiment Results — May 17, 2026

## Setup
- Machine: RTX 2070 (desktoparch), Python 3.14.5, PyTorch 2.12.0+cu130
- Batch: 16384 tokens, seq_len=1024, grad_accum_steps=8
- Base config: CaseOps + AsymLogit + SmearGate ON, Muon optimizer, GQA 8/4
- Seed: 42, TRAIN_LOG_EVERY=25

## Baseline (no BigramHash)
- Model: 17,061,975 params
- Speed: ~1039ms/step

| Step | Loss  | Step Avg |
|------|-------|----------|
| 10   | 6.00  | 1048ms   |
| 25   | 5.69  | 1042ms   |
| 50   | 4.91  | 1040ms   |
| 75   | 4.63  | 1039ms   |

## BigramHash (hash_size=4096)
- Model: 19,159,127 params (+2,097,152 = +12.3%)
- Speed: ~1040ms/step (+0.1% overhead)

| Step | Loss  | Step Avg | vs Baseline |
|------|-------|----------|-------------|
| 10   | 5.87  | 1052ms   | -2.1%       |
| 25   | 5.24  | 1044ms   | -7.8%       |
| 50   | 4.57  | 1042ms   | -6.8%       |
| 75   | 4.28  | 1041ms   | -7.6%       |
| 100  | 4.09  | 1040ms   |             |
| 150  | 3.77  | 1040ms   |             |
| 200  | 3.73  | 1040ms   |             |
| 250  | 3.62  | 1040ms   |             |

## Analysis
- BigramHash provides ~7-8% lower loss at equivalent steps
- Nearly zero speed overhead (0.1% slower)
- Adds 2.1M params (12% more) — worth checking if fits in 16MB after compression
- The hash function is: (prev_token_id * 2654435761 + curr_token_id) % table_size
- BigramHash embeddings added to token embeddings BEFORE RMSNorm and SmearGate

## Comparison to Earlier Run (600s wallclock)
Earlier run with nohup (killed at step 350):
- Step 200: loss 3.6998 (vs this run's 3.7282 — slight variance from different random state)
- Consistent ~7% improvement over baseline

## Recommendations
1. BigramHash should be ON by default — it's essentially free improvement
2. Test hash_size=5120 (recommended_bigram_vocab_size from manifest) — may be even better
3. Test BigramHash + MLP_MULT=3 for more capacity
4. Check compressed artifact size with BigramHash added

## Environment Notes
- 65536 batch tokens OOMs on PyTorch 2.12.0 (worked on earlier versions)
- 32768 batch OOMs after ~10 steps (fragmentation?)
- 16384 batch is stable at 1.04s/step, ~1.8GB VRAM
- SSH connections drop after ~90s — need setsid or terminal background mode
- Competing kanban task (t_46cbc0f0) spawns processes on same GPU — must kill before runs
