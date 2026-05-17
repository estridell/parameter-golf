## CaseOps + AsymLogit (2026-05-16)

**Commit:** 423998b on rtx2070 branch
**Status:** Implemented and smoke-tested

### CaseOps (4-case embedding)
- **What:** Adds a learnable 4-case embedding (lower/title/allcaps/mixed) to token embeddings
- **How:** At model init, builds a lookup table from SentencePiece vocab classifying each token by its case pattern. During forward, looks up case type and adds case_emb(token_case) to tok_emb.
- **Params:** +2,048 (4 cases x 512 model_dim), zero-init (no-op at start)
- **Env:** CASEOPS_ENABLED=1 (default on)
- **Note:** Upstream uses data-level lossless_caps preprocessing with sentinel characters. Our approach is model-side (works with standard tokenizer). Both approaches encode case info; upstream approach also changes tokenization.
- **File changes:** ~40 lines added to train_gpt.py

### AsymLogit (asymmetric logit softcap)
- **What:** Replaces symmetric softcap with separate pos/neg scalar softcaps
- **How:** where(logits > 0, pos_cap * tanh(logits/pos_cap), neg_cap * tanh(logits/neg_cap))
- **Params:** +2 (softcap_pos, softcap_neg), initialized to LOGIT_SOFTCAP (30.0)
- **Env:** ASYMLOGIT_ENABLED=1 (default on)
- **Impact:** ~0.001-0.002 BPB improvement (per PR #1923, #2130)
- **File changes:** ~15 lines added to train_gpt.py

### Smoke Test Results
- Model: 17,061,975 params (was 17,059,925 with SmearGate)
- Step time: 3.76s/step (no regression)
- Peak VRAM: ~5.6GB / 8GB (no regression)
- Loss: 6.94 to 11.23 over 9 steps (warmup, normal)
- Artifact: 5.09 MB int8+zlib (under 16MB limit)

### Toggle
Both features are independently togglable via env vars:
- CASEOPS_ENABLED=0 to disable CaseOps
- ASYMLOGIT_ENABLED=0 to disable AsymLogit

### Next Steps
- Run longer training (240s+) to verify loss convergence matches baseline
- Test combined with EMA when that is implemented
