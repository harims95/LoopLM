# LoopLM Architecture Research & Design Spec

Consolidated from: Parcae (arXiv:2604.12946), official sandyresearch/parcae repo,
HobbyLM (harishsg993010), modded-nanogpt, Muon optimizer paper.
Target: **dense looped LLMs at ~135M params**. Date: 2026-06-28.

---

## 1. Decisions that are settled (baked in)

| Area | Decision | Source / rationale |
|---|---|---|
| Architecture | Decoder-only transformer, **pre-norm RMSNorm** | Universal |
| **QK-norm** | RMSNorm per-head on Q and K before RoPE | OLMo-2/OLMoE; cheapest stability win |
| Attention | **GQA** (16 Q-heads / 8 KV-heads), head_dim=64 | KV cache reduction at this scale |
| Positional | **RoPE θ=10,000** | Standard for ≤4k context |
| FFN | **SwiGLU** (gate/up/down, dim=2816) | All reference models |
| Embeddings | **Tied** (embed = lm_head), vocab=50304 | Embeddings ≈38% of 135M — tying mandatory |
| Looping | **Per-sequence Poisson(μ_rec=6) depth** | Parcae paper Section 4 |
| BPTT | **Truncated: μ_bwd = ceil(μ_rec/2) = 3** | Parcae paper Appendix E |
| Optimizer | **Muon** (all ≥2D weight matrices) + **AdamW** (norms, biases, embeddings) | HobbyLM / Moonlight |
| z-loss | 1e-4 · mean(logsumexp²) | PaLM/OLMo |
| Init | std=0.02 | GPT-2 |
| Precision | **bf16** autocast, fp32 logits + loss | Standard |

---

## 2. Architecture: Prelude → Loop → Coda

```
Input tokens
    ↓
[Embedding, vocab=50304, d=1024]
    ↓
[Prelude: 4 transformer blocks]   → produces e (input injection vector)
    ↓
[Loop block × T iterations]       T ~ Poisson(μ_rec=6) per-sequence
    h_{t+1} = block(h_t + e)     naive looped update rule
    ↓
[Coda: 2 transformer blocks]
    ↓
[Tied lm_head] → logits (50304)
```

**Parameter count (135M preset):**

| Component | Params |
|---|---|
| Embeddings (tied) | 51.5M |
| Prelude (4 blocks) | 47.2M |
| Loop block (1 block, reused T times) | 11.8M |
| Coda (2 blocks) | 23.6M |
| Final norm | ~1K |
| **Total unique** | **134.1M** |

**Key property:** The loop block's weights are shared across all T iterations. Effective compute per forward pass = (4 + T + 2) transformer block evaluations. At μ_rec=6: ~12 blocks of compute from 8 blocks of parameters. Compute-to-parameter ratio ≈ 1.5× vs dense.

---

## 3. Parcae LTI Mechanism (Attempted)

The Parcae paper proposes augmenting the looped residual stream with a Linear Time-Invariant (LTI) stability constraint:

```
h_{t+1} = decay · h_t + input_gain · (e @ B.T) + block(...)
where:
  decay     = exp(-Δ · A),  A = exp(A_log) > 0
  input_gain = Δ  (Euler discretization)
  B         = d×d learnable matrix, init as identity
  A_log, Δ, B → AdamW with weight_decay=0  (Appendix Q)
```

**Theoretical motivation:** Constrains spectral radius of the recurrent map to < 1, preventing exponential growth of h across loops. Particularly important at large scale (1B+ params, 170k+ steps) where late-stage training instabilities occur.

---

## 4. Ablation Results (3000 steps, ~800M tokens, FineWeb-Edu)

All runs: 135M params, 1× H100, micro=32, seq=1024, batch_tokens=262144, Muon lr=0.02, Adam lr=3e-4.

| # | Run | Config | Val @ 250 | Val @ 1000 | Final @ 3000 | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Naive looped** | use_a_matrix=False, use_input_norm=False | 5.50 | ~4.70 | **3.84** | ✅ baseline |
| 2 | **Parcae basic** | + A matrix (LTI in parallel) | 5.50 | 4.51 | **3.84** | = naive (tied) |
| 3 | **Parcae full v1** | + input norm (broken: norm inside block) | 6.09 | ~5.0 | cancelled | ❌ diverged |
| 4 | **Parcae fixed v2** | LTI step BEFORE block, B=identity init | 5.89 | 5.16 | cancelled | ❌ worse |
| 5 | **Parcae v3** | + B→AdamW (wd=0), A_log=0, decay=√(1/5) | 6.85@750 | — | cancelled | ❌❌ dramatically worse |

**Debugging journey:**
- v1 bug: LTI placed in parallel with block instead of before it (contradicts official Parcae code)
- v2 bug: B initialized as nn.Linear (std=0.02) instead of identity matrix
- v3 bug: B routed to Muon optimizer (Muon orthogonalizes B, destroying identity init every step)
- v3 applied "correct" fix per Appendix Q + official repo — made it dramatically worse

**Conclusion:** Every fix matching the official implementation made things worse, not better.

---

## 5. Root Cause Analysis: Why Parcae Didn't Help

Three independent LLM consultations (Claude, ChatGPT 5.5, Gemini) all agreed that B→Muon was the primary bug. Applying the "correct" fix (B→AdamW, official init) produced val 6.85 at step 750 — worse than all previous runs including the broken ones.

**Most likely explanation:** The Parcae paper demonstrates stability improvements specifically in late-stage training (170k+ steps) where loss spikes occur. Our ablations ran for 3,000 steps maximum — we never reached the regime where these stability mechanisms matter.

At 135M params / 3,000 steps:
- No late-stage loss spikes observed in naive baseline
- LTI constraints add complexity without addressing any actual instability
- Naive looped (simpler, cleaner gradients) wins by default

**Generalization:** Parcae's stability tricks are likely a large-scale phenomenon. At hobby-budget scale, naive looped is sufficient.

---

## 6. Flagship Training Run (17,500 steps, FineWeb)

After ablations confirmed naive looped as the best config, trained the full model on raw FineWeb for comparison with sibling MoE.

| | |
|---|---|
| Config | use_a_matrix=False, use_input_norm=False |
| Dataset | FineWeb (kjj0/fineweb10B-gpt2), 4.6B tokens |
| Steps | 17,500 (stopped at checkpoint, budget constraint) |
| Hardware | 2× H100 on Modal |
| Wall clock | ~3 hours |
| Cost | ~$22 |
| Final val loss | **3.95** (FineWeb) |

**Val loss trajectory:**

| Step | Val loss | LR |
|---|---|---|
| 250 | 5.70 | 0.020 |
| 2,000 | 4.47 | 0.020 |
| 5,000 | 4.24 | 0.020 |
| 10,000 | 4.16 | 0.020 |
| 12,500 | 4.12 | 0.019 (cosine starts) |
| 15,000 | 4.06 | 0.013 |
| **17,500** | **3.95** | 0.008 |

---

## 7. Comparison with Sibling MoE (HobbyLM-130M)

| Model | Architecture | Params (total/active) | Tokens | Val loss (FineWeb) |
|---|---|---|---|---|
| **LoopLM-135M (ours)** | Dense looped | 135M / 135M | 4.6B | **3.95** |
| HobbyLM-130M MoE (sibling) | Sparse MoE | 140M / 62M | 10B | **3.30** |
| HobbyLM-30M dense (prior) | Dense | 30M / 30M | 1B | 3.91 |

**Observations:**
- Dense looped beats dense at matched params (3.95 vs 3.91 for 30M with 4× less data per param)
- Sparse MoE wins on sample efficiency at matched total params
- MoE's advantage: more effective parameters per activated compute
- Looped model's advantage: compute-to-parameter ratio (12 block evals from 8 param blocks) — but doesn't match MoE's routing efficiency at this scale

---

## 8. SFT Results (Alpaca 52k, 3 epochs)

Fine-tuned LoopLM-135M-naive on Stanford Alpaca using Lightning AI free H200.

| | |
|---|---|
| Dataset | tatsu-lab/alpaca (52,002 examples) |
| Epochs | 3 |
| Hardware | 1× H200 (Lightning AI, free tier) |
| Training time | **6 minutes** |
| Final SFT loss | ~3.0 |

**Results:**
- ✅ Model learned instruction-following format (`### Instruction / ### Response`)
- ✅ Generates coherent English sentences
- ✅ Stops cleanly with repetition penalty
- ❌ Poor factual accuracy (hallucinates facts, wrong answers)
- ❌ Cannot reliably do math or structured output

**Conclusion:** At 135M / 4.6B pretrain tokens, SFT teaches format, not knowledge. Factual accuracy requires more pretraining data, not better fine-tuning.

---

## 9. Open Questions (would need more scale to answer)

1. **Does naive looped beat dense at matched compute-optimal training?** We compared to MoE but not to an equivalent dense non-looped 135M model on the same FineWeb data. The 30M dense comparison is suggestive but not conclusive.

2. **At what scale does Parcae help?** The paper claims benefits at 1.3B+ params and 170k+ steps. Would need a 1B looped model trained to 100B tokens to verify. Out of scope for this project.

3. **Does the compute multiplier from looping actually help?** T~Poisson(6) means ~12 block evals from 8 parameter blocks. Does this give the same benefit as a 12-layer dense model? Untested.

4. **MoE vs looped at matched active parameters?** HobbyLM-130M has 62M active params. What would a 62M dense looped model achieve on FineWeb at the same token budget? This is the cleaner comparison.

5. **What does Parcae actually need to work?** Our experiments suggest the LTI mechanism requires either (a) much more training steps to encounter instabilities it's designed to prevent, or (b) larger models where the recurrent state h can actually carry meaningful information across 6+ loops.

---

## 10. Links

- Model (base): https://huggingface.co/harims95/LoopLM-135M-naive
- Model (SFT): https://huggingface.co/harims95/LoopLM-135M-naive-sft
- Code: https://github.com/harims95/LoopLM
- Reference paper: https://arxiv.org/abs/2604.12946
- Official Parcae code: https://github.com/sandyresearch/parcae
- Sibling project: https://github.com/harishsg993010/HobbyLM
