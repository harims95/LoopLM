# LoopLM

A 135M parameter **dense looped transformer** trained from scratch on FineWeb, with honest documentation of an architectural exploration that didn't go where the paper promised.

🤗 **Model:** [harims95/LoopLM-135M-naive](https://huggingface.co/harims95/LoopLM-135M-naive)
📄 **Reference paper:** [Parcae (arXiv 2604.12946)](https://arxiv.org/abs/2604.12946)
🧬 **Forked training infra from:** [harishsg993010/HobbyLM](https://github.com/harishsg993010/HobbyLM)

## TL;DR

| What | Result |
|---|---|
| Architecture | 135M dense looped transformer (prelude → loop ×T → coda) |
| Training data | FineWeb (raw), 4.6B tokens |
| Final val loss | **3.95** at step 17,500 |
| Hardware | 2× H100 on Modal, ~3 hours wall clock |
| Total cost | ~$22 |
| Headline finding | Parcae's stability mechanisms didn't help at this scale |

## Quick Start

Load the model from HuggingFace in 3 lines (no clone needed):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("harims95/LoopLM-135M-naive")
model = AutoModelForCausalLM.from_pretrained(
    "harims95/LoopLM-135M-naive",
    trust_remote_code=True,
)

inputs = tok("Once upon a time", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
print(out.logits.shape)  # (1, seq_len, 50304)
```

For text generation with sampling, see [`scripts/generate.py`](scripts/generate.py):

```bash
python scripts/generate.py --prompt "The history of solar energy" --max_new_tokens 80
```

## Architecture

```
Input tokens
    ↓
[Embedding]
    ↓
[Prelude: 4 transformer blocks]
    ↓
e (input injection)
    ↓
[Loop block × T loops]  ← T ~ Poisson(μ=6) sampled per-sequence
    ↓                      h_{t+1} = block(h + e)
h_final
    ↓
[Coda: 2 transformer blocks]
    ↓
[Tied lm_head] → logits
```

**Details:**
- `d_model = 1024`, GQA with 16 query heads / 8 KV heads, head dim 64
- RoPE (θ=10000), QK-norm, RMSNorm pre-norm
- SwiGLU FFN, dim 2816
- GPT-2 BPE tokenizer (vocab 50304), tied embeddings
- Total: **135M params** unique trainable

**Training:**
- Optimizer: Muon (matrices) + AdamW (norms, biases, embeddings)
- Per-sequence Poisson(μ=6) loop depth sampling
- Truncated BPTT: `μ_bwd = ceil(μ_rec/2) = 3` (gradients only through last 3 loops)
- bf16 mixed precision
- 60% constant LR, 40% cosine decay to 0.1× peak

## The Parcae Investigation (Honest Findings)

This project started as an attempt to reproduce [Parcae](https://arxiv.org/abs/2604.12946)'s stability mechanisms for looped LMs. Across **5 ablations**, none of the Parcae variants beat the naive baseline at this scale:

| Ablation | Description | Final Val |
|---|---|---|
| 1. Naive looped | `h_{t+1} = block(h + e)` | **3.84** (FineWeb-Edu) |
| 2. Parcae basic | + A matrix constraint | 3.84 (tied) |
| 3. Parcae full v1 | + input norm (broken arch) | diverged |
| 4. Parcae fixed v2 | + LTI step before block, B identity init | worse |
| 5. Parcae v3 | + B → AdamW, decay init = √(1/5) | dramatically worse |

Each "fix" — matching the [official Parcae implementation](https://github.com/sandyresearch/parcae) more closely — actually made performance worse. After careful debugging with multiple second opinions (Appendix Q of the paper, official repo code, multi-LLM consultations), my conclusion is:

> **Parcae's stability mechanisms appear to require larger scale (1B+ params, 100B+ tokens) to show benefit. At 135M params / 0.8B tokens, naive looped reuse is sufficient.**

The final shipped model uses the naive recipe.

## How It Compares

| Model | Architecture | Params | Tokens | Val Loss |
|---|---|---|---|---|
| This (LoopLM-135M-naive) | Dense looped | 135M | 4.6B | 3.95 |
| [HobbyLM-130M-MoE](https://github.com/harishsg993010/HobbyLM) (sibling) | MoE | 140M / 62M active | 10B | 3.30 |
| [HobbyLM-30M](https://huggingface.co/harims95/HobbyLM-30M) (prior) | Dense | 30M | 1B | 3.91 |

At this scale, sparse MoE wins on sample efficiency. Dense looped is competitive but doesn't surpass it.

## Reproducing

Built on top of brother's [HobbyLM](https://github.com/harishsg993010/HobbyLM) training infrastructure, deployed via Modal H100s.

**Setup:**
```bash
git clone https://github.com/harims95/LoopLM
cd LoopLM
python -m venv .venv
.venv/Scripts/activate  # or `source .venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
pip install modal && modal token new
```

**Download FineWeb shards (Modal volume):**
```bash
python -m modal run --detach training/modal_train.py --action download --dataset fineweb --shards 50
```

**Train (replicates the published run):**
```bash
python -m modal run --detach training/modal_train.py \
    --action train \
    --preset 135M \
    --steps 20000 \
    --run-name looplm_naive_fineweb \
    --gpus 2 \
    --micro 32 \
    --seq-len 1024 \
    --batch-tokens 262144 \
    --dataset fineweb \
    --overrides "use_a_matrix=false,use_input_norm=false"
```

**Try Parcae variants (won't help, but you can verify):**
```bash
# A matrix only
--overrides "use_a_matrix=true,use_input_norm=false"

# Full Parcae
--overrides "use_a_matrix=true,use_input_norm=true"
```

## Repository Structure

```
looplm/                Core model code
  ├── config.py        ModelConfig + TrainConfig dataclasses
  ├── model.py         RMSNorm, Attention (GQA+QK-norm+RoPE), DenseFFN (SwiGLU)
  ├── parcae.py        ParcaeTransformer, ParcaeLoopBlock
  ├── optim.py         Muon + AdamW with parameter group routing
  └── data.py          FineWeb / FineWeb-Edu data loader

training/
  ├── train.py         Main training loop, DDP setup, schedule
  └── modal_train.py   Modal harness (download, smoke, train)

scripts/
  ├── generate.py             Inference / sampling
  ├── prepare_hf_release.py   Convert checkpoint to HF release package
  └── fix_safetensors_prefix.py
```

## Cost & Compute Notes

- **Recommended:** 2× H100 for 135M scale
- **Avoid:** 4× H100 or more — diminishing returns from communication overhead
- **Throughput:** ~530 ms/step at micro=32, seq=1024, batch=262K on 2× H100
- **5B-token training:** ~3-3.5 hours wall clock, ~$20-25 on Modal

## Acknowledgments

- [@harishsg993010](https://github.com/harishsg993010) — training infrastructure (Muon, data loader, Modal harness, optimizer setup); also a brother who answers questions at all hours
- [Sandy Research](https://github.com/sandyresearch/parcae) — official Parcae implementation
- The [Parcae authors](https://arxiv.org/abs/2604.12946) — for the architecture and the honest scaling analysis that explains why our small-scale ablation didn't beat naive
- [Karpathy](https://github.com/karpathy/llm.c) and [kjj0](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2) — FineWeb / FineWeb-Edu shards
- [Modal Labs](https://modal.com) — making H100 training accessible for hobby budgets

## License

Apache 2.0. Use it, study it, build on it. Credit appreciated, not required.

## Author

Hari ([@harims95](https://github.com/harims95) on GitHub, [@harims95](https://huggingface.co/harims95) on HuggingFace)

Built as a hobby research project, late June 2026.
