"""Generate text with LoopLM-135M-naive."""
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def sample(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        # Truncate context if too long (model trained on 1024 seq_len)
        ctx = ids[:, -1024:] if ids.size(1) > 1024 else ids

        with torch.no_grad():
            out = model(ctx)
        logits = out.logits[0, -1] / max(temperature, 1e-6)

        # Top-k filtering
        if top_k and top_k > 0:
            vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits = torch.where(
                logits < vals[-1],
                torch.full_like(logits, float("-inf")),
                logits,
            )

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        ids = torch.cat([ids, next_id], dim=1)

        # Stop on EOS if any
        if next_id.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="harims95/LoopLM-135M-naive")
    ap.add_argument("--prompt", default="The history of the Internet began")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=3)
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Loaded on {device}. Generating {args.n_samples} samples...\n")

    for i in range(args.n_samples):
        print(f"=== Sample {i + 1} (seed={args.seed + i}) ===")
        text = sample(
            model,
            tok,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            seed=args.seed + i,
        )
        print(text)
        print()


if __name__ == "__main__":
    main()
