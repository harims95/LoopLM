"""Modal harness: download FineWeb-Edu, smoke-test, and train LoopLM on H100s.

  python -m modal run training/modal_train.py --action download --chunks 30
  python -m modal run training/modal_train.py --action smoke
  python -m modal run --detach training/modal_train.py --action train \
      --preset 135M --steps 4000 --run-name parcae_baseline --gpus 2
"""
import subprocess
import modal

HF_REPO = "karpathy/fineweb-edu-100B-gpt2-token-shards"
DATA_DIR = "/data/fineweb_edu"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "numpy", "huggingface-hub", "tqdm", "tiktoken", "safetensors")
    .add_local_dir(".", "/root/looplm")
)

app = modal.App("looplm", image=image)
vol = modal.Volume.from_name("fineweb_edu", create_if_missing=True)


@app.function(volumes={"/data": vol}, timeout=6 * 60 * 60,
              secrets=[modal.Secret.from_name("huggingface")])
def download(chunks: int = 30):
    """Download N FineWeb-Edu shards (~200M tokens each) into the Modal volume."""
    import os
    from huggingface_hub import hf_hub_download
    os.makedirs(DATA_DIR, exist_ok=True)
    done = 0

    def get(fname):
        nonlocal done
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            hf_hub_download(repo_id=HF_REPO, filename=fname, repo_type="dataset", local_dir=DATA_DIR)
            done += 1
            if done % 10 == 0:
                print(f"  downloaded {done} shards...", flush=True)
                vol.commit()

    get("edu_fineweb_val_000000.bin")
    for i in range(1, chunks + 1):
        get(f"edu_fineweb_train_{i:06d}.bin")
    vol.commit()
    print(f"data ready: {chunks} train shards (~{chunks*200}M tokens) + val in {DATA_DIR}", flush=True)


@app.function(gpu="H100", timeout=20 * 60)
def smoke():
    """No-data GPU sanity check: model + optimizers + bf16 + compile, fwd/bwd/step."""
    import os, sys, torch
    os.chdir("/root/looplm")
    sys.path.insert(0, "/root/looplm")
    from looplm.config import get_config, TrainConfig
    from looplm.parcae import ParcaeTransformer, count_params
    from looplm.optim import build_optimizers
    dev = torch.device("cuda")
    cfg = get_config("135M")
    cfg.n_prelude = 2
    cfg.n_coda = 1
    cfg.mu_rec = 3
    model = ParcaeTransformer(cfg).to(dev)
    pc = count_params(model)
    print(f"135M (shrunken probe): {pc['total']/1e6:.1f}M params", flush=True)
    muon, adamw, _ = build_optimizers(model, TrainConfig())
    cmodel = model
    x = torch.randint(0, cfg.vocab_size, (4, 256), device=dev)
    y = torch.randint(0, cfg.vocab_size, (4, 256), device=dev)
    T_per_seq = torch.full((4,), cfg.mu_rec, device=dev, dtype=torch.long)
    for step in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = cmodel(x, y, T_per_seq=T_per_seq, n_no_grad=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step(); adamw.step()
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        print(f"step {step}: loss={loss.item():.4f} ce={parts['ce'].item():.4f} "
              f"z={parts['z'].item():.2f} finite={torch.isfinite(loss).item()}", flush=True)
    print("GPU SMOKE OK", flush=True)


def _train_body(preset, steps, run_name, overrides, gpus, micro, seq_len, batch_tokens, save_every=0):
    import os
    os.chdir("/root/looplm")
    over = ["--set", *overrides.split(",")] if overrides else []
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}", "training/train.py",
        "--preset", preset, "--run_name", run_name, "--data_dir", DATA_DIR,
        "--out_dir", "/data/runs", "--max_steps", str(steps), "--micro_batch_seqs", str(micro),
        "--seq_len", str(seq_len), "--batch_tokens", str(batch_tokens),
        "--save_every", str(save_every), *over,
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    import json
    rp = f"/data/runs/{run_name}/result.json"
    return json.load(open(rp)) if os.path.exists(rp) else {}


@app.function(gpu="H100", volumes={"/data": vol}, timeout=24 * 60 * 60)
def train_1(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=0):
    return _train_body(preset, steps, run_name, overrides, 1, micro, seq_len, batch_tokens, save_every)


@app.function(gpu="H100:2", volumes={"/data": vol}, timeout=24 * 60 * 60)
def train_2(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=0):
    return _train_body(preset, steps, run_name, overrides, 2, micro, seq_len, batch_tokens, save_every)


@app.local_entrypoint()
def main(action: str = "train", preset: str = "135M", steps: int = 4000,
         run_name: str = "baseline", overrides: str = "", gpus: int = 1,
         chunks: int = 30, micro: int = 16, seq_len: int = 1024,
         batch_tokens: int = 262144, save_every: int = 0):
    if action == "download":
        download.remote(chunks)
    elif action == "smoke":
        smoke.remote()
    elif action == "train":
        fn = train_2 if gpus == 2 else train_1
        print(f"training on {gpus}x H100", flush=True)
        fn.remote(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every)
    else:
        raise ValueError(f"unknown action: {action}")
