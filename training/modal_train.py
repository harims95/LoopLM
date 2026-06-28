"""Modal harness: download dataset shards, smoke-test, and train LoopLM on H100s.

  python -m modal run training/modal_train.py --action download --dataset fineweb_edu --shards 30
  python -m modal run training/modal_train.py --action download --dataset fineweb --shards 4
  python -m modal run training/modal_train.py --action smoke
  python -m modal run --detach training/modal_train.py --action train \
      --dataset fineweb_edu --preset 135M --steps 4000 --run-name parcae_baseline --gpus 2
"""
import subprocess
import modal

DATASETS = {
    "fineweb_edu": {
        "hf_repo": "karpathy/fineweb-edu-100B-gpt2-token-shards",
        "data_dir": "/data/fineweb_edu",
        "train_pattern": "edu_fineweb_train_*.bin",
        "val_pattern": "edu_fineweb_val_*.bin",
        "train_prefix": "edu_fineweb_train_",
        "train_tokens_millions": 200,
        "download_val": True,
        "volume_name": "fineweb_edu",
    },
    "fineweb": {
        "hf_repo": "kjj0/fineweb10B-gpt2",
        "data_dir": "/data/fineweb",
        "train_pattern": "fineweb_train_*.bin",
        "val_pattern": "fineweb_train_*.bin",
        "train_prefix": "fineweb_train_",
        "train_tokens_millions": 100,
        "download_val": False,
        "holdout_last_for_val": True,
        "volume_name": "fineweb",
    },
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "numpy", "huggingface-hub", "tqdm", "tiktoken", "safetensors")
    .add_local_dir(".", "/root/looplm")
)

app = modal.App("looplm", image=image)
fineweb_edu_vol = modal.Volume.from_name(
    DATASETS["fineweb_edu"]["volume_name"], create_if_missing=True
)
fineweb_vol = modal.Volume.from_name(
    DATASETS["fineweb"]["volume_name"], create_if_missing=True
)
DATASET_VOLUMES = {
    "fineweb_edu": fineweb_edu_vol,
    "fineweb": fineweb_vol,
}


def _download_dataset(dataset: str, shards: int):
    import os
    from huggingface_hub import hf_hub_download

    spec = DATASETS[dataset]
    data_dir = spec["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    done = 0

    def get(fname):
        nonlocal done
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            hf_hub_download(
                repo_id=spec["hf_repo"],
                filename=fname,
                repo_type="dataset",
                local_dir=data_dir,
            )
            done += 1
            if done % 10 == 0:
                print(f"  downloaded {done} shards...", flush=True)
                DATASET_VOLUMES[dataset].commit()

    if spec["download_val"]:
        get("edu_fineweb_val_000000.bin")
    for i in range(1, shards + 1):
        get(f"{spec['train_prefix']}{i:06d}.bin")

    DATASET_VOLUMES[dataset].commit()
    extra = " + val" if spec["download_val"] else ""
    print(
        f"data ready: {shards} train shards (~{shards * spec['train_tokens_millions']}M tokens)"
        f"{extra} in {data_dir}",
        flush=True,
    )


@app.function(volumes={"/data": fineweb_edu_vol}, timeout=6 * 60 * 60,
              secrets=[modal.Secret.from_name("huggingface")])
def download_fineweb_edu(shards: int = 30):
    """Download N FineWeb-Edu shards (~200M tokens each) into the Modal volume."""
    _download_dataset("fineweb_edu", shards)


@app.function(volumes={"/data": fineweb_vol}, timeout=6 * 60 * 60,
              secrets=[modal.Secret.from_name("huggingface")])
def download_fineweb(shards: int = 30):
    """Download N raw FineWeb shards (~100M tokens each) into the Modal volume."""
    _download_dataset("fineweb", shards)


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


def _train_body(dataset, preset, steps, run_name, overrides, gpus, micro, seq_len, batch_tokens, save_every=2500):
    import os

    spec = DATASETS[dataset]
    os.chdir("/root/looplm")
    over = ["--set", *overrides.split(",")] if overrides else []
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}", "training/train.py",
        "--preset", preset, "--run_name", run_name, "--data_dir", spec["data_dir"],
        "--train_pattern", spec["train_pattern"], "--val_pattern", spec["val_pattern"],
        "--out_dir", "/data/runs", "--max_steps", str(steps), "--micro_batch_seqs", str(micro),
        "--seq_len", str(seq_len), "--batch_tokens", str(batch_tokens),
        "--save_every", str(save_every), *over,
    ]
    if spec.get("holdout_last_for_val"):
        cmd.append("--holdout_last_for_val")
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    DATASET_VOLUMES[dataset].commit()
    import json
    rp = f"/data/runs/{run_name}/result.json"
    return json.load(open(rp)) if os.path.exists(rp) else {}


@app.function(gpu="H100", volumes={"/data": fineweb_edu_vol}, timeout=24 * 60 * 60)
def train_1_fineweb_edu(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=2500):
    return _train_body("fineweb_edu", preset, steps, run_name, overrides, 1, micro, seq_len, batch_tokens, save_every)


@app.function(gpu="H100:2", volumes={"/data": fineweb_edu_vol}, timeout=24 * 60 * 60)
def train_2_fineweb_edu(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=2500):
    return _train_body("fineweb_edu", preset, steps, run_name, overrides, 2, micro, seq_len, batch_tokens, save_every)


@app.function(gpu="H100", volumes={"/data": fineweb_vol}, timeout=24 * 60 * 60)
def train_1_fineweb(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=2500):
    return _train_body("fineweb", preset, steps, run_name, overrides, 1, micro, seq_len, batch_tokens, save_every)


@app.function(gpu="H100:2", volumes={"/data": fineweb_vol}, timeout=24 * 60 * 60)
def train_2_fineweb(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=2500):
    return _train_body("fineweb", preset, steps, run_name, overrides, 2, micro, seq_len, batch_tokens, save_every)


@app.local_entrypoint()
def main(action: str = "train", preset: str = "135M", steps: int = 4000,
         run_name: str = "baseline", overrides: str = "", gpus: int = 1,
         dataset: str = "fineweb_edu", shards: int = 30, micro: int = 16, seq_len: int = 1024,
         batch_tokens: int = 262144, save_every: int = 2500):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")

    if action == "download":
        if dataset == "fineweb":
            download_fineweb.remote(shards)
        else:
            download_fineweb_edu.remote(shards)
    elif action == "smoke":
        smoke.remote()
    elif action == "train":
        if dataset == "fineweb":
            fn = train_2_fineweb if gpus == 2 else train_1_fineweb
        else:
            fn = train_2_fineweb_edu if gpus == 2 else train_1_fineweb_edu
        print(f"training on {gpus}x H100 with {dataset}", flush=True)
        fn.remote(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every)
    else:
        raise ValueError(f"unknown action: {action}")
