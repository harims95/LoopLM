"""Training loop for LoopLM (Parcae-style looped transformer).

  python training/train.py --preset 135M --max_steps 4000 --run_name baseline

Per-sequence depth sampling: T_i ~ Poisson(mu_rec) drawn fresh each micro-step.
Truncated BPTT: only the last mu_bwd = ceil(mu_rec/2) loops backprop.
Ablation switches: --set use_a_matrix=false use_input_norm=false mu_rec=4
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import socket
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from looplm.config import TrainConfig, get_config
from looplm.data import CUDAPrefetcher, data_generator, resolve_shards
from looplm.parcae import ParcaeTransformer, count_params
from looplm.optim import build_optimizers


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd="/root/looplm",
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def resolve_data_split(data_dir: str, train_pattern: str, val_pattern: str,
                       holdout_last_for_val: bool) -> tuple[list[str], list[str] | None]:
    train_shards = resolve_shards(Path(data_dir) / train_pattern)
    if holdout_last_for_val:
        if len(train_shards) < 2:
            raise ValueError("need at least 2 shards to hold out the last shard for validation")
        return train_shards[:-1], [train_shards[-1]]

    if not val_pattern:
        return train_shards, None

    val_shards = resolve_shards(Path(data_dir) / val_pattern)
    return train_shards, val_shards


def lr_mult(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return (step + 1) / tc.warmup_steps
    cd_start = int(tc.max_steps * (1 - tc.cooldown_frac))
    if step < cd_start:
        return 1.0
    t = (step - cd_start) / max(1, tc.max_steps - cd_start)
    return 1.0 + t * (tc.final_lr_frac - 1.0)


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif v.replace(".", "", 1).replace("-", "", 1).isdigit():
            out[k] = float(v) if "." in v else int(v)
        else:
            out[k] = v
    return out


def sample_depths(B: int, mu_rec: int, device) -> torch.Tensor:
    """Per-sequence T ~ Poisson(mu_rec), clamped to [1, 4*mu_rec]."""
    T = torch.poisson(torch.full((B,), float(mu_rec), device=device)).long()
    T.clamp_(min=1, max=4 * mu_rec)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="135M")
    ap.add_argument("--run_name", default="baseline")
    ap.add_argument("--data_dir", default="/data/fineweb_edu")
    ap.add_argument("--train_pattern", default="edu_fineweb_train_*.bin")
    ap.add_argument("--val_pattern", default="edu_fineweb_val_*.bin")
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_tokens", type=int, default=256 * 1024)
    ap.add_argument("--micro_batch_seqs", type=int, default=16)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--save_every", type=int, default=2500)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--holdout_last_for_val", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="model config overrides key=value")
    args = ap.parse_args()

    # ---- DDP setup ----
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0

    tc = TrainConfig(seq_len=args.seq_len, batch_tokens=args.batch_tokens,
                     micro_batch_seqs=args.micro_batch_seqs, max_steps=args.max_steps,
                     val_every=args.val_every, run_name=args.run_name,
                     out_dir=args.out_dir, compile=not args.no_compile)
    torch.manual_seed(tc.seed + rank)
    torch.set_float32_matmul_precision("high")

    # ---- model ----
    cfg = get_config(args.preset)
    for k, v in parse_overrides(args.set).items():
        setattr(cfg, k, v)

    out_dir = Path(tc.out_dir) / tc.run_name
    log_fp = None
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_fp = open(out_dir / "train.log", "a", encoding="utf-8", buffering=1)

    def log(*a):
        if not master:
            return
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        if log_fp is not None:
            log_fp.write(msg + "\n")

    if master:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        spec = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_name": args.run_name,
            "git_commit": _git_commit(),
            "cli_args": vars(args),
            "train_config": asdict(tc),
            "model_config": asdict(cfg),
            "hostname": socket.gethostname(),
            "gpu_count": gpu_count,
            "gpu_type": torch.cuda.get_device_name(0) if gpu_count else None,
            "pytorch_version": torch.__version__,
        }
        (out_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    model = ParcaeTransformer(cfg).to(device)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.type == "cuda" and tc.bf16 else nullcontext())
    pc = count_params(model)
    log(f"[{args.preset}] total={pc['total']/1e6:.1f}M non_embed={pc['non_embed']/1e6:.1f}M  "
        f"overrides={parse_overrides(args.set)}")
    log(f"  mu_rec={cfg.mu_rec} mu_bwd={math.ceil(cfg.mu_rec/2)} "
        f"use_a_matrix={cfg.use_a_matrix} use_input_norm={cfg.use_input_norm}")

    raw_model = model
    if tc.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    muon, adamw, (nm, na) = build_optimizers(raw_model, tc)
    log(f"optimizers: Muon over {nm} tensors, AdamW over {na} tensors")

    # ---- data ----
    B, S = tc.micro_batch_seqs, tc.seq_len
    tokens_per_micro = B * S * world
    accum = max(1, tc.batch_tokens // tokens_per_micro)
    train_shards, val_shards = resolve_data_split(
        args.data_dir, args.train_pattern, args.val_pattern, args.holdout_last_for_val
    )
    train_gen = data_generator(train_shards, B, S, device, rank, world, to_device=False)
    train_prefetch = CUDAPrefetcher(train_gen, device) if device.type == "cuda" else None
    log(f"batch_tokens={tc.batch_tokens} micro=({B}x{S})x{world} accum={accum} "
        f"muon_lr={tc.muon_lr:.4f} adam_lr={tc.adam_lr:.2e}")
    log(f"data shards: train={len(train_shards)} val={0 if val_shards is None else len(val_shards)}")
    if val_shards is None:
        log("validation disabled: no validation shards matched")

    if master:
        (out_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "preset": args.preset}, indent=2))

    def save_ckpt(fname, **extra):
        if not master:
            return
        torch.save({"model": raw_model.state_dict(),
                    "config": {**cfg.to_dict(), "preset": args.preset}, **extra}, out_dir / fname)
        log(f"saved checkpoint -> {out_dir / fname}")

    def next_batch():
        if train_prefetch is not None:
            return train_prefetch.next()
        return next(train_gen)

    mu_bwd = math.ceil(cfg.mu_rec / 2)

    def forward_loss(m, x, y):
        T_per_seq = sample_depths(x.size(0), cfg.mu_rec, x.device)
        T_max = int(T_per_seq.max().item())
        n_no_grad = max(0, T_max - mu_bwd)
        return m(x, y, T_per_seq=T_per_seq, n_no_grad=n_no_grad)

    @torch.no_grad()
    def evaluate(max_tokens=tc.val_tokens):
        if val_shards is None:
            return None
        model.eval()
        gen = data_generator(val_shards, B, S, device, rank, world)
        tot_loss, tot_tok, steps = 0.0, 0, max(1, max_tokens // (B * S * world))
        for _ in range(steps):
            x, y = next(gen)
            # eval at fixed T = mu_rec (no sampling), full backprop disabled by no_grad
            T_per_seq = torch.full((x.size(0),), cfg.mu_rec, device=x.device, dtype=torch.long)
            with amp:
                loss, _ = raw_model(x, y, T_per_seq=T_per_seq, n_no_grad=0)
            tot_loss += loss.item() * x.numel()
            tot_tok += x.numel()
        model.train()
        t = torch.tensor([tot_loss, tot_tok], device=device)
        if ddp:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t[0] / t[1]).item()

    # ---- train ----
    model.train()
    t0 = time.time()
    for step in range(tc.max_steps):
        m = lr_mult(step, tc)
        for g in muon.param_groups:
            g["lr"] = tc.muon_lr * m
        for g in adamw.param_groups:
            g["lr"] = tc.adam_lr * m

        loss_accum = torch.zeros((), device=device)
        for micro in range(accum):
            x, y = next_batch()
            sync_ctx = model.no_sync() if (ddp and micro < accum - 1) else nullcontext()
            with sync_ctx:
                with amp:
                    loss, _ = forward_loss(model, x, y)
                (loss / accum).backward()
            loss_accum += loss.detach() / accum

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tc.grad_clip)
        muon.step()
        adamw.step()
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)

        if step % tc.log_every == 0:
            dt = (time.time() - t0) / (step + 1)
            log(f"step {step:5d} | loss {loss_accum.item():.4f} | lr {tc.muon_lr*m:.4f} | {dt*1000:.0f}ms/step")
        if val_shards is not None and tc.val_every and (step + 1) % tc.val_every == 0:
            vl = evaluate()
            log(f"  >> val loss {vl:.4f} @ step {step+1}")
        if args.save_every and (step + 1) % args.save_every == 0:
            save_ckpt(f"ckpt_{step+1}.pt", step=step + 1)

    vl = evaluate()
    if vl is None:
        log("=== final val loss skipped (no validation shards) ===")
    else:
        log(f"=== final val loss {vl:.4f} ===")
    if master:
        (out_dir / "result.json").write_text(json.dumps({"final_val_loss": vl, "steps": tc.max_steps}))
    save_ckpt("model.pt", step=tc.max_steps, val_loss=vl)
    if log_fp is not None:
        log_fp.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
