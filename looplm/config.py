"""Model and training configuration for LoopLM (Parcae-style looped transformer)."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    # ---- shape ----
    vocab_size: int = 50304            # GPT-2 (50257) padded to mult of 128
    d_model: int = 1024
    # ---- architecture: middle-looped (Parcae) ----
    n_prelude: int = 4                 # transformer blocks before the loop
    n_coda: int = 2                    # transformer blocks after the loop
    mu_rec: int = 6                    # mean loop count (Poisson during training)
    # ---- attention (GQA + QK-norm + RoPE) ----
    n_q_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 64
    qk_norm: bool = True
    rope_theta: float = 10000.0
    # ---- FFN (SwiGLU) ----
    dense_ffn: int = 2816
    # ---- output ----
    tie_embeddings: bool = True
    final_z_loss_coef: float = 1e-4
    # ---- ablation switches ----
    use_a_matrix: bool = True          # if False, naive looped (no LTI stability)
    use_input_norm: bool = True        # if False, skip RMSNorm on prelude output
    # ---- init ----
    init_std: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    # data
    data_dir: str = "/data/fineweb_edu"
    train_pattern: str = "edu_fineweb_train_*.bin"
    val_pattern: str = "edu_fineweb_val_*.bin"
    seq_len: int = 1024
    batch_tokens: int = 256 * 1024
    micro_batch_seqs: int = 16
    # schedule
    max_steps: int = 4000
    warmup_steps: int = 100
    cooldown_frac: float = 0.4
    final_lr_frac: float = 0.1
    # optimizer
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_wd: float = 0.1
    muon_ns_steps: int = 5
    adam_lr: float = 3e-4
    adam_betas: tuple = (0.9, 0.95)
    adam_wd: float = 0.1
    grad_clip: float = 1.0
    # eval / logging
    val_every: int = 250
    val_tokens: int = 10 * 1024 * 1024
    log_every: int = 10
    # run
    seed: int = 1337
    compile: bool = True
    bf16: bool = True
    out_dir: str = "runs"
    run_name: str = "default"


# ---- preset architectures ----
PRESETS: dict[str, ModelConfig] = {
    "135M": ModelConfig(
        d_model=1024, n_prelude=4, n_coda=2, mu_rec=6,
        n_q_heads=16, n_kv_heads=8, head_dim=64,
        dense_ffn=2816,
    ),
    # ablation variants (override via CLI in practice; these are starting points)
    "135M_naive": ModelConfig(
        d_model=1024, n_prelude=4, n_coda=2, mu_rec=6,
        n_q_heads=16, n_kv_heads=8, head_dim=64,
        dense_ffn=2816,
        use_a_matrix=False, use_input_norm=False,
    ),
}


def get_config(preset: str) -> ModelConfig:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    return ModelConfig(**PRESETS[preset].to_dict())
