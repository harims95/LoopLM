"""Parcae: stable looped transformer (arXiv 2604.12946).

Architecture:
    e        = LN(Prelude(s))                    # input injection
    h_0      ~ N(0, σ·I)                          # random initial state
    h_{t+1}  = A_bar * h_t + e @ B_bar.T + delta(h_t, e)
    logits   = LMHead(Coda(h_T))

where A_bar = exp(Δ ⊙ A), A = -exp(log_A) is negative diagonal (stable),
B_bar = Δ ⊙ B, and delta is the nonlinear contribution of a transformer
block applied to (h_t + e) with the standard residual carry stripped
(since the LTI pathway A_bar/B_bar replaces it).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .model import RMSNorm, Attention, DenseFFN, precompute_rope


class TransformerBlock(nn.Module):
    """Standard pre-norm block for Prelude / Coda (no loop, no LTI params)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = DenseFFN(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ParcaeLoopBlock(nn.Module):
    """One iteration of the recurrent unit. Called T times by ParcaeTransformer.

    Architecture (matches official sandyresearch/parcae reference):
        y     = decay * h + input_gain * (e @ B.T)    # LTI injection step
        h_new = block(y)                              # transformer with internal residual
    where decay = exp(-dt * A), A = exp(log_A) ≥ 0, B initialized as identity.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = DenseFFN(cfg)
        d = cfg.d_model
        if cfg.use_a_matrix:
            # Stability params: A = exp(A_log) positive, decay = exp(-dt*A) ∈ (0,1)
            self.A_log = nn.Parameter(torch.empty(d))
            self.dt_bias = nn.Parameter(torch.zeros(d))     # softplus(0) ≈ 0.69
            # B initialized as IDENTITY so e @ B.T = e from step 1 (full input signal)
            self.B = nn.Parameter(torch.eye(d))
            with torch.no_grad():
                # A_log init: A ∈ (0.1, 1.0) so decay = exp(-0.69*A) ∈ (0.5, 0.93)
                self.A_log.uniform_(math.log(0.1), math.log(1.0))

    def _lti_step(self, h: Tensor, e: Tensor) -> Tensor:
        """y = decay * h + input_gain * (e @ B.T)"""
        dt = F.softplus(self.dt_bias)                       # (d,) positive
        A = torch.exp(self.A_log)                            # (d,) positive
        decay = torch.exp(-dt * A)                           # (d,) ∈ (0,1)
        # Euler input gain (paper's choice; ZOH-exact is +0.015 improvement, not worth complexity)
        input_gain = dt                                      # (d,)
        input_write = F.linear(e, self.B)                    # e @ B.T, shape (B, S, d)
        return decay * h + input_gain * input_write

    def forward(self, h: Tensor, e: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # Step 1: LTI injection (or naive add if ablated)
        if self.cfg.use_a_matrix:
            y = self._lti_step(h, e)
        else:
            y = h + e                                        # naive looped baseline

        # Step 2: standard pre-norm transformer block with internal residual
        y = y + self.attn(self.attn_norm(y), cos, sin)
        y = y + self.ffn(self.ffn_norm(y))
        return y


class ParcaeTransformer(nn.Module):
    """Prelude (L_P blocks) + Recurrent loop (1 block, T times) + Coda (L_C blocks)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_prelude)])
        self.prelude_norm = RMSNorm(d) if cfg.use_input_norm else nn.Identity()
        self.loop = ParcaeLoopBlock(cfg)
        self.coda = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_coda)])
        self.final_norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.h0_std = 0.02
        self._rope_cache: dict = {}
        self.apply(self._init)

    def _init(self, m: nn.Module):
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def rope(self, S: int, device, dtype):
        key = (S, device, dtype)
        if key not in self._rope_cache:
            cos, sin = precompute_rope(self.cfg.head_dim, S, self.cfg.rope_theta, device)
            self._rope_cache[key] = (cos.to(dtype), sin.to(dtype))
        return self._rope_cache[key]

    def _run_prelude(self, idx: Tensor):
        x = self.embed(idx)
        cos, sin = self.rope(x.size(1), x.device, x.dtype)
        for blk in self.prelude:
            x = blk(x, cos, sin)
        return self.prelude_norm(x), cos, sin

    def _run_coda(self, h: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        for blk in self.coda:
            h = blk(h, cos, sin)
        return self.final_norm(h)

    def _h0(self, B: int, S: int, device, dtype) -> Tensor:
        return torch.randn(B, S, self.cfg.d_model, device=device, dtype=dtype) * self.h0_std

    def forward(self, idx: Tensor, targets: Tensor | None = None,
                T_per_seq: Tensor | None = None, n_no_grad: int = 0):
        """
        T_per_seq: (B,) int tensor of per-sequence loop counts. If None, uses cfg.mu_rec for all.
        n_no_grad: number of initial loops to run under torch.no_grad (truncated BPTT).
        """
        e, cos, sin = self._run_prelude(idx)
        B, S = idx.shape

        if T_per_seq is None:
            T_per_seq = torch.full((B,), self.cfg.mu_rec, device=idx.device, dtype=torch.long)
        T_max = int(T_per_seq.max().item())
        n_no_grad = min(n_no_grad, T_max)

        h = self._h0(B, S, idx.device, e.dtype)

        # Phase 1: no-grad loops
        with torch.no_grad():
            for t in range(n_no_grad):
                active = (t < T_per_seq).view(B, 1, 1).to(h.dtype)
                h_new = self.loop(h, e, cos, sin)
                h = active * h_new + (1.0 - active) * h

        # Phase 2: with-grad loops
        for t in range(n_no_grad, T_max):
            active = (t < T_per_seq).view(B, 1, 1).to(h.dtype)
            h_new = self.loop(h, e, cos, sin)
            h = active * h_new + (1.0 - active) * h

        h = self._run_coda(h, cos, sin)

        if targets is None:
            return self.lm_head(h), h.new_zeros(())

        logits = self.lm_head(h).float()
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                             targets.view(-1), ignore_index=-1)
        z = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        loss = ce + self.cfg.final_z_loss_coef * z
        return loss, {"ce": ce.detach(), "z": z.detach()}


def count_params(model: ParcaeTransformer) -> dict:
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    embed = cfg.vocab_size * cfg.d_model  # tied, counted once
    return {"total": total, "non_embed": total - embed, "embed": embed}
