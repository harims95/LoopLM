"""Shared transformer building blocks: RMSNorm, GQA attention with QK-norm + RoPE, SwiGLU FFN.

Adapted from rootxhacker/HobbyLM (Apache-2.0).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig


def rms_norm(x: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    out = F.rms_norm(x, (x.size(-1),), eps=eps)
    return out * weight if weight is not None else out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


def precompute_rope(head_dim: int, max_seq: int, theta: float, device) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    S, D = x.shape[-2], x.shape[-1]
    cos = cos[:S].view(1, 1, S, D // 2)
    sin = sin[:S].view(1, 1, S, D // 2)
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    """GQA attention with per-head QK-norm before RoPE."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.nq, self.nkv, self.hd = cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim
        assert self.nq % self.nkv == 0, "n_q_heads must be divisible by n_kv_heads"
        self.rep = self.nq // self.nkv
        qkv_out = (self.nq + 2 * self.nkv) * self.hd
        self.qkv = nn.Linear(cfg.d_model, qkv_out, bias=False)
        self.proj = nn.Linear(self.nq * self.hd, cfg.d_model, bias=False)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.hd)
            self.k_norm = RMSNorm(self.hd)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.nq * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = q.view(B, S, self.nq, self.hd).transpose(1, 2)
        k = k.view(B, S, self.nkv, self.hd).transpose(1, 2)
        v = v.view(B, S, self.nkv, self.hd).transpose(1, 2)
        if self.cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, S, self.nq * self.hd)
        return self.proj(o)


class DenseFFN(nn.Module):
    """SwiGLU feed-forward."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w13 = nn.Linear(cfg.d_model, 2 * cfg.dense_ffn, bias=False)
        self.w2 = nn.Linear(cfg.dense_ffn, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)
