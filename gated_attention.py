"""
Gated Self-Attention — full quadratic attention with output gating.

In **Qwen3-NeXT**, the model interleaves Gated DeltaNet layers (linear
attention, sub-quadratic) with a smaller number of **Gated Attention**
layers (full quadratic attention) to preserve the ability to form sharp,
long-range dependencies that pure linear attention may miss.

Design
------
    o = α ⊙ Attention(Q, K, V)
    α = σ(W_α · x)

where ``alpha`` is a sigmoid gate applied element-wise to the attention
output, and the attention itself is standard scaled dot-product with GQA
and optional causal masking.

The gating mechanism:
  - Allows the model to *selectively suppress* attention output per-feature.
  - Provides training stability similar to gating in GRU/LSTM.
  - Is parameter-cheap (one extra linear layer for the gate).

In hybrid architectures, these gated attention layers are placed at regular
intervals (e.g. every 4th or 8th layer) among DeltaNet layers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb
from .rms_norm import RMSNorm


class GatedSelfAttention(nn.Module):
    """Full self-attention with output gating (GQA-compatible).

    Parameters
    ----------
    hidden_size : int
    num_q_heads : int
    num_kv_heads : int
    head_dim : int or None
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_q_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        q_total = num_q_heads * self.head_dim
        kv_total = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, q_total, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_total, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_total, bias=False)
        self.o_proj = nn.Linear(q_total, hidden_size, bias=False)

        # Output gate: α = σ(W_α · x)
        self.gate_proj = nn.Linear(hidden_size, q_total, bias=True)

        # Per-head group norm (applied before gating)
        self.group_norm = nn.GroupNorm(num_q_heads, q_total)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        causal: bool = True,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: ``[B, S, hidden_size]``
            rope_cos, rope_sin: RoPE tensors (broadcastable).
            causal: apply causal mask.
            kv_cache: optional past K/V.

        Returns:
            ``(output, (new_k, new_v))``
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        # RoPE
        if rope_cos is not None and rope_sin is not None:
            q = apply_rotary_emb(q, rope_cos, rope_sin)
            k = apply_rotary_emb(k, rope_cos, rope_sin)

        q = q.transpose(1, 2)  # [B, Hq, S, D]
        k = k.transpose(1, 2)  # [B, Hkv, S, D]
        v = v.transpose(1, 2)

        # KV-cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        S_k = k.size(2)

        # GQA expand
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            row_idx = torch.arange(S, device=x.device).unsqueeze(1) + (S_k - S)
            col_idx = torch.arange(S_k, device=x.device).unsqueeze(0)
            mask = col_idx > row_idx
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Safe softmax
        attn_max = attn.max(dim=-1, keepdim=True).values
        attn = torch.exp(attn - attn_max)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)  # [B, Hq, S, D]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, total]

        # Per-token group norm (preserves causality)
        B_out, S_out, C_out = out.shape
        out = out.reshape(B_out * S_out, C_out, 1)  # [B*S, total, 1]
        out = self.group_norm(out)
        out = out.reshape(B_out, S_out, C_out)        # [B, S, total]

        # Output gate
        alpha = torch.sigmoid(self.gate_proj(x))  # [B, S, total]
        out = out * alpha

        # Output projection
        out = self.o_proj(out)
        return out, new_kv

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, q_heads={self.num_q_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}"
        )
