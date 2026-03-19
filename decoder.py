"""
LLM Decoder Backbone for Qwen VL models.

Implements the standard Transformer decoder block used in Qwen2/2.5/3:

    x → RMSNorm → GQA Self-Attention → Residual →
        RMSNorm → SwiGLU FFN        → Residual → output

Key features
------------
- **Grouped Query Attention (GQA)**: Q heads are grouped; each group shares
  one set of K/V heads, reducing KV-cache by ``num_q_heads / num_kv_heads``.
- **M-RoPE aware**: position cos/sin can be supplied externally (from the
  M-RoPE module) so the same decoder handles text-only and multimodal
  sequences.
- **KV-cache support**: for auto-regressive generation, previously computed
  K/V can be passed and incrementally extended.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .rms_norm import RMSNorm
from .swiglu_mlp import SwiGLUMLP
from .rope import apply_rotary_emb


class GQASelfAttention(nn.Module):
    """Grouped Query Attention with KV-cache support.

    Parameters
    ----------
    hidden_size : int
    num_q_heads : int
    num_kv_heads : int
    head_dim : int or None
    max_seq_len : int
        Maximum sequence length (for causal mask buffer).
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        max_seq_len: int = 8192,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_q_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        q_total = num_q_heads * self.head_dim
        kv_total = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, q_total, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, kv_total, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, kv_total, bias=qkv_bias)
        self.o_proj = nn.Linear(q_total, hidden_size, bias=False)

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
            rope_cos, rope_sin: ``[B, S, 1, head_dim]`` or broadcastable.
            causal: apply causal mask.
            kv_cache: ``(cached_k, cached_v)`` each ``[B, kv_heads, S_past, D]``.

        Returns:
            ``(output, (new_k, new_v))``
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            q = apply_rotary_emb(q, rope_cos, rope_sin)
            k = apply_rotary_emb(k, rope_cos, rope_sin)

        # Transpose to [B, heads, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV-cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # GQA: repeat KV heads to match Q heads
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Attention
        S_k = k.size(2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            # Causal mask: position i can only attend to j <= i
            # For KV-cache, offset correctly
            row_idx = torch.arange(S, device=x.device).unsqueeze(1) + (S_k - S)
            col_idx = torch.arange(S_k, device=x.device).unsqueeze(0)
            mask = col_idx > row_idx  # [S, S_k]
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Safe softmax
        attn_max = attn.max(dim=-1, keepdim=True).values
        attn = torch.exp(attn - attn_max)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.o_proj(out)

        return out, new_kv_cache


class DecoderBlock(nn.Module):
    """Single Transformer decoder block (GQA + SwiGLU + RMSNorm).

    Parameters
    ----------
    hidden_size : int
    num_q_heads : int
    num_kv_heads : int
    intermediate_size : int
    head_dim : int or None
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int | None = None,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = GQASelfAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            qkv_bias=qkv_bias,
        )
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        causal: bool = True,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, new_kv = self.attn(self.attn_norm(x), rope_cos, rope_sin, causal, kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


class DecoderBackbone(nn.Module):
    """Stack of Transformer decoder blocks.

    Parameters
    ----------
    num_layers : int
    hidden_size : int
    num_q_heads : int
    num_kv_heads : int
    intermediate_size : int
    head_dim : int or None
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int | None = None,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_size, num_q_heads, num_kv_heads,
                         intermediate_size, head_dim, qkv_bias=qkv_bias)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        causal: bool = True,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            ``(output, list_of_kv_caches)``
        """
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_kv = layer(x, rope_cos, rope_sin, causal, layer_cache)
            new_kv_caches.append(new_kv)
        return self.norm(x), new_kv_caches
