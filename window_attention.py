"""
Windowed Multi-Head Self-Attention for Vision Transformers.

Qwen2-VL / 2.5-VL uses **non-overlapping window attention** inside its ViT
to keep memory usage linear in the number of visual tokens when processing
high-resolution images.

Algorithm
---------
1. Partition the 2-D feature grid ``(H_patches, W_patches)`` into windows
   of size ``(win_h, win_w)``.
2. Run standard multi-head self-attention **independently** within each
   window (no cross-window communication).
3. Un-partition back to the original grid.

This reduces the attention complexity from ``O(N²)`` (where N = H × W) to
``O(N × win_h × win_w)``.

For long-range interaction the ViT interleaves window-attention layers with
**global attention** layers (full self-attention over all tokens) or relies
on the subsequent LLM decoder layers for cross-patch reasoning.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb
from .rms_norm import RMSNorm


# ------------------------------------------------------------------
#  Utility: partition / un-partition
# ------------------------------------------------------------------

def _window_partition(
    x: torch.Tensor, win_h: int, win_w: int, H: int, W: int,
) -> torch.Tensor:
    """Partition ``[B, H*W, C]`` → ``[B * nH * nW, win_h*win_w, C]``.

    Pads spatially if ``H`` or ``W`` is not divisible by the window size.
    """
    B, _, C = x.shape
    # Reshape to spatial grid
    x = x.view(B, H, W, C)

    # Pad if needed
    pad_h = (win_h - H % win_h) % win_h
    pad_w = (win_w - W % win_w) % win_w
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # pad W then H

    Hp, Wp = H + pad_h, W + pad_w
    nH, nW = Hp // win_h, Wp // win_w

    # [B, nH, win_h, nW, win_w, C]
    x = x.view(B, nH, win_h, nW, win_w, C)
    # [B, nH, nW, win_h, win_w, C] → [B*nH*nW, win_h*win_w, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B * nH * nW, win_h * win_w, C)
    return x, Hp, Wp, nH, nW


def _window_unpartition(
    x: torch.Tensor, B: int, nH: int, nW: int,
    win_h: int, win_w: int, H: int, W: int,
) -> torch.Tensor:
    """Reverse ``_window_partition``: ``[B*nH*nW, win_h*win_w, C]`` → ``[B, H*W, C]``."""
    C = x.shape[-1]
    x = x.view(B, nH, nW, win_h, win_w, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, nH * win_h, nW * win_w, C)
    # Remove padding
    x = x[:, :H, :W, :].contiguous()
    return x.view(B, H * W, C)


# ------------------------------------------------------------------
#  Window Attention
# ------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Multi-Head Self-Attention restricted to local windows.

    Parameters
    ----------
    hidden_size : int
        Feature dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int or None
        Per-head dimension; defaults to ``hidden_size // num_heads``.
    window_size : tuple[int, int]
        ``(win_h, win_w)`` — height and width of each attention window
        measured in patches.
    qkv_bias : bool
        Whether Q/K/V projections include bias.
    use_rope : bool
        If True, the caller is expected to supply pre-computed ``(cos, sin)``
        tensors and they will be applied inside attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        window_size: tuple[int, int] = (7, 7),
        qkv_bias: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.window_size = window_size
        self.use_rope = use_rope
        self.scale = 1.0 / math.sqrt(self.head_dim)

        total = self.num_heads * self.head_dim
        self.qkv = nn.Linear(hidden_size, 3 * total, bias=qkv_bias)
        self.proj = nn.Linear(total, hidden_size, bias=True)

    # ----- core forward -----

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``[B, H*W, C]``
            H, W: spatial dimensions (in patches).
            rope_cos, rope_sin: pre-computed rotary tensors for per-window
                positions; shape ``[num_windows*B, win_tokens, head_dim]``
                or broadcastable.

        Returns:
            ``[B, H*W, C]``
        """
        B = x.size(0)
        win_h, win_w = self.window_size

        # 1. Window partition
        x_win, Hp, Wp, nH, nW = _window_partition(x, win_h, win_w, H, W)
        # x_win: [B*nH*nW, win_tokens, C]

        # Also partition RoPE tensors into the same windows
        if self.use_rope and rope_cos is not None and rope_sin is not None:
            # rope_cos/sin: [B, N, 1, D] → squeeze head dim → partition → unsqueeze
            rc = rope_cos.squeeze(2)  # [B, N, D]
            rs = rope_sin.squeeze(2)
            rc, _, _, _, _ = _window_partition(rc, win_h, win_w, H, W)  # [B*nH*nW, win_tokens, D]
            rs, _, _, _, _ = _window_partition(rs, win_h, win_w, H, W)
            rope_cos = rc.unsqueeze(2)  # [B*nH*nW, win_tokens, 1, D]
            rope_sin = rs.unsqueeze(2)

        Bw, N, _ = x_win.shape  # Bw = B*nH*nW, N = win_h*win_w

        # 2. QKV projection → split heads
        qkv = self.qkv(x_win)  # [Bw, N, 3*total]
        qkv = qkv.view(Bw, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [Bw, N, num_heads, head_dim]

        # 3. Optional RoPE
        if self.use_rope and rope_cos is not None and rope_sin is not None:
            q = apply_rotary_emb(q, rope_cos, rope_sin)
            k = apply_rotary_emb(k, rope_cos, rope_sin)

        # 4. Attention: transpose to [Bw, heads, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [Bw, H, N, N]

        # Safe softmax (subtract max for numerical stability)
        attn_max = attn.max(dim=-1, keepdim=True).values
        attn = torch.exp(attn - attn_max)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)  # [Bw, heads, N, D]
        out = out.transpose(1, 2).contiguous().view(Bw, N, -1)  # [Bw, N, C]

        # 5. Output projection
        out = self.proj(out)

        # 6. Un-partition
        out = _window_unpartition(out, B, nH, nW, win_h, win_w, H, W)
        return out

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, window={self.window_size}"
        )
