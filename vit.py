"""
Vision Transformer (ViT) encoder for Qwen VL models.

Architecture progression
------------------------
- **Qwen2-VL ViT**: ViT-L/14 backbone with 2-D RoPE + window attention.
  Dynamic resolution: images are processed at their native aspect ratio;
  patch tokens are arranged on a variable-size grid and each window gets
  local 2-D positional encoding.

- **Qwen2.5-VL ViT**: Same structure but upgrades to 3-D RoPE, enabling
  native video input where the temporal axis shares the ViT backbone.

Key components
--------------
1. ``PatchEmbed2D`` — 2-D convolution that maps image patches → token
   embeddings.
2. ``ViTBlock`` — Transformer block with optional *window* or *global*
   attention, 2-D/3-D RoPE, SwiGLU FFN, and RMSNorm.
3. ``VisionTransformer`` — Full ViT encoder that stacks blocks and applies
   global attention periodically (e.g. every 4th layer).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .window_attention import WindowAttention
from .rope import RotaryEmbedding2D, RotaryEmbedding3D, apply_rotary_emb

import math


# ======================================================================
#  Patch Embedding
# ======================================================================

class PatchEmbed2D(nn.Module):
    """2-D patch embedding via convolution.

    Maps ``[B, C_in, img_H, img_W]`` → ``[B, num_patches, embed_dim]``.

    Parameters
    ----------
    patch_size : int
        Patch resolution (square patches).
    in_channels : int
        Number of image channels.
    embed_dim : int
        Output embedding dimension.
    """

    def __init__(self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: ``[B, C, img_H, img_W]``

        Returns:
            ``(tokens, H_patches, W_patches)`` where tokens is
            ``[B, H_patches * W_patches, embed_dim]``.
        """
        x = self.proj(x)  # [B, embed_dim, H_p, W_p]
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, Hp*Wp, C]
        return x, Hp, Wp


class PatchEmbed3D(nn.Module):
    """3-D patch embedding for video input via ``Conv3d``.

    Maps ``[B, C_in, T, img_H, img_W]`` → ``[B, T'*H'*W', embed_dim]``
    where T' = T // temporal_patch_size, H' = H // patch_size, etc.

    Parameters
    ----------
    patch_size : int
        Spatial patch resolution (square patches).
    temporal_patch_size : int
        Temporal patch size (number of frames grouped per token).
    in_channels : int
        Number of image channels.
    embed_dim : int
        Output embedding dimension.
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """
        Args:
            x: ``[B, C, T, img_H, img_W]``

        Returns:
            ``(tokens, T_patches, H_patches, W_patches)`` where tokens is
            ``[B, T'*H'*W', embed_dim]``.
        """
        x = self.proj(x)  # [B, embed_dim, T_p, H_p, W_p]
        B, C, Tp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, T_p*H_p*W_p, C]
        return x, Tp, Hp, Wp


# ======================================================================
#  Full Self-Attention (for global layers)
# ======================================================================

class FullSelfAttention(nn.Module):
    """Standard multi-head self-attention over **all** tokens (no windowing)."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int | None = None,
                 qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        total = num_heads * self.head_dim
        self.qkv = nn.Linear(hidden_size, 3 * total, bias=qkv_bias)
        self.proj = nn.Linear(total, hidden_size, bias=True)

    def forward(
        self, x: torch.Tensor, H: int, W: int,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # [B, N, heads, D]

        if rope_cos is not None and rope_sin is not None:
            q = apply_rotary_emb(q, rope_cos, rope_sin)
            k = apply_rotary_emb(k, rope_cos, rope_sin)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, N, D]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_max = attn.max(dim=-1, keepdim=True).values
        attn = torch.exp(attn - attn_max)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.proj(out)


# ======================================================================
#  ViT MLP (fc1 / GELU / fc2 — matches HF Qwen ViT naming)
# ======================================================================

class ViTMLP(nn.Module):
    """Standard 2-layer MLP used in HuggingFace Qwen ViT blocks.

    ``fc1`` → GELU → ``fc2``.  Named to match the HF checkpoint keys
    ``visual.blocks.{i}.mlp.fc1`` / ``visual.blocks.{i}.mlp.fc2``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ======================================================================
#  ViT Block
# ======================================================================

class ViTBlock(nn.Module):
    """Single ViT Transformer block.

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    intermediate_size : int
        SwiGLU FFN intermediate dim.
    window_size : tuple[int, int] or None
        If None → full (global) self-attention.
    use_rope : bool
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        window_size: tuple[int, int] | None = (7, 7),
        use_rope: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        if window_size is not None:
            self.attn = WindowAttention(
                hidden_size, num_heads, window_size=window_size, use_rope=use_rope,
            )
        else:
            self.attn = FullSelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = ViTMLP(hidden_size, intermediate_size)
        self.window_size = window_size

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W, rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ======================================================================
#  Vision Transformer
# ======================================================================

class VisionTransformer(nn.Module):
    """Full ViT encoder for Qwen VL models.

    Parameters
    ----------
    patch_size : int
    in_channels : int
    hidden_size : int
    num_heads : int
    num_layers : int
    intermediate_size : int
    window_size : tuple[int, int]
    global_attn_every : int
        Insert a global-attention layer every N layers.
        Qwen2-VL default: every 4th layer is global.
    rope_dim : int
        2 or 3; selects RotaryEmbedding2D or RotaryEmbedding3D.
    temporal_patch_size : int
        Temporal (frame) patch size for 3-D Conv video embedding.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        intermediate_size: int = 2816,
        window_size: tuple[int, int] = (7, 7),
        global_attn_every: int = 4,
        rope_dim: int = 2,
        temporal_patch_size: int = 2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed2D(patch_size, in_channels, hidden_size)
        # Optional 3-D patch embed for video input
        self.patch_embed_3d: PatchEmbed3D | None = None
        if rope_dim == 3:
            self.patch_embed_3d = PatchEmbed3D(
                patch_size, temporal_patch_size, in_channels, hidden_size,
            )

        # Build blocks
        blocks = []
        for i in range(num_layers):
            is_global = (i + 1) % global_attn_every == 0
            ws = None if is_global else window_size
            blocks.append(ViTBlock(hidden_size, num_heads, intermediate_size,
                                   window_size=ws, use_rope=True))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(hidden_size)

        head_dim = hidden_size // num_heads
        if rope_dim == 3:
            self.rope = RotaryEmbedding3D(head_dim)
        else:
            self.rope = RotaryEmbedding2D(head_dim)
        self.rope_dim = rope_dim

    def forward(
        self,
        pixel_values: torch.Tensor,
        time_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: ``[B, C, img_H, img_W]`` for images or
                ``[B, C, T, img_H, img_W]`` for video.
            time_ids: ``[B, num_patches]`` frame indices (only for 3-D RoPE).

        Returns:
            ``[B, num_patches, hidden_size]``
        """
        if pixel_values.dim() == 5 and self.patch_embed_3d is not None:
            # Video input: [B, C, T, H, W] → 3-D patch embed
            x, Tp, Hp, Wp = self.patch_embed_3d(pixel_values)
            B, N, C = x.shape
            # Build temporal / spatial position IDs
            t_ids = torch.arange(Tp, device=x.device).view(Tp, 1, 1).expand(Tp, Hp, Wp).flatten()
            h_ids = torch.arange(Hp, device=x.device).view(1, Hp, 1).expand(Tp, Hp, Wp).flatten()
            w_ids = torch.arange(Wp, device=x.device).view(1, 1, Wp).expand(Tp, Hp, Wp).flatten()
            t_ids = t_ids.unsqueeze(0).expand(B, -1)
            h_ids = h_ids.unsqueeze(0).expand(B, -1)
            w_ids = w_ids.unsqueeze(0).expand(B, -1)
            rope_cos, rope_sin = self.rope(t_ids, h_ids, w_ids)
            # Flatten temporal into spatial H for window attention: Hp_eff = Tp * Hp
            Hp = Tp * Hp
        else:
            # Image input: [B, C, H, W] → 2-D patch embed
            x, Hp, Wp = self.patch_embed(pixel_values)
            B, N, C = x.shape
            h_ids = torch.arange(Hp, device=x.device).unsqueeze(1).expand(Hp, Wp).flatten()
            w_ids = torch.arange(Wp, device=x.device).unsqueeze(0).expand(Hp, Wp).flatten()
            h_ids = h_ids.unsqueeze(0).expand(B, -1)
            w_ids = w_ids.unsqueeze(0).expand(B, -1)
            if self.rope_dim == 3:
                if time_ids is None:
                    time_ids = torch.zeros(B, N, dtype=torch.long, device=x.device)
                rope_cos, rope_sin = self.rope(time_ids, h_ids, w_ids)
            else:
                rope_cos, rope_sin = self.rope(h_ids, w_ids)

        # Expand for head broadcasting: [B, N, 1, D] for compatibility
        rope_cos = rope_cos.unsqueeze(2)
        rope_sin = rope_sin.unsqueeze(2)

        for block in self.blocks:
            x = block(x, Hp, Wp, rope_cos, rope_sin)

        return self.norm(x)

    def extra_repr(self) -> str:
        return f"rope_dim={self.rope_dim}, num_layers={len(self.blocks)}"
