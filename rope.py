"""
Rotary Position Embedding (RoPE) variants for the Qwen VL model family.

Evolution
---------
- **1D RoPE** (Qwen2 text decoder):
    Standard rotary embedding — pairs of dimensions are rotated by
    ``position * θ_i`` where ``θ_i = base^{-2i/d}``.

- **2D RoPE** (Qwen2-VL ViT):
    Head dimension is split into two halves; the first half is rotated by
    the **height** position, the second by the **width** position.

- **3D RoPE** (Qwen2.5-VL ViT):
    Head dimension is split into three parts for **(time, height, width)**,
    enabling native video understanding with temporal position awareness.

- **M-RoPE** (Multimodal RoPE, Qwen2-VL / 2.5-VL decoder):
    Three independent position-ID streams ``(pos_t, pos_h, pos_w)`` share a
    single head-dim split.  For **text** tokens all three IDs equal the token
    position, degenerating to standard 1D RoPE.  For **image/video** tokens
    each stream carries the corresponding spatial-temporal coordinate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ======================================================================
#  Helper: apply rotary to a (cos, sin) pair
# ======================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension: [x1, x2] → [−x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding ``x * cos + rotate_half(x) * sin``.

    Shapes for *cos* / *sin* are broadcast-compatible with *x*; typically
    ``[1, S, 1, D]`` or ``[1, S, D]``.
    """
    return x * cos + _rotate_half(x) * sin


# ======================================================================
#  1-D RoPE  (standard text decoder)
# ======================================================================

class RotaryEmbedding1D(nn.Module):
    """Standard 1-D Rotary Position Embedding.

    Parameters
    ----------
    dim : int
        Per-head dimension (must be even).
    max_seq_len : int
        Maximum sequence length for pre-computed cache.
    base : float
        Frequency base (default 10 000, Qwen2 uses 1 000 000).
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # θ_i = base^{-2i / dim},  i = 0 .. dim/2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [S, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, dim]
        cos_cache = emb.cos().unsqueeze(0).unsqueeze(2)   # [1, S, 1, dim]
        sin_cache = emb.sin().unsqueeze(0).unsqueeze(2)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` tensors for the given positions.

        Args:
            x: ``[B, S, H, D]`` – used only for device/dtype; actual
               rotary is determined by *position_ids*.
            position_ids: ``[B, S]`` integer positions.  If *None*, uses
                          ``0 .. S-1``.

        Returns:
            ``(cos, sin)`` each ``[B, S, 1, D]``.
        """
        S = x.size(1)
        if position_ids is None:
            cos = self.cos_cache[:, :S]
            sin = self.sin_cache[:, :S]
        else:
            # Gather from cache
            cos = self.cos_cache.squeeze(0).squeeze(1)  # [max_S, D]
            sin = self.sin_cache.squeeze(0).squeeze(1)
            cos = cos[position_ids].unsqueeze(2)  # [B, S, 1, D]
            sin = sin[position_ids].unsqueeze(2)
        return cos.to(x.dtype), sin.to(x.dtype)


# ======================================================================
#  2-D RoPE  (Qwen2-VL ViT — spatial height × width)
# ======================================================================

class RotaryEmbedding2D(nn.Module):
    """2-D Rotary Position Embedding for vision features.

    Splits ``dim`` into two halves:
      - first  ``dim // 2`` dimensions ← height positions
      - second ``dim // 2`` dimensions ← width  positions

    Parameters
    ----------
    dim : int
        Per-head dimension (must be divisible by 4, since each half still
        uses the "pair" trick internally).
    max_spatial : int
        Maximum spatial extent in either direction.
    base : float
        Frequency base.
    """

    def __init__(self, dim: int, max_spatial: int = 512, base: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2-D RoPE"
        self.dim = dim
        self.half_dim = dim // 2
        self.max_spatial = max_spatial

        inv_freq = 1.0 / (base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for 2-D positions.

        Args:
            height_ids: ``[B, S]`` integer row positions.
            width_ids:  ``[B, S]`` integer column positions.

        Returns:
            ``(cos, sin)`` each ``[B, S, dim]``.
        """
        # Height part
        h_freqs = torch.outer(height_ids.flatten().float(), self.inv_freq)  # [B*S, half_dim/2]
        h_emb = torch.cat([h_freqs, h_freqs], dim=-1)  # [B*S, half_dim]

        # Width part
        w_freqs = torch.outer(width_ids.flatten().float(), self.inv_freq)
        w_emb = torch.cat([w_freqs, w_freqs], dim=-1)

        # Concatenate height + width
        emb = torch.cat([h_emb, w_emb], dim=-1)  # [B*S, dim]
        emb = emb.view(*height_ids.shape, self.dim)  # [B, S, dim]

        return emb.cos(), emb.sin()


# ======================================================================
#  3-D RoPE  (Qwen2.5-VL ViT — temporal × height × width)
# ======================================================================

class RotaryEmbedding3D(nn.Module):
    """3-D Rotary Position Embedding for video/image features.

    Splits ``dim`` into **three** equal parts for ``(time, height, width)``.
    This is the key upgrade from Qwen2-VL → Qwen2.5-VL for native video.

    Parameters
    ----------
    dim : int
        Per-head dimension (must be divisible by 6).
    max_spatial : int
        Max spatial extent.
    max_temporal : int
        Max temporal frames.
    base : float
        Frequency base.
    """

    def __init__(self, dim: int, max_spatial: int = 512,
                 max_temporal: int = 256, base: float = 10000.0):
        super().__init__()
        assert dim % 6 == 0, "dim must be divisible by 6 for 3-D RoPE"
        self.dim = dim
        self.part_dim = dim // 3  # each axis gets dim/3

        inv_freq = 1.0 / (base ** (torch.arange(0, self.part_dim, 2).float() / self.part_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        time_ids: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for 3-D positions.

        Args:
            time_ids:   ``[B, S]`` frame indices.
            height_ids: ``[B, S]`` row positions.
            width_ids:  ``[B, S]`` column positions.

        Returns:
            ``(cos, sin)`` each ``[B, S, dim]``.
        """
        def _axis_emb(ids: torch.Tensor) -> torch.Tensor:
            freqs = torch.outer(ids.flatten().float(), self.inv_freq)
            return torch.cat([freqs, freqs], dim=-1)  # [N, part_dim]

        t_emb = _axis_emb(time_ids)
        h_emb = _axis_emb(height_ids)
        w_emb = _axis_emb(width_ids)

        emb = torch.cat([t_emb, h_emb, w_emb], dim=-1)  # [N, dim]
        emb = emb.view(*time_ids.shape, self.dim)

        return emb.cos(), emb.sin()


# ======================================================================
#  M-RoPE  (Multimodal RoPE for decoder, Qwen2-VL / 2.5-VL)
# ======================================================================

class MRoPE(nn.Module):
    """Multimodal Rotary Position Embedding (M-RoPE).

    Splits per-head ``dim`` into **three** equal segments and assigns each
    to an independent position-ID stream ``(pos_t, pos_h, pos_w)``:

    - **Text tokens**: ``pos_t = pos_h = pos_w = text_position``
      → degenerates to standard 1-D RoPE.
    - **Image tokens**: ``pos_t`` = position of the ``<image>`` placeholder
      in text; ``pos_h``, ``pos_w`` = spatial grid coordinates of the patch.
    - **Video tokens**: ``pos_t`` = frame index (scaled); ``pos_h``,
      ``pos_w`` = spatial grid coordinates.

    Parameters
    ----------
    dim : int
        Per-head dimension (divisible by 6).
    max_seq_len : int
        Maximum total sequence length (text + visual tokens).
    base : float
        Frequency base.
    """

    def __init__(self, dim: int, max_seq_len: int = 32768, base: float = 1000000.0):
        super().__init__()
        assert dim % 6 == 0, "dim must be divisible by 6 for M-RoPE"
        self.dim = dim
        self.part_dim = dim // 3

        inv_freq = 1.0 / (base ** (torch.arange(0, self.part_dim, 2).float() / self.part_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        pos_t: torch.Tensor,
        pos_h: torch.Tensor,
        pos_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute M-RoPE cos/sin.

        Args:
            pos_t: ``[B, S]`` temporal / text position IDs.
            pos_h: ``[B, S]`` height position IDs.
            pos_w: ``[B, S]`` width position IDs.

        Returns:
            ``(cos, sin)`` each ``[B, S, dim]``.
        """
        def _axis_emb(ids: torch.Tensor) -> torch.Tensor:
            freqs = torch.outer(ids.flatten().float(), self.inv_freq)
            return torch.cat([freqs, freqs], dim=-1)

        emb = torch.cat([_axis_emb(pos_t), _axis_emb(pos_h), _axis_emb(pos_w)], dim=-1)
        emb = emb.view(*pos_t.shape, self.dim)
        return emb.cos(), emb.sin()
