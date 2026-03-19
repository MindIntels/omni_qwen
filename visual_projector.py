"""
Vision-to-Language Projector / Token Merger for Qwen VL models.

After the ViT encoder produces per-patch visual features, a **projector**
adapts them to the LLM's token embedding space.

Evolution
---------
- **Qwen2-VL**: 2-layer MLP projector with spatial **token merging** —
  adjacent 2×2 patches are concatenated and projected, compressing the
  visual sequence length by 4×.

- **Qwen2.5-VL**: Same architecture but improved training with dynamic
  resolution and better token budget allocation.

- **Qwen3-VL / Qwen3-NeXT**: Further compressed projection; may use
  cross-attention based compression (perceiver-style) for variable
  token counts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen2VLProjector(nn.Module):
    """2×2 spatial merge + MLP projection (Qwen2-VL / 2.5-VL style).

    Takes ViT output ``[B, H*W, vit_dim]``, performs 2×2 spatial merging
    to get ``[B, H/2 * W/2, 4*vit_dim]``, then projects to LLM hidden
    size via a 2-layer MLP.

    Parameters
    ----------
    vit_hidden_size : int
        ViT output dimension.
    llm_hidden_size : int
        LLM embedding dimension.
    merge_size : int
        Spatial merge factor (default 2 → 2×2 patches merged → 4× compression).
    """

    def __init__(
        self,
        vit_hidden_size: int = 1024,
        llm_hidden_size: int = 4096,
        merge_size: int = 2,
    ):
        super().__init__()
        self.merge_size = merge_size
        merged_dim = vit_hidden_size * merge_size * merge_size

        # LayerNorm on merged tokens before MLP (matches HF ``merger.ln_q``)
        self.ln_q = nn.LayerNorm(merged_dim)

        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(
        self,
        vit_features: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Args:
            vit_features: ``[B, H*W, vit_dim]``
            H, W: spatial grid dimensions (in patches).

        Returns:
            ``[B, (H//m)*(W//m), llm_hidden_size]`` where m = merge_size.
        """
        B, N, C = vit_features.shape
        m = self.merge_size

        # Pad if needed
        pad_h = (m - H % m) % m
        pad_w = (m - W % m) % m
        x = vit_features.view(B, H, W, C)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        # Reshape to merge m×m patches
        x = x.view(B, Hp // m, m, Wp // m, m, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, Hp/m, Wp/m, m, m, C]
        x = x.view(B, (Hp // m) * (Wp // m), m * m * C)  # [B, N', merged_dim]

        x = self.ln_q(x)
        return self.mlp(x)


class PerceiverProjector(nn.Module):
    """Cross-attention (Perceiver) based projector for variable token budget.

    Uses a fixed number of learnable query tokens that cross-attend to ViT
    features, allowing flexible compression ratios independent of image
    resolution.

    Parameters
    ----------
    vit_hidden_size : int
    llm_hidden_size : int
    num_queries : int
        Number of output visual tokens.
    num_heads : int
    """

    def __init__(
        self,
        vit_hidden_size: int = 1024,
        llm_hidden_size: int = 4096,
        num_queries: int = 64,
        num_heads: int = 16,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_hidden_size) * 0.02)
        self.kv_proj = nn.Linear(vit_hidden_size, 2 * llm_hidden_size, bias=False)
        self.o_proj = nn.Linear(llm_hidden_size, llm_hidden_size, bias=False)
        self.scale = (llm_hidden_size // num_heads) ** -0.5
        self.num_heads = num_heads
        self.head_dim = llm_hidden_size // num_heads

    def forward(
        self,
        vit_features: torch.Tensor,
        H: int | None = None,
        W: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            vit_features: ``[B, N_vis, vit_dim]``

        Returns:
            ``[B, num_queries, llm_hidden_size]``
        """
        B, N, _ = vit_features.shape

        q = self.queries.expand(B, -1, -1)  # [B, Q, D]
        kv = self.kv_proj(vit_features)  # [B, N, 2*D]
        k, v = kv.chunk(2, dim=-1)

        # Multi-head
        q = q.view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_max = attn.max(dim=-1, keepdim=True).values
        attn = torch.exp(attn - attn_max)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, self.num_queries, -1)
        return self.o_proj(out)
