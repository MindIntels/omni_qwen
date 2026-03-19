"""
SwiGLU Feed-Forward Network.

All Qwen models (decoder layers, ViT, MoE experts) use the SwiGLU variant
of the feed-forward block:

    FFN(x) = W_down · ( SiLU(W_gate · x) ⊙ W_up · x )

where ⊙ is element-wise multiplication and SiLU(x) = x · σ(x).

The intermediate dimension is typically ``hidden_size * 8/3`` rounded to a
multiple of 128 (for tensor-core alignment).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network.

    Parameters
    ----------
    hidden_size : int
        Input / output dimension.
    intermediate_size : int
        Size of the gated intermediate representation.
    bias : bool
        Whether linear layers include bias terms.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S, hidden] → [B, S, hidden]``."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, "
                f"intermediate_size={self.intermediate_size}")
