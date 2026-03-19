"""
RMSNorm — Root Mean Square Layer Normalization.

Used throughout all Qwen models (ViT, decoder, MoE experts) as the default
normalization layer.  Compared to LayerNorm it drops the mean-centering step,
which is slightly more GPU-friendly and empirically works equally well for
Transformer LLMs.

    RMSNorm(x) = x / RMS(x) * γ
    RMS(x)     = sqrt(mean(x²) + ε)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Parameters
    ----------
    hidden_size : int
        Feature dimension (last dim of input tensor).
    eps : float
        Small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
