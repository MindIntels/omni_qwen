"""
Mixture-of-Experts (MoE) with Shared Experts — Qwen3 style.

Qwen3 / Qwen3-VL upgrades the dense FFN to a **sparse MoE** layer:

    MoE(x) = SharedExpert(x) + Σ_k  g_k · Expert_k(x)

Key design choices
------------------
1. **Shared experts** (always active) provide a dense baseline signal,
   ensuring no token is "dropped" entirely.  This stabilises training
   and follows the DeepSeek-MoE / Qwen3 design.

2. **Top-K routing**: a gating network produces logits over N experts;
   only the top-K experts per token are activated.

3. **Load-balancing auxiliary loss**: encourages uniform expert load
   distribution and prevents collapse to a few experts.

4. Each expert is a standard SwiGLU FFN.

Architecture example (Qwen3-235B-A22B)
--------------------------------------
- 128 routed experts, 1 shared expert
- Top-8 routing (8 active experts per token)
- Expert intermediate size ≈ 2048  (vs ≈ 18944 for the shared expert)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swiglu_mlp import SwiGLUMLP


class TopKRouter(nn.Module):
    """Token-level top-K expert router.

    Parameters
    ----------
    hidden_size : int
    num_experts : int
        Total number of routed experts.
    top_k : int
        Number of experts activated per token.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route each token to top-K experts.

        Args:
            x: ``[B*S, hidden_size]`` (tokens flattened).

        Returns:
            router_logits: ``[B*S, num_experts]`` raw logits.
            topk_indices:  ``[B*S, top_k]`` selected expert IDs.
            topk_weights:  ``[B*S, top_k]`` gating weights (softmax over selected).
        """
        logits = self.gate(x)  # [N, E]

        # Top-K selection
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax over selected experts only
        topk_weights = F.softmax(topk_logits, dim=-1)

        return logits, topk_indices, topk_weights


def load_balancing_loss(
    router_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Auxiliary load-balancing loss.

    Encourages uniform expert utilization:

        L_aux = N · Σ_e  f_e · P_e

    where f_e = fraction of tokens routed to expert e,
          P_e = mean routing probability to expert e.

    Returns:
        Scalar loss (multiply by a small coefficient, e.g. 0.01).
    """
    N = router_logits.size(0)
    E = num_experts

    # f_e: token fraction per expert
    one_hot = F.one_hot(topk_indices, E).float().sum(dim=1)  # [N, E]
    f = one_hot.mean(dim=0)  # [E]

    # P_e: mean routing probability per expert
    probs = F.softmax(router_logits, dim=-1)  # [N, E]
    P = probs.mean(dim=0)  # [E]

    return E * (f * P).sum()


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with shared expert(s).

    Parameters
    ----------
    hidden_size : int
    intermediate_size : int
        Per-expert intermediate dimension.
    num_routed_experts : int
    num_shared_experts : int
    shared_intermediate_size : int
        Intermediate dimension for shared expert(s).
    top_k : int
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_routed_experts: int = 8,
        num_shared_experts: int = 1,
        shared_intermediate_size: int | None = None,
        top_k: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k

        # Router
        self.router = TopKRouter(hidden_size, num_routed_experts, top_k)

        # Routed experts
        self.experts = nn.ModuleList([
            SwiGLUMLP(hidden_size, intermediate_size)
            for _ in range(num_routed_experts)
        ])

        # Shared expert(s) — always active
        s_int = shared_intermediate_size or intermediate_size
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLUMLP(hidden_size, s_int)
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = nn.ModuleList()

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, S, hidden_size]``

        Returns:
            ``(output, aux_loss)`` where output has the same shape as x,
            and aux_loss is the load-balancing scalar.
        """
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # Route
        router_logits, topk_indices, topk_weights = self.router(x_flat)
        aux_loss = load_balancing_loss(router_logits, topk_indices, self.num_routed_experts)

        # Compute routed expert outputs (loop-based for clarity)
        routed_output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]   # [N]
            weight = topk_weights[:, k]        # [N]

            for e in range(self.num_routed_experts):
                mask = expert_idx == e
                if mask.any():
                    tokens = x_flat[mask]
                    expert_out = self.experts[e](tokens)
                    routed_output[mask] += weight[mask].unsqueeze(-1) * expert_out

        # Shared experts
        shared_output = torch.zeros_like(x_flat)
        for se in self.shared_experts:
            shared_output = shared_output + se(x_flat)

        output = (routed_output + shared_output).view(B, S, D)
        return output, aux_loss

    def extra_repr(self) -> str:
        return (
            f"routed_experts={self.num_routed_experts}, "
            f"shared_experts={self.num_shared_experts}, top_k={self.top_k}"
        )
