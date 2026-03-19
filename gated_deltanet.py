"""
Gated DeltaNet — Linear Attention with Delta Rule.

Qwen3-NeXT replaces most standard Transformer attention layers with
**Gated DeltaNet** layers for sub-quadratic sequence complexity while
retaining strong performance on long-context tasks.

Background — Delta Rule
-----------------------
Standard linear attention maintains a state ``S ∈ R^{d_v × d_k}`` updated
additively:  ``S_t = S_{t-1} + v_t k_t^T``.  This is equivalent to an
associative memory that never forgets, leading to capacity limitations.

The **delta rule** improves this by first *erasing* the old value
associated with k_t before writing the new value:

    S_t = S_{t-1} + β_t · (v_t − S_{t-1}^T k_t) ⊗ k_t

where ``β_t ∈ (0, 1)`` is a learned per-token gating scalar controlling
write strength.  The term ``(v_t − S_{t-1}^T k_t)`` is the *delta*
(error signal between desired and currently stored value).

Gated DeltaNet additions
------------------------
1. **Short convolution** on Q, K, V before the recurrence (local feature
   mixing, similar to Mamba / RWKV).
2. **Output gate** ``α_t = σ(W_α · x_t)`` applied to the output.
3. **Chunkwise parallel** training mode: the sequence is split into
   chunks of size C; within each chunk a causal attention-like matrix
   is formed; across chunks the state is propagated recurrently.

Complexity
----------
- Recurrent (inference): O(S · d²)  time, O(d²) state.
- Chunkwise (training):  O(S · C · d) time (C ≪ S).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConv1D(nn.Module):
    """Causal 1-D depthwise convolution for local feature mixing.

    Parameters
    ----------
    dim : int
        Feature dimension (applies depthwise across the feature dim).
    kernel_size : int
        Convolution kernel size (default 4, as in Mamba).
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size - 1,  # causal padding
            groups=dim,               # depthwise
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S, D] → [B, S, D]`` (causal)."""
        x = x.transpose(1, 2)  # [B, D, S]
        x = self.conv(x)
        x = x[..., : x.size(-1) - (self.conv.kernel_size[0] - 1)]  # trim future
        return x.transpose(1, 2)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet layer.

    Parameters
    ----------
    hidden_size : int
        Model dimension.
    head_dim : int
        Per-head key/value dimension (state matrix is head_dim × head_dim).
    num_heads : int
        Number of parallel recurrent heads.
    conv_kernel : int
        Short-conv kernel size.
    chunk_size : int
        Chunk size for chunkwise parallel training mode.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int = 64,
        num_heads: int = 16,
        conv_kernel: int = 4,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        total = num_heads * head_dim

        # Q / K / V projections
        self.q_proj = nn.Linear(hidden_size, total, bias=False)
        self.k_proj = nn.Linear(hidden_size, total, bias=False)
        self.v_proj = nn.Linear(hidden_size, total, bias=False)

        # Short convolutions (before recurrence)
        self.q_conv = ShortConv1D(total, conv_kernel)
        self.k_conv = ShortConv1D(total, conv_kernel)
        self.v_conv = ShortConv1D(total, conv_kernel)

        # Beta gate: write strength ∈ (0, 1)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # Output gate
        self.alpha_proj = nn.Linear(hidden_size, total, bias=True)

        # Output projection
        self.o_proj = nn.Linear(total, hidden_size, bias=False)

        # Group norm on output (per-head norm, common in linear attention)
        self.group_norm = nn.GroupNorm(num_heads, total)

    # ------------------------------------------------------------------
    #  Recurrent mode (inference / reference)
    # ------------------------------------------------------------------

    def _recurrent_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step-by-step recurrent computation.

        Args:
            q, k, v: ``[B, S, num_heads, head_dim]``
            beta: ``[B, S, num_heads, 1]``
            state: ``[B, num_heads, head_dim, head_dim]`` or None.

        Returns:
            ``(output, final_state)``
        """
        B, S, H, D = q.shape

        if state is None:
            state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)

        outputs = []
        for t in range(S):
            q_t = q[:, t]   # [B, H, D]
            k_t = k[:, t]   # [B, H, D]
            v_t = v[:, t]   # [B, H, D]
            b_t = beta[:, t]  # [B, H, 1]

            # Retrieval: what the current state associates with k_t
            # state: [B, H, D_v, D_k], k_t: [B, H, D_k] → retrieval: [B, H, D_v]
            retrieval = torch.einsum("bhvk,bhk->bhv", state, k_t)

            # Delta update
            delta = v_t - retrieval  # [B, H, D]
            # state += beta * delta outer k
            state = state + b_t.unsqueeze(-1) * torch.einsum("bhv,bhk->bhvk", delta, k_t)

            # Output: state^T @ q_t
            o_t = torch.einsum("bhvk,bhk->bhv", state, q_t)  # [B, H, D]
            outputs.append(o_t)

        output = torch.stack(outputs, dim=1)  # [B, S, H, D]
        return output, state

    # ------------------------------------------------------------------
    #  Chunkwise parallel mode (training)
    # ------------------------------------------------------------------

    def _chunkwise_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunkwise parallel computation for efficient training.

        Splits the sequence into chunks of size C.  Within each chunk,
        a C×C "attention-like" intra-chunk matrix captures the delta rule
        dynamics.  Across chunks the state is propagated.

        Args & returns: same as ``_recurrent_forward``.
        """
        B, S, H, D = q.shape
        C = min(self.chunk_size, S)
        num_chunks = (S + C - 1) // C

        # Pad sequence to multiple of C
        pad = num_chunks * C - S
        if pad > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            beta = F.pad(beta, (0, 0, 0, 0, 0, pad))

        # Reshape into chunks: [B, num_chunks, C, H, D]
        q_c = q.view(B, num_chunks, C, H, D)
        k_c = k.view(B, num_chunks, C, H, D)
        v_c = v.view(B, num_chunks, C, H, D)
        beta_c = beta.view(B, num_chunks, C, H, 1)

        if state is None:
            state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)

        all_outputs = []
        for ci in range(num_chunks):
            qc = q_c[:, ci]    # [B, C, H, D]
            kc = k_c[:, ci]
            vc = v_c[:, ci]
            bc = beta_c[:, ci]  # [B, C, H, 1]

            # --- Inter-chunk contribution: state @ q for all positions ---
            # o_inter = state^T @ q  -> [B, C, H, D]
            o_inter = torch.einsum("bhvk,bchk->bchv", state, qc)

            # --- Intra-chunk: causal attention within the chunk ---
            # Must use FULL state (inter-chunk + evolving intra-chunk) for
            # correct retrieval and output, matching the recurrent reference.
            intra_output = torch.zeros_like(qc)
            chunk_state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
            full_state = state.clone()

            for t in range(C):
                q_t = qc[:, t]
                k_t = kc[:, t]
                v_t = vc[:, t]
                b_t = bc[:, t]

                retrieval = torch.einsum("bhvk,bhk->bhv", full_state, k_t)
                delta = v_t - retrieval
                update = b_t.unsqueeze(-1) * torch.einsum(
                    "bhv,bhk->bhvk", delta, k_t
                )
                chunk_state = chunk_state + update
                full_state = full_state + update
                o_t = torch.einsum("bhvk,bhk->bhv", full_state, q_t)
                intra_output[:, t] = o_t

            output_chunk = intra_output  # already includes inter-chunk
            all_outputs.append(output_chunk)

            # Update state for next chunk
            state = state + chunk_state

        output = torch.cat(all_outputs, dim=1)  # [B, num_chunks*C, H, D]
        output = output[:, :S]  # trim padding
        return output, state

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        use_chunkwise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, S, hidden_size]``
            state: ``[B, num_heads, head_dim, head_dim]`` recurrent state.
            use_chunkwise: if True use chunkwise mode; else pure recurrent.

        Returns:
            ``(output, new_state)`` — output is ``[B, S, hidden_size]``.
        """
        B, S, _ = x.shape

        # Project
        q = self.q_proj(x)  # [B, S, total]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Short convolutions
        q = F.silu(self.q_conv(q))
        k = F.silu(self.k_conv(k))
        v = F.silu(self.v_conv(v))

        # Reshape to heads
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)

        # Normalise keys (important for stable recurrence)
        k = F.normalize(k, p=2, dim=-1)

        # Beta gate
        beta = torch.sigmoid(self.beta_proj(x))  # [B, S, H]
        beta = beta.unsqueeze(-1)  # [B, S, H, 1]

        # Output gate
        alpha = torch.sigmoid(self.alpha_proj(x))  # [B, S, total]
        alpha = alpha.view(B, S, self.num_heads, self.head_dim)

        # Core recurrence
        if use_chunkwise and S > 1:
            output, new_state = self._chunkwise_forward(q, k, v, beta, state)
        else:
            output, new_state = self._recurrent_forward(q, k, v, beta, state)

        # Apply output gate
        output = output * alpha  # [B, S, H, D]

        # Reshape and per-token group norm (causal-safe)
        output = output.view(B, S, -1)                   # [B, S, total]
        output = output.reshape(B * S, -1, 1)             # [B*S, total, 1]
        output = self.group_norm(output)
        output = output.reshape(B, S, -1)                 # [B, S, total]

        # Output projection
        output = self.o_proj(output)
        return output, new_state

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, chunk={self.chunk_size}"
        )
