"""
Text generation utilities for Qwen VL models.

Provides:
- ``generate()``         — greedy / top-k / top-p decoding (blocking)
- ``generate_stream()``  — same but yields tokens one-at-a-time (streaming)

These work with **any** of the four model variants (Qwen2VL, Qwen25VL,
Qwen3VL, Qwen3NeXT) because they only depend on the common interface:

    model(input_ids=..., pixel_values=...) → logits  (or (logits, aux))
"""

from __future__ import annotations

from typing import Generator

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
#  Sampling helpers
# ──────────────────────────────────────────────────────────────────

def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample a single token from the last-position logits.

    Parameters
    ----------
    logits : Tensor
        ``[B, vocab_size]`` logits at the *last* sequence position.
    temperature : float
        Softmax temperature.  ``0`` → greedy.
    top_k : int
        If >0, keep only the top-k logits.
    top_p : float
        Nucleus sampling threshold.

    Returns
    -------
    Tensor  ``[B, 1]``
    """
    if temperature == 0 or temperature < 1e-8:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative probability above the threshold
        removed = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(removed, float("-inf"))
        # Scatter back to original ordering
        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ──────────────────────────────────────────────────────────────────
#  Forward helper (handles (logits,) vs (logits, aux) return types)
# ──────────────────────────────────────────────────────────────────

def _model_forward(model, **kwargs) -> torch.Tensor:
    """Run model forward and extract logits (handles MoE aux loss tuple)."""
    out = model(**kwargs)
    if isinstance(out, tuple):
        return out[0]  # (logits, aux_loss)
    return out


# ──────────────────────────────────────────────────────────────────
#  generate()
# ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    pixel_values: torch.Tensor | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stop_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Auto-regressive generation (blocking).

    Parameters
    ----------
    model : nn.Module
        Any Qwen VL model instance.
    input_ids : Tensor
        ``[B, S]`` prompt token IDs.
    pixel_values : Tensor or None
        Optional image input ``[B, C, H, W]``.
    max_new_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature.  0 = greedy.
    top_k : int
    top_p : float
    stop_token_ids : set[int] or None
        Token IDs that trigger early stopping (e.g. EOS).

    Returns
    -------
    Tensor  ``[B, S + generated_len]``
        Full sequence including prompt and generated tokens.
    """
    if stop_token_ids is None:
        stop_token_ids = set()

    generated = input_ids.clone()
    B = input_ids.size(0)

    # First forward with optional image
    fwd_kwargs: dict = {"input_ids": generated}
    if pixel_values is not None:
        fwd_kwargs["pixel_values"] = pixel_values

    for step in range(max_new_tokens):
        logits = _model_forward(model, **fwd_kwargs)
        next_logits = logits[:, -1, :]  # [B, vocab]
        next_token = _sample_next_token(
            next_logits, temperature, top_k, top_p,
        )  # [B, 1]

        generated = torch.cat([generated, next_token], dim=1)

        # After first step, don't send pixel_values again
        fwd_kwargs = {"input_ids": generated}

        # Check stop condition (all batches)
        if stop_token_ids:
            last_tokens = next_token.squeeze(-1).tolist()
            if all(t in stop_token_ids for t in last_tokens):
                break

    return generated


# ──────────────────────────────────────────────────────────────────
#  generate_stream()
# ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_stream(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    pixel_values: torch.Tensor | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stop_token_ids: set[int] | None = None,
) -> Generator[int, None, None]:
    """Auto-regressive generation that yields one token at a time.

    Same interface as :func:`generate` but yields each generated token
    ID as it is produced (batch size must be 1).

    Yields
    ------
    int
        Generated token ID.
    """
    if stop_token_ids is None:
        stop_token_ids = set()

    assert input_ids.size(0) == 1, "Streaming generation requires batch_size=1"

    generated = input_ids.clone()

    fwd_kwargs: dict = {"input_ids": generated}
    if pixel_values is not None:
        fwd_kwargs["pixel_values"] = pixel_values

    for step in range(max_new_tokens):
        logits = _model_forward(model, **fwd_kwargs)
        next_logits = logits[:, -1, :]
        next_token = _sample_next_token(
            next_logits, temperature, top_k, top_p,
        )

        token_id = next_token.item()
        generated = torch.cat([generated, next_token], dim=1)
        fwd_kwargs = {"input_ids": generated}

        yield token_id

        if token_id in stop_token_ids:
            return
