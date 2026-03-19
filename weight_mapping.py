"""
Weight-name mapping between HuggingFace Qwen VL checkpoints and our modules.

Each mapping is a dictionary ``{our_param_name: hf_param_name}`` that allows
``from_pretrained`` to load official HF weights directly.

Strategy
--------
1.  Build the mapping programmatically from layer counts / head counts.
2.  Provide helpers ``map_state_dict`` (HF → ours) and ``reverse_map``
    (ours → HF) for bidirectional conversion.
3.  Handle fused QKV projections: HF Qwen checkpoints keep separate
    ``q_proj``, ``k_proj``, ``v_proj``, which matches our layout.

HF naming conventions (Qwen2-VL / 2.5-VL as reference)
-------------------------------------------------------
- ``model.embed_tokens.weight``
- ``model.layers.{i}.self_attn.{q,k,v,o}_proj.weight``
- ``model.layers.{i}.mlp.gate_proj.weight``
- ``model.layers.{i}.mlp.up_proj.weight``
- ``model.layers.{i}.mlp.down_proj.weight``
- ``model.layers.{i}.input_layernorm.weight``
- ``model.layers.{i}.post_attention_layernorm.weight``
- ``model.norm.weight``
- ``lm_head.weight``
- ``visual.patch_embed.proj.{weight,bias}``
- ``visual.blocks.{i}.attn.{qkv,proj}.{weight,bias}``
- ``visual.blocks.{i}.mlp.{fc1,fc2}.{weight,bias}``
- ``visual.blocks.{i}.norm1.{weight,bias}``
- ``visual.blocks.{i}.norm2.{weight,bias}``
- ``visual.merger.{mlp.0,mlp.2,ln_q}.{weight,bias}``
"""

from __future__ import annotations

from typing import Any

import torch


# ──────────────────────────────────────────────────────────────────
#  Building per-model mappings
# ──────────────────────────────────────────────────────────────────

def _vit_mapping(num_layers: int, *, has_3d_patch: bool = False) -> dict[str, str]:
    """Map ViT block parameters: our name → HF name."""
    m: dict[str, str] = {}
    # Patch embedding (2-D)
    m["vit.patch_embed.proj.weight"] = "visual.patch_embed.proj.weight"
    m["vit.patch_embed.proj.bias"] = "visual.patch_embed.proj.bias"

    # Patch embedding (3-D, only for models with rope_dim == 3)
    if has_3d_patch:
        m["vit.patch_embed_3d.proj.weight"] = "visual.patch_embed.proj.weight"
        m["vit.patch_embed_3d.proj.bias"] = "visual.patch_embed.proj.bias"

    for i in range(num_layers):
        ours = f"vit.blocks.{i}"
        hf = f"visual.blocks.{i}"

        # Attention (fused QKV)
        m[f"{ours}.attn.qkv.weight"] = f"{hf}.attn.qkv.weight"
        m[f"{ours}.attn.qkv.bias"] = f"{hf}.attn.qkv.bias"
        m[f"{ours}.attn.proj.weight"] = f"{hf}.attn.proj.weight"
        m[f"{ours}.attn.proj.bias"] = f"{hf}.attn.proj.bias"
        # MLP (fc1 / fc2)
        m[f"{ours}.mlp.fc1.weight"] = f"{hf}.mlp.fc1.weight"
        m[f"{ours}.mlp.fc1.bias"] = f"{hf}.mlp.fc1.bias"
        m[f"{ours}.mlp.fc2.weight"] = f"{hf}.mlp.fc2.weight"
        m[f"{ours}.mlp.fc2.bias"] = f"{hf}.mlp.fc2.bias"
        # LayerNorms (norm1, norm2)
        m[f"{ours}.norm1.weight"] = f"{hf}.norm1.weight"
        m[f"{ours}.norm1.bias"] = f"{hf}.norm1.bias"
        m[f"{ours}.norm2.weight"] = f"{hf}.norm2.weight"
        m[f"{ours}.norm2.bias"] = f"{hf}.norm2.bias"

    return m


def _projector_mapping() -> dict[str, str]:
    """Map Qwen2VL-style spatial merger (projector)."""
    return {
        "projector.ln_q.weight": "visual.merger.ln_q.weight",
        "projector.ln_q.bias": "visual.merger.ln_q.bias",
        "projector.mlp.0.weight": "visual.merger.mlp.0.weight",
        "projector.mlp.0.bias": "visual.merger.mlp.0.bias",
        "projector.mlp.2.weight": "visual.merger.mlp.2.weight",
        "projector.mlp.2.bias": "visual.merger.mlp.2.bias",
    }


def _decoder_mapping(num_layers: int) -> dict[str, str]:
    """Map standard GQA decoder layers (Qwen2-VL / Qwen2.5-VL)."""
    m: dict[str, str] = {}
    # Token embedding
    m["embed_tokens.weight"] = "model.embed_tokens.weight"

    for i in range(num_layers):
        ours = f"decoder.layers.{i}"
        hf = f"model.layers.{i}"

        # GQA attention
        m[f"{ours}.attn_norm.weight"] = f"{hf}.input_layernorm.weight"
        m[f"{ours}.attn.q_proj.weight"] = f"{hf}.self_attn.q_proj.weight"
        m[f"{ours}.attn.q_proj.bias"] = f"{hf}.self_attn.q_proj.bias"
        m[f"{ours}.attn.k_proj.weight"] = f"{hf}.self_attn.k_proj.weight"
        m[f"{ours}.attn.k_proj.bias"] = f"{hf}.self_attn.k_proj.bias"
        m[f"{ours}.attn.v_proj.weight"] = f"{hf}.self_attn.v_proj.weight"
        m[f"{ours}.attn.v_proj.bias"] = f"{hf}.self_attn.v_proj.bias"
        m[f"{ours}.attn.o_proj.weight"] = f"{hf}.self_attn.o_proj.weight"
        # FFN (SwiGLU — separate gate/up/down projections)
        m[f"{ours}.ffn_norm.weight"] = f"{hf}.post_attention_layernorm.weight"
        m[f"{ours}.ffn.gate_proj.weight"] = f"{hf}.mlp.gate_proj.weight"
        m[f"{ours}.ffn.up_proj.weight"] = f"{hf}.mlp.up_proj.weight"
        m[f"{ours}.ffn.down_proj.weight"] = f"{hf}.mlp.down_proj.weight"

    # Final norm
    m["decoder.norm.weight"] = "model.norm.weight"
    # LM head
    m["lm_head.weight"] = "lm_head.weight"

    return m


def _moe_decoder_mapping(num_layers: int, num_experts: int = 8) -> dict[str, str]:
    """Map MoE decoder layers (Qwen3-VL)."""
    m: dict[str, str] = {}
    m["embed_tokens.weight"] = "model.embed_tokens.weight"

    for i in range(num_layers):
        ours = f"layers.{i}"
        hf = f"model.layers.{i}"

        # Attention (same as standard)
        m[f"{ours}.attn_norm.weight"] = f"{hf}.input_layernorm.weight"
        m[f"{ours}.attn.q_proj.weight"] = f"{hf}.self_attn.q_proj.weight"
        m[f"{ours}.attn.q_proj.bias"] = f"{hf}.self_attn.q_proj.bias"
        m[f"{ours}.attn.k_proj.weight"] = f"{hf}.self_attn.k_proj.weight"
        m[f"{ours}.attn.k_proj.bias"] = f"{hf}.self_attn.k_proj.bias"
        m[f"{ours}.attn.v_proj.weight"] = f"{hf}.self_attn.v_proj.weight"
        m[f"{ours}.attn.v_proj.bias"] = f"{hf}.self_attn.v_proj.bias"
        m[f"{ours}.attn.o_proj.weight"] = f"{hf}.self_attn.o_proj.weight"
        # MoE FFN
        m[f"{ours}.ffn_norm.weight"] = f"{hf}.post_attention_layernorm.weight"
        m[f"{ours}.moe.router.gate.weight"] = f"{hf}.mlp.gate.weight"
        # Routed experts
        for e in range(num_experts):
            m[f"{ours}.moe.experts.{e}.gate_proj.weight"] = \
                f"{hf}.mlp.experts.{e}.gate_proj.weight"
            m[f"{ours}.moe.experts.{e}.up_proj.weight"] = \
                f"{hf}.mlp.experts.{e}.up_proj.weight"
            m[f"{ours}.moe.experts.{e}.down_proj.weight"] = \
                f"{hf}.mlp.experts.{e}.down_proj.weight"
        # Shared experts
        m[f"{ours}.moe.shared_experts.0.gate_proj.weight"] = \
            f"{hf}.mlp.shared_expert.gate_proj.weight"
        m[f"{ours}.moe.shared_experts.0.up_proj.weight"] = \
            f"{hf}.mlp.shared_expert.up_proj.weight"
        m[f"{ours}.moe.shared_experts.0.down_proj.weight"] = \
            f"{hf}.mlp.shared_expert.down_proj.weight"

    m["norm.weight"] = "model.norm.weight"
    m["lm_head.weight"] = "lm_head.weight"

    return m


def _hybrid_decoder_mapping(num_layers: int, attn_every: int = 4) -> dict[str, str]:
    """Map hybrid DeltaNet + Gated Attention decoder layers (Qwen3-NeXT)."""
    m: dict[str, str] = {}
    m["embed_tokens.weight"] = "model.embed_tokens.weight"

    for i in range(num_layers):
        ours = f"layers.{i}"
        hf = f"model.layers.{i}"
        is_attn = ((i + 1) % attn_every == 0)

        m[f"{ours}.attn_norm.weight"] = f"{hf}.input_layernorm.weight"

        if is_attn:
            # Gated Attention layer
            m[f"{ours}.attn.q_proj.weight"] = f"{hf}.self_attn.q_proj.weight"
            m[f"{ours}.attn.k_proj.weight"] = f"{hf}.self_attn.k_proj.weight"
            m[f"{ours}.attn.v_proj.weight"] = f"{hf}.self_attn.v_proj.weight"
            m[f"{ours}.attn.o_proj.weight"] = f"{hf}.self_attn.o_proj.weight"
            m[f"{ours}.attn.gate_proj.weight"] = f"{hf}.self_attn.gate_proj.weight"
            m[f"{ours}.attn.gate_proj.bias"] = f"{hf}.self_attn.gate_proj.bias"
            m[f"{ours}.attn.group_norm.weight"] = f"{hf}.self_attn.group_norm.weight"
            m[f"{ours}.attn.group_norm.bias"] = f"{hf}.self_attn.group_norm.bias"
        else:
            # Gated DeltaNet layer
            m[f"{ours}.attn.q_proj.weight"] = f"{hf}.self_attn.q_proj.weight"
            m[f"{ours}.attn.k_proj.weight"] = f"{hf}.self_attn.k_proj.weight"
            m[f"{ours}.attn.v_proj.weight"] = f"{hf}.self_attn.v_proj.weight"
            m[f"{ours}.attn.beta_proj.weight"] = f"{hf}.self_attn.beta_proj.weight"
            m[f"{ours}.attn.beta_proj.bias"] = f"{hf}.self_attn.beta_proj.bias"
            m[f"{ours}.attn.alpha_proj.weight"] = f"{hf}.self_attn.alpha_proj.weight"
            m[f"{ours}.attn.alpha_proj.bias"] = f"{hf}.self_attn.alpha_proj.bias"
            m[f"{ours}.attn.o_proj.weight"] = f"{hf}.self_attn.o_proj.weight"
            m[f"{ours}.attn.group_norm.weight"] = f"{hf}.self_attn.group_norm.weight"
            m[f"{ours}.attn.group_norm.bias"] = f"{hf}.self_attn.group_norm.bias"
            # Short convolutions (q, k, v)
            m[f"{ours}.attn.q_conv.conv.weight"] = f"{hf}.self_attn.q_conv.weight"
            m[f"{ours}.attn.k_conv.conv.weight"] = f"{hf}.self_attn.k_conv.weight"
            m[f"{ours}.attn.v_conv.conv.weight"] = f"{hf}.self_attn.v_conv.weight"

        # FFN (always SwiGLU for hybrid)
        m[f"{ours}.ffn_norm.weight"] = f"{hf}.post_attention_layernorm.weight"
        m[f"{ours}.ffn.gate_proj.weight"] = f"{hf}.mlp.gate_proj.weight"
        m[f"{ours}.ffn.up_proj.weight"] = f"{hf}.mlp.up_proj.weight"
        m[f"{ours}.ffn.down_proj.weight"] = f"{hf}.mlp.down_proj.weight"

    m["norm.weight"] = "model.norm.weight"
    m["lm_head.weight"] = "lm_head.weight"

    return m


# ──────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────

def build_weight_mapping(
    model_type: str,
    num_llm_layers: int,
    num_vit_layers: int,
    *,
    num_experts: int = 0,
    attn_every: int = 4,
) -> dict[str, str]:
    """Build the complete ``{our_name: hf_name}`` weight mapping.

    Parameters
    ----------
    model_type : str
        ``"qwen2_vl"``, ``"qwen25_vl"``, ``"qwen3_vl"``, ``"qwen3_next"``
    num_llm_layers : int
    num_vit_layers : int
    num_experts : int
        Number of routed experts (Qwen3-VL only).
    attn_every : int
        Gated-Attention layer interval (Qwen3-NeXT only).

    Returns
    -------
    dict[str, str]
        Mapping from our parameter names to HF parameter names.
    """
    m: dict[str, str] = {}

    # Vision — Qwen2-VL uses 2-D RoPE (no 3-D patch embed),
    # Qwen2.5-VL and later use 3-D RoPE.
    has_3d = model_type not in ("qwen2_vl",)
    m.update(_vit_mapping(num_vit_layers, has_3d_patch=has_3d))
    m.update(_projector_mapping())

    # Decoder
    if model_type in ("qwen2_vl", "qwen25_vl"):
        m.update(_decoder_mapping(num_llm_layers))
    elif model_type == "qwen3_vl":
        m.update(_moe_decoder_mapping(num_llm_layers, num_experts))
    elif model_type == "qwen3_next":
        m.update(_hybrid_decoder_mapping(num_llm_layers, attn_every))
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return m


def map_state_dict(
    hf_state_dict: dict[str, torch.Tensor],
    mapping: dict[str, str],
    strict: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a HuggingFace state dict to our naming convention.

    Parameters
    ----------
    hf_state_dict : dict
        State dict with HF-style keys.
    mapping : dict
        ``{our_name: hf_name}`` mapping from :func:`build_weight_mapping`.
    strict : bool
        If True, raise :class:`KeyError` when an expected HF key is missing.

    Returns
    -------
    dict[str, torch.Tensor]
        State dict with our parameter names.
    """
    reverse = {v: k for k, v in mapping.items()}
    converted: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for hf_key, tensor in hf_state_dict.items():
        if hf_key in reverse:
            converted[reverse[hf_key]] = tensor
        else:
            unmapped.append(hf_key)

    if strict and unmapped:
        raise KeyError(
            f"The following HF keys were not mapped: {unmapped[:20]}"
        )

    return converted


def reverse_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """Invert the mapping: ``{hf_name: our_name}``."""
    return {v: k for k, v in mapping.items()}
