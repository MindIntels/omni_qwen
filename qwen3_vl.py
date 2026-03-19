"""
Qwen3-VL — Third-generation VL model with Mixture-of-Experts decoder.

Improvements over Qwen2.5-VL
-----------------------------
1. **MoE decoder**: Dense FFN replaced by sparse MoE layers with shared
   experts.  This scales model capacity (total params) while keeping
   inference cost bounded by the number of **active** parameters.

2. **Thinking mode (optional)**: Qwen3 supports chain-of-thought
   "thinking" within special tokens.

3. **Improved vision encoder**: Carries over 3-D RoPE ViT; may add a
   perceiver-style projector for more aggressive visual token compression.

Architecture
------------
    Image/Video → ViT (3-D RoPE) → Visual Projector →
    [ visual tokens ‖ text tokens ] → MoE Decoder (M-RoPE + GQA + MoE FFN)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, VisionConfig
from .vit import VisionTransformer
from .visual_projector import Qwen2VLProjector, PerceiverProjector
from .decoder import GQASelfAttention
from .rms_norm import RMSNorm
from .swiglu_mlp import SwiGLUMLP
from .moe import MoELayer
from .rope import MRoPE, apply_rotary_emb


class MoEDecoderBlock(nn.Module):
    """Decoder block with MoE FFN instead of dense FFN.

    Parameters
    ----------
    hidden_size : int
    num_q_heads : int
    num_kv_heads : int
    expert_intermediate : int
        Per-expert intermediate dimension.
    num_routed_experts : int
    num_shared_experts : int
    shared_intermediate : int
    top_k : int
    head_dim : int or None
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        expert_intermediate: int = 2048,
        num_routed_experts: int = 8,
        num_shared_experts: int = 1,
        shared_intermediate: int = 11008,
        top_k: int = 2,
        head_dim: int | None = None,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = GQASelfAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            qkv_bias=qkv_bias,
        )
        self.ffn_norm = RMSNorm(hidden_size)
        self.moe = MoELayer(
            hidden_size, expert_intermediate,
            num_routed_experts, num_shared_experts,
            shared_intermediate, top_k,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        causal: bool = True,
        kv_cache=None,
    ) -> tuple[torch.Tensor, tuple, torch.Tensor]:
        h, new_kv = self.attn(self.attn_norm(x), rope_cos, rope_sin, causal, kv_cache)
        x = x + h
        moe_out, aux_loss = self.moe(self.ffn_norm(x))
        x = x + moe_out
        return x, new_kv, aux_loss


class Qwen3VL(nn.Module):
    """Qwen3-VL with MoE decoder.

    Parameters
    ----------
    (Simplified parameter set — see docstrings of subcomponents for details.)
    """

    def __init__(
        self,
        vit_hidden: int = 1024,
        llm_hidden: int = 4096,
        llm_layers: int = 32,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,
        vocab_size: int = 152064,
        # MoE params
        num_routed_experts: int = 8,
        num_shared_experts: int = 1,
        expert_intermediate: int = 2048,
        shared_intermediate: int = 11008,
        top_k: int = 2,
        # ViT params
        vit_layers: int = 24,
        vit_heads: int = 16,
        patch_size: int = 14,
    ):
        super().__init__()
        # ---- Vision encoder ----
        self.vit = VisionTransformer(
            patch_size=patch_size, hidden_size=vit_hidden,
            num_heads=vit_heads, num_layers=vit_layers,
            intermediate_size=vit_hidden * 4,
            rope_dim=3,
        )

        # ---- Projector ----
        self.projector = Qwen2VLProjector(vit_hidden, llm_hidden, merge_size=2)

        # ---- Text embedding ----
        self.embed_tokens = nn.Embedding(vocab_size, llm_hidden)

        # ---- MoE Decoder blocks ----
        head_dim = llm_hidden // num_q_heads
        self.layers = nn.ModuleList([
            MoEDecoderBlock(
                llm_hidden, num_q_heads, num_kv_heads,
                expert_intermediate, num_routed_experts,
                num_shared_experts, shared_intermediate, top_k, head_dim,
            )
            for _ in range(llm_layers)
        ])
        self.norm = RMSNorm(llm_hidden)

        # ---- M-RoPE ----
        self.mrope = MRoPE(head_dim, base=1000000.0)

        # ---- LM head ----
        self.lm_head = nn.Linear(llm_hidden, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ``(logits, total_aux_loss)``
        """
        embeddings = []

        if pixel_values is not None:
            vis_features = self.vit(pixel_values)
            B = pixel_values.size(0)
            Hp = pixel_values.size(2) // self.vit.patch_embed.patch_size
            Wp = pixel_values.size(3) // self.vit.patch_embed.patch_size
            vis_tokens = self.projector(vis_features, Hp, Wp)
            embeddings.append(vis_tokens)

        if input_ids is not None:
            embeddings.append(self.embed_tokens(input_ids))

        x = torch.cat(embeddings, dim=1) if len(embeddings) > 1 else embeddings[0]
        B, S, D = x.shape

        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        rope_cos, rope_sin = self.mrope(pos, pos, pos)
        rope_cos = rope_cos.unsqueeze(2)
        rope_sin = rope_sin.unsqueeze(2)

        total_aux = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, aux = layer(x, rope_cos, rope_sin, causal=True)
            total_aux = total_aux + aux

        x = self.norm(x)
        return self.lm_head(x), total_aux

    # ----------------------------------------------------------------
    #  Factory methods
    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, config: ModelConfig) -> "Qwen3VL":
        """Instantiate from a :class:`ModelConfig`."""
        vc = config.vision_config or VisionConfig(rope_dim=3)
        return cls(
            vit_hidden=vc.hidden_size,
            llm_hidden=config.hidden_size,
            llm_layers=config.num_hidden_layers,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            num_routed_experts=config.num_experts,
            num_shared_experts=config.num_shared_experts,
            expert_intermediate=config.expert_intermediate_size,
            shared_intermediate=config.shared_expert_intermediate_size
                                or config.intermediate_size,
            top_k=config.num_experts_per_tok,
            vit_layers=vc.num_hidden_layers,
            vit_heads=vc.num_attention_heads,
            patch_size=vc.patch_size,
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str,
        *,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "Qwen3VL":
        """Load model with HF weights."""
        from .weight_mapping import build_weight_mapping, map_state_dict
        from .loading_utils import load_hf_state_dict, tie_weights

        config = ModelConfig.from_pretrained(path_or_id)
        model = cls.from_config(config)

        vc = config.vision_config or VisionConfig(rope_dim=3)
        mapping = build_weight_mapping(
            "qwen3_vl",
            config.num_hidden_layers,
            vc.num_hidden_layers,
            num_experts=config.num_experts,
        )

        hf_sd = load_hf_state_dict(path_or_id)
        our_sd = map_state_dict(hf_sd, mapping, strict=False)
        model.load_state_dict(our_sd, strict=False)
        tie_weights(model, config)

        if dtype is not None:
            model = model.to(dtype=dtype)
        model = model.to(device=device)
        model.eval()
        return model
