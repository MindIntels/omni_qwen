"""
Qwen3-NeXT — Hybrid Gated-DeltaNet + Gated-Attention architecture.

Qwen3-NeXT is the first production hybrid-architecture VLM from the Qwen
series, replacing the standard all-attention Transformer decoder with a mix
of:

    - **Gated DeltaNet** layers (majority) — linear attention with
      delta-rule recurrence, O(d²) state, sub-quadratic in sequence length.
    - **Gated Attention** layers (minority, e.g. every 4th layer) — full
      quadratic scaled-dot-product attention that preserves the ability to
      form sharp, long-range pairwise dependencies.

Key design principles
---------------------
1. **Efficiency**: DeltaNet layers have O(S · d²) cost instead of O(S²d),
   making very-long-context inference (128K+) practical.

2. **Quality**: Periodic full-attention layers prevent the quality
   degradation seen in purely-recurrent models, especially for in-context
   learning and precise retrieval tasks.

3. **Output gating** on *both* layer types provides training stability and
   lets the model learn per-feature suppression.

4. **Short convolutions** in DeltaNet layers provide local feature mixing
   similar to the induction-like capabilities in Mamba / RWKV.

5. **Same ViT + projector** front-end as Qwen3-VL; the architectural
   change is confined to the LLM decoder.

Layer schedule example (32 layers)
-----------------------------------
    [DN, DN, DN, GA, DN, DN, DN, GA, DN, DN, DN, GA, ...]
    where DN = Gated DeltaNet, GA = Gated Attention
    → 24 DeltaNet + 8 Gated Attention
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, VisionConfig
from .vit import VisionTransformer
from .visual_projector import Qwen2VLProjector
from .rms_norm import RMSNorm
from .swiglu_mlp import SwiGLUMLP
from .gated_deltanet import GatedDeltaNet
from .gated_attention import GatedSelfAttention
from .rope import MRoPE


class HybridDecoderBlock(nn.Module):
    """A decoder block that uses *either* Gated DeltaNet or Gated Attention.

    Parameters
    ----------
    hidden_size : int
    intermediate_size : int
    use_deltanet : bool
        If True → Gated DeltaNet attention; else → Gated Attention.
    num_q_heads : int
    num_kv_heads : int
    head_dim : int or None
    deltanet_chunk : int
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_deltanet: bool = True,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
        deltanet_chunk: int = 64,
    ):
        super().__init__()
        self.use_deltanet = use_deltanet
        hd = head_dim or hidden_size // num_q_heads

        self.attn_norm = RMSNorm(hidden_size)
        if use_deltanet:
            self.attn = GatedDeltaNet(
                hidden_size, head_dim=hd, num_heads=num_q_heads,
                chunk_size=deltanet_chunk,
            )
        else:
            self.attn = GatedSelfAttention(
                hidden_size, num_q_heads, num_kv_heads, hd,
            )

        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        causal: bool = True,
        state: torch.Tensor | None = None,
        kv_cache: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple | None]:
        """
        Returns:
            ``(output, new_state_or_none, new_kv_or_none)``
        """
        normed = self.attn_norm(x)

        if self.use_deltanet:
            h, new_state = self.attn(normed, state=state)
            new_kv = None
        else:
            h, new_kv = self.attn(
                normed, rope_cos=rope_cos, rope_sin=rope_sin,
                causal=causal, kv_cache=kv_cache,
            )
            new_state = None

        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_state, new_kv


class Qwen3NeXT(nn.Module):
    """Qwen3-NeXT: hybrid DeltaNet + Attention decoder with VL front-end.

    Parameters
    ----------
    vit_hidden : int
    llm_hidden : int
    llm_layers : int
    num_q_heads : int
    num_kv_heads : int
    intermediate_size : int
    vocab_size : int
    attn_every : int
        Place a Gated Attention layer every *attn_every* layers.
    deltanet_chunk : int
    vit_layers : int
    vit_heads : int
    patch_size : int
    """

    def __init__(
        self,
        vit_hidden: int = 1024,
        llm_hidden: int = 4096,
        llm_layers: int = 32,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,
        intermediate_size: int = 11008,
        vocab_size: int = 152064,
        attn_every: int = 4,
        deltanet_chunk: int = 64,
        vit_layers: int = 24,
        vit_heads: int = 16,
        patch_size: int = 14,
    ):
        super().__init__()
        # ---- Vision encoder ----
        self.vit = VisionTransformer(
            patch_size=patch_size, hidden_size=vit_hidden,
            num_heads=vit_heads, num_layers=vit_layers,
            intermediate_size=vit_hidden * 4, rope_dim=3,
        )

        # ---- Projector ----
        self.projector = Qwen2VLProjector(vit_hidden, llm_hidden, merge_size=2)

        # ---- Text embedding ----
        self.embed_tokens = nn.Embedding(vocab_size, llm_hidden)

        # ---- Hybrid decoder ----
        head_dim = llm_hidden // num_q_heads
        layers = []
        for i in range(llm_layers):
            use_dn = ((i + 1) % attn_every != 0)
            layers.append(HybridDecoderBlock(
                llm_hidden, intermediate_size,
                use_deltanet=use_dn,
                num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, deltanet_chunk=deltanet_chunk,
            ))
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(llm_hidden)

        # ---- M-RoPE (only used by Gated Attention layers) ----
        self.mrope = MRoPE(head_dim, base=1000000.0)

        # ---- LM head ----
        self.lm_head = nn.Linear(llm_hidden, vocab_size, bias=False)

        self.attn_every = attn_every

    @property
    def num_deltanet_layers(self) -> int:
        return sum(1 for b in self.layers if b.use_deltanet)

    @property
    def num_attn_layers(self) -> int:
        return sum(1 for b in self.layers if not b.use_deltanet)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            Logits ``[B, S, vocab_size]``.
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

        # M-RoPE (for Gated Attention layers)
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        rope_cos, rope_sin = self.mrope(pos, pos, pos)
        rope_cos = rope_cos.unsqueeze(2)
        rope_sin = rope_sin.unsqueeze(2)

        for layer in self.layers:
            x, _, _ = layer(
                x,
                rope_cos=rope_cos, rope_sin=rope_sin,
                causal=True,
            )

        x = self.norm(x)
        return self.lm_head(x)

    # ----------------------------------------------------------------
    #  Factory methods
    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, config: ModelConfig) -> "Qwen3NeXT":
        """Instantiate from a :class:`ModelConfig`."""
        vc = config.vision_config or VisionConfig(rope_dim=3)
        return cls(
            vit_hidden=vc.hidden_size,
            llm_hidden=config.hidden_size,
            llm_layers=config.num_hidden_layers,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            vocab_size=config.vocab_size,
            attn_every=config.attn_every,
            deltanet_chunk=config.deltanet_chunk_size,
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
    ) -> "Qwen3NeXT":
        """Load model with HF weights."""
        from .weight_mapping import build_weight_mapping, map_state_dict
        from .loading_utils import load_hf_state_dict, tie_weights

        config = ModelConfig.from_pretrained(path_or_id)
        model = cls.from_config(config)

        vc = config.vision_config or VisionConfig(rope_dim=3)
        mapping = build_weight_mapping(
            "qwen3_next",
            config.num_hidden_layers,
            vc.num_hidden_layers,
            attn_every=config.attn_every,
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
