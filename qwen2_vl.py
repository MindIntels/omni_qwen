"""
Qwen2-VL — Vision-Language model (first generation).

Architecture
------------
    Image → ViT (2-D RoPE + Window Attention) → Spatial Merger →
    [ visual tokens ‖ text tokens ] → LLM Decoder (M-RoPE + GQA)

Key innovations (vs. generic VLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Dynamic resolution**: images are processed at their native aspect
   ratio.  Patch tokens form a variable-size ``(H, W)`` grid.
2. **2-D RoPE in ViT**: spatial awareness without learned position
   embeddings; generalises to unseen resolutions at inference.
3. **M-RoPE in decoder**: multimodal position encoding that keeps text
   sequential and image tokens spatially indexed.
4. **2×2 spatial token merging**: reduces the visual sequence length
   by 4× before entering the LLM, balancing cost and detail.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, VisionConfig
from .vit import VisionTransformer
from .visual_projector import Qwen2VLProjector
from .decoder import DecoderBackbone
from .rope import MRoPE, RotaryEmbedding1D
from .rms_norm import RMSNorm


class Qwen2VL(nn.Module):
    """Qwen2-VL model (simplified reference implementation).

    Parameters
    ----------
    vit_hidden : int
        ViT hidden dimension.
    llm_hidden : int
        LLM decoder hidden dimension.
    llm_layers : int
    num_q_heads : int
    num_kv_heads : int
    intermediate_size : int
    vocab_size : int
    vit_layers : int
    vit_heads : int
    patch_size : int
    window_size : tuple[int, int]
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
        vit_layers: int = 24,
        vit_heads: int = 16,
        patch_size: int = 14,
        window_size: tuple[int, int] = (7, 7),
    ):
        super().__init__()
        # ---- Vision encoder ----
        self.vit = VisionTransformer(
            patch_size=patch_size,
            hidden_size=vit_hidden,
            num_heads=vit_heads,
            num_layers=vit_layers,
            intermediate_size=vit_hidden * 4,
            window_size=window_size,
            rope_dim=2,  # 2-D RoPE
        )

        # ---- Vision-Language projector ----
        self.projector = Qwen2VLProjector(vit_hidden, llm_hidden, merge_size=2)

        # ---- Text embedding ----
        self.embed_tokens = nn.Embedding(vocab_size, llm_hidden)

        # ---- LLM decoder ----
        head_dim = llm_hidden // num_q_heads
        self.decoder = DecoderBackbone(
            llm_layers, llm_hidden, num_q_heads, num_kv_heads,
            intermediate_size, head_dim,
        )

        # ---- M-RoPE for decoder ----
        self.mrope = MRoPE(head_dim, base=1000000.0)

        # ---- LM head ----
        self.lm_head = nn.Linear(llm_hidden, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Simplified forward (no KV-cache, no generation logic).

        Args:
            input_ids: ``[B, S_text]`` text token IDs.
            pixel_values: ``[B, C, img_H, img_W]`` image pixels.
            image_grid_hw: ``(H_patches, W_patches)`` from the ViT.

        Returns:
            Logits ``[B, S_total, vocab_size]``.
        """
        embeddings = []

        # Vision path
        if pixel_values is not None:
            vis_features = self.vit(pixel_values)  # [B, N, vit_hidden]
            B = pixel_values.size(0)
            Hp = pixel_values.size(2) // self.vit.patch_embed.patch_size
            Wp = pixel_values.size(3) // self.vit.patch_embed.patch_size
            vis_tokens = self.projector(vis_features, Hp, Wp)  # [B, N', llm_hidden]
            embeddings.append(vis_tokens)

        # Text path
        if input_ids is not None:
            text_emb = self.embed_tokens(input_ids)  # [B, S_text, llm_hidden]
            embeddings.append(text_emb)

        x = torch.cat(embeddings, dim=1) if len(embeddings) > 1 else embeddings[0]
        B, S, D = x.shape

        # M-RoPE: for simplicity, use sequential positions for all tokens
        # (in production, image tokens get 2-D spatial positions)
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        rope_cos, rope_sin = self.mrope(pos, pos, pos)
        rope_cos = rope_cos.unsqueeze(2)  # [B, S, 1, D]
        rope_sin = rope_sin.unsqueeze(2)

        h, _ = self.decoder(x, rope_cos, rope_sin, causal=True)
        return self.lm_head(h)

    # ----------------------------------------------------------------
    #  Factory methods
    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, config: ModelConfig) -> "Qwen2VL":
        """Instantiate from a :class:`ModelConfig`."""
        vc = config.vision_config or VisionConfig(rope_dim=2)
        return cls(
            vit_hidden=vc.hidden_size,
            llm_hidden=config.hidden_size,
            llm_layers=config.num_hidden_layers,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            vocab_size=config.vocab_size,
            vit_layers=vc.num_hidden_layers,
            vit_heads=vc.num_attention_heads,
            patch_size=vc.patch_size,
            window_size=vc.window_size if vc.window_size != (0, 0) else (7, 7),
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str,
        *,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "Qwen2VL":
        """Load model with HF weights.

        Parameters
        ----------
        path_or_id : str
            Local checkpoint directory or HuggingFace model ID.
        device : str
            Target device.
        dtype : torch.dtype or None
            Weight dtype (e.g. ``torch.float16``).
        """
        from .weight_mapping import build_weight_mapping, map_state_dict
        from .loading_utils import load_hf_state_dict, tie_weights
        from pathlib import Path

        config = ModelConfig.from_pretrained(path_or_id)
        model = cls.from_config(config)

        vc = config.vision_config or VisionConfig(rope_dim=2)
        mapping = build_weight_mapping(
            "qwen2_vl",
            config.num_hidden_layers,
            vc.num_hidden_layers,
        )

        # Load state dict from safetensors / pytorch files
        hf_sd = load_hf_state_dict(path_or_id)
        our_sd = map_state_dict(hf_sd, mapping, strict=False)
        model.load_state_dict(our_sd, strict=False)
        tie_weights(model, config)

        if dtype is not None:
            model = model.to(dtype=dtype)
        model = model.to(device=device)
        model.eval()
        return model
