"""
Qwen2.5-VL — Second-generation Vision-Language model.

Improvements over Qwen2-VL
---------------------------
1. **3-D RoPE in ViT**: Extends 2-D spatial RoPE to ``(time, height, width)``
   enabling native **video** understanding with temporal position awareness.
   Each video frame is processed by the same ViT and the 3-D RoPE encodes
   its (frame_idx, row, col) coordinate.

2. **Dynamic FPS for video**: Frames are sampled at varying rates based on
   content dynamics, and the 3-D RoPE temporal positions reflect actual
   timestamps (not uniform indices).

3. **Improved token merging**: The spatial merger now handles variable
   aspect ratios more gracefully and can adapt the merge factor.

4. **Larger decoder**: Qwen2.5-VL comes in 3B / 7B / 72B sizes with
   improved pre-training (more multimodal data, better OCR, spatial
   grounding).

Architecture
------------
    Image/Video → ViT (3-D RoPE + Window Attention) → Spatial Merger →
    [ visual tokens ‖ text tokens ] → LLM Decoder (M-RoPE + GQA)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, VisionConfig
from .vit import VisionTransformer
from .visual_projector import Qwen2VLProjector
from .decoder import DecoderBackbone
from .rope import MRoPE
from .rms_norm import RMSNorm


class Qwen25VL(nn.Module):
    """Qwen2.5-VL model (simplified reference implementation).

    Parameters mirror ``Qwen2VL`` with the addition of ``rope_dim=3``
    in the ViT and support for video frame indices.
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
        # ---- Vision encoder (3-D RoPE!) ----
        self.vit = VisionTransformer(
            patch_size=patch_size,
            hidden_size=vit_hidden,
            num_heads=vit_heads,
            num_layers=vit_layers,
            intermediate_size=vit_hidden * 4,
            window_size=window_size,
            rope_dim=3,  # ← key difference: 3-D RoPE for video
        )

        # ---- Projector ----
        self.projector = Qwen2VLProjector(vit_hidden, llm_hidden, merge_size=2)

        # ---- Text embedding ----
        self.embed_tokens = nn.Embedding(vocab_size, llm_hidden)

        # ---- Decoder ----
        head_dim = llm_hidden // num_q_heads
        self.decoder = DecoderBackbone(
            llm_layers, llm_hidden, num_q_heads, num_kv_heads,
            intermediate_size, head_dim,
        )

        # ---- M-RoPE ----
        self.mrope = MRoPE(head_dim, base=1000000.0)

        # ---- LM head ----
        self.lm_head = nn.Linear(llm_hidden, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        video_frame_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: ``[B, S_text]``
            pixel_values: ``[B, C, img_H, img_W]`` (or video frame).
            video_frame_ids: ``[B, num_patches]`` temporal frame indices
                (for 3-D RoPE in ViT).

        Returns:
            Logits ``[B, S_total, vocab_size]``.
        """
        embeddings = []

        if pixel_values is not None:
            vis_features = self.vit(pixel_values, time_ids=video_frame_ids)
            B = pixel_values.size(0)
            Hp = pixel_values.size(2) // self.vit.patch_embed.patch_size
            Wp = pixel_values.size(3) // self.vit.patch_embed.patch_size
            vis_tokens = self.projector(vis_features, Hp, Wp)
            embeddings.append(vis_tokens)

        if input_ids is not None:
            text_emb = self.embed_tokens(input_ids)
            embeddings.append(text_emb)

        x = torch.cat(embeddings, dim=1) if len(embeddings) > 1 else embeddings[0]
        B, S, D = x.shape

        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        rope_cos, rope_sin = self.mrope(pos, pos, pos)
        rope_cos = rope_cos.unsqueeze(2)
        rope_sin = rope_sin.unsqueeze(2)

        h, _ = self.decoder(x, rope_cos, rope_sin, causal=True)
        return self.lm_head(h)

    # ----------------------------------------------------------------
    #  Factory methods
    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, config: ModelConfig) -> "Qwen25VL":
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
    ) -> "Qwen25VL":
        """Load model with HF weights."""
        from .weight_mapping import build_weight_mapping, map_state_dict
        from .loading_utils import load_hf_state_dict, tie_weights

        config = ModelConfig.from_pretrained(path_or_id)
        model = cls.from_config(config)

        vc = config.vision_config or VisionConfig(rope_dim=3)
        mapping = build_weight_mapping(
            "qwen25_vl",
            config.num_hidden_layers,
            vc.num_hidden_layers,
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
