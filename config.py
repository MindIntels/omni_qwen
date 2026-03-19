"""
Unified configuration dataclasses for all Qwen VL model variants.

Provides ``ModelConfig`` (LLM decoder) and ``VisionConfig`` (ViT encoder)
that can be:

1. Instantiated directly with keyword arguments (for testing).
2. Created from a HuggingFace ``config.json`` via ``from_pretrained(path)``.
3. Created from a plain dictionary via ``from_dict(d)``.

Supported model types
---------------------
- ``qwen2_vl``   — standard GQA decoder, 2-D RoPE ViT
- ``qwen25_vl``  — standard GQA decoder, 3-D RoPE ViT (video)
- ``qwen3_vl``   — MoE decoder, 3-D RoPE ViT
- ``qwen3_next`` — hybrid DeltaNet + Gated Attention decoder, 3-D RoPE ViT
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────
#  Vision encoder configuration
# ──────────────────────────────────────────────────────────────────

@dataclass
class VisionConfig:
    """Configuration for the Vision Transformer encoder.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension of the ViT.
    num_hidden_layers : int
        Number of ViT transformer blocks.
    num_attention_heads : int
        Number of attention heads in ViT.
    intermediate_size : int
        FFN intermediate dimension (typically ``hidden_size * 4``).
    patch_size : int
        Spatial patch size.
    rope_dim : int
        Dimensionality of the RoPE: 2 (spatial) or 3 (spatio-temporal).
    window_size : tuple[int, int]
        Window attention size for local ViT attention.  ``(0, 0)`` means
        full (global) attention.
    spatial_merge_size : int
        Merge factor for the visual projector (token compression).
    in_channels : int
        Number of input image channels (typically 3).
    temporal_patch_size : int
        Temporal patch size for video processing.
    """
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    intermediate_size: int = 5120
    patch_size: int = 14
    rope_dim: int = 2
    window_size: tuple[int, int] = (0, 0)
    spatial_merge_size: int = 2
    in_channels: int = 3
    temporal_patch_size: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VisionConfig:
        """Create from a dictionary (e.g. the ``vision_config`` sub-dict of
        a HF ``config.json``)."""
        mapping = {
            "embed_dim": "hidden_size",
            "depth": "num_hidden_layers",
            "num_heads": "num_attention_heads",
        }
        normalised: dict[str, Any] = {}
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        for k, v in d.items():
            key = mapping.get(k, k)
            if key in valid_fields:
                normalised[key] = v
        # Compute intermediate_size if not given
        if "intermediate_size" not in normalised and "hidden_size" in normalised:
            normalised.setdefault("intermediate_size", normalised["hidden_size"] * 4)
        return cls(**normalised)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────
#  LLM decoder configuration
# ──────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Unified configuration for all Qwen VL model variants.

    Parameters
    ----------
    model_type : str
        One of ``"qwen2_vl"``, ``"qwen25_vl"``, ``"qwen3_vl"``,
        ``"qwen3_next"``.
    hidden_size : int
        LLM hidden dimension.
    num_hidden_layers : int
        Number of decoder layers.
    num_attention_heads : int
        Number of query heads.
    num_key_value_heads : int
        Number of KV heads (for GQA).
    intermediate_size : int
        FFN intermediate dimension (dense models).
    vocab_size : int
    head_dim : int or None
        Per-head dimension.  Computed from ``hidden_size // num_attention_heads``
        if not provided.
    rope_theta : float
        RoPE base frequency.
    rms_norm_eps : float
    tie_word_embeddings : bool

    MoE fields (Qwen3-VL)
    ~~~~~~~~~~~~~~~~~~~~~~
    num_experts : int
    num_experts_per_tok : int
    num_shared_experts : int
    expert_intermediate_size : int
    shared_expert_intermediate_size : int

    Hybrid fields (Qwen3-NeXT)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    attn_every : int
    deltanet_chunk_size : int

    Vision
    ~~~~~~
    vision_config : VisionConfig or None
    """
    # ---- Core ----
    model_type: str = "qwen2_vl"
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    intermediate_size: int = 18944
    vocab_size: int = 152064
    head_dim: int | None = None
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    # ---- MoE (Qwen3-VL) ----
    num_experts: int = 0
    num_experts_per_tok: int = 0
    num_shared_experts: int = 0
    expert_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0

    # ---- Hybrid (Qwen3-NeXT) ----
    attn_every: int = 4
    deltanet_chunk_size: int = 64

    # ---- Vision ----
    vision_config: VisionConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Create from a dictionary, mapping HF config.json keys to our names."""
        # ---- Key mappings from various HF config formats ----
        hf_mapping = {
            # HF Qwen2-VL / Qwen2.5-VL / Qwen3-VL field names
            "num_key_value_heads": "num_key_value_heads",
            "num_experts_per_tok": "num_experts_per_tok",
            # Some HF configs use slightly different names
            "n_embed": "hidden_size",
            "n_layer": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "n_kv_heads": "num_key_value_heads",
            "n_mlp": "intermediate_size",
            "n_vocab": "vocab_size",
            "n_experts": "num_experts",
            "n_experts_per_token": "num_experts_per_tok",
            "n_shared_experts": "num_shared_experts",
        }

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        normalised: dict[str, Any] = {}

        for k, v in d.items():
            if k == "vision_config":
                # Parse nested vision config
                if isinstance(v, dict):
                    normalised["vision_config"] = VisionConfig.from_dict(v)
                elif isinstance(v, VisionConfig):
                    normalised["vision_config"] = v
                continue
            key = hf_mapping.get(k, k)
            if key in valid_fields:
                normalised[key] = v

        return cls(**normalised)

    @classmethod
    def from_pretrained(cls, path_or_id: str | Path) -> ModelConfig:
        """Load config from a local checkpoint directory or HF model ID.

        Parameters
        ----------
        path_or_id : str or Path
            Either a local path containing ``config.json``, or a HuggingFace
            model identifier (e.g. ``"Qwen/Qwen2-VL-7B-Instruct"``).

        Returns
        -------
        ModelConfig
        """
        path = Path(path_or_id)
        config_file = path / "config.json"

        if not config_file.exists():
            # Try downloading from HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download
                config_file = Path(hf_hub_download(
                    repo_id=str(path_or_id),
                    filename="config.json",
                ))
            except Exception as e:
                raise FileNotFoundError(
                    f"Cannot find config.json at {path} or download from "
                    f"HuggingFace Hub: {e}"
                ) from e

        with open(config_file) as f:
            raw = json.load(f)

        # ---- Auto-detect model_type from HF config ----
        model_type = raw.get("model_type", "qwen2_vl")

        # Normalise model_type to our naming convention
        type_mapping = {
            "qwen2_vl": "qwen2_vl",
            "qwen2-vl": "qwen2_vl",
            "qwen2.5_vl": "qwen25_vl",
            "qwen2.5-vl": "qwen25_vl",
            "qwen25_vl": "qwen25_vl",
            "qwen3_vl": "qwen3_vl",
            "qwen3-vl": "qwen3_vl",
            "qwen3_next": "qwen3_next",
            "qwen3-next": "qwen3_next",
            "qwen3next": "qwen3_next",
        }
        raw["model_type"] = type_mapping.get(model_type, model_type)

        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.vision_config is not None:
            d["vision_config"] = self.vision_config.to_dict()
        return d

    # ---- Convenience helpers for building models ----

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def is_hybrid(self) -> bool:
        return self.model_type == "qwen3_next"

    @property
    def has_video_support(self) -> bool:
        if self.vision_config is not None:
            return self.vision_config.rope_dim == 3
        return self.model_type in ("qwen25_vl", "qwen3_vl", "qwen3_next")

    def get_vit_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for ``VisionTransformer(...)``."""
        vc = self.vision_config
        if vc is None:
            raise ValueError("No vision_config available.")
        kwargs: dict[str, Any] = {
            "patch_size": vc.patch_size,
            "hidden_size": vc.hidden_size,
            "num_heads": vc.num_attention_heads,
            "num_layers": vc.num_hidden_layers,
            "intermediate_size": vc.intermediate_size,
            "rope_dim": vc.rope_dim,
        }
        if vc.window_size != (0, 0):
            kwargs["window_size"] = vc.window_size
        return kwargs

    def get_decoder_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for ``DecoderBackbone(...)``."""
        return {
            "num_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "num_q_heads": self.num_attention_heads,
            "num_kv_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "head_dim": self.head_dim,
        }
