"""
omni_qwen — Core architecture implementations for the Qwen VL model family.

Covers Qwen2-VL, Qwen2.5-VL, Qwen3-VL (MoE), and Qwen3-NeXT (hybrid
Gated-DeltaNet + Gated-Attention).

Usage
-----
    import omni_qwen
    model = omni_qwen.Qwen2VL.from_config(config)

    from omni_qwen import Qwen3VL, Processor, generate
"""

# ── Configuration ──────────────────────────────────────────────────────────
from .config import ModelConfig, VisionConfig

# ── Positional Embeddings ──────────────────────────────────────────────────
from .rope import (
    RotaryEmbedding1D,
    RotaryEmbedding2D,
    RotaryEmbedding3D,
    MRoPE,
    apply_rotary_emb,
)

# ── Normalization & Feed-Forward ───────────────────────────────────────────
from .rms_norm import RMSNorm
from .swiglu_mlp import SwiGLUMLP

# ── Attention ──────────────────────────────────────────────────────────────
from .window_attention import WindowAttention

# ── Vision Transformer ─────────────────────────────────────────────────────
from .vit import (
    PatchEmbed2D,
    PatchEmbed3D,
    FullSelfAttention,
    ViTMLP,
    ViTBlock,
    VisionTransformer,
)
from .visual_projector import Qwen2VLProjector, PerceiverProjector

# ── Decoder ────────────────────────────────────────────────────────────────
from .decoder import GQASelfAttention, DecoderBlock, DecoderBackbone

# ── MoE ────────────────────────────────────────────────────────────────────
from .moe import TopKRouter, MoELayer, load_balancing_loss

# ── Gated Linear Attention ─────────────────────────────────────────────────
from .gated_deltanet import ShortConv1D, GatedDeltaNet
from .gated_attention import GatedSelfAttention

# ── Weight Mapping & Loading ───────────────────────────────────────────────
from .weight_mapping import build_weight_mapping, map_state_dict, reverse_mapping
from .loading_utils import load_hf_state_dict, tie_weights

# ── Generation ─────────────────────────────────────────────────────────────
from .generate import generate, generate_stream

# ── Processor ──────────────────────────────────────────────────────────────
from .processor import Processor, preprocess_image, preprocess_video

# ── Model Assemblies ───────────────────────────────────────────────────────
from .qwen2_vl import Qwen2VL
from .qwen25_vl import Qwen25VL
from .qwen3_vl import MoEDecoderBlock, Qwen3VL
from .qwen3_next import HybridDecoderBlock, Qwen3NeXT

__all__ = [
    # Config
    "ModelConfig", "VisionConfig",
    # RoPE
    "RotaryEmbedding1D", "RotaryEmbedding2D", "RotaryEmbedding3D",
    "MRoPE", "apply_rotary_emb",
    # Norm & MLP
    "RMSNorm", "SwiGLUMLP",
    # Attention
    "WindowAttention",
    # ViT
    "PatchEmbed2D", "PatchEmbed3D", "FullSelfAttention", "ViTMLP",
    "ViTBlock", "VisionTransformer",
    "Qwen2VLProjector", "PerceiverProjector",
    # Decoder
    "GQASelfAttention", "DecoderBlock", "DecoderBackbone",
    # MoE
    "TopKRouter", "MoELayer", "load_balancing_loss",
    # Gated
    "ShortConv1D", "GatedDeltaNet", "GatedSelfAttention",
    # Weight mapping & loading
    "build_weight_mapping", "map_state_dict", "reverse_mapping",
    "load_hf_state_dict", "tie_weights",
    # Generation
    "generate", "generate_stream",
    # Processor
    "Processor", "preprocess_image", "preprocess_video",
    # Models
    "Qwen2VL", "Qwen25VL",
    "MoEDecoderBlock", "Qwen3VL",
    "HybridDecoderBlock", "Qwen3NeXT",
]
