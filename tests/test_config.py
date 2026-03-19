"""Tests for config.py, weight_mapping.py, loading_utils.py, and generate.py."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
import pytest

from omni_qwen.config import ModelConfig, VisionConfig
from omni_qwen.weight_mapping import (
    build_weight_mapping,
    map_state_dict,
    reverse_mapping,
)
from omni_qwen.generate import generate, generate_stream


# ======================================================================
#  VisionConfig
# ======================================================================

class TestVisionConfig:
    def test_defaults(self):
        vc = VisionConfig()
        assert vc.hidden_size == 1280
        assert vc.num_hidden_layers == 32
        assert vc.patch_size == 14
        assert vc.rope_dim == 2

    def test_from_dict_direct(self):
        vc = VisionConfig.from_dict({
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "patch_size": 7,
            "rope_dim": 3,
        })
        assert vc.hidden_size == 512
        assert vc.rope_dim == 3

    def test_from_dict_hf_names(self):
        """HF uses 'embed_dim', 'depth', 'num_heads' instead."""
        vc = VisionConfig.from_dict({
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "patch_size": 14,
        })
        assert vc.hidden_size == 768
        assert vc.num_hidden_layers == 12
        assert vc.num_attention_heads == 12
        # intermediate_size auto-computed
        assert vc.intermediate_size == 768 * 4

    def test_to_dict(self):
        vc = VisionConfig(hidden_size=256)
        d = vc.to_dict()
        assert isinstance(d, dict)
        assert d["hidden_size"] == 256

    def test_unknown_keys_ignored(self):
        vc = VisionConfig.from_dict({
            "hidden_size": 128,
            "some_unknown_field": 42,
        })
        assert vc.hidden_size == 128


# ======================================================================
#  ModelConfig
# ======================================================================

class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig()
        assert mc.model_type == "qwen2_vl"
        assert mc.hidden_size == 3584
        assert mc.head_dim == 3584 // 28  # auto-computed

    def test_head_dim_auto(self):
        mc = ModelConfig(hidden_size=128, num_attention_heads=4)
        assert mc.head_dim == 32

    def test_head_dim_explicit(self):
        mc = ModelConfig(hidden_size=128, num_attention_heads=4, head_dim=64)
        assert mc.head_dim == 64

    def test_from_dict(self):
        mc = ModelConfig.from_dict({
            "model_type": "qwen3_vl",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        })
        assert mc.model_type == "qwen3_vl"
        assert mc.is_moe
        assert mc.num_experts == 8

    def test_from_dict_with_vision(self):
        mc = ModelConfig.from_dict({
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vision_config": {
                "embed_dim": 512,
                "depth": 6,
                "num_heads": 8,
            },
        })
        assert mc.vision_config is not None
        assert mc.vision_config.hidden_size == 512
        assert mc.vision_config.num_hidden_layers == 6

    def test_from_pretrained_local(self):
        """Create a fake config.json and load from it."""
        cfg = {
            "model_type": "qwen2_vl",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 1000,
            "vision_config": {
                "embed_dim": 128,
                "depth": 2,
                "num_heads": 4,
                "patch_size": 7,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "config.json"), "w") as f:
                json.dump(cfg, f)
            mc = ModelConfig.from_pretrained(td)
            assert mc.hidden_size == 256
            assert mc.model_type == "qwen2_vl"
            assert mc.vision_config.hidden_size == 128

    def test_model_type_normalisation(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "model_type": "qwen2.5-vl",
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            }
            with open(os.path.join(td, "config.json"), "w") as f:
                json.dump(cfg, f)
            mc = ModelConfig.from_pretrained(td)
            assert mc.model_type == "qwen25_vl"

    def test_properties(self):
        mc = ModelConfig(model_type="qwen3_vl", num_experts=8,
                         num_attention_heads=4, num_key_value_heads=2)
        assert mc.is_moe is True
        assert mc.is_hybrid is False

        mc2 = ModelConfig(model_type="qwen3_next",
                          num_attention_heads=4, num_key_value_heads=2)
        assert mc2.is_hybrid is True
        assert mc2.is_moe is False

    def test_to_dict_roundtrip(self):
        mc = ModelConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
            vision_config=VisionConfig(hidden_size=256),
        )
        d = mc.to_dict()
        assert d["hidden_size"] == 512
        assert d["vision_config"]["hidden_size"] == 256

    def test_get_decoder_kwargs(self):
        mc = ModelConfig(
            hidden_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=256,
        )
        dk = mc.get_decoder_kwargs()
        assert dk["num_layers"] == 4
        assert dk["hidden_size"] == 128
        assert dk["num_q_heads"] == 4

    def test_get_vit_kwargs(self):
        mc = ModelConfig(
            num_attention_heads=4, num_key_value_heads=2,
            vision_config=VisionConfig(
                hidden_size=256, num_hidden_layers=6,
                num_attention_heads=4, patch_size=7, rope_dim=2,
            ),
        )
        vk = mc.get_vit_kwargs()
        assert vk["hidden_size"] == 256
        assert vk["num_layers"] == 6
        assert vk["rope_dim"] == 2

    def test_get_vit_kwargs_no_vision(self):
        mc = ModelConfig(num_attention_heads=4, num_key_value_heads=2)
        with pytest.raises(ValueError):
            mc.get_vit_kwargs()


# ======================================================================
#  Weight Mapping
# ======================================================================

class TestWeightMapping:
    def test_qwen2_vl_mapping_keys(self):
        m = build_weight_mapping("qwen2_vl", 2, 2)
        # Check essential keys exist
        assert "embed_tokens.weight" in m
        assert "lm_head.weight" in m
        assert "decoder.layers.0.attn.q_proj.weight" in m
        assert "decoder.layers.1.ffn.gate_proj.weight" in m
        assert "vit.blocks.0.attn.qkv.weight" in m

    def test_qwen3_vl_mapping_keys(self):
        m = build_weight_mapping("qwen3_vl", 2, 2, num_experts=4)
        assert "layers.0.moe.router.gate.weight" in m
        assert "layers.0.moe.experts.0.gate_proj.weight" in m
        assert "layers.0.moe.experts.3.down_proj.weight" in m

    def test_qwen3_next_mapping_keys(self):
        m = build_weight_mapping("qwen3_next", 4, 2, attn_every=4)
        # Layer 3 (index 3) should be Gated Attention ((3+1)%4==0)
        assert "layers.3.attn.gate_proj.weight" in m
        # Layer 0 should be DeltaNet ((0+1)%4!=0)
        assert "layers.0.attn.beta_proj.weight" in m
        assert "layers.0.attn.q_conv.conv.weight" in m

    def test_map_state_dict(self):
        mapping = {"embed_tokens.weight": "model.embed_tokens.weight"}
        hf_sd = {"model.embed_tokens.weight": torch.randn(100, 64)}
        our_sd = map_state_dict(hf_sd, mapping)
        assert "embed_tokens.weight" in our_sd
        assert our_sd["embed_tokens.weight"].shape == (100, 64)

    def test_map_state_dict_strict(self):
        mapping = {"a": "hf_a"}
        hf_sd = {"hf_a": torch.randn(2), "hf_b": torch.randn(2)}
        # Non-strict: no error
        our_sd = map_state_dict(hf_sd, mapping, strict=False)
        assert "a" in our_sd
        # Strict: error for unmapped hf_b
        with pytest.raises(KeyError):
            map_state_dict(hf_sd, mapping, strict=True)

    def test_reverse_mapping(self):
        m = {"a": "hf_a", "b": "hf_b"}
        rev = reverse_mapping(m)
        assert rev == {"hf_a": "a", "hf_b": "b"}

    def test_unknown_model_type(self):
        with pytest.raises(ValueError):
            build_weight_mapping("unknown_model", 2, 2)


# ======================================================================
#  from_config round-trip (create model from config)
# ======================================================================

class TestFromConfig:
    """Test that from_config creates valid models matching direct init."""

    def test_qwen2_vl_from_config(self):
        from omni_qwen.qwen2_vl import Qwen2VL
        vc = VisionConfig(
            hidden_size=96, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=384,
            patch_size=7, rope_dim=2, window_size=(2, 2),
        )
        mc = ModelConfig(
            model_type="qwen2_vl",
            hidden_size=96, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=192, vocab_size=256,
            vision_config=vc,
        )
        model = Qwen2VL.from_config(mc)
        ids = torch.randint(0, 256, (1, 8))
        logits = model(input_ids=ids)
        assert logits.shape == (1, 8, 256)

    def test_qwen25_vl_from_config(self):
        from omni_qwen.qwen25_vl import Qwen25VL
        vc = VisionConfig(
            hidden_size=96, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=384,
            patch_size=7, rope_dim=3, window_size=(2, 2),
        )
        mc = ModelConfig(
            model_type="qwen25_vl",
            hidden_size=96, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=192, vocab_size=256,
            vision_config=vc,
        )
        model = Qwen25VL.from_config(mc)
        ids = torch.randint(0, 256, (1, 8))
        logits = model(input_ids=ids)
        assert logits.shape == (1, 8, 256)

    def test_qwen3_vl_from_config(self):
        from omni_qwen.qwen3_vl import Qwen3VL
        vc = VisionConfig(
            hidden_size=96, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=384,
            patch_size=7, rope_dim=3,
        )
        mc = ModelConfig(
            model_type="qwen3_vl",
            hidden_size=96, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=192, vocab_size=256,
            num_experts=4, num_experts_per_tok=2,
            num_shared_experts=1,
            expert_intermediate_size=64,
            shared_expert_intermediate_size=192,
            vision_config=vc,
        )
        model = Qwen3VL.from_config(mc)
        ids = torch.randint(0, 256, (1, 8))
        logits, aux = model(input_ids=ids)
        assert logits.shape == (1, 8, 256)

    def test_qwen3_next_from_config(self):
        from omni_qwen.qwen3_next import Qwen3NeXT
        vc = VisionConfig(
            hidden_size=96, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=384,
            patch_size=7, rope_dim=3,
        )
        mc = ModelConfig(
            model_type="qwen3_next",
            hidden_size=96, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=192, vocab_size=256,
            attn_every=4, deltanet_chunk_size=8,
            vision_config=vc,
        )
        model = Qwen3NeXT.from_config(mc)
        ids = torch.randint(0, 256, (1, 8))
        logits = model(input_ids=ids)
        assert logits.shape == (1, 8, 256)
        assert model.num_deltanet_layers == 6
        assert model.num_attn_layers == 2


# ======================================================================
#  Generate
# ======================================================================

class TestGenerate:
    """Test generate() and generate_stream() with tiny models."""

    @pytest.fixture
    def tiny_model(self):
        from omni_qwen.qwen2_vl import Qwen2VL
        return Qwen2VL(
            vit_hidden=96, llm_hidden=96, llm_layers=2,
            num_q_heads=4, num_kv_heads=2, intermediate_size=192,
            vocab_size=256, vit_layers=2, vit_heads=4, patch_size=7,
            window_size=(2, 2),
        )

    def test_generate_greedy(self, tiny_model):
        ids = torch.randint(0, 256, (1, 4))
        out = generate(tiny_model, ids, max_new_tokens=5)
        assert out.shape == (1, 9)  # 4 prompt + 5 generated
        assert (out[:, :4] == ids).all()

    def test_generate_with_stop(self, tiny_model):
        ids = torch.randint(0, 256, (1, 4))
        # Early stop if we ever generate token 0 (likely with random weights)
        out = generate(tiny_model, ids, max_new_tokens=50, stop_token_ids={0})
        # Should stop early or reach max
        assert out.shape[1] <= 54

    def test_generate_with_temperature(self, tiny_model):
        ids = torch.randint(0, 256, (1, 4))
        out = generate(tiny_model, ids, max_new_tokens=3, temperature=1.0)
        assert out.shape == (1, 7)

    def test_generate_stream(self, tiny_model):
        ids = torch.randint(0, 256, (1, 4))
        tokens = list(generate_stream(tiny_model, ids, max_new_tokens=5))
        assert len(tokens) == 5
        assert all(isinstance(t, int) for t in tokens)

    def test_generate_stream_stop(self, tiny_model):
        ids = torch.randint(0, 256, (1, 4))
        tokens = list(generate_stream(
            tiny_model, ids, max_new_tokens=100, stop_token_ids={0}
        ))
        # Should have stopped early (or at 100)
        assert len(tokens) <= 100

    def test_generate_moe(self):
        """Verify generate works with MoE model (returns (logits, aux))."""
        from omni_qwen.qwen3_vl import Qwen3VL
        model = Qwen3VL(
            vit_hidden=96, llm_hidden=96, llm_layers=2,
            num_q_heads=4, num_kv_heads=2, vocab_size=256,
            num_routed_experts=4, num_shared_experts=1,
            expert_intermediate=64, shared_intermediate=192, top_k=2,
            vit_layers=2, vit_heads=4, patch_size=7,
        )
        ids = torch.randint(0, 256, (1, 4))
        out = generate(model, ids, max_new_tokens=3)
        assert out.shape == (1, 7)


# ======================================================================
#  Loading utils (unit tests without actual HF download)
# ======================================================================

class TestLoadingUtils:
    def test_load_nonexistent_raises(self):
        from omni_qwen.loading_utils import load_hf_state_dict
        with pytest.raises((FileNotFoundError, ImportError)):
            load_hf_state_dict("/nonexistent/path/that/does/not/exist")
