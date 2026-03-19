"""Integration tests for full model assemblies (Qwen2-VL, 2.5-VL, 3-VL, 3-NeXT)."""

from __future__ import annotations

import torch
import pytest

from omni_qwen.qwen2_vl import Qwen2VL
from omni_qwen.qwen25_vl import Qwen25VL
from omni_qwen.qwen3_vl import Qwen3VL
from omni_qwen.qwen3_next import Qwen3NeXT


# Use tiny configs for fast testing
_TINY = dict(
    vit_hidden=96, llm_hidden=96, llm_layers=2,
    num_q_heads=4, num_kv_heads=2, intermediate_size=192,
    vocab_size=256, vit_layers=2, vit_heads=4, patch_size=7,
)


# ==================================================================
#  Qwen2-VL
# ==================================================================

class TestQwen2VL:
    @pytest.fixture
    def model(self):
        return Qwen2VL(**_TINY, window_size=(2, 2))

    def test_text_only(self, model):
        ids = torch.randint(0, 256, (1, 16))
        logits = model(input_ids=ids)
        assert logits.shape == (1, 16, 256)

    def test_image_only(self, model):
        img = torch.randn(1, 3, 28, 28)
        logits = model(pixel_values=img)
        # 28/7=4 patches per side, 4x4=16 patches, merged 2x2 → 4 tokens
        assert logits.shape[0] == 1
        assert logits.shape[2] == 256

    def test_multimodal(self, model):
        img = torch.randn(1, 3, 28, 28)
        ids = torch.randint(0, 256, (1, 8))
        logits = model(input_ids=ids, pixel_values=img)
        assert logits.shape[0] == 1
        assert logits.shape[2] == 256

    def test_gradient(self, model):
        ids = torch.randint(0, 256, (1, 8))
        logits = model(input_ids=ids)
        logits.sum().backward()
        assert model.embed_tokens.weight.grad is not None


# ==================================================================
#  Qwen2.5-VL
# ==================================================================

class TestQwen25VL:
    @pytest.fixture
    def model(self):
        # Need head_dim divisible by 6 for 3D RoPE → 96 / 4 = 24 → 24 % 6 = 0 ✓
        return Qwen25VL(**_TINY, window_size=(2, 2))

    def test_text_only(self, model):
        ids = torch.randint(0, 256, (1, 16))
        logits = model(input_ids=ids)
        assert logits.shape == (1, 16, 256)

    def test_image(self, model):
        img = torch.randn(1, 3, 28, 28)
        logits = model(pixel_values=img)
        assert logits.shape[2] == 256

    def test_gradient(self, model):
        ids = torch.randint(0, 256, (1, 8))
        logits = model(input_ids=ids)
        logits.sum().backward()
        assert model.embed_tokens.weight.grad is not None


# ==================================================================
#  Qwen3-VL (MoE)
# ==================================================================

class TestQwen3VL:
    @pytest.fixture
    def model(self):
        tiny = {k: v for k, v in _TINY.items() if k != 'intermediate_size'}
        return Qwen3VL(
            **tiny,
            num_routed_experts=4, num_shared_experts=1,
            expert_intermediate=64, shared_intermediate=192, top_k=2,
        )

    def test_text_only(self, model):
        ids = torch.randint(0, 256, (1, 16))
        logits, aux = model(input_ids=ids)
        assert logits.shape == (1, 16, 256)
        assert aux.dim() == 0  # scalar aux loss

    def test_aux_loss_nonzero(self, model):
        ids = torch.randint(0, 256, (1, 16))
        _, aux = model(input_ids=ids)
        assert aux.item() > 0

    def test_gradient(self, model):
        ids = torch.randint(0, 256, (1, 8))
        logits, aux = model(input_ids=ids)
        (logits.sum() + 0.01 * aux).backward()
        assert model.embed_tokens.weight.grad is not None


# ==================================================================
#  Qwen3-NeXT (hybrid)
# ==================================================================

class TestQwen3NeXT:
    @pytest.fixture
    def model(self):
        return Qwen3NeXT(
            vit_hidden=96, llm_hidden=96, llm_layers=8,
            num_q_heads=4, num_kv_heads=2, intermediate_size=192,
            vocab_size=256, attn_every=4, deltanet_chunk=8,
            vit_layers=2, vit_heads=4, patch_size=7,
        )

    def test_layer_schedule(self, model):
        """Should have 6 DeltaNet + 2 Gated Attention layers."""
        assert model.num_deltanet_layers == 6
        assert model.num_attn_layers == 2

    def test_text_only(self, model):
        ids = torch.randint(0, 256, (1, 16))
        logits = model(input_ids=ids)
        assert logits.shape == (1, 16, 256)

    def test_image(self, model):
        img = torch.randn(1, 3, 28, 28)
        logits = model(pixel_values=img)
        assert logits.shape[2] == 256

    def test_gradient(self, model):
        ids = torch.randint(0, 256, (1, 8))
        logits = model(input_ids=ids)
        logits.sum().backward()
        assert model.embed_tokens.weight.grad is not None

    def test_multimodal(self, model):
        img = torch.randn(1, 3, 28, 28)
        ids = torch.randint(0, 256, (1, 8))
        logits = model(input_ids=ids, pixel_values=img)
        assert logits.shape[0] == 1
        assert logits.shape[2] == 256
