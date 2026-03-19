"""Tests for Gated Self-Attention."""

from __future__ import annotations

import torch
import pytest

from omni_qwen.gated_attention import GatedSelfAttention
from omni_qwen.rope import RotaryEmbedding1D


class TestGatedSelfAttention:
    @pytest.fixture
    def attn(self):
        return GatedSelfAttention(
            hidden_size=128, num_q_heads=8, num_kv_heads=2,
        )

    def test_output_shape(self, attn):
        x = torch.randn(2, 16, 128)
        out, kv = attn(x)
        assert out.shape == (2, 16, 128)

    def test_kv_cache(self, attn):
        x1 = torch.randn(1, 4, 128)
        _, kv1 = attn(x1)
        x2 = torch.randn(1, 1, 128)
        out2, kv2 = attn(x2, kv_cache=kv1)
        assert kv2[0].shape[2] == 5  # 4 + 1

    def test_gradient(self, attn):
        x = torch.randn(1, 8, 128, requires_grad=True)
        out, _ = attn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_gating_effect(self, attn):
        """Gate should modulate output values — output should not be trivially zero."""
        x = torch.randn(1, 8, 128)
        out, _ = attn(x)
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_causal_first_token(self):
        """First token should be the same regardless of later tokens."""
        attn = GatedSelfAttention(64, 4, 2)
        attn.eval()
        x_short = torch.randn(1, 1, 64)
        out_short, _ = attn(x_short, causal=True)
        x_long = torch.cat([x_short, torch.randn(1, 5, 64)], dim=1)
        out_long, _ = attn(x_long, causal=True)
        torch.testing.assert_close(out_short[:, 0], out_long[:, 0], atol=1e-5, rtol=1e-5)

    def test_with_rope(self):
        attn = GatedSelfAttention(128, 8, 2)
        rope = RotaryEmbedding1D(dim=16)
        x = torch.randn(1, 8, 128)
        dummy = torch.randn(1, 8, 1, 16)
        cos, sin = rope(dummy)
        out, _ = attn(x, rope_cos=cos, rope_sin=sin)
        assert out.shape == (1, 8, 128)
