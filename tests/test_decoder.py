"""Tests for Decoder backbone (GQA attention, DecoderBlock, DecoderBackbone)."""

from __future__ import annotations

import torch
import pytest

from omni_qwen.decoder import GQASelfAttention, DecoderBlock, DecoderBackbone
from omni_qwen.rope import RotaryEmbedding1D


class TestGQASelfAttention:
    def test_output_shape(self):
        attn = GQASelfAttention(hidden_size=128, num_q_heads=8, num_kv_heads=2)
        x = torch.randn(2, 16, 128)
        out, kv = attn(x)
        assert out.shape == (2, 16, 128)
        assert kv[0].shape[2] == 16  # key seq dim

    def test_gqa_kv_heads(self):
        """KV cache should have num_kv_heads heads, not num_q_heads."""
        attn = GQASelfAttention(128, num_q_heads=8, num_kv_heads=2)
        x = torch.randn(1, 8, 128)
        _, (k, v) = attn(x)
        assert k.shape[1] == 2  # kv heads

    def test_kv_cache_incremental(self):
        attn = GQASelfAttention(128, 8, 2)
        x1 = torch.randn(1, 4, 128)
        _, kv1 = attn(x1)
        x2 = torch.randn(1, 1, 128)
        out2, kv2 = attn(x2, kv_cache=kv1)
        assert kv2[0].shape[2] == 5  # 4 + 1

    def test_causal_mask(self):
        """First token output should be the same with or without later tokens."""
        attn = GQASelfAttention(64, 4, 2)
        attn.eval()
        torch.manual_seed(42)
        x_short = torch.randn(1, 1, 64)
        out_short, _ = attn(x_short, causal=True)
        x_long = torch.cat([x_short, torch.randn(1, 3, 64)], dim=1)
        out_long, _ = attn(x_long, causal=True)
        torch.testing.assert_close(out_short[:, 0], out_long[:, 0], atol=1e-5, rtol=1e-5)

    def test_with_rope(self):
        attn = GQASelfAttention(128, 8, 2)
        rope = RotaryEmbedding1D(dim=16, max_seq_len=64)  # head_dim = 128/8 = 16
        x = torch.randn(1, 8, 128)
        dummy = torch.randn(1, 8, 1, 16)
        cos, sin = rope(dummy)
        out, _ = attn(x, rope_cos=cos, rope_sin=sin)
        assert out.shape == (1, 8, 128)


class TestDecoderBlock:
    def test_output_shape(self):
        block = DecoderBlock(128, 8, 2, 256)
        x = torch.randn(2, 16, 128)
        out, kv = block(x)
        assert out.shape == (2, 16, 128)

    def test_gradient(self):
        block = DecoderBlock(64, 4, 2, 128)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out, _ = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestDecoderBackbone:
    def test_output_shape(self):
        backbone = DecoderBackbone(4, 64, 4, 2, 128)
        x = torch.randn(1, 8, 64)
        out, kv_caches = backbone(x)
        assert out.shape == (1, 8, 64)
        assert len(kv_caches) == 4

    def test_gradient(self):
        backbone = DecoderBackbone(2, 64, 4, 2, 128)
        x = torch.randn(1, 4, 64, requires_grad=True)
        out, _ = backbone(x)
        out.sum().backward()
        assert x.grad is not None
