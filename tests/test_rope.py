"""Tests for RoPE variants: 1-D, 2-D, 3-D, M-RoPE."""

from __future__ import annotations

import math
import torch
import pytest

from omni_qwen.rope import (
    RotaryEmbedding1D, RotaryEmbedding2D, RotaryEmbedding3D, MRoPE,
    apply_rotary_emb, _rotate_half,
)


# ==================================================================
#  1-D RoPE
# ==================================================================

class TestRotaryEmbedding1D:
    def test_output_shape(self):
        rope = RotaryEmbedding1D(dim=64, max_seq_len=128)
        x = torch.randn(2, 16, 4, 64)  # [B, S, H, D]
        cos, sin = rope(x)
        assert cos.shape == (1, 16, 1, 64)
        assert sin.shape == (1, 16, 1, 64)

    def test_with_position_ids(self):
        rope = RotaryEmbedding1D(dim=64, max_seq_len=128)
        x = torch.randn(2, 8, 4, 64)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        cos, sin = rope(x, position_ids=pos_ids)
        assert cos.shape == (2, 8, 1, 64)

    def test_cos_sin_bounded(self):
        rope = RotaryEmbedding1D(dim=32)
        x = torch.randn(1, 64, 2, 32)
        cos, sin = rope(x)
        assert cos.abs().max() <= 1.0 + 1e-6
        assert sin.abs().max() <= 1.0 + 1e-6

    def test_position_zero_cos_is_one(self):
        """At position 0, all cos values should be 1, sin should be 0."""
        rope = RotaryEmbedding1D(dim=32)
        x = torch.randn(1, 1, 1, 32)
        cos, sin = rope(x)
        torch.testing.assert_close(cos[0, 0, 0], torch.ones(32), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sin[0, 0, 0], torch.zeros(32), atol=1e-5, rtol=1e-5)


# ==================================================================
#  2-D RoPE
# ==================================================================

class TestRotaryEmbedding2D:
    def test_output_shape(self):
        rope = RotaryEmbedding2D(dim=64)
        h_ids = torch.tensor([[0, 0, 1, 1]])
        w_ids = torch.tensor([[0, 1, 0, 1]])
        cos, sin = rope(h_ids, w_ids)
        assert cos.shape == (1, 4, 64)

    def test_symmetric_positions(self):
        """Same h/w ids → same embeddings."""
        rope = RotaryEmbedding2D(dim=64)
        h_ids = torch.tensor([[3, 3]])
        w_ids = torch.tensor([[5, 5]])
        cos, sin = rope(h_ids, w_ids)
        torch.testing.assert_close(cos[:, 0], cos[:, 1])

    def test_different_positions(self):
        rope = RotaryEmbedding2D(dim=64)
        h_ids = torch.tensor([[0, 1]])
        w_ids = torch.tensor([[0, 0]])
        cos, _ = rope(h_ids, w_ids)
        # Different heights → different embeddings
        assert not torch.allclose(cos[:, 0], cos[:, 1])


# ==================================================================
#  3-D RoPE
# ==================================================================

class TestRotaryEmbedding3D:
    def test_output_shape(self):
        rope = RotaryEmbedding3D(dim=96)  # 96/3=32 per axis
        t = torch.zeros(2, 8, dtype=torch.long)
        h = torch.arange(8).unsqueeze(0).expand(2, -1)
        w = torch.arange(8).unsqueeze(0).expand(2, -1)
        cos, sin = rope(t, h, w)
        assert cos.shape == (2, 8, 96)

    def test_temporal_invariance_at_zero(self):
        """All-zero time ids → temporal part is identity (cos=1, sin=0)."""
        rope = RotaryEmbedding3D(dim=96)
        t = torch.zeros(1, 4, dtype=torch.long)
        h = torch.arange(4).unsqueeze(0)
        w = torch.arange(4).unsqueeze(0)
        cos, sin = rope(t, h, w)
        # First 32 dims are temporal → at time=0 should be cos=1
        temporal_cos = cos[:, :, :32]
        torch.testing.assert_close(temporal_cos, torch.ones_like(temporal_cos),
                                   atol=1e-5, rtol=1e-5)


# ==================================================================
#  M-RoPE
# ==================================================================

class TestMRoPE:
    def test_output_shape(self):
        mrope = MRoPE(dim=96)
        pos = torch.arange(16).unsqueeze(0)
        cos, sin = mrope(pos, pos, pos)
        assert cos.shape == (1, 16, 96)

    def test_text_mode_degenerates_to_1d(self):
        """When pos_t == pos_h == pos_w, all three parts are identical."""
        mrope = MRoPE(dim=96)
        pos = torch.arange(8).unsqueeze(0)
        cos, sin = mrope(pos, pos, pos)
        part_dim = 32
        # All three parts should be identical
        torch.testing.assert_close(cos[:, :, :part_dim], cos[:, :, part_dim:2*part_dim])
        torch.testing.assert_close(cos[:, :, :part_dim], cos[:, :, 2*part_dim:])


# ==================================================================
#  apply_rotary_emb
# ==================================================================

class TestApplyRotary:
    def test_output_shape(self):
        x = torch.randn(2, 8, 4, 64)
        cos = torch.ones(2, 8, 1, 64)
        sin = torch.zeros(2, 8, 1, 64)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_identity_rotation(self):
        """cos=1, sin=0 → output == input."""
        x = torch.randn(2, 8, 4, 64)
        cos = torch.ones(2, 8, 1, 64)
        sin = torch.zeros(2, 8, 1, 64)
        out = apply_rotary_emb(x, cos, sin)
        torch.testing.assert_close(out, x)

    def test_rotation_preserves_norm(self):
        """RoPE is an orthogonal transform → preserves L2 norm."""
        x = torch.randn(1, 4, 1, 32)
        rope = RotaryEmbedding1D(dim=32)
        cos, sin = rope(x)
        out = apply_rotary_emb(x, cos, sin)
        x_norm = x.norm(dim=-1)
        out_norm = out.norm(dim=-1)
        torch.testing.assert_close(x_norm, out_norm, atol=1e-5, rtol=1e-5)
