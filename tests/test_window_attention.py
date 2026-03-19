"""Tests for Window Attention."""

from __future__ import annotations

import torch
import pytest

from omni_qwen.window_attention import (
    WindowAttention, _window_partition, _window_unpartition,
)


class TestWindowPartition:
    def test_roundtrip(self):
        """Partition → unpartition should recover the original tensor."""
        B, H, W, C = 2, 8, 8, 64
        x = torch.randn(B, H * W, C)
        x_win, Hp, Wp, nH, nW = _window_partition(x, 4, 4, H, W)
        recovered = _window_unpartition(x_win, B, nH, nW, 4, 4, H, W)
        torch.testing.assert_close(recovered, x)

    def test_partition_shape(self):
        B, H, W, C = 1, 8, 8, 32
        x = torch.randn(B, H * W, C)
        x_win, Hp, Wp, nH, nW = _window_partition(x, 4, 4, H, W)
        assert x_win.shape == (1 * 2 * 2, 16, 32)  # 4 windows, 4*4 tokens each

    def test_padding(self):
        """Non-divisible spatial size should be handled via padding."""
        B, H, W, C = 1, 7, 7, 32
        x = torch.randn(B, H * W, C)
        x_win, Hp, Wp, nH, nW = _window_partition(x, 4, 4, H, W)
        assert Hp == 8 and Wp == 8  # padded to multiple of 4
        recovered = _window_unpartition(x_win, B, nH, nW, 4, 4, H, W)
        torch.testing.assert_close(recovered, x)


class TestWindowAttention:
    def test_output_shape(self):
        attn = WindowAttention(hidden_size=64, num_heads=4, window_size=(4, 4),
                               use_rope=False)
        x = torch.randn(2, 64, 64)  # [B, 8*8, 64]
        out = attn(x, H=8, W=8)
        assert out.shape == (2, 64, 64)

    def test_gradient_flows(self):
        attn = WindowAttention(hidden_size=64, num_heads=4, window_size=(4, 4),
                               use_rope=False)
        x = torch.randn(1, 16, 64, requires_grad=True)
        out = attn(x, H=4, W=4)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_deterministic(self):
        attn = WindowAttention(hidden_size=64, num_heads=4, window_size=(4, 4),
                               use_rope=False)
        attn.eval()
        x = torch.randn(1, 16, 64)
        y1 = attn(x, 4, 4)
        y2 = attn(x, 4, 4)
        torch.testing.assert_close(y1, y2)

    def test_different_window_sizes(self):
        for ws in [(2, 2), (4, 4), (8, 8)]:
            attn = WindowAttention(64, 4, window_size=ws, use_rope=False)
            x = torch.randn(1, 64, 64)
            out = attn(x, 8, 8)
            assert out.shape == (1, 64, 64)
