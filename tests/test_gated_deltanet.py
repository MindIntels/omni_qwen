"""Tests for Gated DeltaNet linear attention."""

from __future__ import annotations

import torch
import pytest

from omni_qwen.gated_deltanet import GatedDeltaNet, ShortConv1D


class TestShortConv1D:
    def test_output_shape(self):
        conv = ShortConv1D(dim=64, kernel_size=4)
        x = torch.randn(2, 16, 64)
        out = conv(x)
        assert out.shape == (2, 16, 64)

    def test_causal(self):
        """Output at time t should not depend on future inputs."""
        conv = ShortConv1D(dim=32, kernel_size=4)
        conv.eval()
        x1 = torch.randn(1, 8, 32)
        x2 = torch.cat([x1, torch.randn(1, 4, 32)], dim=1)
        y1 = conv(x1)
        y2 = conv(x2)
        torch.testing.assert_close(y1, y2[:, :8], atol=1e-5, rtol=1e-5)


class TestGatedDeltaNet:
    @pytest.fixture
    def small_deltanet(self):
        return GatedDeltaNet(
            hidden_size=64, head_dim=16, num_heads=4,
            conv_kernel=4, chunk_size=8,
        )

    def test_output_shape(self, small_deltanet):
        x = torch.randn(2, 16, 64)
        out, state = small_deltanet(x)
        assert out.shape == (2, 16, 64)
        assert state.shape == (2, 4, 16, 16)  # [B, H, D, D]

    def test_recurrent_matches_chunkwise(self, small_deltanet):
        """Recurrent and chunkwise modes should produce the same result."""
        small_deltanet.eval()
        x = torch.randn(1, 16, 64)
        out_r, state_r = small_deltanet(x, use_chunkwise=False)
        out_c, state_c = small_deltanet(x, use_chunkwise=True)
        torch.testing.assert_close(out_r, out_c, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state_r, state_c, atol=1e-4, rtol=1e-4)

    def test_state_updates(self, small_deltanet):
        """State should change after processing tokens."""
        x = torch.randn(1, 8, 64)
        _, state = small_deltanet(x, use_chunkwise=False)
        assert not torch.allclose(state, torch.zeros_like(state))

    def test_gradient_recurrent(self, small_deltanet):
        x = torch.randn(1, 8, 64, requires_grad=True)
        out, _ = small_deltanet(x, use_chunkwise=False)
        out.sum().backward()
        assert x.grad is not None

    def test_gradient_chunkwise(self, small_deltanet):
        x = torch.randn(1, 16, 64, requires_grad=True)
        out, _ = small_deltanet(x, use_chunkwise=True)
        out.sum().backward()
        assert x.grad is not None

    def test_single_token(self, small_deltanet):
        """Single-token input should work."""
        x = torch.randn(1, 1, 64)
        out, state = small_deltanet(x, use_chunkwise=False)
        assert out.shape == (1, 1, 64)

    def test_incremental_state(self, small_deltanet):
        """Passing state from previous call should affect output."""
        small_deltanet.eval()
        x1 = torch.randn(1, 4, 64)
        _, state1 = small_deltanet(x1, use_chunkwise=False)
        x2 = torch.randn(1, 4, 64)
        out_no_state, _ = small_deltanet(x2, state=None, use_chunkwise=False)
        out_with_state, _ = small_deltanet(x2, state=state1, use_chunkwise=False)
        # Outputs should differ when state is present
        assert not torch.allclose(out_no_state, out_with_state, atol=1e-3)

    def test_various_seq_lengths(self, small_deltanet):
        for S in [1, 7, 8, 15, 16, 33]:
            x = torch.randn(1, S, 64)
            out, _ = small_deltanet(x, use_chunkwise=True)
            assert out.shape == (1, S, 64)
