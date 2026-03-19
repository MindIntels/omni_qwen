"""Tests for MoE layer (TopKRouter, load balancing, MoELayer)."""

from __future__ import annotations

import torch
import pytest

from omni_qwen.moe import TopKRouter, load_balancing_loss, MoELayer


class TestTopKRouter:
    def test_output_shapes(self):
        router = TopKRouter(hidden_size=64, num_experts=8, top_k=2)
        x = torch.randn(16, 64)
        logits, indices, weights = router(x)
        assert logits.shape == (16, 8)
        assert indices.shape == (16, 2)
        assert weights.shape == (16, 2)

    def test_weights_sum_to_one(self):
        router = TopKRouter(64, 8, top_k=2)
        x = torch.randn(32, 64)
        _, _, weights = router(x)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(32), atol=1e-5, rtol=1e-5)

    def test_top_k_indices_valid(self):
        router = TopKRouter(64, 8, top_k=3)
        x = torch.randn(10, 64)
        _, indices, _ = router(x)
        assert (indices >= 0).all() and (indices < 8).all()

    def test_top_k_unique_per_token(self):
        router = TopKRouter(64, 8, top_k=3)
        x = torch.randn(10, 64)
        _, indices, _ = router(x)
        for i in range(10):
            assert len(set(indices[i].tolist())) == 3


class TestLoadBalancingLoss:
    def test_returns_scalar(self):
        logits = torch.randn(32, 8)
        indices = torch.randint(0, 8, (32, 2))
        loss = load_balancing_loss(logits, indices, 8)
        assert loss.dim() == 0

    def test_loss_nonnegative(self):
        logits = torch.randn(64, 4)
        indices = torch.randint(0, 4, (64, 2))
        loss = load_balancing_loss(logits, indices, 4)
        assert loss.item() >= 0


class TestMoELayer:
    def test_output_shape(self):
        moe = MoELayer(hidden_size=64, intermediate_size=128,
                        num_routed_experts=4, num_shared_experts=1, top_k=2)
        x = torch.randn(2, 8, 64)
        out, aux = moe(x)
        assert out.shape == (2, 8, 64)
        assert aux.dim() == 0

    def test_gradient_flows(self):
        moe = MoELayer(64, 128, num_routed_experts=4, top_k=2)
        x = torch.randn(1, 4, 64, requires_grad=True)
        out, aux = moe(x)
        (out.sum() + 0.01 * aux).backward()
        assert x.grad is not None

    def test_shared_expert_contributes(self):
        """With zero routed weights, shared expert should still produce output."""
        moe = MoELayer(64, 128, num_routed_experts=4,
                        num_shared_experts=1, top_k=2)
        moe.eval()
        x = torch.randn(1, 2, 64)
        out, _ = moe(x)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_no_shared_experts(self):
        moe = MoELayer(64, 128, num_routed_experts=4,
                        num_shared_experts=0, top_k=2)
        x = torch.randn(1, 4, 64)
        out, _ = moe(x)
        assert out.shape == (1, 4, 64)
