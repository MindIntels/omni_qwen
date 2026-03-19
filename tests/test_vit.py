"""Tests for ViT (PatchEmbed, ViTBlock, VisionTransformer)."""

from __future__ import annotations

import torch
import pytest

from omni_qwen.vit import PatchEmbed2D, PatchEmbed3D, ViTBlock, ViTMLP, VisionTransformer


class TestPatchEmbed:
    def test_output_shape(self):
        pe = PatchEmbed2D(patch_size=14, in_channels=3, embed_dim=256)
        img = torch.randn(2, 3, 56, 56)  # 4×4 patches
        tokens, Hp, Wp = pe(img)
        assert tokens.shape == (2, 16, 256)
        assert Hp == 4 and Wp == 4

    def test_non_square_image(self):
        pe = PatchEmbed2D(patch_size=14, in_channels=3, embed_dim=128)
        img = torch.randn(1, 3, 28, 56)  # 2×4 patches
        tokens, Hp, Wp = pe(img)
        assert tokens.shape == (1, 8, 128)
        assert Hp == 2 and Wp == 4


class TestPatchEmbed3D:
    def test_output_shape(self):
        pe = PatchEmbed3D(patch_size=7, temporal_patch_size=2, in_channels=3, embed_dim=96)
        video = torch.randn(1, 3, 4, 28, 28)  # T=4, H=28, W=28
        tokens, Tp, Hp, Wp = pe(video)
        assert Tp == 2  # 4 / 2
        assert Hp == 4  # 28 / 7
        assert Wp == 4  # 28 / 7
        assert tokens.shape == (1, 2 * 4 * 4, 96)

    def test_gradient(self):
        pe = PatchEmbed3D(patch_size=7, temporal_patch_size=2, embed_dim=64)
        video = torch.randn(1, 3, 2, 14, 14, requires_grad=True)
        tokens, Tp, Hp, Wp = pe(video)
        tokens.sum().backward()
        assert video.grad is not None


class TestViTMLP:
    def test_shape(self):
        mlp = ViTMLP(64, 128)
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_has_fc1_fc2(self):
        mlp = ViTMLP(64, 128)
        assert hasattr(mlp, 'fc1')
        assert hasattr(mlp, 'fc2')
        assert mlp.fc1.in_features == 64
        assert mlp.fc1.out_features == 128
        assert mlp.fc2.in_features == 128
        assert mlp.fc2.out_features == 64


class TestPatchEmbed:
    def test_output_shape(self):
        pe = PatchEmbed2D(patch_size=14, in_channels=3, embed_dim=256)
        img = torch.randn(2, 3, 56, 56)  # 4×4 patches
        tokens, Hp, Wp = pe(img)
        assert tokens.shape == (2, 16, 256)
        assert Hp == 4 and Wp == 4

    def test_non_square_image(self):
        pe = PatchEmbed2D(patch_size=14, in_channels=3, embed_dim=128)
        img = torch.randn(1, 3, 28, 56)  # 2×4 patches
        tokens, Hp, Wp = pe(img)
        assert tokens.shape == (1, 8, 128)
        assert Hp == 2 and Wp == 4


class TestViTBlock:
    def test_window_block_shape(self):
        block = ViTBlock(hidden_size=64, num_heads=4, intermediate_size=128,
                         window_size=(4, 4), use_rope=False)
        x = torch.randn(2, 16, 64)
        out = block(x, H=4, W=4)
        assert out.shape == (2, 16, 64)

    def test_global_block_shape(self):
        block = ViTBlock(hidden_size=64, num_heads=4, intermediate_size=128,
                         window_size=None, use_rope=False)
        x = torch.randn(2, 16, 64)
        out = block(x, H=4, W=4)
        assert out.shape == (2, 16, 64)

    def test_gradient(self):
        block = ViTBlock(64, 4, 128, window_size=None, use_rope=False)
        x = torch.randn(1, 9, 64, requires_grad=True)
        out = block(x, 3, 3)
        out.sum().backward()
        assert x.grad is not None


class TestVisionTransformer:
    @pytest.fixture
    def small_vit(self):
        return VisionTransformer(
            patch_size=7, in_channels=3, hidden_size=64, num_heads=4,
            num_layers=4, intermediate_size=128, window_size=(2, 2),
            global_attn_every=2, rope_dim=2,
        )

    def test_output_shape(self, small_vit):
        img = torch.randn(2, 3, 28, 28)  # 4×4 patches
        out = small_vit(img)
        assert out.shape == (2, 16, 64)

    def test_gradient(self, small_vit):
        img = torch.randn(1, 3, 28, 28, requires_grad=True)
        out = small_vit(img)
        out.sum().backward()
        assert img.grad is not None

    def test_3d_rope_vit(self):
        vit = VisionTransformer(
            patch_size=7, in_channels=3, hidden_size=96, num_heads=4,
            num_layers=2, intermediate_size=192, window_size=(2, 2),
            global_attn_every=2, rope_dim=3,
        )
        img = torch.randn(1, 3, 28, 28)
        out = vit(img)
        assert out.shape == (1, 16, 96)

    def test_video_forward(self):
        """Test 3-D RoPE ViT with 5-D video input."""
        vit = VisionTransformer(
            patch_size=7, in_channels=3, hidden_size=96, num_heads=4,
            num_layers=2, intermediate_size=192, window_size=(2, 2),
            global_attn_every=2, rope_dim=3, temporal_patch_size=2,
        )
        video = torch.randn(1, 3, 4, 28, 28)  # [B, C, T, H, W]
        out = vit(video)
        # T=4 / temporal_patch=2 → Tp=2, H=28/7=4, W=4 → 2*4*4=32 tokens
        assert out.shape == (1, 32, 96)

    def test_layernorm_in_blocks(self):
        """ViT blocks should use LayerNorm (with bias), not RMSNorm."""
        import torch.nn as nn
        vit = VisionTransformer(
            patch_size=7, hidden_size=64, num_heads=4,
            num_layers=2, intermediate_size=128, window_size=(2, 2),
        )
        block = vit.blocks[0]
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)
        assert block.norm1.bias is not None  # LayerNorm has bias
