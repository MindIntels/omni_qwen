"""Tests for processor.py (vision preprocessing + tokenization)."""

from __future__ import annotations

import math
import torch
import pytest

from omni_qwen.processor import (
    Processor,
    preprocess_image,
    preprocess_video,
    _smart_resize,
    _to_tensor,
    _normalize,
)

# Skip tests that need PIL if not installed
PIL = pytest.importorskip("PIL")
from PIL import Image


def _dummy_image(w: int = 224, h: int = 224) -> Image.Image:
    """Create a dummy RGB PIL image."""
    return Image.new("RGB", (w, h), color=(128, 64, 32))


class TestSmartResize:
    def test_output_divisible_by_patch_size(self):
        img = _dummy_image(100, 150)
        resized = _smart_resize(img, patch_size=14)
        w, h = resized.size
        assert w % 14 == 0 and h % 14 == 0

    def test_min_pixels(self):
        img = _dummy_image(28, 28)  # 784 pixels
        resized = _smart_resize(img, patch_size=14, min_pixels=14 * 14 * 4)
        w, h = resized.size
        assert w * h >= 14 * 14 * 4

    def test_max_pixels(self):
        img = _dummy_image(1000, 1000)
        resized = _smart_resize(img, patch_size=14, max_pixels=200 * 200)
        w, h = resized.size
        assert w * h <= 200 * 200 * 2  # some tolerance


class TestToTensor:
    def test_shape_and_range(self):
        img = _dummy_image(56, 56)
        t = _to_tensor(img)
        assert t.shape == (3, 56, 56)
        assert t.min() >= 0.0 and t.max() <= 1.0


class TestNormalize:
    def test_output_range(self):
        t = torch.rand(3, 56, 56)
        n = _normalize(t)
        assert n.shape == (3, 56, 56)
        # After normalization, values should be roughly centered
        assert n.mean().abs() < 2.0  # loose check


class TestPreprocessImage:
    def test_output_shape(self):
        img = _dummy_image(224, 224)
        tensor = preprocess_image(img, patch_size=14)
        assert tensor.dim() == 4  # [1, C, H, W]
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3
        assert tensor.shape[2] % 14 == 0
        assert tensor.shape[3] % 14 == 0


class TestPreprocessVideo:
    def test_output_shape(self):
        frames = [_dummy_image(112, 112) for _ in range(4)]
        tensor = preprocess_video(frames, patch_size=14, temporal_patch_size=2)
        assert tensor.dim() == 5  # [1, C, T, H, W]
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3
        assert tensor.shape[2] % 2 == 0  # divisible by temporal_patch_size

    def test_temporal_padding(self):
        # 3 frames should be padded to 4 (nearest multiple of 2)
        frames = [_dummy_image(56, 56) for _ in range(3)]
        tensor = preprocess_video(frames, patch_size=14, temporal_patch_size=2)
        assert tensor.shape[2] == 4


class TestProcessor:
    @pytest.fixture
    def proc(self):
        return Processor(patch_size=14, merge_size=2)

    def test_process_image(self, proc):
        img = _dummy_image(224, 224)
        tensor = proc.process_image(img)
        assert tensor.dim() == 4

    def test_call_with_images(self, proc):
        imgs = [_dummy_image(112, 112)]
        result = proc(images=imgs)
        assert "pixel_values" in result
        assert "image_grid_hw" in result
        assert result["pixel_values"].dim() == 4

    def test_call_text_only_no_tokenizer(self, proc):
        result = proc(text="hello")
        # Without tokenizer, encode_text raises, so input_ids is empty
        assert "input_ids" in result
        assert result["input_ids"].shape == (1, 0)

    def test_count_visual_tokens_image(self, proc):
        # 224x224 image, patch_size=14 → 16x16 patches, merge 2x2 → 8x8=64
        pv = torch.randn(1, 3, 224, 224)
        n = proc._count_visual_tokens(pv)
        assert n == 64

    def test_count_visual_tokens_video(self, proc):
        # [1, 3, 4, 112, 112], patch=14 → T=2, H=8, W=8, merge 2x2 → 2*4*4=32
        pv = torch.randn(1, 3, 4, 112, 112)
        proc_v = Processor(patch_size=14, temporal_patch_size=2, merge_size=2)
        n = proc_v._count_visual_tokens(pv)
        assert n == 32


class TestProcessorWithMockTokenizer:
    """Test Processor with a simple mock tokenizer."""

    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    @pytest.fixture
    def proc(self):
        return Processor(tokenizer=self.MockTokenizer(), patch_size=14)

    def test_encode_decode(self, proc):
        ids = proc.encode_text("hello world")
        assert ids == [0, 1]
        text = proc.decode(ids)
        assert text == "0 1"

    def test_call_multimodal(self, proc):
        imgs = [_dummy_image(112, 112)]
        result = proc(text="hello world", images=imgs)
        assert "input_ids" in result
        assert "pixel_values" in result
        assert result["input_ids"].shape == (1, 2)
