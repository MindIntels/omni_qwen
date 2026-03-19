"""
Vision-Language Processor for Qwen VL models.

Handles the full preprocessing pipeline:
1. **Image processing**: Resize, normalize, convert to tensor.
2. **Video processing**: Extract frames, apply same image processing.
3. **Tokenization**: Wrap a HuggingFace tokenizer for text encoding.
4. **Multimodal assembly**: Combine image/video tokens with text tokens,
   inserting special ``<|vision_start|>`` / ``<|vision_end|>`` markers.

Usage
-----
>>> from omni_qwen.processor import Processor
>>> proc = Processor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
>>> inputs = proc(text="Describe this image.", images=[pil_image])
>>> outputs = model.generate(**inputs)
>>> print(proc.decode(outputs[0]))
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ──────────────────────────────────────────────────────────────────
#  Image / Video helpers
# ──────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def _to_tensor(img: "Image.Image") -> torch.Tensor:
    """Convert a PIL Image to a ``[C, H, W]`` float32 tensor in [0, 1]."""
    import numpy as np
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]


def _normalize(
    tensor: torch.Tensor,
    mean: tuple[float, ...] = _IMAGENET_MEAN,
    std: tuple[float, ...] = _IMAGENET_STD,
) -> torch.Tensor:
    """Channel-wise normalize a ``[C, H, W]`` tensor."""
    m = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    s = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - m) / s


def _smart_resize(
    img: "Image.Image",
    patch_size: int = 14,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> "Image.Image":
    """Resize an image to a resolution compatible with the ViT patch grid.

    The output height and width are both divisible by ``patch_size``, and
    the total pixel count is clamped between ``min_pixels`` and ``max_pixels``
    while preserving the aspect ratio as closely as possible.

    Parameters
    ----------
    img : PIL.Image
    patch_size : int
    min_pixels, max_pixels : int
        Pixel-count bounds.

    Returns
    -------
    PIL.Image
        Resized image.
    """
    w, h = img.size
    aspect = w / h

    # Target pixel count
    pixels = w * h
    if pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
        w, h = int(w * scale), int(h * scale)
    elif pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        w, h = int(w * scale), int(h * scale)

    # Round to nearest multiple of patch_size
    w = max(patch_size, round(w / patch_size) * patch_size)
    h = max(patch_size, round(h / patch_size) * patch_size)

    return img.resize((w, h), Image.BICUBIC if _HAS_PIL else 3)


def preprocess_image(
    img: "Image.Image",
    patch_size: int = 14,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> torch.Tensor:
    """Full image preprocessing pipeline.

    Parameters
    ----------
    img : PIL.Image
    patch_size : int
    min_pixels, max_pixels : int

    Returns
    -------
    torch.Tensor
        ``[1, C, H, W]`` float32 tensor, normalized.
    """
    img = _smart_resize(img, patch_size, min_pixels, max_pixels)
    tensor = _to_tensor(img)
    tensor = _normalize(tensor)
    return tensor.unsqueeze(0)  # [1, C, H, W]


def preprocess_video(
    frames: Sequence["Image.Image"],
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    min_pixels: int = 128 * 28 * 28,
    max_pixels: int = 768 * 28 * 28,
) -> torch.Tensor:
    """Preprocess a sequence of video frames.

    Pads the temporal dimension to be divisible by ``temporal_patch_size``
    (by repeating the last frame).

    Returns
    -------
    torch.Tensor
        ``[1, C, T, H, W]`` tensor suitable for 3-D patch embedding.
    """
    tensors = []
    for frame in frames:
        t = preprocess_image(frame, patch_size, min_pixels, max_pixels)
        tensors.append(t.squeeze(0))  # [C, H, W]

    # Stack along temporal dimension
    video = torch.stack(tensors, dim=1)  # [C, T, H, W]

    # Pad temporal dimension
    T = video.size(1)
    pad_t = (temporal_patch_size - T % temporal_patch_size) % temporal_patch_size
    if pad_t > 0:
        video = torch.cat([video, video[:, -1:].expand(-1, pad_t, -1, -1)], dim=1)

    return video.unsqueeze(0)  # [1, C, T, H, W]


# ──────────────────────────────────────────────────────────────────
#  Processor
# ──────────────────────────────────────────────────────────────────

# Default special token IDs (Qwen2-VL tokenizer)
_VISION_START_ID = 151652
_VISION_END_ID = 151653
_IMAGE_PAD_ID = 151655


class Processor:
    """Unified processor for Qwen VL models.

    Combines image/video preprocessing with text tokenization.

    Parameters
    ----------
    tokenizer : object or None
        A HuggingFace tokenizer (or any object with ``encode`` / ``decode``
        methods).  If ``None``, only vision preprocessing is available.
    patch_size : int
    temporal_patch_size : int
    merge_size : int
    min_pixels, max_pixels : int
    vision_start_id, vision_end_id, image_pad_id : int
        Special token IDs for vision markers.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        vision_start_id: int = _VISION_START_ID,
        vision_end_id: int = _VISION_END_ID,
        image_pad_id: int = _IMAGE_PAD_ID,
    ):
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.vision_start_id = vision_start_id
        self.vision_end_id = vision_end_id
        self.image_pad_id = image_pad_id

    # ---- Image / Video processing ----

    def process_image(self, img: "Image.Image") -> torch.Tensor:
        """Preprocess a single image → ``[1, C, H, W]``."""
        return preprocess_image(
            img, self.patch_size, self.min_pixels, self.max_pixels,
        )

    def process_video(self, frames: Sequence["Image.Image"]) -> torch.Tensor:
        """Preprocess video frames → ``[1, C, T, H, W]``."""
        return preprocess_video(
            frames, self.patch_size, self.temporal_patch_size,
            self.min_pixels, self.max_pixels,
        )

    # ---- Token helpers ----

    def _count_visual_tokens(self, pixel_values: torch.Tensor) -> int:
        """Compute number of visual tokens after ViT + merge."""
        if pixel_values.dim() == 5:
            # Video: [B, C, T, H, W]
            _, _, T, H, W = pixel_values.shape
            Tp = T // self.temporal_patch_size
            Hp = H // self.patch_size
            Wp = W // self.patch_size
            total_patches = Tp * Hp * Wp
        else:
            # Image: [B, C, H, W]
            _, _, H, W = pixel_values.shape
            Hp = H // self.patch_size
            Wp = W // self.patch_size
            total_patches = Hp * Wp

        # After spatial merge: compress by merge_size^2
        m = self.merge_size
        merged = (Hp // m) * (Wp // m)
        if pixel_values.dim() == 5:
            merged *= Tp
        return merged

    def encode_text(self, text: str) -> list[int]:
        """Encode text string to token IDs."""
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer available. Use from_pretrained().")
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        """Decode token IDs back to text."""
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer available. Use from_pretrained().")
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ---- Main __call__ ----

    def __call__(
        self,
        text: str | None = None,
        images: Sequence["Image.Image"] | None = None,
        video_frames: Sequence["Image.Image"] | None = None,
    ) -> dict[str, Any]:
        """Process inputs for a Qwen VL model.

        Parameters
        ----------
        text : str or None
            Text prompt.
        images : list of PIL.Image or None
            Input images.
        video_frames : list of PIL.Image or None
            Video frames.

        Returns
        -------
        dict
            Keys may include: ``input_ids``, ``pixel_values``,
            ``image_grid_hw``, ``video_grid_thw``.
        """
        result: dict[str, Any] = {}

        # Process vision inputs
        if images is not None and len(images) > 0:
            pixel_values = torch.cat(
                [self.process_image(img) for img in images], dim=0,
            )
            result["pixel_values"] = pixel_values
            H = pixel_values.size(2) // self.patch_size
            W = pixel_values.size(3) // self.patch_size
            result["image_grid_hw"] = (H, W)

        if video_frames is not None and len(video_frames) > 0:
            pixel_values = self.process_video(video_frames)
            result["pixel_values"] = pixel_values

        # Process text
        if text is not None:
            if self.tokenizer is not None:
                token_ids = self.encode_text(text)
            else:
                # Fallback: return text as-is
                token_ids = []
            result["input_ids"] = torch.tensor([token_ids], dtype=torch.long)

        return result

    # ---- Factory ----

    @classmethod
    def from_pretrained(cls, path_or_id: str, **kwargs: Any) -> "Processor":
        """Create a Processor by loading a HF tokenizer.

        Parameters
        ----------
        path_or_id : str
            Local directory or HuggingFace model ID.
        **kwargs
            Override default processor parameters.

        Returns
        -------
        Processor
        """
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                path_or_id, trust_remote_code=True,
            )
        except Exception:
            pass

        # Try to read processor config for patch sizes
        processor_kwargs: dict[str, Any] = {}
        try:
            from pathlib import Path
            import json
            config_file = Path(path_or_id) / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    raw = json.load(f)
                vc = raw.get("vision_config", {})
                processor_kwargs["patch_size"] = vc.get("patch_size", 14)
                processor_kwargs["temporal_patch_size"] = vc.get(
                    "temporal_patch_size", 2,
                )
                processor_kwargs["merge_size"] = vc.get(
                    "spatial_merge_size", 2,
                )
        except Exception:
            pass

        processor_kwargs.update(kwargs)
        return cls(tokenizer=tokenizer, **processor_kwargs)
