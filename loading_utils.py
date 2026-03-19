"""
Utility functions for loading HuggingFace model checkpoints.

Supports both ``safetensors`` and legacy ``pytorch_model.bin`` formats,
including sharded checkpoints (``model-00001-of-00005.safetensors`` etc.).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def load_hf_state_dict(path_or_id: str | Path) -> dict[str, torch.Tensor]:
    """Load a HuggingFace checkpoint into a flat state dict.

    Parameters
    ----------
    path_or_id : str or Path
        Either a local directory containing checkpoint files, or a
        HuggingFace model ID (e.g. ``"Qwen/Qwen2-VL-7B-Instruct"``).

    Returns
    -------
    dict[str, torch.Tensor]
        Complete (merged) state dict with all parameters.

    Notes
    -----
    Loading priority:
    1. ``safetensors`` files (faster, memory-mapped)
    2. ``pytorch_model.bin`` files (legacy fallback)
    3. Download from HuggingFace Hub if path doesn't exist locally
    """
    path = Path(path_or_id)

    # ---- If local directory, load directly ----
    if path.is_dir():
        return _load_from_directory(path)

    # ---- Try HuggingFace Hub download ----
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=str(path_or_id),
            allow_patterns=["*.safetensors", "*.bin", "*.json"],
        )
        return _load_from_directory(Path(local_dir))
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install with: pip install huggingface_hub"
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Cannot load checkpoint from {path_or_id}: {e}"
        ) from e


def _load_from_directory(path: Path) -> dict[str, torch.Tensor]:
    """Load all weight files from a local directory."""
    state_dict: dict[str, torch.Tensor] = {}

    # Try safetensors first
    safetensor_files = sorted(path.glob("*.safetensors"))
    if safetensor_files:
        try:
            from safetensors.torch import load_file
            for sf in safetensor_files:
                state_dict.update(load_file(str(sf), device="cpu"))
            return state_dict
        except ImportError:
            pass  # Fall through to pytorch format

    # Try pytorch format
    bin_files = sorted(path.glob("pytorch_model*.bin"))
    if not bin_files:
        bin_files = sorted(path.glob("model*.bin"))
    if bin_files:
        for bf in bin_files:
            sd = torch.load(str(bf), map_location="cpu", weights_only=True)
            state_dict.update(sd)
        return state_dict

    # Single consolidated model file
    single = path / "model.safetensors"
    if single.exists():
        try:
            from safetensors.torch import load_file
            return load_file(str(single), device="cpu")
        except ImportError:
            pass

    single_pt = path / "pytorch_model.bin"
    if single_pt.exists():
        return torch.load(str(single_pt), map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        f"No checkpoint files found in {path}. "
        f"Expected *.safetensors or *.bin files."
    )


def tie_weights(model: torch.nn.Module, config: Any) -> None:
    """Tie ``embed_tokens.weight`` to ``lm_head.weight`` if config says so."""
    if getattr(config, "tie_word_embeddings", False):
        if hasattr(model, "embed_tokens") and hasattr(model, "lm_head"):
            model.lm_head.weight = model.embed_tokens.weight
