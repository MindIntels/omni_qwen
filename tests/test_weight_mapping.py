"""Round-trip validation tests for weight_mapping.py.

For each model variant, we:
1. Instantiate a tiny model.
2. Get its state_dict keys.
3. Build the HF ↔ ours weight mapping.
4. Verify every "our" key in the mapping exists in the model's state_dict.
"""

from __future__ import annotations

import pytest
import torch

from omni_qwen.weight_mapping import build_weight_mapping
from omni_qwen.qwen2_vl import Qwen2VL
from omni_qwen.qwen25_vl import Qwen25VL
from omni_qwen.qwen3_vl import Qwen3VL
from omni_qwen.qwen3_next import Qwen3NeXT


_VIT_LAYERS = 2
_LLM_LAYERS = 2
_TINY = dict(
    vit_hidden=96, llm_hidden=96, llm_layers=_LLM_LAYERS,
    num_q_heads=4, num_kv_heads=2, intermediate_size=192,
    vocab_size=256, vit_layers=_VIT_LAYERS, vit_heads=4, patch_size=7,
)


class TestQwen2VLMapping:
    def test_all_mapped_keys_exist(self):
        model = Qwen2VL(**_TINY, window_size=(2, 2))
        sd_keys = set(model.state_dict().keys())
        mapping = build_weight_mapping("qwen2_vl", _LLM_LAYERS, _VIT_LAYERS)

        missing = []
        for our_key in mapping:
            if our_key not in sd_keys:
                missing.append(our_key)

        assert missing == [], (
            f"Mapping contains {len(missing)} keys not found in model state_dict:\n"
            + "\n".join(missing[:20])
        )

    def test_coverage(self):
        """Mapped keys should cover most model params (allow unmapped RoPE buffers etc)."""
        model = Qwen2VL(**_TINY, window_size=(2, 2))
        sd_keys = set(model.state_dict().keys())
        mapping = build_weight_mapping("qwen2_vl", _LLM_LAYERS, _VIT_LAYERS)
        mapped_set = set(mapping.keys())
        coverage = len(mapped_set & sd_keys) / len(sd_keys) if sd_keys else 0
        assert coverage > 0.7, f"Only {coverage:.0%} of model keys are mapped"


class TestQwen25VLMapping:
    def test_all_mapped_keys_exist(self):
        model = Qwen25VL(**_TINY, window_size=(2, 2))
        sd_keys = set(model.state_dict().keys())
        mapping = build_weight_mapping("qwen25_vl", _LLM_LAYERS, _VIT_LAYERS)

        missing = []
        for our_key in mapping:
            if our_key not in sd_keys:
                missing.append(our_key)

        assert missing == [], (
            f"Mapping contains {len(missing)} keys not found in model state_dict:\n"
            + "\n".join(missing[:20])
        )


class TestQwen3VLMapping:
    def test_all_mapped_keys_exist(self):
        tiny = {k: v for k, v in _TINY.items() if k != 'intermediate_size'}
        model = Qwen3VL(
            **tiny,
            num_routed_experts=4, num_shared_experts=1,
            expert_intermediate=64, shared_intermediate=192, top_k=2,
        )
        sd_keys = set(model.state_dict().keys())
        mapping = build_weight_mapping(
            "qwen3_vl", _LLM_LAYERS, _VIT_LAYERS, num_experts=4,
        )

        missing = []
        for our_key in mapping:
            if our_key not in sd_keys:
                missing.append(our_key)

        assert missing == [], (
            f"Mapping contains {len(missing)} keys not found in model state_dict:\n"
            + "\n".join(missing[:20])
        )


class TestQwen3NeXTMapping:
    def test_all_mapped_keys_exist(self):
        model = Qwen3NeXT(
            vit_hidden=96, llm_hidden=96, llm_layers=8,
            num_q_heads=4, num_kv_heads=2, intermediate_size=192,
            vocab_size=256, attn_every=4, deltanet_chunk=8,
            vit_layers=_VIT_LAYERS, vit_heads=4, patch_size=7,
        )
        sd_keys = set(model.state_dict().keys())
        mapping = build_weight_mapping(
            "qwen3_next", 8, _VIT_LAYERS, attn_every=4,
        )

        missing = []
        for our_key in mapping:
            if our_key not in sd_keys:
                missing.append(our_key)

        assert missing == [], (
            f"Mapping contains {len(missing)} keys not found in model state_dict:\n"
            + "\n".join(missing[:20])
        )


class TestRoundTrip:
    """Verify map_state_dict → load_state_dict works without error."""

    def test_qwen2_vl_roundtrip(self):
        from omni_qwen.weight_mapping import map_state_dict

        model = Qwen2VL(**_TINY, window_size=(2, 2))
        mapping = build_weight_mapping("qwen2_vl", _LLM_LAYERS, _VIT_LAYERS)

        # Simulate HF state dict from our model
        our_sd = model.state_dict()
        reverse = {v: k for k, v in mapping.items()}
        hf_sd = {}
        for our_key, tensor in our_sd.items():
            if our_key in mapping:
                hf_sd[mapping[our_key]] = tensor

        # Map back
        recovered = map_state_dict(hf_sd, mapping, strict=False)

        # Should be able to load without error
        model2 = Qwen2VL(**_TINY, window_size=(2, 2))
        model2.load_state_dict(recovered, strict=False)

        # Verify at least one weight matches
        for key in recovered:
            if key in our_sd:
                assert torch.equal(recovered[key], our_sd[key]), f"Mismatch on {key}"
                break
