#!/usr/bin/env bash
# Run all tests for the omni_qwen project.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."   # project root (llm/)

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python3}"

echo "============================================="
echo "  omni_qwen — Architecture Test Suite"
echo "============================================="

echo ""
echo ">>> Running RoPE tests ..."
$PYTHON -m pytest omni_qwen/tests/test_rope.py -v

echo ""
echo ">>> Running Window Attention tests ..."
$PYTHON -m pytest omni_qwen/tests/test_window_attention.py -v

echo ""
echo ">>> Running ViT tests ..."
$PYTHON -m pytest omni_qwen/tests/test_vit.py -v

echo ""
echo ">>> Running Decoder tests ..."
$PYTHON -m pytest omni_qwen/tests/test_decoder.py -v

echo ""
echo ">>> Running MoE tests ..."
$PYTHON -m pytest omni_qwen/tests/test_moe.py -v

echo ""
echo ">>> Running Gated DeltaNet tests ..."
$PYTHON -m pytest omni_qwen/tests/test_gated_deltanet.py -v

echo ""
echo ">>> Running Gated Attention tests ..."
$PYTHON -m pytest omni_qwen/tests/test_gated_attention.py -v

echo ""
echo ">>> Running Model integration tests ..."
$PYTHON -m pytest omni_qwen/tests/test_models.py -v

echo ""
echo ">>> Running Config / Weight-Mapping / Generate tests ..."
$PYTHON -m pytest omni_qwen/tests/test_config.py -v

echo ""
echo ">>> Running Weight Mapping validation tests ..."
$PYTHON -m pytest omni_qwen/tests/test_weight_mapping.py -v

echo ""
echo ">>> Running Processor tests ..."
$PYTHON -m pytest omni_qwen/tests/test_processor.py -v

echo ""
echo "============================================="
echo "  All omni_qwen tests passed!"
echo "============================================="
