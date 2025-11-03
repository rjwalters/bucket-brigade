#!/bin/bash
# sandbox.sh - Set up isolated Python environment for training
#
# This script creates a clean Python virtual environment for running
# training experiments, avoiding conflicts with other project environments.
#
# Usage:
#   source scripts/sandbox.sh    # Activates sandbox environment
#   ./scripts/sandbox.sh setup   # Just sets up without activating

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔧 Setting up sandbox environment for bucket-brigade..."

# Unset any existing virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "⚠️  Deactivating existing environment: $VIRTUAL_ENV"
    unset VIRTUAL_ENV
    unset PYTHONHOME
fi

cd "$PROJECT_ROOT"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Remove old venv if it exists (optional, for clean setup)
if [ "$1" == "--clean" ]; then
    echo "🧹 Removing existing .venv..."
    rm -rf .venv
fi

# Sync base dependencies
echo "📦 Syncing base dependencies..."
uv sync --quiet

# Install RL dependencies
echo "🤖 Installing RL dependencies (PyTorch, Gymnasium, etc.)..."
uv pip install --quiet -e ".[rl]"

# Verify PyTorch installation
echo ""
echo "✅ Sandbox environment ready!"
echo ""
uv run python -c "
import torch
import sys
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
" 2>/dev/null || echo "⚠️  Warning: Could not verify PyTorch installation"

echo ""
echo "To run training commands, use one of:"
echo "  uv run python scripts/train_simple.py ..."
echo "  source .venv/bin/activate && python scripts/train_simple.py ..."
echo ""

# If sourced (not executed), activate the environment
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    echo "🎯 Activating sandbox environment..."
    source .venv/bin/activate
    echo "   (use 'deactivate' to exit)"
fi
