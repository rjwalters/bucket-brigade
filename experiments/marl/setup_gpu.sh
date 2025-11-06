#!/bin/bash
# GPU Instance Setup Script for Bucket Brigade PPO Training
# Usage: ./setup_gpu.sh [install_dir]
#
# If run from within bucket-brigade repo, uses current directory
# Otherwise clones to specified directory (default: ~/bucket-brigade)

set -e

echo "🚀 Setting up Bucket Brigade on GPU instance..."

# Determine installation directory
if [ -f "pyproject.toml" ] && grep -q "bucket-brigade" pyproject.toml 2>/dev/null; then
    echo "✓ Running from bucket-brigade directory"
    REPO_DIR="$(pwd)"
else
    INSTALL_DIR="${1:-$HOME/bucket-brigade}"
    if [ ! -d "$INSTALL_DIR" ]; then
        echo "📦 Cloning bucket-brigade to $INSTALL_DIR..."
        git clone https://github.com/rjwalters/bucket-brigade.git "$INSTALL_DIR"
    else
        echo "✓ Found existing repo at $INSTALL_DIR"
    fi
    REPO_DIR="$INSTALL_DIR"
    cd "$REPO_DIR"
fi

echo "📁 Working directory: $REPO_DIR"

# Check if we're on a GPU instance
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. Are you on a GPU instance?"
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create Python environment
echo "🐍 Creating Python environment..."
uv venv
source .venv/bin/activate

# Install dependencies with RL extras (torch, pufferlib, etc.)
echo "📚 Installing dependencies (this may take a few minutes)..."
uv sync --extra rl

# Build Rust core with PyO3 bindings
echo "🦀 Building Rust core..."
cd bucket-brigade-core
VIRTUAL_ENV="$(pwd)/../.venv" \
  PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
  ../.venv/bin/maturin develop --release --features python
cd ..

# Verify installation
echo ""
echo "✅ Verifying installation..."

# Check CUDA
.venv/bin/python << 'EOF'
import torch
cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()
print(f"🔍 CUDA Available: {cuda_available}")
if cuda_available:
    print(f"🔍 GPU Count: {device_count}")
    for i in range(device_count):
        print(f"🔍 GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠️  No CUDA devices found - training will use CPU")
EOF

# Check Rust environment
.venv/bin/python << 'EOF'
try:
    from bucket_brigade.envs.puffer_env_rust import make_rust_env
    import bucket_brigade_core as core

    print(f"✅ Rust core imported successfully")
    print(f"✅ Available scenarios: {len(core.SCENARIOS)}")

    # Test environment creation
    env = make_rust_env('trivial_cooperation', num_opponents=3)
    obs, _ = env.reset()
    print(f"✅ Environment created successfully (obs shape: {obs.shape})")

except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
EOF

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Start training:"
echo "     uv run python experiments/marl/train_gpu.py --steps 10000000 --scenario trivial_cooperation"
echo ""
echo "  2. Monitor with TensorBoard (in another terminal):"
echo "     tensorboard --logdir experiments/marl/runs/"
echo ""
echo "  3. Forward TensorBoard port to local machine:"
echo "     ssh -L 6006:localhost:6006 rwalters-sandbox-2"
echo "     Then open: http://localhost:6006"
echo ""
