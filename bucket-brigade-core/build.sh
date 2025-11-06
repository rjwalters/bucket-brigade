#!/bin/bash
set -e

echo "🔧 Building bucket-brigade-core Rust extension..."

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "📦 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Check Rust version
echo "✅ Rust version: $(rustc --version)"

# Install Python build dependencies
echo "📦 Installing Python dependencies..."
uv pip install cffi maturin

# Clean CFFI artifacts but keep main .so file
echo "🧹 Cleaning CFFI artifacts..."
if [ -d bucket_brigade_core/bucket_brigade_core ]; then
    rm -rf bucket_brigade_core/bucket_brigade_core
fi
rm -rf target/wheels

# Build with maturin
echo "🔨 Building with maturin..."
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv run maturin develop --release

# Clean up CFFI artifacts created during build
echo "🧹 Cleaning up CFFI artifacts..."
if [ -d bucket_brigade_core/bucket_brigade_core ]; then
    rm -rf bucket_brigade_core/bucket_brigade_core
    echo "   Removed nested CFFI directory"
fi

echo "✅ Build complete!"
echo ""
echo "Test import with:"
echo "  cd /tmp && uv run python -c 'from bucket_brigade_core import VectorEnv; print(VectorEnv)'"
