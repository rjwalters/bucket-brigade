#!/bin/bash
set -e

# Build script for bucket-brigade-core (Rust PyO3 extension).
#
# IMPORTANT: This script uses the setuptools-rust build path declared in
# pyproject.toml (build-backend = "setuptools.build_meta"). The previous
# maturin-based path silently fell back to CFFI bindings when --features python
# was not passed, producing a non-importable package. See issue #134.
#
# Environment variables (set defensively in this script):
#   PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
#     Allow PyO3 0.22.x to build against newer Python versions.
#   RUSTC_WRAPPER=
#     Unset sccache wrapper. Many docs in this repo export RUSTC_WRAPPER=sccache,
#     but on systems without sccache installed the build either fails or
#     silently produces a broken artifact. We unset it here to guarantee a
#     reproducible build.

echo "Building bucket-brigade-core Rust extension..."

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Defensive: unset RUSTC_WRAPPER (e.g. sccache) so absence of sccache cannot
# cause silent fallback to the CFFI build path.
export RUSTC_WRAPPER=

# Required for PyO3 0.22.x ABI3 forward compatibility (Python >= 3.13)
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Check Rust version
echo "Rust version: $(rustc --version)"

# Install Python build dependencies (setuptools-rust matches pyproject.toml)
echo "Installing Python build dependencies..."
uv pip install setuptools-rust

# Clean previous CFFI-shadow artifacts (from earlier maturin-based builds) and
# any stale wheel output. The PyO3 .so file lives at
#   bucket_brigade_core/bucket_brigade_core.cpython-*.so
# whereas the broken CFFI shadow lives in the nested directory
#   bucket_brigade_core/bucket_brigade_core/
# Removing the nested directory is safe and required to recover from a
# previously broken build.
echo "Cleaning stale CFFI-shadow artifacts..."
if [ -d bucket_brigade_core/bucket_brigade_core ]; then
    rm -rf bucket_brigade_core/bucket_brigade_core
    echo "  Removed nested CFFI-shadow directory"
fi
rm -rf target/wheels

# Build via the setuptools-rust backend declared in pyproject.toml.
# --no-build-isolation lets the build use the active venv's toolchain
# (avoids re-resolving setuptools-rust in an isolated environment).
echo "Building with setuptools-rust (pip install -e .)..."
uv run python -m pip install -e . --no-build-isolation

echo "Build complete."
echo ""
echo "Test import with:"
echo "  uv run python -c 'from bucket_brigade_core import BucketBrigade; print(BucketBrigade)'"
