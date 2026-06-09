#!/bin/bash
set -e

# Always run from the script's own directory so callers can invoke as
# either ``cd bucket-brigade-core && ./build.sh`` or
# ``bash bucket-brigade-core/build.sh`` from the repo root.
cd "$(dirname "$0")"

# Build script for bucket-brigade-core (Rust PyO3 extension).
#
# As of issue #404 the build backend declared in pyproject.toml is
# **maturin** (previously: setuptools-rust). The canonical PyO3 build path
# is now ``maturin develop --release --features python``, which is also
# what CI (``.github/workflows/ci.yml``) and the cibuildwheel release
# workflow (``.github/workflows/wheels.yml``) call.
#
# IMPORTANT: ``--features python`` MUST be passed explicitly to maturin —
# the Rust crate's default feature set deliberately omits pyo3 so the
# same source tree can also build to wasm or as a pure-Rust library. The
# old issue #134 was a regression caused by forgetting this flag.
# ``[tool.maturin].features = ["python"]`` in pyproject.toml is a
# belt-and-braces second line of defence for backends that consult it.
#
# Environment variables (set defensively in this script):
#   PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
#     Allow PyO3 0.22.x to build against newer Python versions.
#   RUSTC_WRAPPER=
#     Unset sccache wrapper. Many docs in this repo export RUSTC_WRAPPER=sccache,
#     but on systems without sccache installed the build either fails or
#     silently produces a broken artifact. We unset it here to guarantee a
#     reproducible build.

echo "Building bucket-brigade-core Rust extension (maturin)..."

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Defensive: unset RUSTC_WRAPPER (e.g. sccache) so absence of sccache cannot
# cause silent fallback to a broken build.
export RUSTC_WRAPPER=

# Required for PyO3 0.22.x ABI3 forward compatibility (Python >= 3.13)
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Check Rust version
echo "Rust version: $(rustc --version)"

# Install Python build dependencies (maturin matches pyproject.toml build-system)
echo "Installing Python build dependencies..."
uv pip install "maturin>=1.5,<2.0"

# Clean previous CFFI-shadow artifacts (from earlier maturin-based builds where
# a nested directory was accidentally created) and any stale wheel output. The
# PyO3 .so file lives at
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

# Build via maturin develop. This installs the editable wheel into the
# currently-active venv, so callers should be in the bucket-brigade venv
# (e.g. invoked as ``uv run bash bucket-brigade-core/build.sh`` or after
# ``source .venv/bin/activate``).
#
# ``--no-project`` keeps uv from walking up to the workspace
# ``pyproject.toml`` and re-rooting the project context; without it
# maturin looks for ``Cargo.toml`` in the wrong directory.
echo "Building with maturin develop --release --features python..."
uv run --no-project maturin develop --release --features python

echo "Build complete."
echo ""
echo "Test import with:"
echo "  uv run python -c 'from bucket_brigade_core import BucketBrigade; print(BucketBrigade)'"
