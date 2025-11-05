#!/bin/bash
# Build script for bucket-brigade-core Rust module
# This script compiles the Rust code and copies the resulting .so file
# to the correct location for Python to import.

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building bucket-brigade-core Rust module ===${NC}"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "Detected Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Error: Python 3.9+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 14 ]; then
    echo -e "${YELLOW}Warning: Python 3.14+ detected. PyO3 0.22.6 only supports up to Python 3.13.${NC}"
    echo -e "${YELLOW}Build may fail. Please use Python 3.12 or 3.13.${NC}"
    echo ""
fi

# Check for Rust/Cargo
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Rust/Cargo not found. Please install Rust:${NC}"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

CARGO_VERSION=$(cargo --version | awk '{print $2}')
echo "Detected Cargo version: $CARGO_VERSION"
echo ""

# Clean previous builds (optional, comment out for incremental builds)
# echo "Cleaning previous builds..."
# cargo clean

# Build Rust library with Python bindings
echo -e "${GREEN}Building Rust library (release mode)...${NC}"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo build --release --features python

# Check if build succeeded
if [ ! -f "target/release/libbucket_brigade_core.so" ] && [ ! -f "target/release/libbucket_brigade_core.dylib" ]; then
    echo -e "${RED}Error: Build failed - no output library found${NC}"
    exit 1
fi

# Determine the platform-specific extension
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    SO_SOURCE="target/release/libbucket_brigade_core.dylib"
    SO_DEST="bucket_brigade_core/bucket_brigade_core.cpython-${PYTHON_MAJOR}${PYTHON_MINOR}-darwin.so"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    SO_SOURCE="target/release/libbucket_brigade_core.so"
    SO_DEST="bucket_brigade_core/bucket_brigade_core.cpython-${PYTHON_MAJOR}${PYTHON_MINOR}-x86_64-linux-gnu.so"
else
    echo -e "${RED}Error: Unsupported platform$(NC)"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p bucket_brigade_core

# Copy the built library
echo -e "${GREEN}Copying library to Python package...${NC}"
cp "$SO_SOURCE" "$SO_DEST"

# Verify the copy
if [ -f "$SO_DEST" ]; then
    SIZE=$(ls -lh "$SO_DEST" | awk '{print $5}')
    echo -e "${GREEN}✓ Successfully built and installed:${NC}"
    echo "  $SO_DEST ($SIZE)"
else
    echo -e "${RED}Error: Failed to copy library to destination${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Build complete! ===${NC}"
echo ""
echo "Test the module with:"
echo "  python -c 'from bucket_brigade.evolution import FitnessEvaluator; print(FitnessEvaluator.__name__)'"
echo ""
echo "Expected output: RustFitnessEvaluator"
