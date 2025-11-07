#!/bin/bash
# Setup and Launch V7 Evolution on Remote Server
#
# This script handles the complete setup needed to run V7 evolution:
# 1. Ensures Rust toolchain is available
# 2. Builds the bucket-brigade-core Rust module
# 3. Launches V7 evolution for all scenarios in tmux sessions

set -e  # Exit on error

echo "========================================"
echo "V7 Evolution: Remote Setup and Launch"
echo "========================================"
echo ""

# Change to bucket-brigade directory
cd ~/bucket-brigade || cd /root/bucket-brigade || { echo "Error: bucket-brigade directory not found"; exit 1; }

echo "Working directory: $(pwd)"
echo ""

# Step 1: Ensure Rust is installed
echo "[1/5] Checking Rust installation..."
if ! command -v rustc &> /dev/null; then
    echo "  Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
else
    echo "  ✓ Rust already installed: $(rustc --version)"
    source ~/.cargo/env 2>/dev/null || true
fi
echo ""

# Step 2: Pull latest code
echo "[2/5] Updating code from git..."
git fetch origin
git reset --hard origin/main
echo "  ✓ Code updated to: $(git rev-parse --short HEAD)"
echo ""

# Step 3: Sync Python dependencies
echo "[3/5] Syncing Python dependencies..."
uv sync
echo "  ✓ Dependencies synced"
echo ""

# Step 4: Build Rust module
echo "[4/5] Building bucket-brigade-core Rust module..."

# Install build dependencies (from parent dir to use correct venv)
uv pip install cffi maturin

# Build and install the Rust module
cd bucket-brigade-core

# Clean any previous CFFI build artifacts
rm -rf bucket_brigade_core/bucket_brigade_core/

export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
export VIRTUAL_ENV="$(pwd)/../.venv"

# Build with PyO3 (not CFFI)
~/.local/bin/maturin develop --release --features python

echo "  ✓ Rust module built and installed"
cd ..
echo ""

# Step 5: Verify installation
echo "[5/5] Verifying installation..."
if uv run python -c "import bucket_brigade_core as core; print('  ✓ bucket_brigade_core:', list(core.SCENARIOS.keys())[:3])"; then
    echo "  ✓ Rust module import successful"
else
    echo "  ✗ Rust module import failed"
    exit 1
fi
echo ""

# Step 6: Launch V7 evolution sessions
echo "========================================"
echo "Launching V7 Evolution Sessions"
echo "========================================"
echo ""

mkdir -p logs/evolution

SCENARIOS="chain_reaction deceptive_calm default early_containment easy greedy_neighbor hard mixed_motivation overcrowding rest_trap sparse_heroics trivial_cooperation"

for scenario in $SCENARIOS; do
    echo "Launching: v7_$scenario"

    # Kill existing session if it exists
    tmux kill-session -t "v7_$scenario" 2>/dev/null || true

    # Create new tmux session
    tmux new-session -d -s "v7_$scenario" bash -c "
        cd ~/bucket-brigade 2>/dev/null || cd /root/bucket-brigade
        source ~/.cargo/env 2>/dev/null || true

        uv run python scripts/evolve_v7.py \
          --scenario $scenario \
          --population 200 \
          --generations 200 \
          --mutation-rate 0.15 \
          --games-per-eval 100 \
          --workers 4 \
          --num-agents 4 \
          --output experiments/scenarios/$scenario/evolved_v7/ \
          2>&1 | tee logs/evolution/v7_$scenario.log

        echo ''
        echo 'Evolution complete! Press Enter to exit.'
        read
    "

    sleep 0.5
done

echo ""
echo "========================================"
echo "All V7 Sessions Launched!"
echo "========================================"
echo ""
echo "Sessions:"
tmux ls | grep v7_ || echo "  (checking...)"
echo ""
echo "Monitor progress:"
echo "  tmux attach -t v7_easy"
echo "  tmux ls | grep v7_"
echo ""
echo "Check logs:"
echo "  tail -f logs/evolution/v7_easy.log"
echo "  tail -f logs/evolution/v7_*.log"
echo ""
echo "Estimated completion: 12-24 hours"
echo ""
