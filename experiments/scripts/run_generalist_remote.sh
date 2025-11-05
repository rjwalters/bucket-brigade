#!/bin/bash
# Run generalist agent evolution on remote machine
#
# Usage:
#   ./experiments/scripts/run_generalist_remote.sh [remote-host]
#
# Default remote host can be set via REMOTE_HOST environment variable

set -e

# Configuration
REMOTE_HOST="${1:-${REMOTE_HOST:-my-gpu-server}}"
POPULATION="${POPULATION:-200}"
GENERATIONS="${GENERATIONS:-12000}"
GAMES="${GAMES:-50}"
SEED="${SEED:-42}"

echo "===================================================================="
echo "Generalist Agent Evolution - Remote Execution"
echo "===================================================================="
echo "Remote host:  $REMOTE_HOST"
echo "Population:   $POPULATION"
echo "Generations:  $GENERATIONS"
echo "Games/scenario: $GAMES"
echo "Seed:         $SEED"
echo ""
echo "Expected runtime: ~10 hours"
echo "===================================================================="
echo ""

# Check if remote host is reachable
if ! ssh -q "$REMOTE_HOST" exit 2>/dev/null; then
    echo "❌ Error: Cannot connect to $REMOTE_HOST"
    echo ""
    echo "Make sure:"
    echo "  1. SSH config is set up (~/.ssh/config)"
    echo "  2. Remote host is accessible"
    echo "  3. SSH keys are configured"
    echo ""
    exit 1
fi

echo "✅ Connected to $REMOTE_HOST"
echo ""

# Generate timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_NAME="generalist_${TIMESTAMP}"

# Create tmux session and run evolution
echo "Starting evolution in tmux session: $SESSION_NAME"
echo ""

ssh "$REMOTE_HOST" bash <<EOF
    set -e

    # Navigate to repo
    cd ~/bucket-brigade || cd bucket-brigade || {
        echo "❌ Error: bucket-brigade directory not found"
        exit 1
    }

    # Pull latest code
    echo "Pulling latest code..."
    git pull

    # Ensure environment is set up
    if [ ! -d .venv ]; then
        echo "Setting up Python environment..."
        uv venv
    fi

    source .venv/bin/activate

    # Build Rust module
    echo "Building Rust module..."
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    maturin develop --quiet

    # Verify Rust module
    python -c "import bucket_brigade_core; print('✅ Rust module loaded')"

    # Create output directory
    mkdir -p experiments/generalist/evolved

    # Start tmux session with evolution
    echo ""
    echo "Starting evolution in tmux session: $SESSION_NAME"
    echo ""

    tmux new-session -d -s "$SESSION_NAME" bash -c "
        cd ~/bucket-brigade || cd bucket-brigade
        source .venv/bin/activate
        export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

        echo '===================================================================='
        echo 'Generalist Evolution Starting'
        echo '===================================================================='
        echo 'Population:   $POPULATION'
        echo 'Generations:  $GENERATIONS'
        echo 'Games/scenario: $GAMES'
        echo 'Seed:         $SEED'
        echo 'Started:      \$(date)'
        echo '===================================================================='
        echo ''

        uv run python experiments/scripts/run_generalist_evolution.py \\
            --population $POPULATION \\
            --generations $GENERATIONS \\
            --games-per-scenario $GAMES \\
            --seed $SEED \\
            --output-dir experiments/generalist/evolved \\
            2>&1 | tee experiments/generalist/evolution_${TIMESTAMP}.log

        echo ''
        echo '===================================================================='
        echo 'Generalist Evolution Complete!'
        echo 'Finished:     \$(date)'
        echo '===================================================================='

        # Keep session open for inspection
        exec bash
    "

    echo ""
    echo "✅ Evolution started in tmux session: $SESSION_NAME"
    echo ""
    echo "To monitor progress:"
    echo "  ssh $REMOTE_HOST -t 'tmux attach -t $SESSION_NAME'"
    echo ""
    echo "To check log:"
    echo "  ssh $REMOTE_HOST 'tail -f bucket-brigade/experiments/generalist/evolution_${TIMESTAMP}.log'"
    echo ""
    echo "To retrieve results when complete:"
    echo "  rsync -avz $REMOTE_HOST:~/bucket-brigade/experiments/generalist/ experiments/generalist/"
    echo ""
EOF

echo ""
echo "✅ Remote execution started successfully!"
echo ""
echo "Monitor with: ssh $REMOTE_HOST -t 'tmux attach -t $SESSION_NAME'"
