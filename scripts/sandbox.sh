#!/bin/bash
# sandbox.sh - Set up isolated Python environment and run training
#
# This script creates a clean Python virtual environment for running
# training experiments, avoiding conflicts with other project environments.
#
# Usage:
#   ./scripts/sandbox.sh setup           # Just setup environment
#   ./scripts/sandbox.sh train 500000    # Train for 500K steps in tmux
#   ./scripts/sandbox.sh train 1000000   # Train for 1M steps in tmux
#   ./scripts/sandbox.sh status          # Check training status
#   ./scripts/sandbox.sh logs            # View training logs
#   ./scripts/sandbox.sh attach          # Attach to tmux training session
#   ./scripts/sandbox.sh kill            # Kill training session

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Function to setup environment
setup_environment() {
    echo "🔧 Setting up sandbox environment for bucket-brigade..."

    # Unset any existing virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "⚠️  Detected existing environment: $VIRTUAL_ENV"
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

    # Remove old venv if clean flag
    if [ "$CLEAN" == "true" ]; then
        echo "🧹 Removing existing .venv..."
        rm -rf .venv
    fi

    # Sync base dependencies
    echo "📦 Syncing base dependencies..."
    uv sync --quiet

    # Install RL dependencies
    echo "🤖 Installing RL dependencies (PyTorch, Gymnasium, etc.)..."
    if ! uv pip install -e ".[rl]" 2>&1 | tee /tmp/rl-install.log | grep -q "Installed"; then
        echo "⚠️  Warning: Some dependencies may have failed to install"
        echo "   Check /tmp/rl-install.log for details"
        echo ""
        echo "   Attempting to install tensorboard separately..."
        uv pip install tensorboard
    fi

    # Verify PyTorch installation
    echo ""
    echo "✅ Sandbox environment ready!"
    echo ""
    .venv/bin/python -c "
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
}

# Function to start training in tmux
train() {
    local steps=$1
    local session_name="training-${steps}"

    if [ -z "$steps" ]; then
        echo "❌ Error: Please specify number of training steps"
        echo "   Usage: $0 train <steps>"
        echo "   Example: $0 train 500000"
        exit 1
    fi

    # Check if .venv exists and has required dependencies
    local need_setup=false

    if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        echo "⚠️  Virtual environment not found. Setting up first..."
        need_setup=true
    elif ! "$PROJECT_ROOT/.venv/bin/python" -c "import torch, tensorboard" 2>/dev/null; then
        echo "⚠️  Missing critical dependencies. Re-running setup..."
        need_setup=true
    else
        echo "✅ Using existing virtual environment"
    fi

    if [ "$need_setup" = true ]; then
        setup_environment
    fi

    # Check if tmux is installed
    if ! command -v tmux &> /dev/null; then
        echo "❌ Error: tmux is not installed"
        echo "   Install with: apt-get install tmux (Ubuntu) or brew install tmux (macOS)"
        exit 1
    fi

    # Check if session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "⚠️  Training session '$session_name' already exists"
        echo "   Attach with: $0 attach"
        echo "   Or kill with: $0 kill"
        exit 1
    fi

    # Create directories
    mkdir -p models logs

    # Determine model path based on steps
    local model_path="models/policy_${steps}.pt"
    local log_path="logs/training_${steps}.log"
    local run_name="training_${steps}"

    echo ""
    echo "🚀 Starting training in tmux session '$session_name'"
    echo "   Steps: $steps"
    echo "   Model: $model_path"
    echo "   Log: $log_path"
    echo ""
    echo "Commands:"
    echo "  - Attach to session: $0 attach"
    echo "  - View logs: $0 logs"
    echo "  - Check status: $0 status"
    echo "  - Kill training: $0 kill"
    echo ""

    # Create tmux session and run training
    tmux new-session -d -s "$session_name" -c "$PROJECT_ROOT"
    tmux send-keys -t "$session_name" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$session_name" "source .venv/bin/activate" C-m
    tmux send-keys -t "$session_name" "python scripts/train_simple.py --num-steps $steps --num-opponents 3 --batch-size 2048 --hidden-size 128 --save-path $model_path --run-name $run_name | tee $log_path" C-m

    echo "✅ Training started in background!"
    echo "   Run '$0 attach' to view progress"
}

# Function to check training status
status() {
    echo "📊 Training Status:"
    echo ""

    # Check for tmux sessions
    if command -v tmux &> /dev/null; then
        local sessions=$(tmux list-sessions 2>/dev/null | grep "training-" || echo "")
        if [ -n "$sessions" ]; then
            echo "Active training sessions:"
            echo "$sessions"
        else
            echo "No active training sessions found"
        fi
    fi

    echo ""
    echo "Training processes:"
    ps aux | grep "[t]rain_simple.py" || echo "No training processes found"

    echo ""
    echo "Log files:"
    ls -lh logs/training_*.log 2>/dev/null || echo "No training logs found"

    echo ""
    echo "Model files:"
    ls -lh models/policy_*.pt 2>/dev/null || echo "No models found"
}

# Function to view logs
view_logs() {
    local latest_log=$(ls -t logs/training_*.log 2>/dev/null | head -1)

    if [ -z "$latest_log" ]; then
        echo "❌ No training logs found"
        exit 1
    fi

    echo "📜 Viewing latest log: $latest_log"
    echo "   (Press 'q' to exit)"
    echo ""

    # Use less for safe viewing (q to quit)
    less +F "$latest_log"
}

# Function to attach to tmux session
attach() {
    local session=$(tmux list-sessions 2>/dev/null | grep "training-" | head -1 | cut -d: -f1)

    if [ -z "$session" ]; then
        echo "❌ No active training session found"
        echo "   Start one with: $0 train <steps>"
        exit 1
    fi

    echo "🔗 Attaching to session: $session"
    echo "   (Press Ctrl+b then d to detach)"
    echo ""

    tmux attach -t "$session"
}

# Function to kill training
kill_training() {
    echo "🛑 Killing training sessions..."

    # Kill tmux sessions
    local sessions=$(tmux list-sessions 2>/dev/null | grep "training-" | cut -d: -f1 || echo "")
    if [ -n "$sessions" ]; then
        echo "$sessions" | while read session; do
            echo "   Killing session: $session"
            tmux kill-session -t "$session"
        done
    fi

    # Kill any stray processes
    pkill -f "train_simple.py" || true

    echo "✅ Training killed"
}

# Main command handling
case "${1:-setup}" in
    setup|--setup)
        CLEAN="${2}"
        setup_environment
        ;;
    train|--train)
        train "$2"
        ;;
    status|--status)
        status
        ;;
    logs|--logs)
        view_logs
        ;;
    attach|--attach)
        attach
        ;;
    kill|--kill)
        kill_training
        ;;
    --clean)
        CLEAN="true"
        setup_environment
        ;;
    -h|--help|help)
        echo "Sandbox Training Script"
        echo ""
        echo "Usage:"
        echo "  $0 setup              Setup environment"
        echo "  $0 train <steps>      Start training in tmux (e.g., 500000 or 1000000)"
        echo "  $0 status             Check training status"
        echo "  $0 logs               View latest training logs"
        echo "  $0 attach             Attach to training session"
        echo "  $0 kill               Kill all training sessions"
        echo "  $0 --clean            Clean setup (remove .venv first)"
        echo ""
        echo "Examples:"
        echo "  $0 train 500000       Train for 500K steps"
        echo "  $0 train 1000000      Train for 1M steps"
        echo "  $0 attach             Watch training progress"
        echo "  $0 logs               View logs (press 'q' to exit)"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "   Run '$0 --help' for usage"
        exit 1
        ;;
esac
