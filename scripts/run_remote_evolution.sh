#!/usr/bin/env bash
#
# Helper script to run evolutionary training in a persistent tmux session.
#
# Usage:
#   ./scripts/run_remote_evolution.sh [session_name] [additional_args...]
#
# Examples:
#   # Start default run
#   ./scripts/run_remote_evolution.sh
#
#   # Start with custom session name
#   ./scripts/run_remote_evolution.sh my_experiment
#
#   # Start with custom parameters
#   ./scripts/run_remote_evolution.sh exp1 --generations 1000 --population-size 200
#
#   # Resume from checkpoint
#   ./scripts/run_remote_evolution.sh resume --resume runs/remote_evolution/checkpoint_gen0100.json
#
# The script will:
# - Create a tmux session for persistent execution
# - Run the evolution script with optimal settings
# - Log output to files for monitoring
# - Allow detaching/reattaching without interrupting training

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION_NAME="${1:-evolution_$(date +%Y%m%d_%H%M%S)}"
shift || true  # Remove session name from args if present

# Default settings optimized for 48 vCPUs + 4 GPUs
DEFAULT_WORKERS=48
DEFAULT_POPULATION=100
DEFAULT_GENERATIONS=500
DEFAULT_GAMES=50
DEFAULT_CHECKPOINT_INTERVAL=10

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/runs/remote_evolution"
mkdir -p "${OUTPUT_DIR}"

# Log file for tmux session
LOG_FILE="${OUTPUT_DIR}/${SESSION_NAME}.log"

echo "=================================================="
echo "Remote Evolution Training"
echo "=================================================="
echo "Session name: ${SESSION_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "=================================================="
echo ""

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "⚠️  Session '${SESSION_NAME}' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t ${SESSION_NAME}"
    echo "  2. Kill existing session: tmux kill-session -t ${SESSION_NAME}"
    echo "  3. Choose a different session name"
    exit 1
fi

# Build command
CMD="cd ${PROJECT_ROOT} && source .venv/bin/activate && "
CMD+="python scripts/evolve_remote.py "
CMD+="--num-workers ${DEFAULT_WORKERS} "
CMD+="--population-size ${DEFAULT_POPULATION} "
CMD+="--generations ${DEFAULT_GENERATIONS} "
CMD+="--games-per-individual ${DEFAULT_GAMES} "
CMD+="--checkpoint-interval ${DEFAULT_CHECKPOINT_INTERVAL} "
CMD+="--output-dir ${OUTPUT_DIR} "

# Add any additional arguments
if [ $# -gt 0 ]; then
    CMD+="$* "
fi

# Add output redirection
CMD+="2>&1 | tee ${LOG_FILE}"

echo "Starting tmux session: ${SESSION_NAME}"
echo ""
echo "Command:"
echo "  ${CMD}"
echo ""

# Create tmux session and run command
tmux new-session -d -s "${SESSION_NAME}" "${CMD}"

echo "✅ Session started successfully!"
echo ""
echo "To monitor progress:"
echo "  # Attach to session (Ctrl+B then D to detach)"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "  # View live log"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "  # List all tmux sessions"
echo "  tmux ls"
echo ""
echo "  # Kill session (if needed)"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo ""
echo "To check results:"
echo "  ls -lh ${OUTPUT_DIR}"
echo ""

# Optionally attach to the session
read -p "Attach to session now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tmux attach -t "${SESSION_NAME}"
fi
