#!/bin/bash
#
# Launch population training on remote GPU server
#
# Usage:
#   ./scripts/launch_remote_training.sh [options]
#
# Options:
#   -h, --host       Remote host (default: rwalters-sandbox-2)
#   -s, --scenario   Scenario name (default: trivial_cooperation)
#   -p, --pop-size   Population size (default: 16)
#   -e, --episodes   Number of episodes (default: 100000)
#   -n, --name       Run name (auto-generated if not specified)
#   -t, --tmux       Use tmux session (default: nohup)
#   -d, --detach     Detach from tmux immediately (only with --tmux)
#   --help           Show this help message
#
# Examples:
#   # Quick test (tmux, manual monitor)
#   ./scripts/launch_remote_training.sh -s trivial_cooperation -p 4 -e 1000 -t
#
#   # Production run (background with nohup)
#   ./scripts/launch_remote_training.sh -s mixed_motivation -p 32 -e 1000000
#
#   # Custom name, detached tmux
#   ./scripts/launch_remote_training.sh -s greedy_neighbor -p 16 -e 500000 -n greedy_exp_v1 -t -d
#

set -e

# Defaults
REMOTE_HOST="rwalters-sandbox-2"
SCENARIO="trivial_cooperation"
POP_SIZE=16
EPISODES=100000
RUN_NAME=""
USE_TMUX=false
DETACH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    -s|--scenario)
      SCENARIO="$2"
      shift 2
      ;;
    -p|--pop-size)
      POP_SIZE="$2"
      shift 2
      ;;
    -e|--episodes)
      EPISODES="$2"
      shift 2
      ;;
    -n|--name)
      RUN_NAME="$2"
      shift 2
      ;;
    -t|--tmux)
      USE_TMUX=true
      shift
      ;;
    -d|--detach)
      DETACH=true
      shift
      ;;
    --help)
      head -n 30 "$0" | grep '^#' | sed 's/^# //; s/^#//'
      exit 0
      ;;
    *)
      echo "Error: Unknown option $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Generate run name if not specified
if [ -z "$RUN_NAME" ]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RUN_NAME="${SCENARIO}_pop${POP_SIZE}_${TIMESTAMP}"
fi

# Print configuration
echo "========================================"
echo "Remote Population Training Launch"
echo "========================================"
echo "Host:         $REMOTE_HOST"
echo "Scenario:     $SCENARIO"
echo "Population:   $POP_SIZE agents"
echo "Episodes:     $EPISODES"
echo "Run name:     $RUN_NAME"
echo "Mode:         $([ "$USE_TMUX" = true ] && echo "tmux" || echo "nohup background")"
echo "========================================"
echo

# Check SSH connectivity
echo "Checking SSH connection to $REMOTE_HOST..."
if ! ssh -o ConnectTimeout=5 "$REMOTE_HOST" "echo 'Connected successfully'" &>/dev/null; then
  echo "Error: Cannot connect to $REMOTE_HOST"
  echo "Please check your SSH configuration in ~/.ssh/config"
  exit 1
fi
echo "✓ SSH connection OK"
echo

# Pull latest code on remote
echo "Updating code on remote..."
ssh "$REMOTE_HOST" "cd bucket-brigade && git pull"
echo "✓ Code updated"
echo

# Create logs directory
ssh "$REMOTE_HOST" "mkdir -p bucket-brigade/logs"

# Build training command
TRAIN_CMD="cd bucket-brigade && uv run python experiments/marl/train_population.py \
  --scenario $SCENARIO \
  --population-size $POP_SIZE \
  --num-episodes $EPISODES \
  --device cuda \
  --run-name $RUN_NAME \
  2>&1 | tee logs/training_${RUN_NAME}.log"

# Launch based on mode
if [ "$USE_TMUX" = true ]; then
  # Tmux mode
  SESSION_NAME="train_${RUN_NAME}"

  echo "Starting tmux session: $SESSION_NAME"
  echo

  if [ "$DETACH" = true ]; then
    # Detached tmux
    ssh "$REMOTE_HOST" "tmux new -d -s $SESSION_NAME '$TRAIN_CMD'"

    echo "✓ Training started in detached tmux session"
    echo
    echo "To monitor:"
    echo "  ssh $REMOTE_HOST -t \"tmux attach -t $SESSION_NAME\""
    echo
    echo "To check logs:"
    echo "  ssh $REMOTE_HOST \"tail -f bucket-brigade/logs/training_${RUN_NAME}.log\""
  else
    # Attached tmux
    echo "Attaching to tmux session..."
    echo "Press Ctrl+B then D to detach and leave training running"
    echo
    ssh "$REMOTE_HOST" -t "tmux new -s $SESSION_NAME '$TRAIN_CMD'"
  fi
else
  # Nohup background mode
  echo "Starting training in background with nohup..."

  ssh "$REMOTE_HOST" "nohup bash -c '$TRAIN_CMD' > /dev/null 2>&1 &"

  echo "✓ Training started in background"
  echo
  echo "To monitor:"
  echo "  ssh $REMOTE_HOST \"tail -f bucket-brigade/logs/training_${RUN_NAME}.log\""
  echo
  echo "To check GPU:"
  echo "  ssh $REMOTE_HOST \"nvidia-smi\""
  echo
  echo "To check process:"
  echo "  ssh $REMOTE_HOST \"ps aux | grep train_population\""
fi

echo
echo "Results will be saved to:"
echo "  $REMOTE_HOST:~/bucket-brigade/experiments/marl/checkpoints/$RUN_NAME/"
echo
echo "To retrieve results:"
echo "  rsync -avz --progress $REMOTE_HOST:~/bucket-brigade/experiments/marl/checkpoints/$RUN_NAME/ ./experiments/marl/checkpoints/$RUN_NAME/"
echo
echo "Happy training! 🚀"
