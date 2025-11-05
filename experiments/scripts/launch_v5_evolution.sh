#!/bin/bash
# Launch V5 Evolution - Rust Single Source of Truth
#
# This script launches evolution for all 9 scenarios using Rust as the
# single source of truth for both training and testing (no Python mismatch).
#
# Configuration: 6-hour budget approach
# - Population: 200 (proven effective in v3/v4)
# - Generations: 12000 (fits 6-hour budget)
# - Games/eval: 50 (stable evaluation)
# - Seed: 43 (different from v4's 42 for exploration)
#
# Expected runtime: ~6 hours wall-clock
# Goal: Match or beat v3/v4's 58.50 near-Nash result

set -e

# Configuration
SCENARIOS=(
    "chain_reaction"
    "deceptive_calm"
    "early_containment"
    "greedy_neighbor"
    "mixed_motivation"
    "overcrowding"
    "rest_trap"
    "sparse_heroics"
    "trivial_cooperation"
)

POPULATION=200
GENERATIONS=12000
GAMES=50
SEED=43

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/logs/evolution"
mkdir -p "$LOG_DIR"

echo "=== V5 Evolution Launch ==="
echo "Date: $(date)"
echo "Config: pop=$POPULATION, gen=$GENERATIONS, games=$GAMES, seed=$SEED"
echo "Scenarios: ${#SCENARIOS[@]}"
echo "Rust: Single source of truth (train/test consistency guaranteed)"
echo ""

# Kill any existing evolution sessions
echo "Cleaning up old tmux sessions..."
for scenario in "${SCENARIOS[@]}"; do
    tmux kill-session -t "v5_${scenario}" 2>/dev/null || true
done

# Launch each scenario in a tmux session
for scenario in "${SCENARIOS[@]}"; do
    session_name="v5_${scenario}"
    log_file="$LOG_DIR/${scenario}_v5_$(date +%Y%m%d_%H%M%S).log"
    output_dir="$PROJECT_ROOT/experiments/scenarios/${scenario}/evolved_v5"

    echo "Launching: $scenario"
    echo "  Session: $session_name"
    echo "  Log: $log_file"
    echo "  Output: $output_dir"

    # Create tmux session and run evolution
    tmux new-session -d -s "$session_name" \
        "cd $PROJECT_ROOT && \
         uv run python experiments/scripts/run_evolution.py $scenario \
           --population $POPULATION \
           --generations $GENERATIONS \
           --games $GAMES \
           --output-dir $output_dir \
           --seed $SEED \
           2>&1 | tee $log_file"

    echo "  ✓ Started"
    echo ""
done

echo "=== All scenarios launched ==="
echo ""
echo "Monitor progress:"
echo "  tmux ls                              # List all sessions"
echo "  tmux attach -t v5_chain_reaction     # Attach to specific session"
echo "  tail -f $LOG_DIR/*_v5_*.log          # Follow logs"
echo ""
echo "Check status:"
echo "  ls -lh $LOG_DIR/*_v5_*.log                    # Log sizes"
echo "  grep 'Generation' $LOG_DIR/*_v5_*.log | tail  # Latest progress"
echo ""
echo "Estimated completion: $(date -d '+6 hours' 2>/dev/null || date -v +6H 2>/dev/null || echo '6 hours from now')"
echo ""
echo "Expected result: ≥ 58.50 payoff (matching v3/v4 near-Nash performance)"
