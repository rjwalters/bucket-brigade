#!/bin/bash
# Launch V4 Evolution - Fixed Rust Evaluator
#
# This script launches evolution for all 9 scenarios with the corrected
# multi-agent Rust evaluator (fixing the v3 single-agent bug).
#
# Configuration: Intensive 10-hour approach
# - Population: 200 (2x original, same as v3)
# - Generations: 15000 (75x original, 6x v3)
# - Games/eval: 50 (stable evaluation)
# - Seed: 42 (reproducibility)
#
# Expected runtime: ~6.5 hours wall-clock (leaves 3.5hr buffer)
# Based on v3 actual performance: 2500 gen in 64 min = ~39 gen/min

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
GENERATIONS=15000
GAMES=50
SEED=42

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/logs/evolution"
mkdir -p "$LOG_DIR"

echo "=== V4 Evolution Launch ==="
echo "Date: $(date)"
echo "Config: pop=$POPULATION, gen=$GENERATIONS, games=$GAMES, seed=$SEED"
echo "Scenarios: ${#SCENARIOS[@]}"
echo ""

# Kill any existing evolution sessions
echo "Cleaning up old tmux sessions..."
for scenario in "${SCENARIOS[@]}"; do
    tmux kill-session -t "v4_${scenario}" 2>/dev/null || true
done

# Launch each scenario in a tmux session
for scenario in "${SCENARIOS[@]}"; do
    session_name="v4_${scenario}"
    log_file="$LOG_DIR/${scenario}_v4_$(date +%Y%m%d_%H%M%S).log"
    output_dir="$PROJECT_ROOT/experiments/scenarios/${scenario}/evolved_v4"

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
echo "  tmux attach -t v4_chain_reaction     # Attach to specific session"
echo "  tail -f $LOG_DIR/*_v4_*.log          # Follow logs"
echo ""
echo "Check status:"
echo "  ./experiments/scripts/check_v4_progress.sh"
echo ""
echo "Estimated completion: $(date -d '+30 minutes' 2>/dev/null || date -v +30M)"
