#!/bin/bash
# Check V4 Evolution Progress
#
# Shows current generation and fitness for each scenario

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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/logs/evolution"

echo "=== V4 Evolution Progress ==="
echo "Time: $(date)"
echo ""

# Check tmux sessions
echo "Active tmux sessions:"
tmux ls | grep "v4_" || echo "  None running"
echo ""

# Check latest progress from logs
echo "Latest progress by scenario:"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    log_file=$(ls -t "$LOG_DIR/${scenario}_v4_"*.log 2>/dev/null | head -1)

    if [ -z "$log_file" ]; then
        printf "%-25s: No log found\n" "$scenario"
        continue
    fi

    # Extract latest generation and fitness
    latest=$(tail -20 "$log_file" | grep "Generation" | tail -1)

    if [ -z "$latest" ]; then
        printf "%-25s: Starting...\n" "$scenario"
    else
        printf "%-25s: %s\n" "$scenario" "$latest"
    fi
done

echo ""
echo "Full logs: $LOG_DIR/*_v4_*.log"
