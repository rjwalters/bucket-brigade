#!/bin/bash
# Launch overnight specialist evolution (9 agents, no generalist)
# Fixed fitness metric, optimized for 10-hour runtime

set -euo pipefail

# Configuration (optimized for 10-hour runtime based on timing analysis)
POPULATION=${POPULATION:-200}
GENERATIONS=${GENERATIONS:-2500}  # Increased for full 10-hour run
GAMES=${GAMES:-50}

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

echo "========================================"
echo "Overnight Specialist Evolution"
echo "========================================"
echo "Population:  $POPULATION"
echo "Generations: $GENERATIONS"
echo "Games/eval:  $GAMES"
echo "========================================"
echo ""
echo "9 Specialist Agents"
echo "  - Fixed fitness metric (scenario payoff)"
echo "  - One expert per scenario"
echo "  - Full evolution run (no early stopping)"
echo "========================================"
echo

# Create master session
tmux new-session -d -s specialist-evolution

# Create logs directory
mkdir -p logs/evolution

# Windows 0-8: Specialist evolution for each scenario
for i in "${!SCENARIOS[@]}"; do
    scenario="${SCENARIOS[$i]}"
    window_num=$i
    log_file="logs/evolution/${scenario}_v3_\$(date +%Y%m%d_%H%M%S).log"

    if [ $window_num -eq 0 ]; then
        # First window already exists
        tmux rename-window -t specialist-evolution:0 "$scenario"
        tmux send-keys -t specialist-evolution:0 "cd ~/bucket-brigade && source .venv/bin/activate" C-m
        tmux send-keys -t specialist-evolution:0 "python experiments/scripts/run_evolution.py $scenario --population $POPULATION --generations $GENERATIONS --games $GAMES --output-dir experiments/scenarios/$scenario/evolved_v3 2>&1 | tee $log_file" C-m
    else
        tmux new-window -t specialist-evolution:$window_num -n "$scenario"
        tmux send-keys -t specialist-evolution:$window_num "cd ~/bucket-brigade && source .venv/bin/activate" C-m
        tmux send-keys -t specialist-evolution:$window_num "python experiments/scripts/run_evolution.py $scenario --population $POPULATION --generations $GENERATIONS --games $GAMES --output-dir experiments/scenarios/$scenario/evolved_v3 2>&1 | tee $log_file" C-m
    fi

    echo "✓ Launched $scenario (specialist) in window $window_num"
done

echo
echo "========================================"
echo "All specialists launched!"
echo "========================================"
echo
echo "Commands:"
echo "  Attach to session: tmux attach -t specialist-evolution"
echo "  Switch windows:    Ctrl+B then 0-8 (or n/p)"
echo "  Kill all:          tmux kill-session -t specialist-evolution"
echo
echo "Expected runtime: ~10 hours"
echo ""
echo "Configuration:"
echo "  Population: $POPULATION"
echo "  Generations: $GENERATIONS"
echo "  Games/eval: $GAMES"
echo ""
echo "Expected results:"
echo "  - 9 specialist agents with fixed fitness metric"
echo "  - Fitness values in tournament scale (±100)"
echo "  - Results in experiments/scenarios/*/evolved_v3/"
echo
