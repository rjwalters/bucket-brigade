#!/bin/bash
# Launch evolution experiments in separate tmux windows

set -euo pipefail

# Configuration
POPULATION=${POPULATION:-100}
GENERATIONS=${GENERATIONS:-200}
GAMES=${GAMES:-20}

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
echo "Launching Evolution Experiments"
echo "========================================"
echo "Population:  $POPULATION"
echo "Generations: $GENERATIONS"
echo "Games/eval:  $GAMES"
echo "Scenarios:   ${#SCENARIOS[@]}"
echo "========================================"
echo

# Create master session
tmux new-session -d -s evolution-master

# Create logs directory
mkdir -p logs/evolution

# Launch each scenario in its own window
for i in "${!SCENARIOS[@]}"; do
    scenario="${SCENARIOS[$i]}"
    log_file="logs/evolution/${scenario}_$(date +%Y%m%d_%H%M%S).log"

    if [ $i -eq 0 ]; then
        # First window is window 0
        tmux rename-window -t evolution-master:0 "$scenario"
        tmux send-keys -t evolution-master:0 "cd ~/bucket-brigade && source .venv/bin/activate" C-m
        tmux send-keys -t evolution-master:0 "python experiments/scripts/run_evolution.py $scenario --population $POPULATION --generations $GENERATIONS --games $GAMES 2>&1 | tee $log_file" C-m
    else
        # Create new window for other scenarios
        tmux new-window -t evolution-master -n "$scenario"
        tmux send-keys -t evolution-master:$i "cd ~/bucket-brigade && source .venv/bin/activate" C-m
        tmux send-keys -t evolution-master:$i "python experiments/scripts/run_evolution.py $scenario --population $POPULATION --generations $GENERATIONS --games $GAMES 2>&1 | tee $log_file" C-m
    fi

    echo "✓ Launched $scenario in tmux window $i"
done

echo
echo "========================================"
echo "All experiments launched!"
echo "========================================"
echo
echo "Commands:"
echo "  List windows:      tmux list-windows -t evolution-master"
echo "  Attach to session: tmux attach -t evolution-master"
echo "  Switch windows:    Ctrl+B then 0-8 (or n/p for next/previous)"
echo "  Kill all:          tmux kill-session -t evolution-master"
echo
