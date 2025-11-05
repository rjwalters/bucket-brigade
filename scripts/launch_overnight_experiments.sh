#!/bin/bash
# Launch overnight evolution experiments:
# 1. Improved specialists (9 agents with fixed fitness metric)
# 2. Generalist agent (1 agent trained across all scenarios)

set -euo pipefail

# Configuration (optimized for 8-9 hour runtime based on timing analysis)
POPULATION=${POPULATION:-200}
GENERATIONS=${GENERATIONS:-2000}  # 2x more than previous run
GAMES=${GAMES:-50}
GENERALIST_GENERATIONS=${GENERALIST_GENERATIONS:-1200}  # Adjusted for multi-scenario overhead

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
echo "Overnight Evolution Experiments"
echo "========================================"
echo "Population:  $POPULATION"
echo "Generations: $GENERATIONS"
echo "Games/eval:  $GAMES"
echo "========================================"
echo ""
echo "EXPERIMENT 1: Improved Specialists (9 agents)"
echo "  - Fixed fitness metric (scenario payoff)"
echo "  - One expert per scenario"
echo ""
echo "EXPERIMENT 2: Generalist Agent (1 agent)"
echo "  - Trained across all 9 scenarios"
echo "  - Robust to different game dynamics"
echo "========================================"
echo

# Create master session
tmux new-session -d -s overnight-evolution

# Create logs directory
mkdir -p logs/evolution

# Window 0: Generalist evolution
tmux rename-window -t overnight-evolution:0 "generalist"
tmux send-keys -t overnight-evolution:0 "cd ~/bucket-brigade && source .venv/bin/activate" C-m
tmux send-keys -t overnight-evolution:0 "python experiments/scripts/run_generalist_evolution.py --population $POPULATION --generations $GENERALIST_GENERATIONS --games-per-scenario 10 2>&1 | tee logs/evolution/generalist_\$(date +%Y%m%d_%H%M%S).log" C-m

# Windows 1-9: Specialist evolution for each scenario
for i in "${!SCENARIOS[@]}"; do
    scenario="${SCENARIOS[$i]}"
    window_num=$((i + 1))
    log_file="logs/evolution/${scenario}_v2_\$(date +%Y%m%d_%H%M%S).log"
    
    tmux new-window -t overnight-evolution:$window_num -n "$scenario"
    tmux send-keys -t overnight-evolution:$window_num "cd ~/bucket-brigade && source .venv/bin/activate" C-m
    tmux send-keys -t overnight-evolution:$window_num "python experiments/scripts/run_evolution.py $scenario --population $POPULATION --generations $GENERATIONS --games $GAMES --output-dir experiments/scenarios/$scenario/evolved_v2 2>&1 | tee $log_file" C-m
    
    echo "✓ Launched $scenario (specialist) in window $window_num"
done

echo "✓ Launched generalist in window 0"
echo

echo "========================================"
echo "All experiments launched!"
echo "========================================"
echo
echo "Commands:"
echo "  Attach to session: tmux attach -t overnight-evolution"
echo "  Switch windows:    Ctrl+B then 0-9 (or n/p)"
echo "  Kill all:          tmux kill-session -t overnight-evolution"
echo
echo "Expected runtime: 8-9 hours"
echo ""
echo "Configuration:"
echo "  Specialists: 200 pop, $GENERATIONS gen, 50 games (2x more than v1)"
echo "  Generalist:  200 pop, $GENERALIST_GENERATIONS gen, 10 games/scenario"
echo ""
echo "Expected results:"
echo "  - 9 improved specialist agents (with fixed fitness metric)"
echo "  - 1 generalist agent (robust across all scenarios)"
echo "  - Fitness values in tournament scale (±100, interpretable!)"
echo "  - Better convergence without regression"
echo
