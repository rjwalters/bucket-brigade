#!/bin/bash
# Monitor evolution progress across all scenarios

echo "Evolution Progress Monitor"
echo "=========================="
echo

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

LOG_DIR="logs/evolution"

# Find latest logs for each scenario
echo "Latest Runs:"
echo "------------"
for scenario in "${SCENARIOS[@]}"; do
    latest_log=$(ls -t $LOG_DIR/${scenario}_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        # Get last line showing generation progress
        last_gen=$(grep "Gen " "$latest_log" | tail -1)
        if [ -n "$last_gen" ]; then
            echo "$scenario: $last_gen"
        else
            echo "$scenario: Starting..."
        fi
    else
        echo "$scenario: No log found"
    fi
done

echo
echo "Summary Statistics:"
echo "-------------------"

# Aggregate statistics
total_scenarios=0
completed_scenarios=0

for scenario in "${SCENARIOS[@]}"; do
    latest_log=$(ls -t $LOG_DIR/${scenario}_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        ((total_scenarios++))

        # Check if completed
        if grep -q "Evolution Complete!" "$latest_log"; then
            ((completed_scenarios++))
        fi
    fi
done

echo "Scenarios running: $total_scenarios"
echo "Scenarios completed: $completed_scenarios"

echo
echo "System Resources:"
echo "-----------------"
echo "Load average: $(uptime | awk -F'load average:' '{print $2}')"

echo
echo "Commands:"
echo "  Attach to tmux:  tmux attach -t evolution-master"
echo "  Kill all:        tmux kill-session -t evolution-master"
echo "  Watch this:      watch -n 30 ./scripts/monitor_evolution.sh"
