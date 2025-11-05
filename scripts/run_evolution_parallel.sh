#!/bin/bash
# Parallel evolution experiment execution script
# Runs multiple scenarios simultaneously using GNU parallel or background jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
POPULATION=${POPULATION:-100}
GENERATIONS=${GENERATIONS:-200}
GAMES=${GAMES:-20}
MAX_PARALLEL=${MAX_PARALLEL:-8}  # Run 8 scenarios simultaneously on 64 vCPU machine

# Scenarios to process
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
echo "Evolution Parallel Execution"
echo "========================================"
echo "Population:  $POPULATION"
echo "Generations: $GENERATIONS"
echo "Games/eval:  $GAMES"
echo "Max parallel: $MAX_PARALLEL"
echo "Scenarios:   ${#SCENARIOS[@]}"
echo "========================================"
echo

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs/evolution"

# Function to run evolution for a single scenario
run_scenario() {
    local scenario=$1
    local log_file="$PROJECT_ROOT/logs/evolution/${scenario}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting evolution for $scenario"

    cd "$PROJECT_ROOT"
    source .venv/bin/activate

    python experiments/scripts/run_evolution.py "$scenario" \
        --population "$POPULATION" \
        --generations "$GENERATIONS" \
        --games "$GAMES" \
        2>&1 | tee "$log_file"

    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ Completed: $scenario"
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ❌ Failed: $scenario (exit code: $exit_code)"
    fi

    return $exit_code
}

export -f run_scenario
export PROJECT_ROOT POPULATION GENERATIONS GAMES

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for execution"
    printf '%s\n' "${SCENARIOS[@]}" | parallel -j "$MAX_PARALLEL" run_scenario {}
else
    echo "GNU parallel not found, using background jobs"

    # Track background jobs
    pids=()
    active_jobs=0

    for scenario in "${SCENARIOS[@]}"; do
        # Wait if we've hit max parallel jobs
        while [ $active_jobs -ge $MAX_PARALLEL ]; do
            # Check for completed jobs
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}"
                    unset 'pids[$i]'
                    ((active_jobs--))
                fi
            done
            sleep 1
        done

        # Start new job
        run_scenario "$scenario" &
        pids+=($!)
        ((active_jobs++))
    done

    # Wait for all remaining jobs
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
fi

echo
echo "========================================"
echo "All evolution experiments complete!"
echo "========================================"
echo "Results in: $PROJECT_ROOT/experiments/scenarios/*/evolved/"
echo "Logs in: $PROJECT_ROOT/logs/evolution/"
echo

# Generate insights for all scenarios
echo "Generating research insights..."
cd "$PROJECT_ROOT"
source .venv/bin/activate
python experiments/scripts/generate_insights.py --all

echo "✅ Complete! Check the research tab to see updated insights."
