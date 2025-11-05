#!/bin/bash
#
# Run Nash Equilibrium V2 analysis for all scenarios
#
# This script runs the complete V2 Nash analysis including evolved agents
# from V3/V4/V5 experiments. Should be run on a machine with good compute.
#
# Usage:
#   ./experiments/scripts/run_nash_v2_all.sh               # Run all scenarios
#   ./experiments/scripts/run_nash_v2_all.sh --quick       # Quick test (200 sims)
#   ./experiments/scripts/run_nash_v2_all.sh --full        # Full analysis (2000 sims)
#

set -e  # Exit on error

# Configuration
SIMULATIONS=1000
EVOLVED_VERSIONS="v4"
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            SIMULATIONS=200
            echo "🚀 Quick mode: 200 simulations"
            shift
            ;;
        --full)
            SIMULATIONS=2000
            echo "🔬 Full mode: 2000 simulations"
            shift
            ;;
        --evolved-versions)
            EVOLVED_VERSIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Standard research scenarios (9 total)
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

echo "========================================================================"
echo "Nash Equilibrium V2 - Batch Run"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Scenarios:        ${#SCENARIOS[@]} scenarios"
echo "  Simulations:      $SIMULATIONS per evaluation"
echo "  Evolved versions: $EVOLVED_VERSIONS"
echo "  Seed:             $SEED"
echo "  Output:           experiments/nash/v2_results/"
echo ""
echo "Scenarios to run:"
for scenario in "${SCENARIOS[@]}"; do
    echo "  - $scenario"
done
echo ""

# Create output directory
mkdir -p experiments/nash/v2_results
mkdir -p logs/nash_v2

# Record start time
START_TIME=$(date +%s)
echo "Started at: $(date)"
echo ""

# Run each scenario
SUCCESSFUL=0
FAILED=0

for scenario in "${SCENARIOS[@]}"; do
    echo "========================================================================"
    echo "Running Nash V2 for: $scenario"
    echo "========================================================================"
    echo ""

    LOG_FILE="logs/nash_v2/${scenario}_$(date +%Y%m%d_%H%M%S).log"

    if uv run python experiments/scripts/compute_nash_v2.py "$scenario" \
        --simulations "$SIMULATIONS" \
        --evolved-versions $EVOLVED_VERSIONS \
        --seed "$SEED" \
        2>&1 | tee "$LOG_FILE"; then

        echo ""
        echo "✅ $scenario completed successfully"
        echo "   Log: $LOG_FILE"
        echo ""
        ((SUCCESSFUL++))
    else
        echo ""
        echo "❌ $scenario failed!"
        echo "   Log: $LOG_FILE"
        echo ""
        ((FAILED++))
    fi

    echo ""
done

# Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "========================================================================"
echo "Nash V2 Batch Run Complete"
echo "========================================================================"
echo ""
echo "Results:"
echo "  ✅ Successful: $SUCCESSFUL / ${#SCENARIOS[@]}"
echo "  ❌ Failed:     $FAILED / ${#SCENARIOS[@]}"
echo ""
echo "Time elapsed: ${MINUTES}m ${SECONDS}s"
echo "Finished at: $(date)"
echo ""
echo "Results saved to: experiments/nash/v2_results/"
echo "Logs saved to: logs/nash_v2/"
echo ""

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    echo "⚠️  Warning: $FAILED scenario(s) failed"
    exit 1
fi

echo "🎉 All scenarios completed successfully!"
