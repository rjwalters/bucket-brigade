#!/bin/bash
# Run extended evolution for all scenarios
# This uses the existing run_evolution.py with production hyperparameters

set -e

# Production hyperparameters
POPULATION=100
GENERATIONS=100
GAMES=30

# Scenarios to process
SCENARIOS=(
    "greedy_neighbor"
    "trivial_cooperation"
    "sparse_heroics"
    "early_containment"
    "rest_trap"
    "chain_reaction"
    "deceptive_calm"
    "overcrowding"
    "mixed_motivation"
)

echo "================================================================================"
echo "EXTENDED EVOLUTION - ALL SCENARIOS"
echo "================================================================================"
echo ""
echo "Hyperparameters:"
echo "  Population:  $POPULATION"
echo "  Generations: $GENERATIONS"
echo "  Games:       $GAMES"
echo ""
echo "Scenarios: ${#SCENARIOS[@]}"
for scenario in "${SCENARIOS[@]}"; do
    echo "  - $scenario"
done
echo ""

# Process each scenario
for scenario in "${SCENARIOS[@]}"; do
    echo "================================================================================"
    echo "Processing: $scenario"
    echo "================================================================================"
    echo ""

    # Output to extended subdirectory
    OUTPUT_DIR="experiments/scenarios/$scenario/evolved/extended"

    # Run evolution
    .venv/bin/python experiments/scripts/run_evolution.py \
        "$scenario" \
        --population "$POPULATION" \
        --generations "$GENERATIONS" \
        --games "$GAMES" \
        --output-dir "$OUTPUT_DIR"

    echo ""
    echo "✅ Completed: $scenario"
    echo ""
done

echo "================================================================================"
echo "ALL SCENARIOS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: experiments/scenarios/*/evolved/extended/"
