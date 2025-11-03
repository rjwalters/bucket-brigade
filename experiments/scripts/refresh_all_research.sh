#!/bin/bash
# Refresh all research data for all scenarios
# This script runs the complete research pipeline: heuristics, evolution, comparison
# Ideal for running on a fast machine to regenerate all data

set -e

# Configuration
POPULATION=${POPULATION:-100}
GENERATIONS=${GENERATIONS:-100}
GAMES_EVOLUTION=${GAMES_EVOLUTION:-30}
GAMES_HEURISTICS=${GAMES_HEURISTICS:-100}
GAMES_COMPARISON=${GAMES_COMPARISON:-50}

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
echo "REFRESH ALL RESEARCH DATA"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Population:           $POPULATION"
echo "  Generations:          $GENERATIONS"
echo "  Games (evolution):    $GAMES_EVOLUTION"
echo "  Games (heuristics):   $GAMES_HEURISTICS"
echo "  Games (comparison):   $GAMES_COMPARISON"
echo ""
echo "Scenarios to process: ${#SCENARIOS[@]}"
for scenario in "${SCENARIOS[@]}"; do
    echo "  - $scenario"
done
echo ""
echo "This will overwrite existing results in:"
echo "  experiments/scenarios/*/heuristics/"
echo "  experiments/scenarios/*/evolved/"
echo "  experiments/scenarios/*/comparison/"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi
echo ""

# Track timing
START_TIME=$(date +%s)
COMPLETED=0
FAILED=0

# Process each scenario
for scenario in "${SCENARIOS[@]}"; do
    echo "================================================================================"
    echo "Processing: $scenario ($((COMPLETED + FAILED + 1))/${#SCENARIOS[@]})"
    echo "================================================================================"
    echo ""

    SCENARIO_START=$(date +%s)

    # Step 1: Heuristics Analysis
    echo "Step 1/3: Heuristics Analysis"
    if .venv/bin/python experiments/scripts/analyze_heuristics.py \
        "$scenario" \
        --num-games "$GAMES_HEURISTICS"; then
        echo "✅ Heuristics complete"
    else
        echo "❌ Heuristics failed"
        FAILED=$((FAILED + 1))
        continue
    fi
    echo ""

    # Step 2: Evolution
    echo "Step 2/3: Evolutionary Optimization"
    if .venv/bin/python experiments/scripts/run_evolution.py \
        "$scenario" \
        --population "$POPULATION" \
        --generations "$GENERATIONS" \
        --games "$GAMES_EVOLUTION"; then
        echo "✅ Evolution complete"
    else
        echo "❌ Evolution failed"
        FAILED=$((FAILED + 1))
        continue
    fi
    echo ""

    # Step 3: Comparison
    echo "Step 3/3: Cross-Method Comparison"
    if .venv/bin/python experiments/scripts/run_comparison.py \
        "$scenario" \
        --num-games "$GAMES_COMPARISON"; then
        echo "✅ Comparison complete"
    else
        echo "❌ Comparison failed"
        FAILED=$((FAILED + 1))
        continue
    fi
    echo ""

    SCENARIO_END=$(date +%s)
    SCENARIO_TIME=$((SCENARIO_END - SCENARIO_START))

    echo "✅ $scenario complete in ${SCENARIO_TIME}s"
    echo ""

    COMPLETED=$((COMPLETED + 1))
done

# Final summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "================================================================================"
echo "RESEARCH REFRESH COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  Completed:  $COMPLETED/${#SCENARIOS[@]}"
echo "  Failed:     $FAILED"
echo "  Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m)"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All scenarios completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  experiments/scenarios/*/heuristics/results.json"
    echo "  experiments/scenarios/*/evolved/best_agent.json"
    echo "  experiments/scenarios/*/evolved/evolution_trace.json"
    echo "  experiments/scenarios/*/comparison/comparison.json"
    echo ""
    echo "Next steps:"
    echo "  1. Review results: python experiments/scripts/summarize_results.py"
    echo "  2. Commit changes: git add experiments/ && git commit -m 'research: Refresh all scenario data'"
    echo "  3. Push to remote: git push"
else
    echo "⚠️  Some scenarios failed. Check logs above for details."
    exit 1
fi
