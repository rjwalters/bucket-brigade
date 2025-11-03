#!/bin/bash
# Automated research refresh (no prompts)
# Perfect for remote execution or CI/CD

set -e

# Configuration (can be overridden with environment variables)
POPULATION=${POPULATION:-100}
GENERATIONS=${GENERATIONS:-100}
GAMES_EVOLUTION=${GAMES_EVOLUTION:-30}
GAMES_HEURISTICS=${GAMES_HEURISTICS:-100}
GAMES_COMPARISON=${GAMES_COMPARISON:-50}

# Scenarios to process (can override with SCENARIOS env var)
if [ -z "$SCENARIOS" ]; then
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
else
    # Parse comma-separated list
    IFS=',' read -ra SCENARIOS <<< "$SCENARIOS"
fi

echo "================================================================================"
echo "AUTOMATED RESEARCH REFRESH"
echo "================================================================================"
echo ""
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo ""
echo "Configuration:"
echo "  Population:           $POPULATION"
echo "  Generations:          $GENERATIONS"
echo "  Games (evolution):    $GAMES_EVOLUTION"
echo "  Games (heuristics):   $GAMES_HEURISTICS"
echo "  Games (comparison):   $GAMES_COMPARISON"
echo ""
echo "Scenarios: ${#SCENARIOS[@]}"
for scenario in "${SCENARIOS[@]}"; do
    echo "  - $scenario"
done
echo ""

# Track timing and results
START_TIME=$(date +%s)
COMPLETED=0
FAILED=0
declare -a FAILED_SCENARIOS

# Process each scenario
for scenario in "${SCENARIOS[@]}"; do
    echo "================================================================================"
    echo "[$((COMPLETED + FAILED + 1))/${#SCENARIOS[@]}] Processing: $scenario"
    echo "================================================================================"

    SCENARIO_START=$(date +%s)
    SCENARIO_FAILED=0

    # Step 1: Heuristics
    echo "[1/3] Heuristics Analysis..."
    if ! .venv/bin/python experiments/scripts/analyze_heuristics.py \
        "$scenario" \
        --num-games "$GAMES_HEURISTICS" 2>&1 | grep -v "Gym has been"; then
        echo "❌ Heuristics failed"
        SCENARIO_FAILED=1
    fi

    # Step 2: Evolution (only if heuristics succeeded)
    if [ $SCENARIO_FAILED -eq 0 ]; then
        echo "[2/3] Evolutionary Optimization..."
        if ! .venv/bin/python experiments/scripts/run_evolution.py \
            "$scenario" \
            --population "$POPULATION" \
            --generations "$GENERATIONS" \
            --games "$GAMES_EVOLUTION" 2>&1 | grep -v "Gym has been"; then
            echo "❌ Evolution failed"
            SCENARIO_FAILED=1
        fi
    fi

    # Step 3: Comparison (only if evolution succeeded)
    if [ $SCENARIO_FAILED -eq 0 ]; then
        echo "[3/3] Cross-Method Comparison..."
        if ! .venv/bin/python experiments/scripts/run_comparison.py \
            "$scenario" \
            --num-games "$GAMES_COMPARISON" 2>&1 | grep -v "Gym has been"; then
            echo "❌ Comparison failed"
            SCENARIO_FAILED=1
        fi
    fi

    SCENARIO_END=$(date +%s)
    SCENARIO_TIME=$((SCENARIO_END - SCENARIO_START))

    if [ $SCENARIO_FAILED -eq 0 ]; then
        echo "✅ $scenario complete in ${SCENARIO_TIME}s"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "❌ $scenario failed after ${SCENARIO_TIME}s"
        FAILED=$((FAILED + 1))
        FAILED_SCENARIOS+=("$scenario")
    fi
    echo ""
done

# Final summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "================================================================================"
echo "AUTOMATED RESEARCH REFRESH COMPLETE"
echo "================================================================================"
echo ""
echo "Timestamp: $(date)"
echo ""
echo "Summary:"
echo "  Completed:  $COMPLETED/${#SCENARIOS[@]}"
echo "  Failed:     $FAILED"
echo "  Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m)"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Failed scenarios:"
    for failed in "${FAILED_SCENARIOS[@]}"; do
        echo "  - $failed"
    done
    echo ""
fi

if [ $FAILED -eq 0 ]; then
    echo "✅ All scenarios completed successfully!"
    exit 0
else
    echo "⚠️  Some scenarios failed."
    exit 1
fi
