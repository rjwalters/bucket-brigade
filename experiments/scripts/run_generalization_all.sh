#!/bin/bash
# Run Phase 1.5: Cross-Scenario Generalization Analysis
#
# This script evaluates all 9 evolved agents across all 9 scenarios (81 evaluations total)
#
# Usage:
#   ./experiments/scripts/run_generalization_all.sh                    # Sequential mode
#   ./experiments/scripts/run_generalization_all.sh --parallel          # Parallel mode (faster)
#   ./experiments/scripts/run_generalization_all.sh --quick             # Quick mode (200 sims)

set -e  # Exit on error

# Default configuration
SIMULATIONS=2000
EVOLVED_VERSION="v4"
SEED=42
PARALLEL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --quick)
            SIMULATIONS=200
            shift
            ;;
        --full)
            SIMULATIONS=2000
            shift
            ;;
        --simulations)
            SIMULATIONS="$2"
            shift 2
            ;;
        --evolved-version)
            EVOLVED_VERSION="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel              Run evaluations in parallel (9 jobs)"
            echo "  --quick                 Quick mode: 200 simulations (default: 2000)"
            echo "  --full                  Full mode: 2000 simulations"
            echo "  --simulations N         Custom number of simulations"
            echo "  --evolved-version VER   Evolution version: v3, v4, v5 (default: v4)"
            echo "  --seed N                Random seed (default: 42)"
            echo "  --help                  Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --parallel           # Fast parallel execution (~10 min)"
            echo "  $0 --quick              # Quick sequential test (~20 min)"
            echo "  $0                      # Full sequential run (~80 min)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "================================================================================"
echo "Phase 1.5: Cross-Scenario Generalization Analysis"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Mode:              $(if $PARALLEL; then echo 'Parallel (9 agents in parallel)'; else echo 'Sequential'; fi)"
echo "  Simulations:       $SIMULATIONS"
echo "  Evolution Version: $EVOLVED_VERSION"
echo "  Seed:              $SEED"
echo ""

# Scenarios list
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

# Create output directory
OUTPUT_DIR="experiments/generalization"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Logging
LOG_FILE="$OUTPUT_DIR/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Function to run evaluation for single agent across all test scenarios
run_agent_evaluations() {
    local agent_scenario=$1
    local log_file="$OUTPUT_DIR/logs/${agent_scenario}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date +%H:%M:%S)] Starting evaluations for agent: $agent_scenario" | tee -a "$LOG_FILE"

    for test_scenario in "${SCENARIOS[@]}"; do
        echo "  $agent_scenario → $test_scenario..." | tee -a "$LOG_FILE"

        uv run python experiments/scripts/evaluate_cross_scenario.py \
            --agent-scenario "$agent_scenario" \
            --test-scenario "$test_scenario" \
            --evolved-version "$EVOLVED_VERSION" \
            --simulations "$SIMULATIONS" \
            --seed "$SEED" \
            --output "$OUTPUT_DIR/individual/${agent_scenario}_to_${test_scenario}.json" \
            >> "$log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "    ✓ Complete" | tee -a "$LOG_FILE"
        else
            echo "    ✗ ERROR - see $log_file" | tee -a "$LOG_FILE"
        fi
    done

    echo "[$(date +%H:%M:%S)] Completed evaluations for agent: $agent_scenario" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Export function for parallel execution
export -f run_agent_evaluations
export OUTPUT_DIR LOG_FILE SIMULATIONS EVOLVED_VERSION SEED
export SCENARIOS

# Create individual results directory
mkdir -p "$OUTPUT_DIR/individual"

START_TIME=$(date +%s)

if $PARALLEL; then
    echo "Running evaluations in parallel (9 agents × 9 scenarios each)..."
    echo "This will spawn 9 parallel processes"
    echo ""

    # Run evaluations in parallel using xargs
    printf '%s\n' "${SCENARIOS[@]}" | xargs -P 9 -I {} bash -c 'run_agent_evaluations "$@"' _ {}

else
    echo "Running evaluations sequentially (81 total)..."
    echo ""

    # Sequential execution
    for agent_scenario in "${SCENARIOS[@]}"; do
        run_agent_evaluations "$agent_scenario"
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo "================================================================================"
echo "Individual Evaluations Complete"
echo "================================================================================"
echo ""
echo "Time elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""

# Now combine all results into performance matrix using Python script
echo "Combining results into performance matrix..."
echo ""

uv run python experiments/scripts/evaluate_cross_scenario.py \
    --all \
    --evolved-version "$EVOLVED_VERSION" \
    --simulations "$SIMULATIONS" \
    --seed "$SEED" \
    --output "$OUTPUT_DIR/performance_matrix.json" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "================================================================================"
echo "Phase 1.5 Data Collection Complete"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - Performance Matrix: $OUTPUT_DIR/performance_matrix.json"
echo "  - Individual Results: $OUTPUT_DIR/individual/*.json"
echo "  - Logs: $OUTPUT_DIR/logs/"
echo ""
echo "Next steps:"
echo "  1. Run analysis: python experiments/scripts/analyze_generalization.py"
echo "  2. Generate visualizations"
echo "  3. Document findings in GENERALIZATION_RESULTS.md"
echo ""
