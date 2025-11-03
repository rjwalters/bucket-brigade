#!/bin/bash
# Evaluate cross-scenario transfer learning
#
# This script tests how well a policy trained on one scenario
# performs when evaluated on different scenarios. This helps identify:
# - Which skills transfer across scenarios
# - Which scenarios require specialized strategies
# - Optimal curriculum ordering
#
# Usage:
#   ./scripts/evaluate_transfer.sh <train_scenario>
#
# Example:
#   ./scripts/evaluate_transfer.sh trivial_cooperation

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <train_scenario>"
    echo ""
    echo "Available scenarios:"
    uv run python scripts/train_simple.py --list-scenarios
    exit 1
fi

TRAIN_SCENARIO=$1
MODEL_PATH="models/policy_${TRAIN_SCENARIO}.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo "Train the model first with:"
    echo "  uv run python scripts/train_simple.py --scenario $TRAIN_SCENARIO --save-path $MODEL_PATH"
    exit 1
fi

# Test scenarios (excluding meta-scenarios)
TEST_SCENARIOS=(
    "trivial_cooperation"
    "early_containment"
    "greedy_neighbor"
    "sparse_heroics"
    "rest_trap"
    "chain_reaction"
    "deceptive_calm"
    "overcrowding"
    "mixed_motivation"
)

NUM_EPISODES=100

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 Cross-Scenario Transfer Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model trained on: $TRAIN_SCENARIO"
echo "Model path: $MODEL_PATH"
echo "Episodes per scenario: $NUM_EPISODES"
echo ""

mkdir -p results

# Create results file
RESULTS_FILE="results/transfer_${TRAIN_SCENARIO}.txt"
echo "Transfer Learning Results" > "$RESULTS_FILE"
echo "Trained on: $TRAIN_SCENARIO" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Scenario                  Mean Reward    Std Dev" >> "$RESULTS_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$RESULTS_FILE"

for test_scenario in "${TEST_SCENARIOS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Testing on: $test_scenario"

    # Capture output and parse statistics
    OUTPUT=$(uv run python scripts/evaluate_simple.py \
        --model-path "$MODEL_PATH" \
        --scenario "$test_scenario" \
        --num-episodes "$NUM_EPISODES" 2>&1)

    echo "$OUTPUT"

    # Extract mean reward (this assumes the output format from evaluate_simple.py)
    MEAN_REWARD=$(echo "$OUTPUT" | grep "Mean Reward:" | awk '{print $3}')
    STD_DEV=$(echo "$OUTPUT" | grep "Mean Reward:" | awk '{print $5}')

    # Log to results file
    printf "%-25s %12s   %s\n" "$test_scenario" "$MEAN_REWARD" "$STD_DEV" >> "$RESULTS_FILE"

    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Transfer evaluation complete!"
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Next steps:"
echo "  1. View results: cat $RESULTS_FILE"
echo "  2. Compare with other models: ./scripts/evaluate_transfer.sh <other_scenario>"
echo "  3. Analyze difficulty: uv run python scripts/analyze_scenarios.py"
