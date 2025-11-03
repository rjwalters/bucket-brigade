#!/bin/bash
# Train policies for all named Bucket Brigade scenarios
#
# This script trains a policy on each of the 9 predefined scenarios
# using consistent hyperparameters. Training duration: ~9 hours total
# (can be parallelized if you have multiple cores).
#
# Usage:
#   ./scripts/train_all_scenarios.sh

set -e  # Exit on error

# Training hyperparameters
NUM_STEPS=200000
NUM_OPPONENTS=3
BATCH_SIZE=2048
HIDDEN_SIZE=128
SEED=42

# Scenarios to train (excluding meta-scenarios like 'default', 'easy', 'hard')
SCENARIOS=(
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

echo "🚀 Starting batch training for ${#SCENARIOS[@]} scenarios"
echo "   Steps per scenario: $NUM_STEPS"
echo "   Estimated time: ~$(( ${#SCENARIOS[@]} * 60 )) minutes total"
echo ""

mkdir -p models

for scenario in "${SCENARIOS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎮 Training on scenario: $scenario"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    uv run python scripts/train_simple.py \
        --scenario "$scenario" \
        --num-steps "$NUM_STEPS" \
        --num-opponents "$NUM_OPPONENTS" \
        --batch-size "$BATCH_SIZE" \
        --hidden-size "$HIDDEN_SIZE" \
        --seed "$SEED" \
        --save-path "models/policy_${scenario}.pt"

    echo "✅ Completed: $scenario"
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All scenarios trained successfully!"
echo ""
echo "Trained models saved to:"
for scenario in "${SCENARIOS[@]}"; do
    echo "  - models/policy_${scenario}.pt"
done
echo ""
echo "Next steps:"
echo "  1. Evaluate models: ./scripts/evaluate_transfer.sh"
echo "  2. Analyze scenarios: uv run python scripts/analyze_scenarios.py"
