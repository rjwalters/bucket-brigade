#!/usr/bin/env bash
# Run the 6-condition P3 plateau ablation specified in issue #174.
# Each invocation of run_sweep.py runs default scenario, lambda=0, seeds 42..46
# under one (normalize_returns, value_coef, entropy_coef) combination.
#
# Output layout (per-cell; gitignored under the top-level runs/ pattern):
#   experiments/p3_specialization/runs/p3_174_ablation/<condition>/default/lambda_0e0/seed_<N>/metrics.json
#
# Aggregated outputs (committed) live in experiments/p3_specialization/results_174_ablation/
# after running analyze_174.py.
#
# See research_notebook/2026-05-15_p3_plateau_ablation.md for the writeup.

set -euo pipefail

BASE="experiments/p3_specialization/runs/p3_174_ablation"
mkdir -p "$BASE"

# Conditions: name | normalize_returns | value_coef | entropy_coef
declare -a CONDITIONS=(
    "baseline:false:0.5:0.01"
    "L1_norm:true:0.5:0.01"
    "L2_low_vf:false:0.05:0.01"
    "L3_high_ent:false:0.5:0.1"
    "L1L2:true:0.05:0.01"
    "L1L3:true:0.5:0.1"
)

for COND in "${CONDITIONS[@]}"; do
    IFS=':' read -r NAME NORM VC EC <<< "$COND"
    OUT="$BASE/$NAME"

    NORM_FLAG=""
    if [[ "$NORM" == "true" ]]; then
        NORM_FLAG="--normalize-returns"
    fi

    echo ""
    echo "================================================================"
    echo "Condition: $NAME"
    echo "  normalize_returns=$NORM  value_coef=$VC  entropy_coef=$EC"
    echo "  output_root=$OUT"
    echo "  start=$(date -u +%FT%TZ)"
    echo "================================================================"

    uv run python -m experiments.p3_specialization.run_sweep \
        --output-root "$OUT" \
        --scenarios default \
        --lambdas 0 \
        --seeds 42 43 44 45 46 \
        --num-iterations 50 \
        --rollout-steps 2048 \
        --device cpu \
        --value-coef "$VC" \
        --entropy-coef "$EC" \
        $NORM_FLAG

    echo "  end=$(date -u +%FT%TZ)"
done

echo ""
echo "All 6 conditions complete: $(date -u +%FT%TZ)"
