#!/usr/bin/env bash
# Issue #183 — Phase 3 long-horizon retest of #174's L1-anchored configs.
#
# Three conditions x two scenarios x five seeds x 500 iters = 30 cells.
# Conditions are the three L1-anchored variants from #174 (all normalize_returns=True):
#
#     L1_norm:  value_coef=0.5,  entropy_coef=0.01
#     L1L2:     value_coef=0.05, entropy_coef=0.01
#     L1L3:     value_coef=0.5,  entropy_coef=0.10
#
# Scenarios: default + chain_reaction (the May-14 P3 behavioral-signal scenario).
# Lambda fixed at 0 (this issue isolates the learning question; lambda>0 is #176).
#
# Output layout (per-cell; gitignored under the top-level runs/ pattern):
#   experiments/p3_specialization/runs/p3_183_phase3/<condition>/<scenario>/lambda_0e0/seed_<N>/metrics.json
#
# Aggregated outputs (committed) live in experiments/p3_specialization/results_183_phase3/
# after running analyze_183.py.
#
# Compute guidance (per CLAUDE.md): NEVER run locally. Use a remote host
# (rwalters-sandbox-2 or equivalent). Env is CPU-bound; --device cpu is correct.
# Per-cell wall-clock ~10-12 min single-threaded (50-iter cells took ~63s in #174,
# scaling linearly to 500 iters); full sweep is ~5-6 cell-hours single-threaded.
# Shard across processes with --shard-seeds / --shard-scenarios / --shard-conditions
# flags below to parallelize on multi-core hosts.
#
# Usage:
#   # Full sweep (default: all conditions, all scenarios, all seeds):
#   bash experiments/p3_specialization/run_183_phase3.sh
#
#   # Shard examples (split work across N shells/processes):
#   CONDITIONS=L1_norm bash experiments/p3_specialization/run_183_phase3.sh
#   SEEDS="42 43" bash experiments/p3_specialization/run_183_phase3.sh
#   SCENARIOS=chain_reaction CONDITIONS="L1L2 L1L3" bash experiments/p3_specialization/run_183_phase3.sh
#
#   # Smoke test (10 iters x 256 steps, 1 seed, 1 scenario, 1 condition):
#   NUM_ITERATIONS=10 ROLLOUT_STEPS=256 SEEDS=42 SCENARIOS=default CONDITIONS=L1_norm \
#     OUTPUT_SUBDIR=p3_183_smoke bash experiments/p3_specialization/run_183_phase3.sh
#
# Environment variables (all optional; defaults match the issue spec):
#   CONDITIONS       : space-separated subset of {L1_norm, L1L2, L1L3}. Default: all three.
#   SCENARIOS        : space-separated subset of {default, chain_reaction}. Default: both.
#   SEEDS            : space-separated subset of {42 43 44 45 46}. Default: all five.
#   NUM_ITERATIONS   : training iters per cell. Default: 500.
#   ROLLOUT_STEPS    : env steps per iter. Default: 2048.
#   OUTPUT_SUBDIR    : directory under experiments/p3_specialization/runs/. Default: p3_183_phase3.
#   SKIP_EXISTING    : set to "1" to skip cells with metrics.json already on disk. Default: unset.

set -euo pipefail

# Defaults; overridable via environment variables.
CONDITIONS="${CONDITIONS:-L1_norm L1L2 L1L3}"
SCENARIOS="${SCENARIOS:-default chain_reaction}"
SEEDS="${SEEDS:-42 43 44 45 46}"
NUM_ITERATIONS="${NUM_ITERATIONS:-500}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-2048}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-p3_183_phase3}"
SKIP_EXISTING="${SKIP_EXISTING:-}"

BASE="experiments/p3_specialization/runs/${OUTPUT_SUBDIR}"
mkdir -p "$BASE"

# Condition parameter table (name -> normalize_returns/value_coef/entropy_coef).
# All three conditions have normalize_returns=True (the L1 anchor).
get_params() {
    case "$1" in
        L1_norm) echo "true 0.5 0.01" ;;
        L1L2)    echo "true 0.05 0.01" ;;
        L1L3)    echo "true 0.5 0.1" ;;
        *)
            echo "ERROR: unknown condition '$1' (valid: L1_norm L1L2 L1L3)" >&2
            exit 1
            ;;
    esac
}

SKIP_FLAG=""
if [[ -n "$SKIP_EXISTING" ]]; then
    SKIP_FLAG="--skip-existing"
fi

echo "============================================================"
echo "Issue #183 Phase 3 long-horizon retest"
echo "  conditions    = $CONDITIONS"
echo "  scenarios     = $SCENARIOS"
echo "  seeds         = $SEEDS"
echo "  num_iters     = $NUM_ITERATIONS"
echo "  rollout_steps = $ROLLOUT_STEPS"
echo "  base output   = $BASE"
echo "  skip_existing = ${SKIP_EXISTING:-no}"
echo "  start         = $(date -u +%FT%TZ)"
echo "============================================================"

for COND in $CONDITIONS; do
    read -r NORM VC EC <<< "$(get_params "$COND")"
    OUT="$BASE/$COND"

    NORM_FLAG=""
    if [[ "$NORM" == "true" ]]; then
        NORM_FLAG="--normalize-returns"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Condition: $COND"
    echo "  normalize_returns=$NORM  value_coef=$VC  entropy_coef=$EC"
    echo "  output_root=$OUT"
    echo "  start=$(date -u +%FT%TZ)"
    echo "------------------------------------------------------------"

    # shellcheck disable=SC2086  # word-splitting is intentional for $SCENARIOS / $SEEDS / flags
    uv run python -m experiments.p3_specialization.run_sweep \
        --output-root "$OUT" \
        --scenarios $SCENARIOS \
        --lambdas 0 \
        --seeds $SEEDS \
        --num-iterations "$NUM_ITERATIONS" \
        --rollout-steps "$ROLLOUT_STEPS" \
        --device cpu \
        --value-coef "$VC" \
        --entropy-coef "$EC" \
        $NORM_FLAG \
        $SKIP_FLAG

    echo "  end=$(date -u +%FT%TZ)"
done

echo ""
echo "============================================================"
echo "Issue #183 sweep complete: $(date -u +%FT%TZ)"
echo "============================================================"
