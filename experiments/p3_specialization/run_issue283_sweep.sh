#!/usr/bin/env bash
# Issue #283 sweep driver — potential-based team-welfare shaping calibration
# on minimal_specialization. Ng-Harada-Russell (1999) aligned shaping.
#
# Pre-registered grid:
#   lambda in {0.0, 0.1, 0.5, 1.0}                              = 4 cells
#   kind   = "team_welfare_closed_form"  (option B)
#   gamma  = 0.99 (matches typical PPO discount)
#   seeds  {42, 43, 44}                                          = 3 each
#   total  = 12 cells
#
# Per cell: 50 iterations x 2048 rollout steps x IPPO (no MAPPO).
#
# The lambda=0 cell is the in-sweep baseline (env fast-path skip; reward
# stream byte-identical to pre-#283). It anchors the within-sweep gap
# comparison so we're not comparing to a stale historical baseline.
#
# Layout (intended for the future analyze_283_potential_shaping.py):
#   experiments/p3_specialization/runs/issue283_potential_shaping/
#       kind_{KIND}/lambda_{LAMBDA}/seed_{SEED}/
#           metrics.json, config.json, policies/, train.log
#
# Wall-clock: ~1.5 min/cell at 50 iters; 12 cells / 4-way parallel
# ~5 min ideal. Estimated ~10-15 min realistic on COMPUTE_HOST_PRIMARY.
#
# Per CLAUDE.md compute guidelines: DO NOT RUN LOCALLY. This is a sweep
# script — execute it inside tmux on COMPUTE_HOST_PRIMARY only. Smoke
# tests (single seed, ~10 iters, lambda=1) can run locally but the full
# sweep is bound for the remote host.
#
# Modeled on experiments/p3_specialization/run_issue262_sweep.sh
# (xargs -P parallelism, per-cell stdout, git-rev stamping).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/GitHub/bucket-brigade}"
if [[ ! -d "$REPO_ROOT" ]]; then
  # Fallback used on the Mac Studio.
  if [[ -d "$HOME/bucket-brigade" ]]; then
    REPO_ROOT="$HOME/bucket-brigade"
  fi
fi
cd "$REPO_ROOT"

GIT_REV="$(git rev-parse HEAD)"
echo "Pinned to commit: $GIT_REV"

OUTPUT_ROOT="experiments/p3_specialization/runs"
SCENARIO="${SCENARIO:-minimal_specialization}"
NUM_ITERATIONS="${NUM_ITERATIONS:-50}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-2048}"
SEEDS=(42 43 44)
PARALLEL_PER_POOL="${PARALLEL_PER_POOL:-4}"

# Pre-registered grid. The baseline cell (lambda=0) anchors the in-sweep
# comparison; the other 3 cells span the calibration grid.
LAMBDAS=(0.0 0.1 0.5 1.0)
# Default to option B (closed-form). Add "specialist_mc_regressor" here
# when option A lands (#283 followup).
KIND="${TEAM_WELFARE_KIND:-team_welfare_closed_form}"
# NHR discount. Matched to a typical PPO gamma so the invariance argument
# is tight; configurable via env var.
TEAM_WELFARE_GAMMA="${TEAM_WELFARE_GAMMA:-0.99}"

# Build job list (lambda|seed|outdir). The lambda=0 cell uses kind="none"
# automatically via the env fast-path skip; we still pass the kind so the
# config.json records the intended (non-)treatment cleanly.
jobs=()
for lam in "${LAMBDAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    outdir="$OUTPUT_ROOT/issue283_potential_shaping/kind_${KIND}/lambda_${lam}/seed_${seed}"
    jobs+=("${lam}|${seed}|${outdir}")
  done
done

echo
echo "============================================================"
echo "Issue #283 potential-based team-welfare shaping sweep"
echo "  scenario:           $SCENARIO"
echo "  num_iterations:     $NUM_ITERATIONS"
echo "  rollout_steps:      $ROLLOUT_STEPS"
echo "  kind:               $KIND"
echo "  team_welfare_gamma: $TEAM_WELFARE_GAMMA"
echo "  lambdas:            ${LAMBDAS[*]}"
echo "  seeds:              ${SEEDS[*]}"
echo "  cells:              ${#jobs[@]}"
echo "  parallelism:        $PARALLEL_PER_POOL"
echo "============================================================"

run_cell() {
  local spec="$1"
  IFS='|' read -r lam seed outdir <<< "$spec"
  mkdir -p "$outdir"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START $outdir (lambda=$lam)"
  if uv run python -m experiments.p3_specialization.train \
       --scenario "$SCENARIO" --lambda-red 0.0 --seed "$seed" \
       --num-iterations "$NUM_ITERATIONS" --rollout-steps "$ROLLOUT_STEPS" \
       --team-welfare-lambda "$lam" \
       --team-welfare-gamma "$TEAM_WELFARE_GAMMA" \
       --team-welfare-kind "$KIND" \
       --output-dir "$outdir" \
       > "$outdir/train.log" 2>&1; then
    if [[ -f "$outdir/config.json" ]]; then
      python3 -c "
import json
p = '$outdir/config.json'
with open(p) as f: c = json.load(f)
c['_git_rev'] = '$GIT_REV'
c['_issue'] = 283
c['_team_welfare_lambda'] = float('$lam')
c['_team_welfare_gamma'] = float('$TEAM_WELFARE_GAMMA')
c['_team_welfare_kind'] = '$KIND'
with open(p, 'w') as f: json.dump(c, f, indent=2)
"
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DONE  $outdir"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAIL  $outdir (see train.log)" >&2
    return 1
  fi
}

export -f run_cell
export SCENARIO NUM_ITERATIONS ROLLOUT_STEPS GIT_REV TEAM_WELFARE_GAMMA KIND

echo
echo "--- Launching ${#jobs[@]} cells, ${PARALLEL_PER_POOL}-way parallel ---"
printf "%s\n" "${jobs[@]}" | xargs -n1 -P "$PARALLEL_PER_POOL" -I{} bash -c 'run_cell "$@"' _ {}

echo
echo "============================================================"
echo "Issue #283 sweep complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
touch "$OUTPUT_ROOT/.issue283_potential_shaping_done"
