#!/usr/bin/env bash
# Issue #290 sweep driver — Jaques-2019 social-influence intrinsic motivation
# on minimal_specialization.
#
# Pre-registered grid (per issue #290 curator block):
#   influence_coef (alpha) in {0.0, 0.1, 0.5, 1.0}             = 4 alphas
#   seeds {42, 43, 44}                                          = 3 seeds
#   total = 12 cells
#
# Per cell: 50 iterations x 2048 rollout steps x IPPO (no MAPPO, no redundancy).
# The alpha=0 cell is the IPPO control — must be bit-identical to the
# pre-#290 baseline (this is the regression test
# ``test_influence_coef_zero_bit_identical``).
#
# Layout (consumed by analyze_290.py — to be added in a follow-up):
#   experiments/p3_specialization/runs/issue290_influence/
#       alpha_{ALPHA}/seed_{SEED}/
#           metrics.json, config.json, policies/, train.log
#
# Wall-clock: ~30 min/cell × 12 cells / 4-way parallel ≈ 1.5 h wall, ~6 h CPU.
# Run on COMPUTE_HOST_PRIMARY per CLAUDE.md compute guidelines — local
# machine is NOT a compute platform.
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
INFLUENCE_MC_SAMPLES="${INFLUENCE_MC_SAMPLES:-4}"

# Pre-registered grid. alpha=0 is the IPPO control.
ALPHAS=(0.0 0.1 0.5 1.0)

jobs=()
for alpha in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    outdir="$OUTPUT_ROOT/issue290_influence/alpha_${alpha}/seed_${seed}"
    jobs+=("${alpha}|${seed}|${outdir}")
  done
done

echo
echo "============================================================"
echo "Issue #290 social-influence sweep"
echo "  scenario:           $SCENARIO"
echo "  num_iterations:     $NUM_ITERATIONS"
echo "  rollout_steps:      $ROLLOUT_STEPS"
echo "  alphas:             ${ALPHAS[*]}"
echo "  seeds:              ${SEEDS[*]}"
echo "  mc_samples:         $INFLUENCE_MC_SAMPLES"
echo "  cells:              ${#jobs[@]}"
echo "  parallelism:        $PARALLEL_PER_POOL"
echo "============================================================"

run_cell() {
  local spec="$1"
  IFS='|' read -r alpha seed outdir <<< "$spec"
  mkdir -p "$outdir"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START $outdir (alpha=$alpha seed=$seed)"
  if uv run python -m experiments.p3_specialization.train \
       --scenario "$SCENARIO" --lambda-red 0.0 --seed "$seed" \
       --num-iterations "$NUM_ITERATIONS" --rollout-steps "$ROLLOUT_STEPS" \
       --influence-coef "$alpha" \
       --influence-mc-samples "$INFLUENCE_MC_SAMPLES" \
       --output-dir "$outdir" \
       > "$outdir/train.log" 2>&1; then
    if [[ -f "$outdir/config.json" ]]; then
      python3 -c "
import json
p = '$outdir/config.json'
with open(p) as f: c = json.load(f)
c['_git_rev'] = '$GIT_REV'
c['_issue'] = 290
c['_influence_coef'] = float('$alpha')
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
export SCENARIO NUM_ITERATIONS ROLLOUT_STEPS INFLUENCE_MC_SAMPLES GIT_REV

echo
echo "--- Launching ${#jobs[@]} cells, ${PARALLEL_PER_POOL}-way parallel ---"
printf "%s\n" "${jobs[@]}" | xargs -n1 -P "$PARALLEL_PER_POOL" -I{} bash -c 'run_cell "$@"' _ {}

echo
echo "============================================================"
echo "Issue #290 sweep complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
touch "$OUTPUT_ROOT/.issue290_influence_done"
