#!/usr/bin/env bash
# Issue #261/#262 sweep driver — action-shaping calibration on minimal_specialization.
#
# Pre-registered grid (per issue #262 body):
#   alpha in {0.1, 0.5, 2.0} x beta in {0.0, 0.1, 0.5}        = 9 cells
#   plus baseline (alpha=0, beta=0)                            = 10 configs
#   seeds {42, 43, 44}                                         = 3 each
#   total = 30 cells
#
# Per cell: 50 iterations x 2048 rollout steps x IPPO (no MAPPO).
#
# Layout (consumed by analyze_261_calibration.py):
#   experiments/p3_specialization/runs/issue261_calibration/
#       alpha_{ALPHA}/beta_{BETA}/seed_{SEED}/
#           metrics.json, config.json, policies/, train.log
#
# Wall-clock: ~1.5 min/cell at 50 iters; 30 cells / 4-way parallel
# ~12 min ideal, ~20-30 min realistic. Curator confirmed the issue
# body's 3-4h estimate is conservative — half-iters of the #239 sweep.
#
# Modeled on experiments/p3_specialization/run_issue239_sweep.sh
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

# Pre-registered grid. The baseline cell (alpha=0, beta=0) anchors the
# in-sweep entropy reference; the other 9 cells span the calibration grid.
ALPHAS=(0.0 0.1 0.5 2.0)
BETAS=(0.0 0.1 0.5)

# Build job list (alpha|beta|seed|outdir). Skip nothing — the (0,0) cell
# is the in-sweep baseline used by analyze_261_calibration.py to compute
# entropy collapse multiples; we want it from the same code path as the
# treatment cells for a fair within-sweep comparison.
jobs=()
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      outdir="$OUTPUT_ROOT/issue261_calibration/alpha_${alpha}/beta_${beta}/seed_${seed}"
      jobs+=("${alpha}|${beta}|${seed}|${outdir}")
    done
  done
done

echo
echo "============================================================"
echo "Issue #261/#262 calibration sweep"
echo "  scenario:        $SCENARIO"
echo "  num_iterations:  $NUM_ITERATIONS"
echo "  rollout_steps:   $ROLLOUT_STEPS"
echo "  alphas:          ${ALPHAS[*]}"
echo "  betas:           ${BETAS[*]}"
echo "  seeds:           ${SEEDS[*]}"
echo "  cells:           ${#jobs[@]}"
echo "  parallelism:     $PARALLEL_PER_POOL"
echo "============================================================"

run_cell() {
  local spec="$1"
  IFS='|' read -r alpha beta seed outdir <<< "$spec"
  mkdir -p "$outdir"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START $outdir (alpha=$alpha beta=$beta)"
  if uv run python -m experiments.p3_specialization.train \
       --scenario "$SCENARIO" --lambda-red 0.0 --seed "$seed" \
       --num-iterations "$NUM_ITERATIONS" --rollout-steps "$ROLLOUT_STEPS" \
       --action-shaping-alpha "$alpha" --action-shaping-beta "$beta" \
       --output-dir "$outdir" \
       > "$outdir/train.log" 2>&1; then
    if [[ -f "$outdir/config.json" ]]; then
      python3 -c "
import json
p = '$outdir/config.json'
with open(p) as f: c = json.load(f)
c['_git_rev'] = '$GIT_REV'
c['_issue'] = 262
c['_action_shaping_alpha'] = float('$alpha')
c['_action_shaping_beta'] = float('$beta')
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
export SCENARIO NUM_ITERATIONS ROLLOUT_STEPS GIT_REV

echo
echo "--- Launching ${#jobs[@]} cells, ${PARALLEL_PER_POOL}-way parallel ---"
printf "%s\n" "${jobs[@]}" | xargs -n1 -P "$PARALLEL_PER_POOL" -I{} bash -c 'run_cell "$@"' _ {}

echo
echo "============================================================"
echo "Issue #261/#262 sweep complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
touch "$OUTPUT_ROOT/.issue261_calibration_done"
