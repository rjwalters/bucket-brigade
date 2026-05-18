#!/usr/bin/env bash
# Issue #265 sweep driver — dense Δsafe progress shaping on minimal_specialization.
#
# Pre-registered grid (per the curator enhancement on #265):
#   progress_shaping_coef in {0.0, 1.0, 5.0, 25.0}             = 4 configs
#   seeds {42, 43, 44}                                          = 3 each
#   total = 12 cells
#
# Per cell: 50 iterations x 2048 rollout steps x IPPO (no MAPPO).
#
# Layout (consumed by analyze_265.py):
#   experiments/p3_specialization/runs/issue265_progress_signal/
#       coef_{COEF}/seed_{SEED}/
#           metrics.json, config.json, policies/, train.log
#
# Wall-clock: ~1.5 min/cell at 50 iters; 12 cells / 4-way parallel
# ~5 min ideal, ~10-15 min realistic. The curator's issue-body estimate
# (~30 min wall clock at 4-way parallel) is conservative.
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

# Pre-registered grid. The baseline cell (coef=0.0) is the in-sweep
# reference used by analyze_265.py to compute the gap_closed delta. The
# magnitudes are chosen against the existing
# ``team_reward_house_survives=10.0`` on minimal_specialization:
# 1.0 is small relative to team signal, 25.0 is comparable, 5.0 intermediate.
COEFS=(0.0 1.0 5.0 25.0)

jobs=()
for coef in "${COEFS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    outdir="$OUTPUT_ROOT/issue265_progress_signal/coef_${coef}/seed_${seed}"
    jobs+=("${coef}|${seed}|${outdir}")
  done
done

echo
echo "============================================================"
echo "Issue #265 dense progress shaping sweep"
echo "  scenario:        $SCENARIO"
echo "  num_iterations:  $NUM_ITERATIONS"
echo "  rollout_steps:   $ROLLOUT_STEPS"
echo "  coefs:           ${COEFS[*]}"
echo "  seeds:           ${SEEDS[*]}"
echo "  cells:           ${#jobs[@]}"
echo "  parallelism:     $PARALLEL_PER_POOL"
echo "============================================================"

run_cell() {
  local spec="$1"
  IFS='|' read -r coef seed outdir <<< "$spec"
  mkdir -p "$outdir"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START $outdir (coef=$coef)"
  if uv run python -m experiments.p3_specialization.train \
       --scenario "$SCENARIO" --lambda-red 0.0 --seed "$seed" \
       --num-iterations "$NUM_ITERATIONS" --rollout-steps "$ROLLOUT_STEPS" \
       --progress-shaping-coef "$coef" \
       --output-dir "$outdir" \
       > "$outdir/train.log" 2>&1; then
    if [[ -f "$outdir/config.json" ]]; then
      python3 -c "
import json
p = '$outdir/config.json'
with open(p) as f: c = json.load(f)
c['_git_rev'] = '$GIT_REV'
c['_issue'] = 265
c['_progress_shaping_coef'] = float('$coef')
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
echo "Issue #265 sweep complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
touch "$OUTPUT_ROOT/.issue265_progress_signal_done"
