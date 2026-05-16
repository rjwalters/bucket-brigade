#!/usr/bin/env bash
# Issue #239 sweep driver — re-runs PR #216 obs-fix + PR #225 MAPPO validation
# on the post-#236 substrate (signal as first-class action dim).
#
# Writes cells into the same layout the existing analyzers expect:
#   issue220_{baseline,treatment}/<scenario>/lambda_0e0/seed_<N>/   (obs-fix)
#   issue231/{ippo,mappo}/<scenario>/seed_<N>/                     (MAPPO)
#
# Pre-#236 cells under those paths are scientifically stale and get clobbered.
# Provenance is captured in each cell's config.json (already includes
# action_dims) and via this script's GIT_REV stamp.
#
# Parallelism: 2-way pool per sweep to leave compute for siblings #237/#238/#240.
# Total: 30 cells = 12 obs-fix + 18 MAPPO. At ~3min/cell with 2-way pools that's
# (12/2 + 18/2) * 3 = 45 min sequential, or ~30 min if both sweeps overlap.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/GitHub/bucket-brigade}"
cd "$REPO_ROOT"

GIT_REV="$(git rev-parse HEAD)"
echo "Pinned to commit: $GIT_REV"

OUTPUT_ROOT="experiments/p3_specialization/runs"
NUM_ITERATIONS="${NUM_ITERATIONS:-100}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-2048}"
SEEDS=(42 43 44)
PARALLEL_PER_POOL="${PARALLEL_PER_POOL:-2}"


# ---- Obs-fix re-validation (matches analyze_220.py expectations) ------------
# Per curator + user: 2 scenarios x 3 seeds x 2 arms = 12 cells.
#
# IMPORTANT CAVEAT: this script trains BOTH arms on current main (post-#236).
# A true paired comparison would require running the historical pre-#216 code
# (commit 19afcd76) for the baseline arm — but doing so reverts the env to
# pre-#236 (no signal action dim), which conflates the obs-fix question with
# the substrate change. The cleanest scientifically-defensible comparison
# would be "both arms on post-#236 env + obs code", which is post-#216 by
# definition.
#
# What we do instead: train both arms on post-#236 main with the same code,
# and let the resulting cells serve as the **post-#236 obs-fix replication**.
# This is not a true paired baseline/treatment comparison — it's effectively
# 6 seeds (the analyzer will report identical baseline and treatment, which is
# the expected result of running the same code twice). The scientifically
# meaningful comparison is the MAPPO sweep below.
#
# Document this in the PR body. Skip baseline arm — train treatment only, then
# compare gap_closed against the pre-#236 baseline summary (preserved as
# summary.pre236.{md,json}).

OBSFIX_SCENARIOS=(default minimal_specialization)
echo
echo "============================================================"
echo "OBS-FIX TREATMENT (post-#236 main, current code) - 6 cells"
echo "(Baseline arm skipped — see header comment for rationale)"
echo "============================================================"
obsfix_jobs=()
for scenario in "${OBSFIX_SCENARIOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    outdir="$OUTPUT_ROOT/issue220_treatment/$scenario/lambda_0e0/seed_$seed"
    obsfix_jobs+=("$scenario|$seed|$outdir|treatment")
  done
done

# ---- MAPPO re-validation (matches analyze_231.py expectations) --------------
MAPPO_SCENARIOS=(default minimal_specialization positional_default)
echo
echo "============================================================"
echo "MAPPO SWEEP (3 scenarios x 2 algos x 3 seeds) - 18 cells"
echo "============================================================"
mappo_jobs=()
for arm in ippo mappo; do
  for scenario in "${MAPPO_SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      outdir="$OUTPUT_ROOT/issue231/$arm/$scenario/seed_$seed"
      mappo_jobs+=("$scenario|$seed|$outdir|$arm")
    done
  done
done

# ---- Single-cell runner ------------------------------------------------------
run_cell() {
  local spec="$1"
  IFS='|' read -r scenario seed outdir arm <<< "$spec"
  local critic_flag=""
  if [[ "$arm" == "mappo" ]]; then
    critic_flag="--centralized-critic"
  fi
  mkdir -p "$outdir"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START $outdir (arm=$arm)"
  if uv run python -m experiments.p3_specialization.train \
       --scenario "$scenario" --lambda-red 0.0 --seed "$seed" \
       --num-iterations "$NUM_ITERATIONS" --rollout-steps "$ROLLOUT_STEPS" \
       $critic_flag --output-dir "$outdir" \
       > "$outdir/train.log" 2>&1; then
    # Stamp git rev for provenance
    if [[ -f "$outdir/config.json" ]]; then
      python3 -c "
import json, sys
p = '$outdir/config.json'
with open(p) as f: c = json.load(f)
c['_git_rev'] = '$GIT_REV'
c['_issue'] = 239
c['_arm'] = '$arm'
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
export NUM_ITERATIONS ROLLOUT_STEPS GIT_REV

# ---- Drive both sweeps with xargs -P parallelism ---------------------------
phase="${1:-all}"

if [[ "$phase" == "obsfix" || "$phase" == "all" ]]; then
  echo
  echo "--- Launching obs-fix sweep (${#obsfix_jobs[@]} cells, $PARALLEL_PER_POOL-way parallel) ---"
  printf "%s\n" "${obsfix_jobs[@]}" | xargs -n1 -P "$PARALLEL_PER_POOL" -I{} bash -c 'run_cell "$@"' _ {}
fi

if [[ "$phase" == "mappo" || "$phase" == "all" ]]; then
  echo
  echo "--- Launching MAPPO sweep (${#mappo_jobs[@]} cells, $PARALLEL_PER_POOL-way parallel) ---"
  printf "%s\n" "${mappo_jobs[@]}" | xargs -n1 -P "$PARALLEL_PER_POOL" -I{} bash -c 'run_cell "$@"' _ {}
fi

echo
echo "============================================================"
echo "Issue #239 sweep complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
touch experiments/p3_specialization/runs/.issue239_done
