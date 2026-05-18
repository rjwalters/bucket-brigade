#!/usr/bin/env bash
# Issue #282: high-λ GAE PPO smoke — is bootstrap bias the misalignment source?
#
# 4 λ cells × 3 seeds = 12 runs on minimal_specialization (IPPO, random init).
# λ ∈ {0.95 (current default — re-baseline), 0.99, 0.999, 1.0}.
# 50 iterations × 2048 rollout steps, matches the #270 baseline configuration.
#
# REMOTE-ONLY. This script must NOT be executed on localhost. Per
# ``CLAUDE.md`` compute guidelines (no PPO sweeps locally), run on
# ``COMPUTE_HOST_PRIMARY`` in tmux. Expected wall-clock ~30-60 min for
# the full grid with 2 seeds in parallel.
#
# Usage:
#   ./run_issue282_lambda_sweep.sh [seed1 seed2 ...]
#
# Defaults seeds: 42 43 44. Cells land at
#   experiments/p3_specialization/runs/issue282_lambda_sweep/minimal_specialization/lambda_<L>/seed_<S>/
#
# Verdict: see ``analyze_282.py`` for tier classification (≥0.5 / 0.25-0.5 / <0.25).
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- 42 43 44
fi

LAMBDAS=(0.95 0.99 0.999 1.0)
OUT_ROOT="experiments/p3_specialization/runs/issue282_lambda_sweep/minimal_specialization"
mkdir -p "$OUT_ROOT"
export OUT_ROOT

run_one() {
  local lam="$1"
  local s="$2"
  # Sanitize lambda value for path: replace '.' with '_' to keep it shell-safe.
  local lam_tag="${lam//./_}"
  local out="$OUT_ROOT/lambda_${lam_tag}/seed_$s"
  mkdir -p "$out"
  python experiments/p3_specialization/train.py \
    --scenario minimal_specialization \
    --lambda-red 0.0 \
    --seed "$s" \
    --gae-lambda "$lam" \
    --num-iterations 50 \
    --rollout-steps 2048 \
    --num-agents 4 \
    --output-dir "$out" 2>&1 | tee "$out/train.log"
}
export -f run_one

# Build (lambda, seed) job list and run two cells in parallel to share CPU
# with parallel builders on this host. Matches the #270 driver convention.
JOBS=()
for lam in "${LAMBDAS[@]}"; do
  for s in "$@"; do
    JOBS+=("$lam $s")
  done
done

printf "%s\n" "${JOBS[@]}" | xargs -P 2 -n 2 bash -c 'run_one "$0" "$1"'

echo "All λ cells finished."
