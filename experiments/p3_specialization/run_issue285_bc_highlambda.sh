#!/usr/bin/env bash
# Issue #285: BC-init + PPO with high-λ — combined warm-start + Monte Carlo credit.
#
# 4 λ cells × 3 seeds = 12 runs on minimal_specialization. Each cell warm-starts
# the actor from the #270/#278 specialist BC checkpoint (40k pairs/agent × 30
# epochs, gap_closed = 0.934) and then continues with PPO for 50 iters at the
# given λ. The λ=0.95 cell is the BC-init baseline (matched-seed control vs.
# the #270 PPO continuation arm); λ ∈ {0.99, 0.999, 1.0} are the high-λ probes
# of the curator's 2×2 verdict matrix (#270 verdict × this issue's high-λ
# verdict — see issue body).
#
# REMOTE-ONLY. This script must NOT be executed on localhost. Per
# ``CLAUDE.md`` compute guidelines (no PPO sweeps locally), run on
# ``COMPUTE_HOST_PRIMARY`` in tmux. Expected wall-clock ~3-4 hours for
# the full grid with 2 cells in parallel.
#
# Usage:
#   ./run_issue285_bc_highlambda.sh <bc_checkpoint_dir> [seed1 seed2 ...]
#
# Defaults seeds: 42 43 44. Cells land at
#   experiments/p3_specialization/runs/issue285_bc_highlambda/minimal_specialization/lambda_<L>/seed_<S>/
#
# Verdict: see ``analyze_285.py`` for per-λ basin-trap / anti-attractor /
# partial classification (reuses analyze_270.classify_verdict).
set -euo pipefail

BC_DIR=${1:?"missing BC checkpoint dir (e.g., experiments/p3_specialization/runs/issue270_bc_init/specialist_bc_v2)"}
shift || true
if [ "$#" -eq 0 ]; then
  set -- 42 43 44
fi

LAMBDAS=(0.95 0.99 0.999 1.0)
OUT_ROOT="experiments/p3_specialization/runs/issue285_bc_highlambda/minimal_specialization"
mkdir -p "$OUT_ROOT"
export BC_DIR OUT_ROOT

run_one() {
  local lam="$1"
  local s="$2"
  # Sanitize lambda value for path: replace '.' with '_' to keep it shell-safe.
  # Matches analyze_285.py's _lambda_tag().
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
    --bc-init-checkpoint-dir "$BC_DIR" \
    --output-dir "$out" 2>&1 | tee "$out/train.log"
}
export -f run_one

# Build (lambda, seed) job list and run two cells in parallel to share CPU
# with parallel builders on this host. Matches the #270 / #282 driver
# convention.
JOBS=()
for lam in "${LAMBDAS[@]}"; do
  for s in "$@"; do
    JOBS+=("$lam $s")
  done
done

printf "%s\n" "${JOBS[@]}" | xargs -P 2 -n 2 bash -c 'run_one "$0" "$1"'

echo "All λ × seed cells finished."
