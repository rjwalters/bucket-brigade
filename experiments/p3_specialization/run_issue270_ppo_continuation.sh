#!/usr/bin/env bash
# Issue #270: 50-iter IPPO continuation from BC-pretrained init, 3 seeds.
#
# Usage:
#   ./run_issue270_ppo_continuation.sh <bc_checkpoint_dir> [seed1 seed2 ...]
#
# Defaults seeds: 42 43 44. Cells land at
#   experiments/p3_specialization/runs/issue270_bc_continuation/minimal_specialization/lambda_0e0/seed_<S>/
set -euo pipefail

BC_DIR=${1:?"missing BC checkpoint dir (e.g., experiments/p3_specialization/runs/issue270_bc_init/specialist_bc_v2)"}
shift || true
if [ "$#" -eq 0 ]; then
  set -- 42 43 44
fi

OUT_ROOT="experiments/p3_specialization/runs/issue270_bc_continuation/minimal_specialization/lambda_0e0"
mkdir -p "$OUT_ROOT"
export BC_DIR OUT_ROOT

run_one() {
  local s="$1"
  local out="$OUT_ROOT/seed_$s"
  mkdir -p "$out"
  python experiments/p3_specialization/train.py \
    --scenario minimal_specialization \
    --lambda-red 0.0 \
    --seed "$s" \
    --num-iterations 50 \
    --rollout-steps 2048 \
    --num-agents 4 \
    --bc-init-checkpoint-dir "$BC_DIR" \
    --output-dir "$out" 2>&1 | tee "$out/train.log"
}
export -f run_one

# Run two seeds at a time to share CPU with parallel builders on this host.
printf "%s\n" "$@" | xargs -P 2 -I{} bash -c 'run_one "$@"' _ {}

echo "All seeds finished."
