#!/usr/bin/env bash
# Issue #289: Hindsight Credit Assignment (HCA) verdict sweep on
# minimal_specialization. 50 iters x 3 seeds, plus a baseline GAE arm at
# the same config so the gap_closed comparison against the PPO baseline
# (PR #257: 0.182) is on a level playing field.
#
# DO NOT run this on localhost --- HCA adds a per-agent hindsight network
# (one extra MLP forward per step) and 50 iters x 2048 rollout-steps is
# squarely in "use a compute host" territory. See CLAUDE.md /
# experiments/REMOTE_EXECUTION.md for the remote workflow.
#
# Usage:
#   # On a compute host, inside tmux:
#   ./experiments/p3_specialization/run_issue289_sweep.sh [seed1 seed2 ...]
#
# Defaults seeds: 42 43 44. Cells land at:
#   experiments/p3_specialization/runs/issue289_hca/minimal_specialization/<arm>/seed_<S>/
# where <arm> is one of "hca_default" or "gae_baseline".
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- 42 43 44
fi

OUT_ROOT="experiments/p3_specialization/runs/issue289_hca/minimal_specialization"
mkdir -p "$OUT_ROOT"
export OUT_ROOT

run_hca() {
  local s="$1"
  local out="$OUT_ROOT/hca_default/seed_$s"
  mkdir -p "$out"
  python experiments/p3_specialization/train.py \
    --scenario minimal_specialization \
    --lambda-red 0.0 \
    --seed "$s" \
    --num-iterations 50 \
    --rollout-steps 2048 \
    --num-agents 4 \
    --use-hca \
    --hindsight-num-return-buckets 8 \
    --hindsight-ratio-clip 10.0 \
    --output-dir "$out" 2>&1 | tee "$out/train.log"
}

run_gae() {
  local s="$1"
  local out="$OUT_ROOT/gae_baseline/seed_$s"
  mkdir -p "$out"
  python experiments/p3_specialization/train.py \
    --scenario minimal_specialization \
    --lambda-red 0.0 \
    --seed "$s" \
    --num-iterations 50 \
    --rollout-steps 2048 \
    --num-agents 4 \
    --advantage-estimator gae \
    --output-dir "$out" 2>&1 | tee "$out/train.log"
}

export -f run_hca run_gae

# Run two seeds at a time to share CPU with parallel builders on this host.
echo "== HCA arm =="
printf "%s\n" "$@" | xargs -P 2 -I{} bash -c 'run_hca "$@"' _ {}

echo "== GAE baseline arm =="
printf "%s\n" "$@" | xargs -P 2 -I{} bash -c 'run_gae "$@"' _ {}

echo "All cells finished. Run experiments/p3_specialization/analyze_289.py next."
