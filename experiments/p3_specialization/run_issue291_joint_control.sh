#!/usr/bin/env bash
# Issue #291: single-controller (joint-action) PPO on minimal_specialization.
#
# Budget matches the #270 IPPO continuation sweep: 50 iter * 2048 steps * 3
# seeds on minimal_specialization. The single change is the env wrapper —
# one controller emits the joint action for all 4 sub-agents.
#
# CPU-bound workload per CLAUDE.md compute guidelines — run on
# COMPUTE_HOST_PRIMARY (Mac Studio class) in tmux. Do NOT run on localhost.
#
# Usage:
#   ./run_issue291_joint_control.sh [seed1 seed2 ...]
#
# Defaults to seeds 42 43 44. Output cells land at:
#   experiments/p3_specialization/runs/issue291_joint_control/minimal_specialization/seed_<S>/
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- 42 43 44
fi

OUT_ROOT="experiments/p3_specialization/runs/issue291_joint_control/minimal_specialization"
mkdir -p "$OUT_ROOT"
export OUT_ROOT

run_one() {
  local s="$1"
  local out="$OUT_ROOT/seed_$s"
  mkdir -p "$out"
  python experiments/p3_specialization/train_single_agent.py \
    --scenario minimal_specialization \
    --seed "$s" \
    --num-iterations 50 \
    --rollout-steps 2048 \
    --num-subagents 4 \
    --joint-control \
    --output-dir "$out" 2>&1 | tee "$out/train.log"
}
export -f run_one

# Two seeds at a time, matching the #270 driver's parallelism budget.
printf "%s\n" "$@" | xargs -P 2 -I{} bash -c 'run_one "$@"' _ {}

echo "All seeds finished."
