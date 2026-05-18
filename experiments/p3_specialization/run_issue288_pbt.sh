#!/usr/bin/env bash
# Issue #288: Population-Based Training (PBT) basin-escape probe.
#
# Wraps experiments/p3_specialization/run_issue288_pbt.py with the curator's
# default knobs (pop=16, gen=6, iters=50, 3 PBT seeds). MUST run on
# COMPUTE_HOST_PRIMARY per CLAUDE.md guidelines — ~18-27 wall-clock hours.
#
# Usage (remote, in tmux):
#   ./run_issue288_pbt.sh [seed1 seed2 ...]
#
# Outputs land at:
#   experiments/p3_specialization/runs/issue288_pbt/seed_<S>/gen_<G>/lineage_<L>/
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- 42 43 44
fi

OUT_ROOT="experiments/p3_specialization/runs/issue288_pbt"
mkdir -p "$OUT_ROOT"

python experiments/p3_specialization/run_issue288_pbt.py \
  --output-dir "$OUT_ROOT" \
  --seeds "$@" \
  --population-size 16 \
  --generations 6 \
  --iters-per-gen 50 \
  --rollout-steps 2048 \
  --num-agents 4 \
  --scenario minimal_specialization \
  --lambda-red 0.0 \
  --initial-lr 3e-4 \
  --initial-entropy-coef 0.01 \
  --weight-noise 0.01 \
  --truncation-frac 0.25 2>&1 | tee "$OUT_ROOT/pbt_$(date +%Y%m%d_%H%M%S).log"

echo "PBT run complete. Run analyze_288.py to produce the verdict."
