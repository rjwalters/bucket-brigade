#!/usr/bin/env bash
# Issue #281: TransformerPolicyNetwork BC-fit, single cell (3 seeds).
#
# This script is the Phase-1.5 escalation harness for issue #281: swap the
# default MLP for TransformerPolicyNetwork (~350K params vs ~14K) on the
# specialist BC-fit task and compare against PR #278's MLP baseline at
# gap_closed = 0.934.
#
# NB: PR #278 already showed the default MLP reaches gap_closed = 0.934 at
# 40k pairs / 30 epochs on `minimal_specialization`, which contradicts the
# original `INDUCTIVE_BIAS_GAP` premise. Curator recommended close-without-
# execution if #279 confirms. Run this only if the human decides the
# transformer-vs-MLP comparison is still wanted.
#
# Usage:
#   ./run_issue281_transformer_bc.sh [seed1 seed2 ...]
#
# Defaults: seeds = 42 43 44. Results land at
#   experiments/p3_specialization/diagnostics/results/issue281_transformer_bc/seed_<S>/
#
# Cost gate: ~1 hour on COMPUTE_HOST_PRIMARY (Mac Studio CPU); abort and
# re-scope if it blows past 2 hours.
#
# DO NOT RUN LOCALLY. Use the remote workflow described in CLAUDE.md
# (see "Where to Run Different Tasks").
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- 42 43 44
fi

OUT_ROOT="experiments/p3_specialization/diagnostics/results/issue281_transformer_bc"
mkdir -p "$OUT_ROOT"

run_one() {
  local s="$1"
  local out="$OUT_ROOT/seed_$s"
  mkdir -p "$out"
  # Match #278's BC budget for an apples-to-apples comparison:
  #   40k pairs/agent × 30 epochs, lr 1e-3, batch 64, 4 agents.
  # The `bc_fit_only.py` num-steps argument multiplies by num-agents to
  # produce the pair count, so num-steps = 40000 / num-agents = 10000.
  python experiments/p3_specialization/bc_fit_only.py \
    --scenario minimal_specialization \
    --num-agents 4 \
    --num-steps 10000 \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-3 \
    --epsilon 0.1 \
    --seed "$s" \
    --architecture transformer \
    --transformer-d-model 256 \
    --transformer-nhead 4 \
    --transformer-num-layers 3 \
    --transformer-dim-feedforward 512 \
    --transformer-dropout 0.1 \
    --out "$out/bc_fit_only_result.json" 2>&1 | tee "$out/train.log"
}
export -f run_one
export OUT_ROOT

# Run one seed at a time — transformer is memory-heavier than MLP; share
# the CPU host conservatively. Override with `xargs -P 2` if you have
# headroom and want to parallelize.
printf "%s\n" "$@" | xargs -P 1 -I{} bash -c 'run_one "$@"' _ {}

echo
echo "All seeds finished. Aggregate with:"
echo "  python experiments/p3_specialization/analyze_281.py --results-dir $OUT_ROOT"
