#!/usr/bin/env bash
# Orchestration script for issue #292 — minimal-dilemma basin-trap demo.
#
# Heavy compute — run on COMPUTE_HOST_PRIMARY (per CLAUDE.md). The local
# smoke version is the test_smoke.py pytest case; this script is the
# full-protocol producer.
#
# Phases:
#   1. IPPO baseline ×5 seeds (random init).         ~15 min total.
#   2. Best-of-N random nets (K=20 → K=200, N=1000). ~10 min total.
#   3. BC-init (always_cooperate) + PPO continuation ×3 seeds. ~10 min total.
#   4. Verdict classifier (4-gate read).
#
# Total wall ~45 min on a single fast machine. Parallelize the IPPO seeds
# externally (GNU parallel or N tmux panes) if you have cores to spare.
#
# Usage (from repo root):
#   ./experiments/p3_specialization/minimal_dilemma/run_all.sh OUTDIR

set -euo pipefail

OUTDIR="${1:-experiments/p3_specialization/minimal_dilemma/results}"
mkdir -p "$OUTDIR"

echo "=== Phase 1: IPPO baseline (5 seeds, 100 iters each) ==="
for seed in 0 1 2 3 4; do
  uv run python -m experiments.p3_specialization.minimal_dilemma.train_ippo \
    --seed "$seed" --num-iterations 100 --rollout-steps 2048 \
    --output-dir "$OUTDIR/ippo_seed${seed}"
done

echo "=== Phase 2: Best-of-N (1000 seeds, K=20 → K=200) ==="
uv run python -m experiments.p3_specialization.minimal_dilemma.best_of_n \
  --seeds 1000 --episodes-per-seed 20 --restability-episodes 200 \
  --protocol independent \
  --output-dir "$OUTDIR/best_of_n"

echo "=== Phase 3a: BC fit (always_cooperate) ==="
uv run python -m experiments.p3_specialization.minimal_dilemma.bc_init \
  --specialist always_cooperate \
  --num-pairs-per-agent 5000 \
  --output-dir "$OUTDIR/bc_alwaysC"

echo "=== Phase 3b: PPO continuation from BC init (3 seeds, 100 iters) ==="
for seed in 100 101 102; do
  uv run python -m experiments.p3_specialization.minimal_dilemma.train_ippo \
    --seed "$seed" --num-iterations 100 --rollout-steps 2048 \
    --bc-init-checkpoint-dir "$OUTDIR/bc_alwaysC" \
    --output-dir "$OUTDIR/bc_continuation_seed${seed}"
done

echo "=== Phase 4: Verdict ==="
uv run python -m experiments.p3_specialization.minimal_dilemma.verdict \
  --ippo-runs "$OUTDIR/ippo_seed0/metrics.json" \
              "$OUTDIR/ippo_seed1/metrics.json" \
              "$OUTDIR/ippo_seed2/metrics.json" \
              "$OUTDIR/ippo_seed3/metrics.json" \
              "$OUTDIR/ippo_seed4/metrics.json" \
  --bc-continuation-runs "$OUTDIR/bc_continuation_seed100/metrics.json" \
                         "$OUTDIR/bc_continuation_seed101/metrics.json" \
                         "$OUTDIR/bc_continuation_seed102/metrics.json" \
  --bc-summary "$OUTDIR/bc_alwaysC/bc_summary.json" \
  --bestofn-summary "$OUTDIR/best_of_n/summary.json" \
  --output "$OUTDIR/verdict.json"

echo "All done. Verdict at $OUTDIR/verdict.json"
