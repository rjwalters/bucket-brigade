#!/usr/bin/env bash
# Issue #273 — off-PPO baseline (REINFORCE) sweep driver.
#
# Compares vanilla REINFORCE against the IPPO baseline (PR #257) on
# ``minimal_specialization``: 3 seeds × {normalize_returns on, off} = 6
# cells per algorithm. Total env steps matched to the IPPO baseline
# (``num_iterations=50`` × ``rollout_steps=2048`` × ``num_agents=4``).
#
# **Compute guideline**: do NOT run this locally. Per ``CLAUDE.md``
# Compute Resource Guidelines, dispatch to ``$COMPUTE_HOST_PRIMARY``
# (Mac Studio). REINFORCE is cheaper than PPO per step but the full
# 6-cell sweep is still ~hours of CPU. Use ``tmux`` for persistence.
#
# Usage on remote host:
#     source .env
#     ssh "$COMPUTE_HOST_PRIMARY"
#     cd ~/GitHub/bucket-brigade  # or ~/bucket-brigade
#     tmux new -s issue273
#     bash experiments/p3_specialization/run_issue273_reinforce_sweep.sh
#     # Ctrl+B, D to detach
#
# Verdict logic (from issue #273):
#     Same plateau as PPO (~0.182 gap_closed) → failure is RL-general,
#         not PPO-specific. Double down on env-side fixes.
#     REINFORCE > PPO                         → PPO's clip/GAE is hurting.
#                                              Revisit clip schedule.
#     REINFORCE < PPO                         → PPO is mitigating real
#                                              variance. Search orthogonal
#                                              axes.

set -euo pipefail

SCENARIO="${SCENARIO:-minimal_specialization}"
NUM_ITERATIONS="${NUM_ITERATIONS:-50}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-2048}"
NUM_AGENTS="${NUM_AGENTS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/p3_specialization/runs_reinforce}"
DEVICE="${DEVICE:-cpu}"
# Matched seeds with IPPO baseline (PR #257).
SEEDS=(42 43 44)
# Both settings — un-normalized is the canonical-REINFORCE baseline; the
# normalized variant is the standard variance-reduction trick.
NORMALIZE_OPTS=(false true)

mkdir -p "${OUTPUT_ROOT}"

echo "Issue #273 REINFORCE sweep"
echo "  scenario:        ${SCENARIO}"
echo "  num_iterations:  ${NUM_ITERATIONS}"
echo "  rollout_steps:   ${ROLLOUT_STEPS}"
echo "  num_agents:      ${NUM_AGENTS}"
echo "  output_root:     ${OUTPUT_ROOT}"
echo "  device:          ${DEVICE}"
echo "  seeds:           ${SEEDS[*]}"
echo "  normalize:       ${NORMALIZE_OPTS[*]}"
echo

CELL=0
TOTAL=$(( ${#SEEDS[@]} * ${#NORMALIZE_OPTS[@]} ))

for norm in "${NORMALIZE_OPTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CELL=$(( CELL + 1 ))
        cell_dir="${OUTPUT_ROOT}/${SCENARIO}/norm_${norm}/seed_${seed}"
        if [[ -f "${cell_dir}/metrics.json" ]]; then
            echo "[${CELL}/${TOTAL}] skip (exists): ${cell_dir}"
            continue
        fi
        echo "[${CELL}/${TOTAL}] ${cell_dir}"
        norm_flag=""
        if [[ "${norm}" == "true" ]]; then
            norm_flag="--normalize-returns"
        fi
        uv run python -m experiments.p3_specialization.train \
            --algorithm reinforce \
            --scenario "${SCENARIO}" \
            --seed "${seed}" \
            --num-iterations "${NUM_ITERATIONS}" \
            --rollout-steps "${ROLLOUT_STEPS}" \
            --num-agents "${NUM_AGENTS}" \
            --output-dir "${cell_dir}" \
            --device "${DEVICE}" \
            ${norm_flag}
    done
done

echo
echo "All ${TOTAL} cells complete. Output: ${OUTPUT_ROOT}"
echo "Next: analyze with experiments/p3_specialization/analyze_issue273.py"
