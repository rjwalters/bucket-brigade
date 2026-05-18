#!/usr/bin/env bash
# Issue #280 — hidden_size capacity probe for BC-fit (Phase 1.5 escalation).
#
# Path A (default — recommended after PR #278's cross-finding):
#   hidden_size in {64, 128} at short budget (10k pairs x 10 epochs).
#
# Path B (fallback — only run if Path A is inconclusive):
#   Set HIDDEN_SIZES="64 128 256 512" via env var.
#
# Curator note (in issue body): PR #278 already hit gap_closed=0.934 at the
# default hidden_size=64 with 40k pairs x 30 epochs. The capacity hypothesis
# may already be settled. Path A cheaply confirms whether widening helps at
# the *short-budget* regime where #272/PR #276 first saw failure.
#
# Layout (consumed by analyze_280.py):
#   experiments/p3_specialization/runs/issue280_hidden_size/
#       hs_{HS}/seed_{SEED}/result.json
#
# Wall-clock estimate (CPU): ~15-30 min per cell. Path A (2 cells x 1 seed)
# ~30-60 min total. Path B (4 cells x 1 seed) ~2-3 h total.
#
# Run on COMPUTE_HOST_PRIMARY per CLAUDE.md. Do NOT run locally.
#
# Modeled on experiments/p3_specialization/run_issue262_sweep.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/GitHub/bucket-brigade}"
if [[ ! -d "$REPO_ROOT" ]]; then
  if [[ -d "$HOME/bucket-brigade" ]]; then
    REPO_ROOT="$HOME/bucket-brigade"
  fi
fi
cd "$REPO_ROOT"

GIT_REV="$(git rev-parse HEAD)"
echo "Pinned to commit: $GIT_REV"

OUTPUT_ROOT="experiments/p3_specialization/runs"
SCENARIO="${SCENARIO:-minimal_specialization}"
NUM_AGENTS="${NUM_AGENTS:-4}"
# Short budget — matches #272/PR #276 regime that first reported the failure.
NUM_STEPS="${NUM_STEPS:-2500}"     # 2500 env steps x 4 agents = 10k pairs
EPOCHS="${EPOCHS:-10}"
SEEDS=(${SEEDS:-0})                # single seed by default; pass SEEDS="0 1 2" to add
# Path A: 2-cell probe. Override to "64 128 256 512" for Path B.
HIDDEN_SIZES=(${HIDDEN_SIZES:-64 128})
PARALLEL_PER_POOL="${PARALLEL_PER_POOL:-2}"

jobs=()
for hs in "${HIDDEN_SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    outdir="$OUTPUT_ROOT/issue280_hidden_size/hs_${hs}/seed_${seed}"
    jobs+=("${hs}|${seed}|${outdir}")
  done
done

echo
echo "============================================================"
echo "Issue #280 hidden_size BC-fit sweep"
echo "  scenario:        $SCENARIO"
echo "  num_agents:      $NUM_AGENTS"
echo "  num_steps:       $NUM_STEPS  (pairs = num_steps * num_agents)"
echo "  epochs:          $EPOCHS"
echo "  hidden_sizes:    ${HIDDEN_SIZES[*]}"
echo "  seeds:           ${SEEDS[*]}"
echo "  cells:           ${#jobs[@]}"
echo "  parallelism:     $PARALLEL_PER_POOL"
echo "============================================================"

run_cell() {
  local spec="$1"
  IFS='|' read -r hs seed outdir <<< "$spec"
  mkdir -p "$outdir"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START $outdir (hidden_size=$hs seed=$seed)"
  if uv run python experiments/p3_specialization/bc_fit_only.py \
       --scenario "$SCENARIO" --num-agents "$NUM_AGENTS" \
       --num-steps "$NUM_STEPS" --epochs "$EPOCHS" \
       --hidden-size "$hs" --seed "$seed" \
       --out "$outdir/result.json" \
       > "$outdir/train.log" 2>&1; then
    # Stamp git rev into the result JSON for provenance.
    python3 -c "
import json
p = '$outdir/result.json'
with open(p) as f: c = json.load(f)
c['_git_rev'] = '$GIT_REV'
c['_issue'] = 280
c['_hidden_size'] = int('$hs')
with open(p, 'w') as f: json.dump(c, f, indent=2)
"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DONE  $outdir"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAIL  $outdir (see train.log)" >&2
    return 1
  fi
}

export -f run_cell
export SCENARIO NUM_AGENTS NUM_STEPS EPOCHS GIT_REV

echo
echo "--- Launching ${#jobs[@]} cells, ${PARALLEL_PER_POOL}-way parallel ---"
printf "%s\n" "${jobs[@]}" | xargs -n1 -P "$PARALLEL_PER_POOL" -I{} bash -c 'run_cell "$@"' _ {}

echo
echo "============================================================"
echo "Issue #280 sweep complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Next: uv run python experiments/p3_specialization/analyze_280.py"
echo "============================================================"
touch "$OUTPUT_ROOT/.issue280_hidden_size_done"
