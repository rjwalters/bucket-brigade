#!/usr/bin/env bash
# Issue #239 finalize — runs after the sweep completes.
#
# 1. For each training cell, run the pairwise_action_kl diagnostic so the
#    analyzers can read kl_off_diag_mean (this also exercises the new
#    40-class joint code path on every cell).
# 2. Run analyze_220.py + analyze_231.py to render the post-#236 summaries.
# 3. Print the verdicts.
#
# Notes:
# - Pre-#236 baseline cells (if any) on disk are stale and were re-trained
#   into the same paths by run_issue239_sweep.sh; the analyzers are
#   path-driven and see only the new cells.
# - If sibling PRs #243/#244 (issue #237/#238) have landed on main, run
#   `git fetch origin && git rebase origin/main` BEFORE this script so the
#   analyzers pick up updated random/specialist constants.
# - The analyzers write summary.md / summary.json which clobber any prior
#   summaries. The pre-#236 summaries are preserved as summary.pre236.*
#   (already done at issue-239 sweep start time).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/GitHub/bucket-brigade}"
cd "$REPO_ROOT"

RUNS_ROOT="experiments/p3_specialization/runs"

echo "=== Computing pairwise action-KL for every issue 239 cell ==="
# Iterate over both layouts (obs-fix + MAPPO).
declare -a cell_globs=(
  "$RUNS_ROOT/issue220_treatment/*/lambda_0e0/seed_*"
  "$RUNS_ROOT/issue231/ippo/*/seed_*"
  "$RUNS_ROOT/issue231/mappo/*/seed_*"
)

n_kl=0; n_fail=0
for glob in "${cell_globs[@]}"; do
  for cell in $glob; do
    if [[ ! -f "$cell/metrics.json" ]]; then
      echo "  SKIP $cell (no metrics.json — incomplete cell)"
      continue
    fi
    # Stamp git rev if missing (idempotent).
    if [[ ! -f "$cell/pairwise_action_kl.json" ]]; then
      if uv run python experiments/p3_specialization/diagnostics/pairwise_action_kl.py \
           --cell "$cell" --rollout-steps 512 > /dev/null 2>&1; then
        n_kl=$((n_kl + 1))
        echo "  KL  $cell"
      else
        n_fail=$((n_fail + 1))
        echo "  FAIL $cell" >&2
      fi
    fi
  done
done
echo "Computed KL for $n_kl cells ($n_fail failures)"

echo
echo "=== Running analyze_220.py (obs-fix re-validation) ==="
uv run python experiments/p3_specialization/analyze_220.py

echo
echo "=== Running analyze_231.py (MAPPO re-validation) ==="
uv run python experiments/p3_specialization/analyze_231.py

echo
echo "=== Post-#236 summaries ==="
echo "--- analyze_220 summary ---"
cat experiments/p3_specialization/diagnostics/results/issue220_obsfix/summary.md
echo
echo "--- analyze_231 summary ---"
cat experiments/p3_specialization/diagnostics/results/issue231_mappo/summary.md
