#!/usr/bin/env bash
# Env-health diagnostic aggregator (issue #201).
#
# Runs H1 (rollout-reward inspector, hermetic mode), H2 (reward attribution),
# and H3 (random baseline) sequentially against a single scenario, captures
# their logs, and prints a final summary table.
#
# Usage:
#   bash experiments/p3_specialization/diagnostics/check_env_health.sh [scenario]
#
# Defaults to scenario=default. Exits non-zero if any sub-script crashes;
# does NOT enforce numeric thresholds (that is the job of the pytest
# regression suite, tests/test_env_health_diagnostics.py).

set -uo pipefail

SCENARIO="${1:-default}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Resolve repo root from this script's location so the script works whether
# invoked from the repo root, the diagnostics dir, or any other cwd.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../../.." && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

LOG_PREFIX="${RESULTS_DIR}/check_env_health_${SCENARIO}_${TIMESTAMP}"
H1_LOG="${LOG_PREFIX}_h1.log"
H2_LOG="${LOG_PREFIX}_h2.log"
H3_LOG="${LOG_PREFIX}_h3.log"

H1_SCRIPT="${SCRIPT_DIR}/inspect_rollout_rewards.py"
H2_SCRIPT="${SCRIPT_DIR}/audit_reward_attribution.py"
H3_SCRIPT="${SCRIPT_DIR}/random_baseline.py"

# Track per-stage exit codes so we can produce a useful summary even when one
# stage crashes. The script's final exit code is the bitwise-OR of these so
# a single sub-crash is detectable in CI/wrapper scripts.
H1_RC=0
H2_RC=0
H3_RC=0

echo "============================================================"
echo "env-health diagnostics — scenario=${SCENARIO}"
echo "  H1 log: ${H1_LOG}"
echo "  H2 log: ${H2_LOG}"
echo "  H3 log: ${H3_LOG}"
echo "============================================================"

cd "${REPO_ROOT}"

# ---- H1: rollout-reward inspector (hermetic random-init mode) -------------
echo
echo "--- H1: inspect_rollout_rewards.py --hermetic --scenario=${SCENARIO} ---"
uv run python "${H1_SCRIPT}" --hermetic --scenario "${SCENARIO}" \
  2>&1 | tee "${H1_LOG}"
H1_RC="${PIPESTATUS[0]}"

# ---- H2: reward attribution audit -----------------------------------------
echo
echo "--- H2: audit_reward_attribution.py --scenario=${SCENARIO} ---"
uv run python "${H2_SCRIPT}" --scenario "${SCENARIO}" \
  2>&1 | tee "${H2_LOG}"
H2_RC="${PIPESTATUS[0]}"

# ---- H3: random baseline (always scenario=default per script design) ------
echo
echo "--- H3: random_baseline.py (scenario hard-coded to default) ---"
# H3 only supports the "default" scenario because it re-derives a specific
# cited number (308) from issue #145. Skip with a clear note if the user
# asked for a different scenario.
if [[ "${SCENARIO}" == "default" ]]; then
  uv run python "${H3_SCRIPT}" --episodes-per-seed 50 --seeds 2 --no-mlp \
    2>&1 | tee "${H3_LOG}"
  H3_RC="${PIPESTATUS[0]}"
else
  echo "H3 skipped: random_baseline.py hard-codes scenario=default" \
    | tee "${H3_LOG}"
fi

# ---- Summary --------------------------------------------------------------
echo
echo "============================================================"
echo "SUMMARY — scenario=${SCENARIO}"
echo "============================================================"

extract() {
  local file="$1"
  local pattern="$2"
  if [[ -f "${file}" ]]; then
    grep -E "${pattern}" "${file}" | head -5 || true
  fi
}

echo
echo "H1 (per-agent reward CV + action-reward R²):"
extract "${H1_LOG}" '^agent_[0-9]+:|CV=|R²\(packed|--> H1'

echo
echo "H2 (team/ownership ratio + pairwise reward correlation):"
extract "${H2_LOG}" 'team\|ownership\|mean_abs|median=|Mean off-diagonal|Min off-diagonal|Team-share|H2 VERDICT'

echo
echo "H3 (uniform-random baseline reward/step):"
extract "${H3_LOG}" 'per-step|per-episode|Verdict|Uniform-random per-step'

echo
echo "Exit codes:  H1=${H1_RC}  H2=${H2_RC}  H3=${H3_RC}"
echo "============================================================"

# Aggregate: non-zero if any stage crashed.
exit $(( H1_RC | H2_RC | H3_RC ))
