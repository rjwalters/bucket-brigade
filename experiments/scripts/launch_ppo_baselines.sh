#!/usr/bin/env bash
# Launch the issue #384 PPO baseline training sweep on a remote host. This
# is the operator-launch wrapper for the "actually run PPO on the canonical
# scenario set" task that feeds the frozen-baseline release tracked by
# issue #371 (slice 3/5 of #365) and ultimately the M4 release infra in
# #357.
#
# The trained checkpoints written by this sweep land under
# ``experiments/p3_specialization/baselines/<scenario>/`` on the remote
# host. Issue #371 owns the downstream step of packaging them into
# ``bucket_brigade/baselines/release/ppo/`` with a loader API and
# ``scores.json``; this script only produces the raw artifacts.
#
# The dispatcher is the same ``run_tier1_cell.py --trainer ippo`` used by
# the Tier-1 sweep — the "ippo" entry in TRAINERS is Independent PPO with
# no extras (see run_tier1_cell.py:201–205), which is exactly the
# "baseline PPO checkpoint" #371 / #365 want to ship.
#
# The script does NOT compute anything locally. It only:
#   1. Reads $COMPUTE_HOST_* aliases from .env to resolve a target host
#      (unless --host is passed explicitly). Per CLAUDE.md the env is
#      CPU-bound, so COMPUTE_HOST_PRIMARY (Mac Studio) and
#      COMPUTE_HOST_CLUSTER (alc-*) are both reasonable.
#   2. Verifies the host is reachable via SSH BatchMode.
#   3. SSHs in, pulls latest main, builds the Rust extension, and starts
#      a detached tmux session running run_tier1_cell.py with
#      --trainer ippo over the requested scenarios × seeds.
#   4. Prints monitor / rsync / handoff-to-#371 instructions.
#
# CLAUDE.md is explicit: long unattended cluster runs are operator-driven,
# not safe for an autonomous agent to spawn. This script is the operator's
# launch tool. It does its job and exits — the actual cells fill over
# hours inside the remote tmux session.
#
# Usage
# -----
#   # Auto-resolve host from .env, train the full release scenario set
#   # (6 scenarios × 3 seeds, ~18 GPU-hours per #384 estimate).
#   ./experiments/scripts/launch_ppo_baselines.sh
#
#   # Subset for a fast turnaround (e.g. just minimal_specialization).
#   ./experiments/scripts/launch_ppo_baselines.sh \
#       --scenarios minimal_specialization \
#       --seeds 42,43,44 \
#       --num-iterations 25
#
#   # Explicit host, smaller seed set for a single-scenario sanity rerun.
#   ./experiments/scripts/launch_ppo_baselines.sh \
#       --host alc-2 --scenarios minimal_specialization --seeds 42
#
#   # Dry-run prints the ssh + driver command without launching anything.
#   ./experiments/scripts/launch_ppo_baselines.sh --dry-run
#
# Flags:
#   --host HOST                     SSH alias (or auto-resolved from .env)
#   --scenarios LIST                Comma-separated scenario names
#                                    (default: the #384 release scenario set)
#   --seeds LIST                    Comma-separated seeds (default: 42,43,44)
#   --num-iterations N              PPO iterations per seed (default: 50)
#   --rollout-steps N               Rollout buffer size (default: 2048)
#   --output-dir PATH               Remote output root
#                                    (default: experiments/p3_specialization/baselines)
#   --session-name NAME             tmux session name (auto-synthesized)
#   --dry-run                       Print plan; do not ssh anywhere
#   --skip-connectivity-check       For CI / smoke tests
#
# See experiments/p3_specialization/PPO_BASELINES_RUNBOOK.md for the
# canonical scenario list, host assignments, per-cell wall-clock, and the
# handoff procedure to issue #371.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults & helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Where the bucket-brigade repo lives on remote hosts. Tried in order; first
# that exists wins. Matches CLAUDE.md guidance ("~/GitHub/bucket-brigade
# first, fall back to ~/bucket-brigade").
REMOTE_REPO_CANDIDATES=("\$HOME/GitHub/bucket-brigade" "\$HOME/bucket-brigade")

# Default scenario set per issue #384 acceptance criteria. These are the
# scenarios the paper references in headline numbers, all backed by the
# scenario registry shipped in #369 (see
# bucket_brigade/envs/scenarios_generated.py::SCENARIO_REGISTRY and
# bucket_brigade/envs/registry.py::SCENARIO_VERSIONS — the bare names below
# are the registry keys consumed by get_scenario_by_name, which is what
# run_tier1_cell.py dispatches to).
#
# Sources:
#   - minimal_specialization: canonical P3 substrate (#199-family)
#   - default: vanilla 10-house environment
#   - positional_default: positional variant of default (no -v1 yet)
#   - chain_reaction: cascade dynamics scenario
#   - trivial_cooperation: simplest cooperation diagnostic
#   - v2_minimal: 2-house x 4-agent PPO learnability diagnostic (#254)
DEFAULT_SCENARIOS="minimal_specialization,default,positional_default,chain_reaction,trivial_cooperation,v2_minimal"
DEFAULT_SEEDS="42,43,44"
DEFAULT_NUM_ITERATIONS=50
DEFAULT_ROLLOUT_STEPS=2048
DEFAULT_OUTPUT_DIR="experiments/p3_specialization/baselines"
DEFAULT_SESSION_PREFIX="ppo-baselines"

HOST=""
SCENARIOS="$DEFAULT_SCENARIOS"
SEEDS="$DEFAULT_SEEDS"
NUM_ITERATIONS="$DEFAULT_NUM_ITERATIONS"
ROLLOUT_STEPS="$DEFAULT_ROLLOUT_STEPS"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
SESSION_NAME=""
DRY_RUN=0
SKIP_CONNECTIVITY_CHECK=0

print_usage() {
    sed -n '2,72p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Resolve a host alias from .env using the documented priority order.
# Echoes the first non-empty option; empty string if none configured.
resolve_host_from_env() {
    local env_file="$REPO_ROOT/.env"
    if [[ ! -f "$env_file" ]]; then
        return 0
    fi
    local primary cluster lambda gcp
    primary=$(grep -E '^COMPUTE_HOST_PRIMARY=' "$env_file" | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'")
    cluster=$(grep -E '^COMPUTE_HOST_CLUSTER=' "$env_file" | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'")
    lambda=$(grep -E '^COMPUTE_HOST_LAMBDA=' "$env_file" | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'")
    gcp=$(grep -E '^COMPUTE_HOST_GCP=' "$env_file" | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'")
    for h in "$primary" "$cluster" "$lambda" "$gcp"; do
        if [[ -n "$h" ]]; then
            echo "$h"
            return 0
        fi
    done
    return 0
}

# Confirm an SSH alias resolves and accepts a non-interactive connection.
check_ssh_reachable() {
    local host="$1"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" echo ok >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --scenarios)
            SCENARIOS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --num-iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --rollout-steps)
            ROLLOUT_STEPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --session-name)
            SESSION_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-connectivity-check)
            # Useful in CI / smoke tests where we can't actually ssh out.
            SKIP_CONNECTIVITY_CHECK=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# Scenarios must be non-empty (the operator can override the default but
# cannot launch a sweep with zero cells).
if [[ -z "$SCENARIOS" ]]; then
    echo "ERROR: --scenarios cannot be empty" >&2
    echo "       Pass --scenarios <comma-separated-list> or omit the flag" >&2
    echo "       to use the default release set." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Resolve host
# ---------------------------------------------------------------------------

if [[ -z "$HOST" ]]; then
    HOST=$(resolve_host_from_env)
fi

if [[ -z "$HOST" ]]; then
    echo "ERROR: no host specified and none resolvable from .env" >&2
    echo "       Pass --host <alias> or set COMPUTE_HOST_* in .env" >&2
    exit 3
fi

if [[ "$SKIP_CONNECTIVITY_CHECK" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
    if ! check_ssh_reachable "$HOST"; then
        echo "ERROR: cannot reach SSH host '$HOST' (BatchMode, 5s timeout)" >&2
        echo "       Try: ssh -v $HOST" >&2
        exit 4
    fi
fi

# ---------------------------------------------------------------------------
# Build driver command + tmux session name
# ---------------------------------------------------------------------------

# Convert comma-separated seeds into space-separated for the driver.
SEEDS_SPACED="${SEEDS//,/ }"

if [[ -z "$SESSION_NAME" ]]; then
    # Compact tag so two concurrent launches on the same host don't collide.
    # Use the first scenario as the tag; if multiple, append count.
    first_scenario="${SCENARIOS%%,*}"
    n_scenarios=$(echo "$SCENARIOS" | tr ',' '\n' | wc -l | tr -d ' ')
    if [[ "$n_scenarios" -gt 1 ]]; then
        SESSION_NAME="${DEFAULT_SESSION_PREFIX}-${first_scenario}-and${n_scenarios}more"
    else
        SESSION_NAME="${DEFAULT_SESSION_PREFIX}-${first_scenario}"
    fi
fi

# Compose the driver invocation as a loop over scenarios so a single tmux
# session covers the whole sweep. Each scenario is one run_tier1_cell.py
# invocation (which itself loops over seeds and writes its own
# cell_summary.json + per-seed checkpoint under <OUTPUT_DIR>/ippo_<scenario>/).
#
# We use `;` not `&&` so one failing scenario doesn't abort the rest — the
# driver writes cell_summary.json with verdict=no_data on failure and #371's
# packager can still pick up whichever scenarios succeeded.
DRIVER_CMD=""
IFS=',' read -ra SCENARIO_LIST <<< "$SCENARIOS"
for scen in "${SCENARIO_LIST[@]}"; do
    cell_cmd="echo '=== ippo on $scen ===' && uv run python experiments/p3_specialization/run_tier1_cell.py --trainer ippo --scenario '$scen' --seeds $SEEDS_SPACED --num-iterations $NUM_ITERATIONS --rollout-steps $ROLLOUT_STEPS --output-root '$OUTPUT_DIR'"
    if [[ -z "$DRIVER_CMD" ]]; then
        DRIVER_CMD="$cell_cmd"
    else
        DRIVER_CMD+=" ; $cell_cmd"
    fi
done

# Remote bootstrap: try canonical repo paths, pull, build Rust ext, run.
# Notes on the remote env (see CLAUDE.md):
#   - PATH is bare under non-interactive SSH on macOS; prepend Homebrew + cargo.
#   - PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 + unset RUSTC_WRAPPER for the build.
#   - bucket-brigade-core/build.sh uses `python -m pip install -e .`, so the
#     venv must have pip; uv ships venvs without pip by default.
read -r -d '' REMOTE_BOOTSTRAP <<REMOTE_SCRIPT || true
set -euo pipefail
export PATH="/opt/homebrew/bin:\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
unset RUSTC_WRAPPER || true

REPO=""
for candidate in ${REMOTE_REPO_CANDIDATES[*]}; do
    if [[ -d "\$candidate/.git" ]]; then
        REPO="\$candidate"
        break
    fi
done
if [[ -z "\$REPO" ]]; then
    echo "ERROR: bucket-brigade not found in any of: ${REMOTE_REPO_CANDIDATES[*]}" >&2
    exit 10
fi
cd "\$REPO"
echo "Remote repo: \$REPO"

git fetch origin
git checkout main
git pull --ff-only origin main

# Seed pip if needed (uv-created venvs ship without pip; build.sh needs it).
if [[ -d .venv ]]; then
    .venv/bin/python -m pip --version >/dev/null 2>&1 || uv pip install pip
else
    uv venv
    uv pip install pip
fi
uv sync --extra rl

# Build Rust extension (idempotent — skipped if .so is fresh).
bash bucket-brigade-core/build.sh

# Sanity-check the ippo trainer dispatch before consuming a tmux session
# for a multi-hour run. If 'ippo' is missing from TRAINERS the launcher
# has no business burning compute.
uv run python -c "from experiments.p3_specialization.run_tier1_cell import TRAINERS; assert 'ippo' in TRAINERS, 'ippo missing from dispatch — run from a main checkout'; print('ippo dispatch OK')"

mkdir -p $OUTPUT_DIR

# Kill any prior session with the same name so re-launches are idempotent.
tmux kill-session -t '$SESSION_NAME' 2>/dev/null || true

tmux new-session -d -s '$SESSION_NAME' "cd \$REPO && export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 && ( $DRIVER_CMD ) 2>&1 | tee -a '$OUTPUT_DIR/${SESSION_NAME}.log'"

echo ""
echo "Launched tmux session: $SESSION_NAME"
echo "Log: \$REPO/$OUTPUT_DIR/${SESSION_NAME}.log"
echo "Attach with: tmux attach -t $SESSION_NAME"
REMOTE_SCRIPT

# ---------------------------------------------------------------------------
# Print plan / execute
# ---------------------------------------------------------------------------

echo "===================================================================="
echo "Issue #384 — PPO baseline training launch (feeds #371)"
echo "===================================================================="
echo "Host:           $HOST"
echo "Session name:   $SESSION_NAME"
echo "Output dir:     $OUTPUT_DIR    (on remote)"
echo "Scenarios:      $SCENARIOS"
echo "Seeds:          $SEEDS"
echo "Num iterations: $NUM_ITERATIONS"
echo "Rollout steps:  $ROLLOUT_STEPS"
echo ""
echo "Per-scenario driver commands (one run_tier1_cell.py invocation each,"
echo "chained with ';' so one failure doesn't abort the sweep):"
echo "$DRIVER_CMD" | tr ';' '\n' | sed 's/^ *//; s/^/  /'
echo "===================================================================="

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "[dry-run] not connecting to $HOST. Remote bootstrap script would be:"
    echo "---"
    echo "$REMOTE_BOOTSTRAP"
    echo "---"
    exit 0
fi

echo ""
echo "Connecting to $HOST and bootstrapping..."
ssh "$HOST" bash <<EOF
$REMOTE_BOOTSTRAP
EOF

echo ""
echo "===================================================================="
echo "Launched. Useful follow-ups:"
echo ""
echo "  # Watch progress live"
echo "  ssh $HOST -t 'tmux attach -t $SESSION_NAME'"
echo ""
echo "  # Tail the log without attaching"
echo "  ssh $HOST 'tail -f bucket-brigade/${OUTPUT_DIR}/${SESSION_NAME}.log'"
echo ""
echo "  # When done, rsync results back to this checkout:"
echo "  rsync -avz $HOST:bucket-brigade/${OUTPUT_DIR}/ ${OUTPUT_DIR}/"
echo ""
echo "  # Handoff to #371: each scenario produces a cell_summary.json +"
echo "  # per-seed checkpoint under ${OUTPUT_DIR}/ippo_<scenario>/."
echo "  # Issue #371 owns the packaging step into"
echo "  # bucket_brigade/baselines/release/ppo/<scenario>/checkpoint.pt"
echo "  # + scores.json — see experiments/p3_specialization/PPO_BASELINES_RUNBOOK.md."
echo "===================================================================="
