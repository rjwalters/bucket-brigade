#!/usr/bin/env bash
# Launch the Tier-1 sweep matrix (issue #343 / driver #346) on a remote
# host. Dispatches experiments/p3_specialization/run_tier1_cell.py once per
# trainer, then runs aggregate_tier1.py to produce the verdict table.
#
# The script does NOT compute anything locally. It only:
#   1. Reads $COMPUTE_HOST_* aliases from .env to resolve a target host
#      (unless --host is passed explicitly).
#   2. Verifies the host is reachable via SSH BatchMode.
#   3. SSHs in, pulls latest main, builds the Rust extension, and starts
#      a detached tmux session running the Tier-1 cells sequentially.
#   4. Prints monitor / rsync / aggregate instructions.
#
# CLAUDE.md is explicit: long unattended cluster runs are operator-driven,
# not safe for an autonomous agent to spawn. This script is the operator's
# launch tool. It does its job and exits — the actual cells fill over
# hours inside the remote tmux session.
#
# Usage
# -----
#   # Auto-resolve host from .env, run the full Tier-1 launch set (12 cells)
#   ./experiments/scripts/launch_tier1_sweep.sh
#
#   # Explicit host, only the cheap PPO-family arms (3 cells)
#   ./experiments/scripts/launch_tier1_sweep.sh \
#       --host alc-2 \
#       --trainers mappo,high_lambda,reinforce
#
#   # Add COMA back in (deprioritized but useful for the full grid)
#   ./experiments/scripts/launch_tier1_sweep.sh \
#       --trainers mappo,high_lambda,coma,lola
#
#   # Cross-scenario robustness check after a candidate emerges
#   ./experiments/scripts/launch_tier1_sweep.sh \
#       --trainers bc_init_continuation \
#       --scenarios minimal_specialization,default,chain_reaction
#
#   # Dry-run prints the ssh + driver commands without launching anything
#   ./experiments/scripts/launch_tier1_sweep.sh --trainers mappo --dry-run
#
# See experiments/p3_specialization/TIER1_LAUNCH_RUNBOOK.md for the
# canonical host assignments and merge / aggregate procedure.

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

# Canonical Tier-1 launch set (see TIER1_SWEEP_MATRIX.md). 12 of the 14
# trainers in the parent issue's table — ippo (baseline) and coma
# (author-deprioritized per #271) are excluded by default but can be
# re-enabled via --trainers explicitly.
DEFAULT_TRAINERS="mappo,high_lambda,bc_init_continuation,bc_init_high_lambda,lola,hca,influence,nhr,progress,macro_actions,reinforce,pbt"
DEFAULT_SCENARIOS="minimal_specialization"
DEFAULT_SEEDS="42 43 44"
DEFAULT_NUM_ITERATIONS="50"
DEFAULT_ROLLOUT_STEPS="2048"
DEFAULT_OUTPUT_ROOT="experiments/p3_specialization/tier1_runs"
DEFAULT_SESSION_PREFIX="tier1-sweep"

HOST=""
TRAINERS="$DEFAULT_TRAINERS"
SCENARIOS="$DEFAULT_SCENARIOS"
SEEDS="$DEFAULT_SEEDS"
NUM_ITERATIONS="$DEFAULT_NUM_ITERATIONS"
ROLLOUT_STEPS="$DEFAULT_ROLLOUT_STEPS"
OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
SESSION_NAME=""
DRY_RUN=0
SKIP_CONNECTIVITY_CHECK=0
SKIP_AGGREGATE=0

print_usage() {
    sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
        --trainers)
            TRAINERS="$2"
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
        --output-root)
            OUTPUT_ROOT="$2"
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
        --skip-aggregate)
            # Skip the post-sweep aggregate_tier1.py call. Useful when
            # sharding across multiple hosts — the operator aggregates
            # after the rsync-merge step.
            SKIP_AGGREGATE=1
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

# ---------------------------------------------------------------------------
# Validate trainer list against the driver's known names
# ---------------------------------------------------------------------------

# Source the trainer set from the driver itself (issue #405 — DRY the
# launcher KNOWN_TRAINERS): the previous design kept a hand-maintained bash
# array in lockstep with run_tier1_cell.py's TRAINERS dict, which drifted
# (see PR #398). Calling --list-trainers on the driver eliminates the
# duplicate at a ~200ms startup cost, paid once per launch.
KNOWN_TRAINERS=()
while IFS= read -r line; do
    [[ -n "$line" ]] && KNOWN_TRAINERS+=("$line")
done < <(uv run --quiet python "$REPO_ROOT/experiments/p3_specialization/run_tier1_cell.py" --list-trainers 2>/dev/null)

if [[ "${#KNOWN_TRAINERS[@]}" -eq 0 ]]; then
    echo "ERROR: could not enumerate trainers from run_tier1_cell.py --list-trainers" >&2
    echo "       (is the local Python env / Rust extension set up?)" >&2
    echo "       Try: uv sync && bash bucket-brigade-core/build.sh" >&2
    exit 6
fi

is_known_trainer() {
    local name="$1"
    for t in "${KNOWN_TRAINERS[@]}"; do
        if [[ "$t" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

# Parse the comma-separated trainer list into an array and check every entry.
IFS=',' read -r -a TRAINER_ARRAY <<<"$TRAINERS"
for tr in "${TRAINER_ARRAY[@]}"; do
    if ! is_known_trainer "$tr"; then
        echo "ERROR: unknown trainer '$tr'" >&2
        echo "       Known trainers: ${KNOWN_TRAINERS[*]}" >&2
        echo "       Source of truth: experiments/p3_specialization/run_tier1_cell.py" >&2
        exit 5
    fi
done

IFS=',' read -r -a SCENARIO_ARRAY <<<"$SCENARIOS"

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
# Build per-cell driver commands + tmux session name
# ---------------------------------------------------------------------------

if [[ -z "$SESSION_NAME" ]]; then
    # Compact tag so two concurrent launches on the same host don't collide.
    # Use the trainer count + first trainer name as a stable fingerprint.
    tag="n${#TRAINER_ARRAY[@]}-${TRAINER_ARRAY[0]}"
    SESSION_NAME="${DEFAULT_SESSION_PREFIX}-${tag}"
fi

# Compose the per-cell driver invocations as a sequential shell-safe string.
# Each cell runs to completion before the next starts — the driver's
# subprocess plumbing already streams output, so tee'ing the whole tmux
# session captures everything in one log.
DRIVER_CMDS=""
for sc in "${SCENARIO_ARRAY[@]}"; do
    for tr in "${TRAINER_ARRAY[@]}"; do
        cell_cmd="uv run python experiments/p3_specialization/run_tier1_cell.py"
        cell_cmd+=" --trainer '$tr'"
        cell_cmd+=" --scenario '$sc'"
        cell_cmd+=" --seeds $SEEDS"
        cell_cmd+=" --num-iterations $NUM_ITERATIONS"
        cell_cmd+=" --rollout-steps $ROLLOUT_STEPS"
        cell_cmd+=" --output-root '$OUTPUT_ROOT'"
        if [[ -z "$DRIVER_CMDS" ]]; then
            DRIVER_CMDS="$cell_cmd"
        else
            # Use `;` not `&&` so one failing cell doesn't abort the sweep —
            # the driver writes cell_summary.json with verdict=no_data on
            # failure and we want the aggregator to surface that, not stop.
            DRIVER_CMDS+=" ; $cell_cmd"
        fi
    done
done

if [[ "$SKIP_AGGREGATE" -eq 0 ]]; then
    AGG_CMD="uv run python experiments/p3_specialization/aggregate_tier1.py --tier1-root '$OUTPUT_ROOT'"
    DRIVER_CMDS+=" ; $AGG_CMD"
fi

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
# Activate the venv so subsequent `uv pip install` and `bash build.sh` see
# VIRTUAL_ENV and write to .venv/bin (issue #418). Without this, `uv pip
# install maturin` inside build.sh has occasionally landed in a different
# location and the subsequent bare `maturin develop` call fails with
# "Failed to spawn: maturin — No such file or directory".
source .venv/bin/activate
uv sync

# Build Rust extension (idempotent — skipped if .so is fresh).
bash bucket-brigade-core/build.sh

# Sanity-check the import before consuming a tmux session for a multi-hour run.
uv run python -c "from experiments.p3_specialization.run_tier1_cell import TRAINERS; print(f'driver ok, {len(TRAINERS)} trainers')"

mkdir -p $OUTPUT_ROOT

# Kill any prior session with the same name so re-launches are idempotent.
tmux kill-session -t '$SESSION_NAME' 2>/dev/null || true

tmux new-session -d -s '$SESSION_NAME' "cd \$REPO && export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 && ( $DRIVER_CMDS ) 2>&1 | tee -a '$OUTPUT_ROOT/${SESSION_NAME}.log'"

echo ""
echo "Launched tmux session: $SESSION_NAME"
echo "Log: \$REPO/$OUTPUT_ROOT/${SESSION_NAME}.log"
echo "Attach with: tmux attach -t $SESSION_NAME"
REMOTE_SCRIPT

# ---------------------------------------------------------------------------
# Print plan / execute
# ---------------------------------------------------------------------------

echo "===================================================================="
echo "Tier-1 sweep launch (issue #343)"
echo "===================================================================="
echo "Host:           $HOST"
echo "Session name:   $SESSION_NAME"
echo "Output root:    $OUTPUT_ROOT    (on remote)"
echo "Trainers:       $TRAINERS    (${#TRAINER_ARRAY[@]} cells per scenario)"
echo "Scenarios:      $SCENARIOS"
echo "Seeds:          $SEEDS"
echo "Num iterations: $NUM_ITERATIONS"
echo "Rollout steps:  $ROLLOUT_STEPS"
echo "Aggregate:      $([ "$SKIP_AGGREGATE" -eq 0 ] && echo "yes (aggregate_tier1.py)" || echo "no (--skip-aggregate)")"
echo ""
echo "Per-cell driver commands (to be run inside tmux on remote, sequentially):"
# Pretty-print one per line for the operator's sanity check.
echo "$DRIVER_CMDS" | tr ';' '\n' | sed 's/^ *//; s/^/  /'
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
echo "  ssh $HOST 'tail -f bucket-brigade/${OUTPUT_ROOT}/${SESSION_NAME}.log'"
echo ""
echo "  # When done, rsync results back to this checkout:"
echo "  rsync -avz $HOST:bucket-brigade/${OUTPUT_ROOT}/ ${OUTPUT_ROOT}/"
echo ""
echo "  # Regenerate the verdict table locally once results are back:"
echo "  uv run python experiments/p3_specialization/aggregate_tier1.py \\"
echo "      --tier1-root ${OUTPUT_ROOT}   # writes tier1_verdict.md + .json"
echo "===================================================================="
