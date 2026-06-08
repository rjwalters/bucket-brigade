#!/usr/bin/env bash
# Launch the `rest_trap` Nash re-derivation on a remote host to fill the
# 12th (and only missing) cell of the V1 post-#240 sweep. Tracks issue
# #349, which exists because the original sweep (issue #256, tmux session
# `nash256` on COMPUTE_HOST_PRIMARY, 2026-05-16) crashed at the
# `equilibrium.json` write step with ENOSPC. The df-precheck from PR #315
# (closes #269, now wired in compute_nash.py:329) makes a safe single-cell
# re-run cheap (~25-35 min on a Mac Studio, vs. ~5 h for the whole sweep).
#
# The script does NOT compute anything locally. It only:
#   1. Reads $COMPUTE_HOST_* aliases from .env to resolve a target host
#      (unless --host is passed explicitly).
#   2. Verifies the host is reachable via SSH BatchMode.
#   3. SSHs in, pulls latest main, builds the Rust extension, and starts
#      a detached tmux session running compute_nash.py with the canonical
#      sweep parameters (seed=42, simulations=200, max-iterations=50,
#      epsilon=0.01) — matching the other 11 sibling scenarios so the
#      result is schema-comparable.
#   4. Prints monitor / rsync / verify instructions.
#
# CLAUDE.md is explicit: long unattended cluster runs are operator-driven,
# not safe for an autonomous agent to spawn. This script is the operator's
# launch tool. It does its job and exits — the actual rest_trap cell
# completes ~30 minutes later inside the remote tmux session.
#
# Usage
# -----
#   # Auto-resolve host from .env, run with the canonical sweep params
#   ./experiments/scripts/launch_rest_trap_rerun.sh
#
#   # Explicit host
#   ./experiments/scripts/launch_rest_trap_rerun.sh --host alc-2
#
#   # Override one of the sweep params (NOT recommended — the schema-match
#   # acceptance criterion depends on the canonical values)
#   ./experiments/scripts/launch_rest_trap_rerun.sh --seed 7 --simulations 500
#
#   # Dry-run prints the ssh + driver command without launching anything
#   ./experiments/scripts/launch_rest_trap_rerun.sh --dry-run
#
# See experiments/nash/v1_results_python_post240/RERUN_RUNBOOK.md for the
# canonical operator procedure, the post-run verification checklist, and
# the diff/commit workflow.

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

# Canonical sweep parameters from issue #256 / PR #347. Changing any of
# these breaks the schema-match acceptance criterion (algorithm.seed = 42,
# num_simulations = 200, max_iterations = 50, epsilon = 0.01) so they are
# locked-in defaults that --simulations / --seed / etc. can override only
# if the operator explicitly chooses to do so.
DEFAULT_SCENARIO="rest_trap"
DEFAULT_SIMULATIONS="200"
DEFAULT_MAX_ITERATIONS="50"
DEFAULT_EPSILON="0.01"
DEFAULT_SEED="42"
DEFAULT_OUTPUT_DIR="experiments/nash/v1_results_python_post240/rest_trap"
DEFAULT_SESSION_NAME="nash-rest-trap"

HOST=""
SCENARIO="$DEFAULT_SCENARIO"
SIMULATIONS="$DEFAULT_SIMULATIONS"
MAX_ITERATIONS="$DEFAULT_MAX_ITERATIONS"
EPSILON="$DEFAULT_EPSILON"
SEED="$DEFAULT_SEED"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
SESSION_NAME="$DEFAULT_SESSION_NAME"
DRY_RUN=0
SKIP_CONNECTIVITY_CHECK=0

print_usage() {
    sed -n '2,43p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --simulations)
            SIMULATIONS="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --epsilon)
            EPSILON="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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
# Build driver command
# ---------------------------------------------------------------------------

# Compose the driver invocation as a single shell-safe string. Note that
# `scenario` is a POSITIONAL argument in compute_nash.py:291 — not a flag.
# The original issue body had a bug here (it said --scenario rest_trap)
# that the curator caught; we lock the positional form into the launcher
# so an operator cannot accidentally reintroduce that bug.
DRIVER_CMD="uv run python experiments/scripts/compute_nash.py"
DRIVER_CMD+=" '$SCENARIO'"
DRIVER_CMD+=" --simulations $SIMULATIONS"
DRIVER_CMD+=" --max-iterations $MAX_ITERATIONS"
DRIVER_CMD+=" --epsilon $EPSILON"
DRIVER_CMD+=" --seed $SEED"
DRIVER_CMD+=" --output-dir '$OUTPUT_DIR'"

# Remote bootstrap: try canonical repo paths, pull, build Rust ext, run.
# Notes on the remote env (see CLAUDE.md):
#   - PATH is bare under non-interactive SSH on macOS; prepend Homebrew + cargo.
#   - PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 + unset RUSTC_WRAPPER for the build.
#   - bucket-brigade-core/build.sh uses `python -m pip install -e .`, so the
#     venv must have pip; uv ships venvs without pip by default.
read -r -d '' REMOTE_BOOTSTRAP <<REMOTE_SCRIPT || true
set -euo pipefail
export PATH="/opt/homebrew/bin:\$HOME/.cargo/bin:\$PATH"
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
uv sync

# Build Rust extension (idempotent — skipped if .so is fresh).
bash bucket-brigade-core/build.sh

# Sanity-check the import before consuming a tmux session for a ~30-min run.
uv run python -c "from bucket_brigade.equilibrium import DoubleOracle; print('ok')"

mkdir -p $OUTPUT_DIR

# Kill any prior session with the same name so re-launches are idempotent.
tmux kill-session -t '$SESSION_NAME' 2>/dev/null || true

tmux new-session -d -s '$SESSION_NAME' "cd \$REPO && export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 && $DRIVER_CMD 2>&1 | tee -a '$OUTPUT_DIR/${SESSION_NAME}.log'"

echo ""
echo "Launched tmux session: $SESSION_NAME"
echo "Log: \$REPO/$OUTPUT_DIR/${SESSION_NAME}.log"
echo "Attach with: tmux attach -t $SESSION_NAME"
REMOTE_SCRIPT

# ---------------------------------------------------------------------------
# Print plan / execute
# ---------------------------------------------------------------------------

echo "===================================================================="
echo "rest_trap Nash re-run launch (issue #349)"
echo "===================================================================="
echo "Host:           $HOST"
echo "Session name:   $SESSION_NAME"
echo "Output dir:     $OUTPUT_DIR    (on remote)"
echo "Scenario:       $SCENARIO"
echo "Simulations:    $SIMULATIONS"
echo "Max iterations: $MAX_ITERATIONS"
echo "Epsilon:        $EPSILON"
echo "Seed:           $SEED"
echo ""
echo "Driver command (to be run inside tmux on remote):"
echo "  $DRIVER_CMD"
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
echo "  # When done (~30 min), rsync result back to this checkout:"
echo "  rsync -avz $HOST:bucket-brigade/${OUTPUT_DIR}/ ${OUTPUT_DIR}/"
echo ""
echo "  # Verify the artifact is well-formed:"
echo "  uv run python -c \"import json; d = json.load(open('${OUTPUT_DIR}/equilibrium.json')); print(d['algorithm'])\""
echo ""
echo "  # Re-run the post-240 diff and update docs/NASH_BENCHMARKS.md:"
echo "  uv run python experiments/nash/scripts/diff_post240.py"
echo "===================================================================="
