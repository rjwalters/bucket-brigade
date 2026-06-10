#!/usr/bin/env bash
# Launch the heterogeneous Nash phase-diagram driver on a remote host to
# fill a specified slice of (β, κ, c) cells. Designed for the gap-fill
# operation tracked in issue #390 (β=0.1 row, c=1.0, c=2.0 planes), but
# generic: any --beta-values / --kappa-values / --c-values list can be
# passed and the driver will run that subset, caching previously-completed
# cells via summary.json (see compute_nash_phase_diagram.py --force).
#
# The script does NOT compute anything locally. It only:
#   1. Reads $COMPUTE_HOST_* aliases from .env to resolve a target host
#      (unless --host is passed explicitly).
#   2. Verifies the host is reachable via SSH BatchMode.
#   3. SSHs in, pulls latest main, builds the Rust extension, and starts
#      a detached tmux session running compute_nash_phase_diagram.py with
#      the requested cell filter and --num-workers.
#   4. Prints monitor / rsync / replot instructions.
#
# CLAUDE.md is explicit: long unattended cluster runs are operator-driven,
# not safe for an autonomous agent to spawn. This script is the operator's
# launch tool. It does its job and exits — the actual cells fill over
# hours/days inside the remote tmux session.
#
# Usage
# -----
#   # Auto-resolve host from .env, fill all of c=2.0 (the alc-10 gap)
#   ./experiments/scripts/launch_phase_diagram_fill.sh \
#       --c-values 2.0
#
#   # Explicit host, just the β=0.1 row gap at c=0.5 (κ ∈ {0.5, 0.9})
#   ./experiments/scripts/launch_phase_diagram_fill.sh \
#       --host alc-2 --beta-values 0.1 --kappa-values 0.5,0.9 --c-values 0.5
#
#   # Dry-run prints the ssh + driver command without launching anything
#   ./experiments/scripts/launch_phase_diagram_fill.sh --c-values 1.0 --dry-run
#
# See experiments/nash/phase_diagram/LAUNCH_RUNBOOK.md for the canonical
# host assignments for the issue #390 fill operation.

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

# Defaults that the operator usually does not need to override.
DEFAULT_SESSION_PREFIX="nash-fill"
DEFAULT_NUM_WORKERS=""        # empty => let driver default (cpu_count())
DEFAULT_BETA_VALUES=""        # empty => omit flag (driver uses --preview defaults if --preview is on, else FULL)
DEFAULT_KAPPA_VALUES=""
DEFAULT_C_VALUES=""

HOST=""
BETA_VALUES="$DEFAULT_BETA_VALUES"
KAPPA_VALUES="$DEFAULT_KAPPA_VALUES"
C_VALUES="$DEFAULT_C_VALUES"
NUM_WORKERS="$DEFAULT_NUM_WORKERS"
SESSION_NAME=""
DRY_RUN=0
SKIP_CONNECTIVITY_CHECK=0
OUTPUT_DIR="experiments/nash/phase_diagram/preview"

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
    # Source in a subshell-safe way: only export the COMPUTE_HOST_* vars.
    # Using `set -a` would also export anything else in .env which we
    # don't want.
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
        --beta-values)
            BETA_VALUES="$2"
            shift 2
            ;;
        --kappa-values)
            KAPPA_VALUES="$2"
            shift 2
            ;;
        --c-values)
            C_VALUES="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --session-name)
            SESSION_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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
# Build driver command + tmux session name
# ---------------------------------------------------------------------------

if [[ -z "$SESSION_NAME" ]]; then
    # Compact tag so two concurrent launches on the same host don't collide.
    tag_parts=()
    [[ -n "$BETA_VALUES" ]] && tag_parts+=("b$(echo "$BETA_VALUES" | tr ',' '-')")
    [[ -n "$KAPPA_VALUES" ]] && tag_parts+=("k$(echo "$KAPPA_VALUES" | tr ',' '-')")
    [[ -n "$C_VALUES" ]] && tag_parts+=("c$(echo "$C_VALUES" | tr ',' '-')")
    if [[ ${#tag_parts[@]} -gt 0 ]]; then
        SESSION_NAME="${DEFAULT_SESSION_PREFIX}-$(IFS=_; echo "${tag_parts[*]}")"
    else
        SESSION_NAME="$DEFAULT_SESSION_PREFIX"
    fi
fi

# Compose the driver invocation as a single shell-safe string.
DRIVER_CMD="uv run python experiments/scripts/compute_nash_phase_diagram.py --output-dir '$OUTPUT_DIR'"
[[ -n "$BETA_VALUES" ]] && DRIVER_CMD+=" --beta-values '$BETA_VALUES'"
[[ -n "$KAPPA_VALUES" ]] && DRIVER_CMD+=" --kappa-values '$KAPPA_VALUES'"
[[ -n "$C_VALUES" ]] && DRIVER_CMD+=" --c-values '$C_VALUES'"
[[ -n "$NUM_WORKERS" ]] && DRIVER_CMD+=" --num-workers $NUM_WORKERS"

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
uv sync

# Build Rust extension (idempotent — skipped if .so is fresh).
bash bucket-brigade-core/build.sh

# Sanity-check the import before consuming a tmux session for a multi-hour run.
uv run python -c "from bucket_brigade.equilibrium.double_oracle_heterogeneous import HeterogeneousDoubleOracle; print('ok')"

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
echo "Phase-diagram fill launch"
echo "===================================================================="
echo "Host:           $HOST"
echo "Session name:   $SESSION_NAME"
echo "Output dir:     $OUTPUT_DIR    (on remote)"
echo "Beta values:    ${BETA_VALUES:-<driver default>}"
echo "Kappa values:   ${KAPPA_VALUES:-<driver default>}"
echo "C values:       ${C_VALUES:-<driver default>}"
echo "Num workers:    ${NUM_WORKERS:-<driver default = cpu_count()>}"
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
echo "  # When done, rsync results back to this checkout:"
echo "  rsync -avz $HOST:bucket-brigade/${OUTPUT_DIR}/ ${OUTPUT_DIR}/"
echo ""
echo "  # Regenerate aggregate + figures locally once results are back:"
echo "  uv run python experiments/scripts/compute_nash_phase_diagram.py --preview \\"
echo "      --output-dir ${OUTPUT_DIR}   # picks up cached cells, writes results.json"
echo "  uv run python experiments/scripts/plot_phase_diagram.py \\"
echo "      --results ${OUTPUT_DIR}/results.json"
echo "===================================================================="
