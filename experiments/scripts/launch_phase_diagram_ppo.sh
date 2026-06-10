#!/usr/bin/env bash
# Launch the phase-diagram PPO sweep (issue #360 / parent #357 M2.1) on a
# remote host. Dispatches experiments/p3_specialization/run_phase_diagram_ppo.py
# which itself drives experiments/p3_specialization/train.py with the per-cell
# (β, κ, c) override flags added in #360.
#
# Default host: alc-2 (RTX 4090 CPU host). The CPU-bound bucket-brigade env
# benefits from alc-2's core count + reliability (see
# memory/reference_cluster_host_reliability.md). If alc-2 isn't reachable the
# script falls back to whatever .env's COMPUTE_HOST_* aliases resolve to, in
# the same priority order as launch_tier1_sweep.sh.
#
# The script does NOT compute anything locally. It only:
#   1. Resolves a target host (--host wins; else COMPUTE_HOST_* from .env;
#      else exits non-zero).
#   2. Verifies the host is reachable via SSH BatchMode.
#   3. SSHs in, pulls latest main, builds the Rust extension, and starts a
#      detached tmux session running the per-cell driver.
#   4. Prints monitor / rsync instructions.
#
# CLAUDE.md is explicit: long unattended cluster runs are operator-driven,
# not safe for an autonomous agent to spawn. This script is the operator's
# launch tool. It does its job and exits — the actual cells fill over hours
# inside the remote tmux session.
#
# Usage
# -----
#   # Default: alc-2, 7 NE-phase-diagram cells × 4 seeds × minimal_specialization
#   ./experiments/scripts/launch_phase_diagram_ppo.sh
#
#   # Pick a different host (alc-6 / alc-9 are also reliable, see memory)
#   ./experiments/scripts/launch_phase_diagram_ppo.sh --host alc-6
#
#   # Sanity-check a single cell at smoke budget before the full sweep
#   ./experiments/scripts/launch_phase_diagram_ppo.sh \
#       --limit-cells 1 --num-iterations 5 --rollout-steps 256
#
#   # Dry-run prints the ssh + driver command without launching
#   ./experiments/scripts/launch_phase_diagram_ppo.sh --dry-run
#
#   # Use a different NE-search results source (e.g. the freqtest variant)
#   ./experiments/scripts/launch_phase_diagram_ppo.sh \
#       --cells-source experiments/nash/phase_diagram/results_freqtest.json
#
# See experiments/p3_specialization/run_phase_diagram_ppo.py for the
# per-(cell × seed) summary schema and the per-cell aggregation.

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

# Documented default host for this sweep (operator may override with --host).
# Resolver still falls back to .env's COMPUTE_HOST_* if alc-2 isn't reachable
# or isn't configured.
DEFAULT_HOST="alc-2"

DEFAULT_CELLS_SOURCE="experiments/nash/phase_diagram/results.json"
DEFAULT_SCENARIO="minimal_specialization"
# The gap_closed metric is calibrated against the MINSPEC_RANDOM /
# MINSPEC_SPECIALIST baselines in bucket_brigade.baselines (see
# run_tier1_cell.py:306-312 and run_phase_diagram_ppo.py:334-335). Running
# the sweep on any other scenario produces uncalibrated gap_closed numbers
# — we lock the scenario at the CLI to prevent the silent-garbage failure
# mode. Operators who know what they're doing can opt out with
# --allow-non-minspec-gap.
MINSPEC_LOCKED_SCENARIO="minimal_specialization"
DEFAULT_SEEDS="42 43 44 45"
# Mirror launch_tier1_sweep.sh's budget — 50 PPO iterations × 2048 rollout
# steps is the Tier-1 baseline gap_closed measurement budget.
DEFAULT_NUM_ITERATIONS="50"
DEFAULT_ROLLOUT_STEPS="2048"
DEFAULT_OUTPUT_ROOT="experiments/p3_specialization/phase_diagram_ppo"
DEFAULT_SESSION_PREFIX="phase-diagram-ppo"

HOST=""
CELLS_SOURCE="$DEFAULT_CELLS_SOURCE"
SCENARIO="$DEFAULT_SCENARIO"
SEEDS="$DEFAULT_SEEDS"
NUM_ITERATIONS="$DEFAULT_NUM_ITERATIONS"
ROLLOUT_STEPS="$DEFAULT_ROLLOUT_STEPS"
OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
SESSION_NAME=""
LIMIT_CELLS=""
DRY_RUN=0
SKIP_CONNECTIVITY_CHECK=0
ALLOW_NON_MINSPEC_GAP=0

print_usage() {
    sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Resolve a host alias from .env using the documented priority order.
# Echoes the first non-empty option; empty string if none configured.
# Copied verbatim from launch_tier1_sweep.sh so both launchers see the same
# host inventory.
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
        --cells-source)
            CELLS_SOURCE="$2"
            shift 2
            ;;
        --scenario)
            SCENARIO="$2"
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
        --limit-cells)
            LIMIT_CELLS="$2"
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
        --allow-non-minspec-gap)
            # Escape hatch for an operator who knows the gap_closed metric
            # will be uncalibrated and wants to run the sweep anyway (e.g.
            # for raw trajectory inspection). See MINSPEC_LOCKED_SCENARIO
            # comment above.
            ALLOW_NON_MINSPEC_GAP=1
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
# Validate scenario (gap_closed lock)
# ---------------------------------------------------------------------------
#
# The gap_closed metric written by run_phase_diagram_ppo.py is hard-coded to
# the MINSPEC_RANDOM / MINSPEC_SPECIALIST baselines. Running with any other
# --scenario produces summary.json / cell_summary.json files whose
# gap_closed numbers are not comparable to anything — the operator concern
# from PR #410's review is that this fails silently. Reject non-minspec
# scenarios at the CLI unless --allow-non-minspec-gap is set.

if [[ "$SCENARIO" != "$MINSPEC_LOCKED_SCENARIO" && "$ALLOW_NON_MINSPEC_GAP" -eq 0 ]]; then
    echo "ERROR: --scenario '$SCENARIO' rejected." >&2
    echo "       gap_closed metric is calibrated only for minimal_specialization;" >&2
    echo "       other scenarios will produce uncalibrated gap_closed values." >&2
    echo "       Re-run with --allow-non-minspec-gap to override." >&2
    exit 5
fi

# ---------------------------------------------------------------------------
# Resolve host
# ---------------------------------------------------------------------------
#
# Priority:
#   1. --host explicit
#   2. DEFAULT_HOST (alc-2) if SSH-reachable
#   3. .env COMPUTE_HOST_PRIMARY / _CLUSTER / _LAMBDA / _GCP in that order
#
# The .env fallback keeps the script useful on a fresh clone that doesn't
# happen to have alc-2 wired up. We do the reachability check up-front
# (unless --dry-run / --skip-connectivity-check) so misconfigured operators
# fail fast, not after committing a tmux session.

if [[ -z "$HOST" ]]; then
    if [[ "$SKIP_CONNECTIVITY_CHECK" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
        if check_ssh_reachable "$DEFAULT_HOST"; then
            HOST="$DEFAULT_HOST"
        else
            HOST=$(resolve_host_from_env)
        fi
    else
        # Skip the probe in dry-run; just take the documented default.
        HOST="$DEFAULT_HOST"
    fi
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
# Build the driver command + tmux session name
# ---------------------------------------------------------------------------

if [[ -z "$SESSION_NAME" ]]; then
    # Compact tag so two concurrent launches on the same host don't collide.
    # Use the scenario + seed count as a stable fingerprint.
    seed_count=$(echo "$SEEDS" | wc -w | tr -d ' ')
    SESSION_NAME="${DEFAULT_SESSION_PREFIX}-${SCENARIO}-n${seed_count}"
fi

# Compose the driver invocation. One python entrypoint covers all cells; the
# driver writes per-(cell × seed) summary.json and per-cell cell_summary.json.
DRIVER_CMD="uv run python experiments/p3_specialization/run_phase_diagram_ppo.py"
DRIVER_CMD+=" --cells-source '$CELLS_SOURCE'"
DRIVER_CMD+=" --scenario '$SCENARIO'"
DRIVER_CMD+=" --seeds $SEEDS"
DRIVER_CMD+=" --num-iterations $NUM_ITERATIONS"
DRIVER_CMD+=" --rollout-steps $ROLLOUT_STEPS"
DRIVER_CMD+=" --output-root '$OUTPUT_ROOT'"
if [[ -n "$LIMIT_CELLS" ]]; then
    DRIVER_CMD+=" --limit-cells $LIMIT_CELLS"
fi
if [[ "$ALLOW_NON_MINSPEC_GAP" -eq 1 ]]; then
    # Forward the opt-out so the driver's own scenario lock also lets it
    # through. (We've already passed the launcher-side lock above.)
    DRIVER_CMD+=" --allow-non-minspec-gap"
fi

# Remote bootstrap: try canonical repo paths, pull, build Rust ext, run.
# Notes on the remote env (see CLAUDE.md):
#   - PATH is bare under non-interactive SSH; prepend Homebrew (macOS),
#     $HOME/.local/bin (Linux — uv installer's default), and $HOME/.cargo/bin.
#   - PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 + unset RUSTC_WRAPPER for the build.
#   - bucket-brigade-core/build.sh uses `python -m pip install -e .`, so the
#     venv must have pip; uv ships venvs without pip by default. (See
#     memory/feedback_cluster_bootstrap_uv_sync.md — uv sync must come
#     before the editable install of bucket-brigade-core.)
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
# --extra rl is mandatory for this sweep: experiments/p3_specialization/train.py
# imports torch unconditionally. A bare `uv sync` produces a venv without
# torch, every cell × seed crashes immediately at import, and the launcher
# reports "0/7 cells produced metrics" after a few seconds. See
# launch_ppo_baselines.sh:309 for the matching precedent (PPO baselines also
# need --extra rl); launch_tier1_sweep.sh does NOT pass --extra rl because
# its Rust-only trainers don't import torch.
uv sync --extra rl

# Build Rust extension (idempotent — skipped if .so is fresh).
bash bucket-brigade-core/build.sh

# Sanity-check the import + cells file exists before consuming a tmux
# session for a multi-hour run.
uv run python -c "from experiments.p3_specialization.run_phase_diagram_ppo import load_cells; cells = load_cells(__import__('pathlib').Path('$CELLS_SOURCE')); print(f'driver ok, {len(cells)} cells')"

mkdir -p $OUTPUT_ROOT

# Kill any prior session with the same name so re-launches are idempotent.
tmux kill-session -t '$SESSION_NAME' 2>/dev/null || true

tmux new-session -d -s '$SESSION_NAME' "cd \$REPO && export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 && ( $DRIVER_CMD ) 2>&1 | tee -a '$OUTPUT_ROOT/${SESSION_NAME}.log'"

echo ""
echo "Launched tmux session: $SESSION_NAME"
echo "Log: \$REPO/$OUTPUT_ROOT/${SESSION_NAME}.log"
echo "Attach with: tmux attach -t $SESSION_NAME"
REMOTE_SCRIPT

# ---------------------------------------------------------------------------
# Print plan / execute
# ---------------------------------------------------------------------------

seed_count=$(echo "$SEEDS" | wc -w | tr -d ' ')

echo "===================================================================="
echo "Phase-diagram PPO sweep launch (issue #360)"
echo "===================================================================="
echo "Host:           $HOST"
echo "Session name:   $SESSION_NAME"
echo "Output root:    $OUTPUT_ROOT    (on remote)"
echo "Cells source:   $CELLS_SOURCE"
echo "Scenario:       $SCENARIO"
echo "Seeds:          $SEEDS    ($seed_count per cell)"
echo "Num iterations: $NUM_ITERATIONS"
echo "Rollout steps:  $ROLLOUT_STEPS"
if [[ -n "$LIMIT_CELLS" ]]; then
    echo "Limit cells:    $LIMIT_CELLS"
fi
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
echo "  ssh $HOST 'tail -f bucket-brigade/${OUTPUT_ROOT}/${SESSION_NAME}.log'"
echo ""
echo "  # When done, rsync results back to this checkout:"
echo "  rsync -avz $HOST:bucket-brigade/${OUTPUT_ROOT}/ ${OUTPUT_ROOT}/"
echo ""
echo "  # Once results are local, the Tier-1 aggregator already understands"
echo "  # the cell_summary.json schema (we reused build_cell_summary), so:"
echo "  uv run python experiments/p3_specialization/aggregate_tier1.py \\"
echo "      --tier1-root ${OUTPUT_ROOT}"
echo "===================================================================="
