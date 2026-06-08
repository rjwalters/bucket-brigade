#!/usr/bin/env bash
# Launch the issue #386 asymmetry-aware ("het_ppo") sweep on a remote host.
# The het_ppo trainer arm (see experiments/p3_specialization/run_tier1_cell.py
# TRAINERS dict and bucket_brigade/training/joint_trainer.py
# per_agent_init_seed_offset kwarg) is the HetGPPO-style positive baseline
# for asymmetric_only phase-diagram cells (Bettini et al. AAMAS 2023,
# arXiv:2301.07137): each per-position policy is initialized from a
# maximally-distinct RNG stream so SGD cannot trap into a symmetric basin.
#
# The script does NOT compute anything locally. It only:
#   1. Reads $COMPUTE_HOST_* aliases from .env to resolve a target host
#      (unless --host is passed explicitly).
#   2. Verifies the host is reachable via SSH BatchMode.
#   3. SSHs in, pulls latest main, builds the Rust extension, and starts
#      a detached tmux session running run_tier1_cell.py with
#      --trainer het_ppo over the requested scenarios × seeds.
#   4. Prints monitor / rsync follow-up instructions.
#
# CLAUDE.md is explicit: long unattended cluster runs are operator-driven,
# not safe for an autonomous agent to spawn. This script is the operator's
# launch tool. It does its job and exits — the actual cells fill over
# hours inside the remote tmux session.
#
# Usage
# -----
#   # Phase 1: anchor on rest_trap (20 seeds, default budget)
#   ./experiments/scripts/launch_het_ppo_sweep.sh --scenarios rest_trap
#
#   # Phase 2: every asymmetric_only cell from the phase diagram
#   ./experiments/scripts/launch_het_ppo_sweep.sh \
#       --scenarios rest_trap,asym_b05_k05_c09,asym_b05_k09_c09
#
#   # Explicit host, smaller seed set for a fast positive-control rerun
#   ./experiments/scripts/launch_het_ppo_sweep.sh \
#       --host alc-2 --scenarios rest_trap --seeds 42,43,44 \
#       --num-iterations 25 --rollout-steps 1024
#
#   # Dry-run prints the ssh + driver command without launching anything
#   ./experiments/scripts/launch_het_ppo_sweep.sh --scenarios rest_trap --dry-run
#
# Flags:
#   --host HOST                     SSH alias (or auto-resolved from .env)
#   --scenarios LIST                Comma-separated scenario names (required)
#   --seeds LIST                    Comma-separated seeds (default: 20 seeds)
#   --num-iterations N              PPO iterations per seed (default: 50)
#   --rollout-steps N               Rollout buffer size (default: 2048)
#   --output-dir PATH               Remote output root (default: tier1_runs)
#   --session-name NAME             tmux session name (auto-synthesized)
#   --dry-run                       Print plan; do not ssh anywhere
#   --skip-connectivity-check       For CI / smoke tests
#
# See experiments/p3_specialization/het_ppo_runbook.md for the canonical
# scenario list, host assignments, and the per-cell expected wall-clock.

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
DEFAULT_SESSION_PREFIX="het-ppo"
DEFAULT_SEEDS="42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61"
DEFAULT_NUM_ITERATIONS=50
DEFAULT_ROLLOUT_STEPS=2048
DEFAULT_OUTPUT_DIR="experiments/p3_specialization/tier1_runs"

HOST=""
SCENARIOS=""
SEEDS="$DEFAULT_SEEDS"
NUM_ITERATIONS="$DEFAULT_NUM_ITERATIONS"
ROLLOUT_STEPS="$DEFAULT_ROLLOUT_STEPS"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
SESSION_NAME=""
DRY_RUN=0
SKIP_CONNECTIVITY_CHECK=0

print_usage() {
    sed -n '2,64p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

# Required: --scenarios
if [[ -z "$SCENARIOS" ]]; then
    echo "ERROR: --scenarios is required (comma-separated scenario names)" >&2
    echo "       e.g. --scenarios rest_trap" >&2
    echo "       See experiments/p3_specialization/het_ppo_runbook.md for the canonical list." >&2
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

# Convert comma-separated scenarios/seeds into space-separated for the driver.
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
# cell_summary.json under <OUTPUT_DIR>/het_ppo_<scenario>/).
DRIVER_CMD="set -e"
IFS=',' read -ra SCENARIO_LIST <<< "$SCENARIOS"
for scen in "${SCENARIO_LIST[@]}"; do
    DRIVER_CMD+=" && echo '=== het_ppo on $scen ===' && uv run python experiments/p3_specialization/run_tier1_cell.py --trainer het_ppo --scenario '$scen' --seeds $SEEDS_SPACED --num-iterations $NUM_ITERATIONS --rollout-steps $ROLLOUT_STEPS --output-root '$OUTPUT_DIR'"
done

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
uv sync --extra rl

# Build Rust extension (idempotent — skipped if .so is fresh).
bash bucket-brigade-core/build.sh

# Sanity-check the trainer import + het_ppo dispatch before consuming a tmux
# session for a multi-hour run.
uv run python -c "from experiments.p3_specialization.run_tier1_cell import TRAINERS; assert 'het_ppo' in TRAINERS, 'het_ppo missing from dispatch — run from a main checkout that has issue #386 merged'; print('het_ppo dispatch OK')"

mkdir -p $OUTPUT_DIR

# Kill any prior session with the same name so re-launches are idempotent.
tmux kill-session -t '$SESSION_NAME' 2>/dev/null || true

tmux new-session -d -s '$SESSION_NAME' "cd \$REPO && export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 && ($DRIVER_CMD) 2>&1 | tee -a '$OUTPUT_DIR/${SESSION_NAME}.log'"

echo ""
echo "Launched tmux session: $SESSION_NAME"
echo "Log: \$REPO/$OUTPUT_DIR/${SESSION_NAME}.log"
echo "Attach with: tmux attach -t $SESSION_NAME"
REMOTE_SCRIPT

# ---------------------------------------------------------------------------
# Print plan / execute
# ---------------------------------------------------------------------------

echo "===================================================================="
echo "Issue #386 — het_ppo sweep launch"
echo "===================================================================="
echo "Host:           $HOST"
echo "Session name:   $SESSION_NAME"
echo "Output dir:     $OUTPUT_DIR    (on remote)"
echo "Scenarios:      $SCENARIOS"
echo "Seeds:          $SEEDS"
echo "Num iterations: $NUM_ITERATIONS"
echo "Rollout steps:  $ROLLOUT_STEPS"
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
echo "  # Aggregate verdicts across scenarios once results are back:"
echo "  uv run python experiments/p3_specialization/aggregate_tier1.py \\"
echo "      --root ${OUTPUT_DIR}   # writes tier1_verdict.{json,md}"
echo "===================================================================="
