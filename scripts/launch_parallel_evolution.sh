#!/bin/bash
# Launch parallel evolution for all 12 scenarios on remote sandbox
#
# Usage:
#   ./scripts/launch_parallel_evolution.sh [OPTIONS]
#
# Options:
#   --host HOST        Remote host (default: from .env or prompt)
#   --generations N    Number of generations (default: 500)
#   --population N     Population size (default: 50)
#   --workers N        Workers per scenario (default: 4)
#   --dry-run          Show what would be done without executing
#   --help             Show this help message

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_DIR="${REMOTE_DIR:-bucket-brigade}"
GENERATIONS=500
POPULATION=50
WORKERS_PER_SCENARIO=4
DRY_RUN=false

# All 12 scenarios
SCENARIOS=(
    chain_reaction
    deceptive_calm
    default
    early_containment
    easy
    greedy_neighbor
    hard
    mixed_motivation
    overcrowding
    rest_trap
    sparse_heroics
    trivial_cooperation
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --generations)
            GENERATIONS="$2"
            shift 2
            ;;
        --population)
            POPULATION="$2"
            shift 2
            ;;
        --workers)
            WORKERS_PER_SCENARIO="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Load .env if exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo -e "${BLUE}ℹ Loaded configuration from .env${NC}"
fi

# Prompt for host if not set
if [ -z "$REMOTE_HOST" ]; then
    echo -e "${YELLOW}⚠ Remote host not specified${NC}"
    read -p "Enter remote host (e.g., rwalters-sandbox-2): " REMOTE_HOST

    if [ -z "$REMOTE_HOST" ]; then
        echo -e "${RED}Error: Remote host is required${NC}"
        exit 1
    fi
fi

# Configuration summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Parallel Evolution Launch Configuration           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Remote host:${NC}        $REMOTE_HOST"
echo -e "  ${BLUE}Scenarios:${NC}          ${#SCENARIOS[@]} (all available)"
echo -e "  ${BLUE}Generations:${NC}        $GENERATIONS"
echo -e "  ${BLUE}Population:${NC}         $POPULATION"
echo -e "  ${BLUE}Workers/scenario:${NC}   $WORKERS_PER_SCENARIO"
echo -e "  ${BLUE}Total workers:${NC}      $((${#SCENARIOS[@]} * WORKERS_PER_SCENARIO))"
echo ""

# Dry run check
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🔍 DRY RUN MODE - No commands will be executed${NC}"
    echo ""
fi

# Pre-flight checks
echo -e "${BLUE}🔍 Running pre-flight checks...${NC}"
echo ""

# Check SSH connection
echo -n "  [1/4] Checking SSH connection... "
if [ "$DRY_RUN" = false ]; then
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$REMOTE_HOST" "echo ok" &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        echo -e "${RED}Error: Cannot connect to $REMOTE_HOST${NC}"
        echo "Please check:"
        echo "  - SSH keys are configured"
        echo "  - Host is accessible"
        echo "  - ~/.ssh/config has the correct entry"
        exit 1
    fi
else
    echo -e "${YELLOW}[skipped]${NC}"
fi

# Check remote directory
echo -n "  [2/4] Checking remote directory... "
if [ "$DRY_RUN" = false ]; then
    if ssh "$REMOTE_HOST" "[ -d $REMOTE_DIR ]"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        echo -e "${RED}Error: $REMOTE_DIR directory not found on remote${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[skipped]${NC}"
fi

# Check Python environment
echo -n "  [3/4] Checking Python environment... "
if [ "$DRY_RUN" = false ]; then
    if ssh "$REMOTE_HOST" "[ -f bucket-brigade/.venv/bin/python ]"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        echo -e "${RED}Error: Python virtual environment not found${NC}"
        echo "Run on remote: cd $REMOTE_DIR && uv venv && uv sync"
        exit 1
    fi
else
    echo -e "${YELLOW}[skipped]${NC}"
fi

# Check evolution script
echo -n "  [4/4] Checking evolution script... "
if [ "$DRY_RUN" = false ]; then
    if ssh "$REMOTE_HOST" "[ -f $REMOTE_DIR/scripts/evolve_scenario_expert.py ]"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        echo -e "${RED}Error: scripts/evolve_scenario_expert.py not found${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[skipped]${NC}"
fi

echo ""
echo -e "${GREEN}✓ All pre-flight checks passed${NC}"
echo ""

# Confirm launch
if [ "$DRY_RUN" = false ]; then
    read -p "Launch evolution for all ${#SCENARIOS[@]} scenarios? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Aborted${NC}"
        exit 0
    fi
fi

# Launch evolution for each scenario
echo ""
echo -e "${GREEN}🚀 Launching evolution sessions...${NC}"
echo ""

# Create logs directory on remote
if [ "$DRY_RUN" = false ]; then
    ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR/logs"
fi

for scenario in "${SCENARIOS[@]}"; do
    SESSION_NAME="evolve_${scenario}"

    echo -n "  Launching: ${scenario}... "

    # Build command
    CMD="cd ${REMOTE_DIR} && ../bucket-brigade/.venv/bin/python scripts/evolve_scenario_expert.py \
        --scenario ${scenario} \
        --output-dir experiments/evolved_experts/${scenario} \
        --generations ${GENERATIONS} \
        --population-size ${POPULATION} \
        --workers ${WORKERS_PER_SCENARIO} \
        --early-stopping \
        --convergence-threshold 0.005 \
        --convergence-generations 10 \
        2>&1 | tee logs/evolve_${scenario}.log"

    if [ "$DRY_RUN" = false ]; then
        # Launch in tmux
        ssh "$REMOTE_HOST" "tmux new-session -d -s '${SESSION_NAME}' '${CMD}'" 2>/dev/null || {
            echo -e "${YELLOW}[session exists, killing and relaunching]${NC}"
            ssh "$REMOTE_HOST" "tmux kill-session -t '${SESSION_NAME}' 2>/dev/null || true"
            ssh "$REMOTE_HOST" "tmux new-session -d -s '${SESSION_NAME}' '${CMD}'"
        }
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}[dry-run]${NC}"
        echo "    Command: $CMD"
    fi
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Evolution Sessions Launched!                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Monitor progress:"
echo "     ./scripts/monitor_evolution.sh --host $REMOTE_HOST"
echo ""
echo "  2. Attach to a specific scenario:"
echo "     ssh $REMOTE_HOST -t 'tmux attach -t evolve_easy'"
echo ""
echo "  3. List all sessions:"
echo "     ssh $REMOTE_HOST 'tmux ls | grep evolve_'"
echo ""
echo "  4. Collect results when complete:"
echo "     ./scripts/collect_evolution_results.sh --host $REMOTE_HOST"
echo ""
echo -e "${BLUE}ℹ Estimated completion time: 30-60 minutes${NC}"
echo ""
