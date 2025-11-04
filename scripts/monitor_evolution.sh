#!/bin/bash
# Monitor progress of parallel evolution runs on remote sandbox
#
# Usage:
#   ./scripts/monitor_evolution.sh [OPTIONS]
#
# Options:
#   --host HOST        Remote host (default: from .env or prompt)
#   --watch            Continuous monitoring (refresh every 10s)
#   --detailed         Show more log lines per scenario
#   --scenario NAME    Monitor only specific scenario
#   --help             Show this help message

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
REMOTE_HOST="${REMOTE_HOST:-}"
WATCH_MODE=false
DETAILED=false
SPECIFIC_SCENARIO=""
LOG_LINES=3

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
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --detailed)
            DETAILED=true
            LOG_LINES=10
            shift
            ;;
        --scenario)
            SPECIFIC_SCENARIO="$2"
            shift 2
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

# Function to display monitoring output
monitor_evolution() {
    clear

    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         Evolution Progress Monitor                        ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${BLUE}Remote host:${NC} $REMOTE_HOST"
    echo -e "  ${BLUE}Time:${NC}        $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Show active tmux sessions
    echo -e "${CYAN}═══ Active Evolution Sessions ═══${NC}"
    echo ""

    ACTIVE_SESSIONS=$(ssh "$REMOTE_HOST" "tmux ls 2>/dev/null | grep -c 'evolve_' || echo 0")

    if [ "$ACTIVE_SESSIONS" -eq 0 ]; then
        echo -e "  ${YELLOW}⚠ No active evolution sessions found${NC}"
        echo ""
        return
    fi

    echo -e "  ${GREEN}✓ $ACTIVE_SESSIONS active session(s)${NC}"
    echo ""

    # Get session details
    ssh "$REMOTE_HOST" "tmux ls 2>/dev/null | grep 'evolve_'" || true
    echo ""

    # Monitor specific scenario or all
    if [ -n "$SPECIFIC_SCENARIO" ]; then
        MONITOR_SCENARIOS=("$SPECIFIC_SCENARIO")
    else
        MONITOR_SCENARIOS=("${SCENARIOS[@]}")
    fi

    # Show progress for each scenario
    echo -e "${CYAN}═══ Scenario Progress ═══${NC}"
    echo ""

    for scenario in "${MONITOR_SCENARIOS[@]}"; do
        LOG_FILE="bucket-brigade/logs/evolve_${scenario}.log"

        echo -e "${BLUE}▸ ${scenario}${NC}"

        # Check if log exists
        if ! ssh "$REMOTE_HOST" "[ -f $LOG_FILE ]" 2>/dev/null; then
            echo -e "  ${YELLOW}⚠ No log file yet${NC}"
            echo ""
            continue
        fi

        # Get latest log lines
        LATEST_LOGS=$(ssh "$REMOTE_HOST" "tail -$LOG_LINES $LOG_FILE 2>/dev/null" || echo "  [No logs available]")

        # Check for completion or errors
        if echo "$LATEST_LOGS" | grep -q "Evolution Complete"; then
            echo -e "  ${GREEN}✓ COMPLETED${NC}"
        elif echo "$LATEST_LOGS" | grep -q "Error\|Traceback"; then
            echo -e "  ${RED}✗ ERROR DETECTED${NC}"
        else
            # Extract generation info if available
            GEN_INFO=$(echo "$LATEST_LOGS" | grep "Gen " | tail -1 || echo "")
            if [ -n "$GEN_INFO" ]; then
                echo -e "  ${CYAN}→ $GEN_INFO${NC}"
            fi
        fi

        # Show log excerpt
        echo "$LATEST_LOGS" | sed 's/^/    /'
        echo ""
    done

    # Summary statistics
    echo -e "${CYAN}═══ Summary ═══${NC}"
    echo ""

    COMPLETED=0
    ERRORS=0
    RUNNING=0

    for scenario in "${SCENARIOS[@]}"; do
        LOG_FILE="bucket-brigade/logs/evolve_${scenario}.log"

        if ssh "$REMOTE_HOST" "[ -f $LOG_FILE ]" 2>/dev/null; then
            CONTENT=$(ssh "$REMOTE_HOST" "tail -20 $LOG_FILE 2>/dev/null" || echo "")

            if echo "$CONTENT" | grep -q "Evolution Complete"; then
                ((COMPLETED++))
            elif echo "$CONTENT" | grep -q "Error\|Traceback"; then
                ((ERRORS++))
            else
                ((RUNNING++))
            fi
        fi
    done

    echo -e "  ${GREEN}Completed:${NC} $COMPLETED / ${#SCENARIOS[@]}"
    echo -e "  ${CYAN}Running:${NC}   $RUNNING / ${#SCENARIOS[@]}"
    echo -e "  ${RED}Errors:${NC}    $ERRORS / ${#SCENARIOS[@]}"
    echo ""

    # Progress bar
    TOTAL=${#SCENARIOS[@]}
    PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED / $TOTAL) * 100}")
    BAR_LENGTH=50
    FILLED=$(awk "BEGIN {printf \"%.0f\", ($COMPLETED / $TOTAL) * $BAR_LENGTH}")

    echo -n "  Progress: ["
    for ((i=0; i<$BAR_LENGTH; i++)); do
        if [ $i -lt $FILLED ]; then
            echo -n "█"
        else
            echo -n "░"
        fi
    done
    echo "] $PROGRESS%"
    echo ""

    if [ $COMPLETED -eq $TOTAL ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║         ✓ ALL SCENARIOS COMPLETED!                        ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "Next step: Collect results"
        echo "  ./scripts/collect_evolution_results.sh --host $REMOTE_HOST"
        echo ""
    fi
}

# Main loop
if [ "$WATCH_MODE" = true ]; then
    echo -e "${BLUE}ℹ Watch mode enabled (Ctrl+C to exit)${NC}"
    echo ""

    while true; do
        monitor_evolution

        # Check if all completed
        COMPLETED=0
        for scenario in "${SCENARIOS[@]}"; do
            LOG_FILE="bucket-brigade/logs/evolve_${scenario}.log"
            if ssh "$REMOTE_HOST" "[ -f $LOG_FILE ] && grep -q 'Evolution Complete' $LOG_FILE" 2>/dev/null; then
                ((COMPLETED++))
            fi
        done

        if [ $COMPLETED -eq ${#SCENARIOS[@]} ]; then
            echo -e "${GREEN}✓ All scenarios complete, exiting watch mode${NC}"
            exit 0
        fi

        echo -e "${YELLOW}  Refreshing in 10 seconds... (Ctrl+C to exit)${NC}"
        sleep 10
    done
else
    monitor_evolution
fi
