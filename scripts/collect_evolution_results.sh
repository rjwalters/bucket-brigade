#!/bin/bash
# Collect evolution results from remote sandbox
#
# Downloads evolved genomes, logs, and fitness histories from all scenarios
# and organizes them in experiments/evolved_experts/
#
# Usage:
#   ./scripts/collect_evolution_results.sh [OPTIONS]
#
# Options:
#   --host HOST        Remote host (default: from .env or prompt)
#   --output-dir DIR   Local output directory (default: experiments/evolved_experts)
#   --scenario NAME    Collect only specific scenario
#   --force            Overwrite existing local files
#   --verify           Verify collected files after download
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
OUTPUT_DIR="experiments/evolved_experts"
SPECIFIC_SCENARIO=""
FORCE=false
VERIFY=true

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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --scenario)
            SPECIFIC_SCENARIO="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --no-verify)
            VERIFY=false
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

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Evolution Results Collection                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Remote host:${NC}   $REMOTE_HOST"
echo -e "  ${BLUE}Output dir:${NC}    $OUTPUT_DIR"
echo -e "  ${BLUE}Scenarios:${NC}     ${SPECIFIC_SCENARIO:-all (${#SCENARIOS[@]})}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine which scenarios to collect
if [ -n "$SPECIFIC_SCENARIO" ]; then
    COLLECT_SCENARIOS=("$SPECIFIC_SCENARIO")
else
    COLLECT_SCENARIOS=("${SCENARIOS[@]}")
fi

# Check SSH connection
echo -n "Checking SSH connection... "
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$REMOTE_HOST" "echo ok" &>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}Error: Cannot connect to $REMOTE_HOST${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}═══ Collecting Results ═══${NC}"
echo ""

COLLECTED=0
FAILED=0
SKIPPED=0

for scenario in "${COLLECT_SCENARIOS[@]}"; do
    REMOTE_DIR="bucket-brigade/experiments/evolved_experts/${scenario}"
    LOCAL_DIR="${OUTPUT_DIR}/${scenario}"

    echo -n "  ${scenario}... "

    # Check if remote directory exists
    if ! ssh "$REMOTE_HOST" "[ -d $REMOTE_DIR ]" 2>/dev/null; then
        echo -e "${YELLOW}[no data]${NC}"
        ((SKIPPED++))
        continue
    fi

    # Check if local directory exists and --force not set
    if [ -d "$LOCAL_DIR" ] && [ "$FORCE" = false ]; then
        echo -e "${YELLOW}[exists, use --force to overwrite]${NC}"
        ((SKIPPED++))
        continue
    fi

    # Create local directory
    mkdir -p "$LOCAL_DIR"

    # Download all files from scenario directory
    if rsync -az --progress "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/" &>/dev/null; then
        # Also collect the log file
        REMOTE_LOG="bucket-brigade/logs/evolve_${scenario}.log"
        if ssh "$REMOTE_HOST" "[ -f $REMOTE_LOG ]" 2>/dev/null; then
            scp -q "$REMOTE_HOST:$REMOTE_LOG" "$LOCAL_DIR/" 2>/dev/null || true
        fi

        echo -e "${GREEN}✓${NC}"
        ((COLLECTED++))
    else
        echo -e "${RED}✗ (download failed)${NC}"
        ((FAILED++))
    fi
done

echo ""
echo -e "${CYAN}═══ Collection Summary ═══${NC}"
echo ""
echo -e "  ${GREEN}Collected:${NC} $COLLECTED"
echo -e "  ${YELLOW}Skipped:${NC}   $SKIPPED"
echo -e "  ${RED}Failed:${NC}    $FAILED"
echo ""

# Verify collected files
if [ "$VERIFY" = true ] && [ $COLLECTED -gt 0 ]; then
    echo -e "${CYAN}═══ Verifying Results ═══${NC}"
    echo ""

    for scenario in "${COLLECT_SCENARIOS[@]}"; do
        LOCAL_DIR="${OUTPUT_DIR}/${scenario}"

        if [ ! -d "$LOCAL_DIR" ]; then
            continue
        fi

        echo -n "  ${scenario}... "

        # Check for required files
        ERRORS=0

        if [ ! -f "$LOCAL_DIR/best_genome.json" ]; then
            echo -e "${RED}✗ missing best_genome.json${NC}"
            ((ERRORS++))
        fi

        if [ ! -f "$LOCAL_DIR/fitness_history.json" ]; then
            echo -e "${YELLOW}⚠ missing fitness_history.json${NC}"
        fi

        if [ ! -f "$LOCAL_DIR/config.json" ]; then
            echo -e "${YELLOW}⚠ missing config.json${NC}"
        fi

        # Validate best_genome.json is valid JSON
        if [ -f "$LOCAL_DIR/best_genome.json" ]; then
            if ! python3 -m json.tool "$LOCAL_DIR/best_genome.json" &>/dev/null; then
                echo -e "${RED}✗ best_genome.json is not valid JSON${NC}"
                ((ERRORS++))
            fi
        fi

        if [ $ERRORS -eq 0 ]; then
            # Count files
            FILE_COUNT=$(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ')
            SIZE=$(du -sh "$LOCAL_DIR" | cut -f1)
            echo -e "${GREEN}✓${NC} ($FILE_COUNT files, $SIZE)"
        fi
    done

    echo ""
fi

# Final summary
if [ $COLLECTED -eq ${#COLLECT_SCENARIOS[@]} ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         ✓ ALL RESULTS COLLECTED SUCCESSFULLY!             ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results:"
    echo "     python experiments/scripts/analyze_evolved_experts.py"
    echo ""
    echo "  2. View best genomes:"
    echo "     cat experiments/evolved_experts/*/best_genome.json"
    echo ""
    echo "  3. Compare across scenarios:"
    echo "     python scripts/compare_evolved_experts.py"
    echo ""
elif [ $COLLECTED -gt 0 ]; then
    echo -e "${YELLOW}⚠ Partial collection: $COLLECTED / ${#COLLECT_SCENARIOS[@]} scenarios${NC}"
    echo ""
    echo "Some scenarios may still be running. Check status:"
    echo "  ./scripts/monitor_evolution.sh --host $REMOTE_HOST"
    echo ""
elif [ $SKIPPED -gt 0 ]; then
    echo -e "${YELLOW}⚠ All scenarios already collected (use --force to re-download)${NC}"
    echo ""
else
    echo -e "${RED}✗ No results collected${NC}"
    echo ""
    echo "Possible reasons:"
    echo "  - Evolution runs haven't started"
    echo "  - Remote directory doesn't exist"
    echo "  - No data generated yet"
    echo ""
    echo "Check evolution status:"
    echo "  ./scripts/monitor_evolution.sh --host $REMOTE_HOST"
    echo ""
fi
