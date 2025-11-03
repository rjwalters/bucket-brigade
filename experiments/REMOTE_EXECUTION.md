# Remote Execution Guide

Instructions for running research experiments on a fast remote machine.

## Quick Start

```bash
# 1. Clone repository on remote machine
git clone https://github.com/rjwalters/bucket-brigade.git
cd bucket-brigade

# 2. Setup environment
uv venv
source .venv/bin/activate  # or: .venv/bin/activate on Linux
uv sync

# 3. Run automated research refresh
./experiments/scripts/refresh_all_research_auto.sh
```

## Scripts Available

### 1. `refresh_all_research.sh` (Interactive)

Runs complete research pipeline with confirmation prompt.

```bash
./experiments/scripts/refresh_all_research.sh
```

**Default Configuration:**
- Population: 100
- Generations: 100
- Games (evolution): 30
- Games (heuristics): 100
- Games (comparison): 50

**Customization:**
```bash
# Override specific parameters
POPULATION=200 GENERATIONS=200 ./experiments/scripts/refresh_all_research.sh

# Quick test run
POPULATION=20 GENERATIONS=20 GAMES_EVOLUTION=10 GAMES_HEURISTICS=20 GAMES_COMPARISON=10 \
  ./experiments/scripts/refresh_all_research.sh
```

### 2. `refresh_all_research_auto.sh` (Automated)

Non-interactive version perfect for remote/unattended execution.

```bash
# Run with defaults
./experiments/scripts/refresh_all_research_auto.sh

# Run specific scenarios only
SCENARIOS="greedy_neighbor,trivial_cooperation" ./experiments/scripts/refresh_all_research_auto.sh

# Production run with high-quality settings
POPULATION=200 GENERATIONS=200 GAMES_EVOLUTION=50 GAMES_HEURISTICS=200 GAMES_COMPARISON=100 \
  ./experiments/scripts/refresh_all_research_auto.sh
```

**Features:**
- No user prompts (fully automated)
- Progress tracking
- Error handling and retry logic
- Filters out gym deprecation warnings
- Detailed timing statistics

### 3. `run_all_extended_evolution.sh` (Evolution Only)

Run only the evolution step (faster, useful for testing evolution improvements).

```bash
./experiments/scripts/run_all_extended_evolution.sh
```

## Recommended Workflows

### Development/Testing

Quick validation run (~30-60 minutes):

```bash
POPULATION=50 \
GENERATIONS=50 \
GAMES_EVOLUTION=20 \
GAMES_HEURISTICS=50 \
GAMES_COMPARISON=20 \
./experiments/scripts/refresh_all_research_auto.sh
```

### Production Research

High-quality results (~3-6 hours on fast machine):

```bash
POPULATION=200 \
GENERATIONS=200 \
GAMES_EVOLUTION=50 \
GAMES_HEURISTICS=200 \
GAMES_COMPARISON=100 \
./experiments/scripts/refresh_all_research_auto.sh
```

### Overnight Run

Maximum quality (~8-12 hours):

```bash
POPULATION=500 \
GENERATIONS=500 \
GAMES_EVOLUTION=100 \
GAMES_HEURISTICS=500 \
GAMES_COMPARISON=200 \
./experiments/scripts/refresh_all_research_auto.sh
```

## Remote Machine Setup

### SSH and Run in Background

```bash
# SSH to remote machine
ssh user@remote-host

# Start tmux/screen session (so it continues if disconnected)
tmux new -s research

# Clone and setup
git clone https://github.com/rjwalters/bucket-brigade.git
cd bucket-brigade
uv venv && source .venv/bin/activate && uv sync

# Run research (logs to file)
./experiments/scripts/refresh_all_research_auto.sh 2>&1 | tee research_$(date +%Y%m%d_%H%M%S).log

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t research
```

### Using nohup (Alternative)

```bash
nohup ./experiments/scripts/refresh_all_research_auto.sh > research.log 2>&1 &

# Check progress
tail -f research.log

# Check if still running
ps aux | grep refresh_all_research
```

## Retrieving Results

### Option 1: Commit and Push

```bash
# On remote machine after completion
git add experiments/scenarios/
git commit -m "research: High-quality data from remote execution"
git push

# On local machine
git pull
```

### Option 2: Direct Copy with rsync

```bash
# From local machine
rsync -avz --progress \
  user@remote-host:~/bucket-brigade/experiments/scenarios/ \
  ./experiments/scenarios/
```

### Option 3: Archive and Download

```bash
# On remote machine
tar czf research_results_$(date +%Y%m%d).tar.gz experiments/scenarios/

# On local machine
scp user@remote-host:~/bucket-brigade/research_results_*.tar.gz .
tar xzf research_results_*.tar.gz
```

## Monitoring Progress

### Check Current Scenario

```bash
# Find which scenario is currently processing
ps aux | grep python | grep experiments/scripts

# Check evolution progress (shows generation number)
tail -f logs/evolution_*.log  # if logging enabled
```

### Estimate Time Remaining

Approximate times per scenario (on modern CPU):
- Heuristics (100 games): 2-5 minutes
- Evolution (100 pop, 100 gen, 30 games): 20-40 minutes
- Comparison (50 games): 1-3 minutes

**Total per scenario**: ~25-50 minutes
**All 9 scenarios**: ~4-8 hours

## Hardware Recommendations

### Minimum
- CPU: 4 cores
- RAM: 8GB
- Time: ~12 hours for production run

### Recommended
- CPU: 8+ cores (multiprocessing helps)
- RAM: 16GB
- SSD storage
- Time: ~4-6 hours for production run

### Optimal
- CPU: 16+ cores
- RAM: 32GB
- NVMe SSD
- Time: ~2-3 hours for production run

## Troubleshooting

### Out of Memory

Reduce population or games:
```bash
POPULATION=50 GAMES_EVOLUTION=20 ./experiments/scripts/refresh_all_research_auto.sh
```

### Slow Performance

Check CPU usage and consider reducing parallelism in the evolution code, or run scenarios sequentially.

### Failed Scenarios

Check logs for specific error, then re-run just the failed scenarios:
```bash
SCENARIOS="failed_scenario1,failed_scenario2" ./experiments/scripts/refresh_all_research_auto.sh
```

## Expected Output Structure

```
experiments/scenarios/{scenario}/
├── config.json
├── README.md
├── heuristics/
│   └── results.json          # Heuristic archetype performance
├── evolved/
│   ├── best_agent.json       # Best evolved agent
│   ├── evolution_trace.json  # Generation-by-generation history
│   └── generation_*/         # Population snapshots
│       └── snapshot.json
└── comparison/
    └── comparison.json       # Head-to-head tournament results
```

## Data Quality Indicators

Check these to validate results:

1. **Evolution Convergence**: Check `evolution_trace.json` - best fitness should improve over generations
2. **Diversity**: Should stay above 0.1 (check trace)
3. **Performance Gap**: Compare evolved vs heuristic in `comparison.json`
4. **Consistency**: Std deviation in payoffs should be reasonable

## Next Steps After Completion

1. **Review results**: Use web dashboard or manual inspection
2. **Commit data**: `git add experiments/ && git commit && git push`
3. **Analyze patterns**: Look for scenarios where evolution beats heuristics
4. **Iterate**: Adjust evolution parameters if needed and re-run

---

*Last updated: 2025-11-03*
