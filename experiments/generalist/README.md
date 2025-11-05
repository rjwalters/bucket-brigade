# Generalist Agent Evolution

This directory contains results from evolving a single generalist agent that performs well across all scenarios.

## Overview

The generalist agent is trained simultaneously on all 9 scenarios:
- chain_reaction
- deceptive_calm
- early_containment
- greedy_neighbor
- mixed_motivation
- overcrowding
- rest_trap
- sparse_heroics
- trivial_cooperation

Unlike specialist agents (optimized for a single scenario), the generalist must balance trade-offs and find strategies that work across diverse game dynamics.

## Running Evolution

### Quick Test (Local)

Test with small parameters (~2 minutes):

```bash
uv run python experiments/scripts/run_generalist_evolution.py \
  --population 10 \
  --generations 5 \
  --games-per-scenario 2 \
  --seed 42 \
  --output-dir experiments/generalist/test
```

### Full Evolution (Remote Recommended)

**Recommended parameters** (based on timing analysis):
- Population: 200
- Generations: 12,000
- Games per scenario: 50
- Expected runtime: ~10 hours

**Run on remote machine**:

```bash
# SSH to remote machine
ssh my-gpu-server

# Start tmux session
tmux new -s generalist

# Navigate to repo
cd bucket-brigade
git pull

# Ensure Rust module is built
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop

# Run evolution with logging
uv run python experiments/scripts/run_generalist_evolution.py \
  --population 200 \
  --generations 12000 \
  --games-per-scenario 50 \
  --seed 42 \
  --output-dir experiments/generalist/evolved \
  2>&1 | tee experiments/generalist/evolution_$(date +%Y%m%d_%H%M%S).log

# Detach from tmux: Ctrl+B, D
# Reattach later: tmux attach -t generalist
```

**Monitor progress**:

```bash
# Check latest output
ssh my-gpu-server "tail -f bucket-brigade/experiments/generalist/evolution_*.log"

# Check snapshots
ssh my-gpu-server "ls -lh bucket-brigade/experiments/generalist/evolved/generation_*/"
```

### Retrieve Results

Once complete, sync results back:

```bash
# Option 1: Via git (recommended)
ssh my-gpu-server "cd bucket-brigade && git add experiments/generalist/ && git commit -m 'feat: Add generalist agent evolution results' && git push"
git pull

# Option 2: Via rsync
rsync -avz --progress \
  my-gpu-server:~/bucket-brigade/experiments/generalist/evolved/ \
  experiments/generalist/evolved/
```

## Results Structure

```
experiments/generalist/
├── README.md                    # This file
├── evolved/                     # Main results
│   ├── best_agent.json          # Best generalist agent
│   ├── evolution_trace.json     # Generation history
│   └── generation_NNNN/         # Snapshots every 10 generations
│       └── snapshot.json
└── evolution_YYYYMMDD_HHMMSS.log  # Execution logs
```

## Performance Expectations

Based on timing analysis from specialist evolution (v3):
- Single scenario: 1.53 sec/generation
- Cross-scenario (9 scenarios): ~2.3-3.1 sec/generation
- Total evaluations: 200 pop × 12,000 gen × 50 games × 9 scenarios = 108M game episodes

**Success criteria**:
- Generalist achieves >60% of specialist performance per scenario
- No catastrophic failures in any scenario
- Robust to different team compositions

## Implementation Details

The generalist evolution uses:
- `RustFitnessEvaluator` for 100x speedup over Python
- `CrossScenarioFitnessEvaluator` to evaluate across all scenarios
- Fitness = mean performance across all 9 scenarios
- Same genetic algorithm parameters as specialist evolution

See `experiments/scripts/run_generalist_evolution.py` for implementation.

## Dependencies

Requires:
- ✅ Rust module built (`bucket_brigade_core`)
- ✅ Python 3.12+ with `uv`
- ✅ All 9 scenarios configured
- ✅ Team rewards implemented in Rust (issue #119 - resolved)

## See Also

- `experiments/EVOLUTION_TIMING_ANALYSIS.md` - Performance benchmarks
- `experiments/REMOTE_EXECUTION.md` - Remote execution guide
- `experiments/scripts/run_generalist_evolution.py` - Implementation
