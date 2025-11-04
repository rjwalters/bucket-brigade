# Remote GPU Sandbox Training Guide

This guide explains how to run evolved agent training on the remote GPU sandbox with 48 vCPUs and 4 NVIDIA L4 GPUs.

## Overview

The remote sandbox (`rwalters-sandbox-1`) provides:
- **CPU**: 48 vCPUs (24 cores × 2 threads)
- **GPU**: 4× NVIDIA L4 GPUs with CUDA 12.8
- **Python**: 3.12.3 with uv package manager
- **Storage**: Fast SSD storage for checkpoints

## Quick Start

### 1. Access the Remote

The repository is already cloned at `/workspace/bucket-brigade`. SSH access is configured via MCP server.

### 2. Run Evolution Training

```bash
# SSH to remote
ssh rwalters-sandbox-1

# Navigate to project
cd /workspace/bucket-brigade

# Activate environment
source .venv/bin/activate

# Run short test (2 generations, small population)
python scripts/evolve_remote.py \
  --population-size 10 \
  --generations 2 \
  --games-per-individual 5 \
  --output-dir runs/test

# Run full training (500 generations, large population)
python scripts/evolve_remote.py \
  --num-workers 48 \
  --population-size 100 \
  --generations 500 \
  --games-per-individual 50 \
  --output-dir runs/production
```

### 3. Run in Persistent tmux Session

For long-running experiments that survive disconnects:

```bash
# Start training in tmux
./scripts/run_remote_evolution.sh my_experiment \
  --generations 1000 \
  --population-size 200

# Detach from tmux: Ctrl+B then D

# Reattach later
tmux attach -t my_experiment

# Monitor log file
tail -f runs/remote_evolution/my_experiment.log
```

## Training Script Options

### Population & Evolution
- `--population-size N`: Number of individuals (default: 100)
- `--generations N`: Number of generations (default: 500)
- `--games-per-individual N`: Games per fitness evaluation (default: 50)
- `--elite-size N`: Number of elite individuals to preserve (default: 10)

### Hardware
- `--num-workers N`: Parallel workers for fitness evaluation (default: 48)
  - Set to match vCPU count for maximum throughput
  - Python evaluator doesn't use parallelism (Rust bindings required)

### Genetic Operators
- `--selection {tournament,roulette,rank}`: Selection strategy
- `--crossover {uniform,single_point,arithmetic}`: Crossover method
- `--crossover-rate FLOAT`: Crossover probability (0.0-1.0)
- `--mutation {gaussian,uniform,adaptive}`: Mutation strategy
- `--mutation-rate FLOAT`: Per-gene mutation probability
- `--mutation-scale FLOAT`: Mutation magnitude

### Fitness Function
- `--fitness-type {mean_reward,win_rate,robustness,multi_objective}`
  - `mean_reward`: Average reward across games (default)
  - `win_rate`: Fraction of successful games
  - `robustness`: Performance across diverse scenarios
  - `multi_objective`: Weighted combination of metrics

### Checkpointing & Output
- `--checkpoint-interval N`: Save checkpoint every N generations (default: 10)
- `--output-dir PATH`: Output directory (default: runs/remote_evolution)
- `--resume PATH`: Resume from checkpoint file
- `--seed N`: Random seed for reproducibility

## Output Files

Training produces the following outputs in `--output-dir`:

```
runs/remote_evolution/
├── evolution_TIMESTAMP.log              # Training log
├── evolution_TIMESTAMP_final.json       # Final results
├── checkpoint_gen0010.json              # Checkpoints (every 10 gens)
├── checkpoint_gen0020.json
└── ...
```

### Final Results File

Contains:
- Best individual genome and fitness
- Final population statistics
- Fitness history across generations
- Diversity metrics
- Training configuration
- Elapsed time

### Checkpoint Files

Enable resuming interrupted training:

```bash
# Resume from checkpoint
python scripts/evolve_remote.py \
  --resume runs/remote_evolution/checkpoint_gen0100.json \
  --generations 500  # Will run 400 more generations
```

## Performance Notes

### Rust vs Python Evaluator

The codebase supports two fitness evaluators:

1. **Rust Evaluator** (100x faster, parallel)
   - Requires `bucket_brigade_core` Rust bindings
   - Uses multiprocessing with `--num-workers` parallel processes
   - ~100x faster than Python implementation
   - **Status**: Currently not built on remote (import errors)

2. **Python Evaluator** (slower, fallback)
   - Pure Python implementation
   - Sequential evaluation (no parallelism)
   - Used automatically when Rust bindings unavailable
   - Sufficient for small experiments

### Building Rust Bindings (Optional)

To enable the fast Rust evaluator:

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install maturin and cffi
uv pip install maturin cffi

# Build Rust bindings
cd bucket-brigade-core
maturin develop --release
```

Once built, the evolution script will automatically use the Rust evaluator.

### CPU Utilization

With 48 vCPUs:
- **Rust evaluator**: Near-linear scaling with `--num-workers 48`
- **Python evaluator**: Single-threaded (1 vCPU utilized)

For maximum throughput, build Rust bindings or run multiple independent experiments in parallel.

## Monitoring Training

### Live Progress

View training progress in real-time:

```bash
# Tail log file
tail -f runs/remote_evolution/evolution_*.log

# Or attach to tmux session
tmux attach -t my_experiment
```

### Generation Output

Each generation logs:
```
Gen  123/500 | Best:  15.234 | Mean:   8.456 | Std:  2.123 | Diversity: 0.856
```

- **Best**: Highest fitness in population
- **Mean**: Average fitness
- **Std**: Fitness standard deviation
- **Diversity**: Population genetic diversity

### Checkpoints

Checkpoints save every 10 generations (configurable):
```
💾 Checkpoint saved: runs/remote_evolution/checkpoint_gen0100.json
```

## Best Practices

### For Quick Experiments
```bash
python scripts/evolve_remote.py \
  --population-size 20 \
  --generations 50 \
  --games-per-individual 10 \
  --output-dir runs/quick_test
```
Runtime: ~5-15 minutes (Python evaluator)

### For Production Training
```bash
./scripts/run_remote_evolution.sh production_run \
  --population-size 100 \
  --generations 500 \
  --games-per-individual 50 \
  --checkpoint-interval 25
```
Runtime: Several hours (depends on evaluator)

### For Long-Running Experiments

Always use tmux for multi-hour runs:
- Survives SSH disconnects
- Allows reattaching to monitor progress
- Can manage multiple experiments in parallel

## Troubleshooting

### Training Crashes

If training crashes, resume from last checkpoint:
```bash
python scripts/evolve_remote.py \
  --resume runs/remote_evolution/checkpoint_gen0100.json
```

### Out of Memory

Reduce population or batch size:
```bash
python scripts/evolve_remote.py \
  --population-size 50  # Instead of 100
```

### Slow Training

- Build Rust bindings for 100x speedup
- Reduce `--games-per-individual` for faster (noisier) evaluation
- Use smaller `--population-size`

### tmux Session Lost

List and reattach:
```bash
# List all sessions
tmux ls

# Attach to session
tmux attach -t my_experiment
```

## Example Workflows

### Baseline Experiment

Establish baseline performance:
```bash
python scripts/evolve_remote.py \
  --seed 42 \
  --population-size 50 \
  --generations 100 \
  --output-dir runs/baseline
```

### Parameter Sweep

Test different mutation rates:
```bash
for rate in 0.05 0.1 0.2; do
  python scripts/evolve_remote.py \
    --mutation-rate $rate \
    --output-dir runs/mutation_$rate &
done
wait
```

### Resume and Extend

Continue previous training:
```bash
# Original run: 100 generations
python scripts/evolve_remote.py \
  --generations 100 \
  --output-dir runs/experiment1

# Extend to 200 generations
python scripts/evolve_remote.py \
  --resume runs/experiment1/checkpoint_gen0100.json \
  --generations 200
```

## Remote Access via MCP

This project includes MCP (Model Context Protocol) server for remote access:

```python
# In Claude Code, use MCP tools:
# - mcp__remote-ssh__remote_bash: Execute commands
# - mcp__remote-ssh__remote_file_read: Read files
# - mcp__remote-ssh__remote_bash_output: Check background jobs
```

See `MCP_SETUP.md` for configuration details.

## Next Steps

1. **Run test evolution** to verify setup
2. **Build Rust bindings** for 100x speedup (optional)
3. **Launch production training** in tmux
4. **Monitor progress** via logs
5. **Analyze results** from final JSON files

For questions or issues, check the main `README.md` or open an issue on GitHub.
