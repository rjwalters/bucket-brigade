# Population-Based Multi-Agent Training Guide

## Overview

This guide explains how to train multiple agents simultaneously using population-based training with PPO. This approach allows agents to learn diverse strategies by playing against each other during training.

## Architecture

```
CPU Process: Game Simulator
  - Runs Rust BucketBrigade environments
  - Manages matchmaking (round-robin or random)
  - Distributes experiences to learners
    ↓ Experience Queues (multiprocessing)

GPU Processes: Policy Learners (parallel)
  - Each trains one agent with PPO
  - On-policy learning (agent learns from own experience)
  - Periodically sends policy updates back
    ↑ Policy Update Queue

CPU Process: Policy Repository
  - Maintains latest policy for each agent
  - Used by simulator for gameplay
```

## Quick Start

### Local Training (CPU)

```bash
# Test run (4 agents, 100 episodes)
uv run python experiments/marl/train_population.py \
  --scenario trivial_cooperation \
  --population-size 4 \
  --num-episodes 100 \
  --device cpu \
  --run-name test

# Small training (8 agents, 10K episodes)
uv run python experiments/marl/train_population.py \
  --scenario trivial_cooperation \
  --population-size 8 \
  --num-episodes 10000 \
  --device cpu \
  --run-name pop_baseline_v1
```

### GPU Training (Remote)

```bash
# Medium training (16 agents, 100K episodes)
uv run python experiments/marl/train_population.py \
  --scenario mixed_motivation \
  --population-size 16 \
  --num-games 128 \
  --num-episodes 100000 \
  --hidden-size 512 \
  --device cuda \
  --run-name mixed_mot_pop16_v1

# Large training (32 agents, 1M episodes)
uv run python experiments/marl/train_population.py \
  --scenario greedy_neighbor \
  --population-size 32 \
  --num-games 256 \
  --num-episodes 1000000 \
  --hidden-size 1024 \
  --device cuda \
  --run-name greedy_pop32_v1
```

## Parameters

### Required

- **`--scenario`**: Scenario name (default: `trivial_cooperation`)
  - Options: `trivial_cooperation`, `easy`, `greedy_neighbor`, `mixed_motivation`, etc.
  - See `bucket_brigade_core/SCENARIOS` for all available scenarios

- **`--population-size`**: Number of agents in population (default: `8`)
  - **Must be >= `num-agents-per-game`** (validation enforced)
  - Larger populations = more diversity, slower training
  - Typical: 4-32 agents

### Simulation

- **`--num-games`**: Number of parallel game environments (default: `64`)
  - More games = faster experience collection
  - Balance with memory/CPU constraints
  - Typical: 32-256 games

- **`--num-agents-per-game`**: Agents per game instance (default: `4`)
  - Defined by scenario, typically 4
  - Must be <= `population-size`

- **`--num-episodes`**: Total simulation episodes (default: `10000`)
  - Total training length
  - 10K = quick test, 100K = small training, 1M+ = production

### Network Architecture

- **`--hidden-size`**: Neural network hidden layer size (default: `512`)
  - Larger = more capacity, slower training
  - Typical: 128 (small), 512 (medium), 1024 (large)

- **`--learning-rate`**: Learning rate (default: `3e-4`)
  - Standard PPO learning rate
  - Lower if training unstable

### PPO Hyperparameters

- **`--batch-size`**: Learner batch size (default: `256`)
  - Larger = more stable but slower
  - Must fit in GPU memory

- **`--num-epochs`**: PPO epochs per batch (default: `4`)
  - More epochs = more updates per batch
  - 3-10 typical for PPO

- **`--update-interval`**: Policy update frequency (default: `100`)
  - Learners send policy to simulator every N batches
  - Lower = fresher policies, higher overhead

### System

- **`--device`**: Device for learners (default: `cuda`)
  - Options: `cuda` (GPU), `cpu`
  - Use `cpu` for testing, `cuda` for production

- **`--matchmaking`**: Matchmaking strategy (default: `round_robin`)
  - `round_robin`: Fair distribution, each agent plays equally
  - `random`: Random sampling
  - Recommend `round_robin` for population training

- **`--seed`**: Random seed for reproducibility

### Logging & Checkpoints

- **`--run-name`**: Name for this training run
  - Auto-generated if not specified: `{scenario}_pop{size}_{timestamp}`

- **`--log-interval`**: Log progress every N episodes (default: `100`)

- **`--checkpoint-interval`**: Save checkpoint every N episodes (default: `1000`)

## Output Structure

```
experiments/marl/checkpoints/{run_name}/
├── config.json                # Training configuration
├── final_checkpoint.pt        # Final checkpoint
└── checkpoint_N.pt           # Intermediate checkpoints (if enabled)
```

### Checkpoint Contents

```python
checkpoint = {
    'scenario_name': str,
    'population_size': int,
    'total_episodes': int,
    'policies': {
        agent_id: state_dict,  # PyTorch state dict for each agent
        ...
    },
    'statistics': {
        'total_episodes': int,
        'total_steps': int,
        'match_counts': list[int],  # Games played per agent
        'episode_rewards': dict,     # Rewards history per agent
    },
}
```

## Remote Training

### Setup

```bash
# On remote machine
git clone https://github.com/rjwalters/bucket-brigade.git
cd bucket-brigade

# Setup environment
uv venv
source .venv/bin/activate
uv sync --extra rl

# Build Rust core
cd bucket-brigade-core
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
cd ..
```

### Launch Training

#### Option 1: Direct SSH

```bash
# From local machine
ssh remote-gpu "cd bucket-brigade && \
  uv run python experiments/marl/train_population.py \
    --scenario trivial_cooperation \
    --population-size 16 \
    --num-episodes 100000 \
    --device cuda \
    --run-name remote_test_v1 \
    2>&1 | tee logs/training_remote_test_v1.log"
```

#### Option 2: Tmux Session

```bash
# Connect and start tmux
ssh remote-gpu
tmux new -s training

# Inside tmux
cd bucket-brigade
uv run python experiments/marl/train_population.py \
  --scenario mixed_motivation \
  --population-size 32 \
  --num-episodes 1000000 \
  --device cuda \
  --run-name mixed_mot_1M_v1 \
  2>&1 | tee logs/training_mixed_mot_1M_v1.log

# Detach: Ctrl+B, D
# Training continues in background

# Reconnect later
ssh remote-gpu
tmux attach -s training
```

#### Option 3: Nohup Background

```bash
ssh remote-gpu "cd bucket-brigade && \
  nohup uv run python experiments/marl/train_population.py \
    --scenario greedy_neighbor \
    --population-size 32 \
    --num-episodes 1000000 \
    --device cuda \
    --run-name greedy_1M_v1 \
    > logs/training_greedy_1M_v1.log 2>&1 &"
```

### Monitor Training

```bash
# Watch logs
ssh remote-gpu "tail -f bucket-brigade/logs/training_*.log"

# Check GPU usage
ssh remote-gpu "nvidia-smi"

# Check running processes
ssh remote-gpu "ps aux | grep train_population"
```

### Retrieve Results

```bash
# From local machine
rsync -avz --progress \
  remote-gpu:~/bucket-brigade/experiments/marl/checkpoints/greedy_1M_v1/ \
  ./experiments/marl/checkpoints/greedy_1M_v1/
```

## Multi-Scenario Training

Train on multiple scenarios sequentially:

```bash
# Create training script
cat > run_multi_scenario.sh <<'EOF'
#!/bin/bash

SCENARIOS="trivial_cooperation easy greedy_neighbor mixed_motivation"
POP_SIZE=16
EPISODES=50000

for scenario in $SCENARIOS; do
  echo "=== Training on $scenario ==="

  uv run python experiments/marl/train_population.py \
    --scenario $scenario \
    --population-size $POP_SIZE \
    --num-episodes $EPISODES \
    --device cuda \
    --run-name ${scenario}_pop${POP_SIZE}_v1 \
    2>&1 | tee logs/training_${scenario}.log

  echo "=== Completed $scenario ==="
done

echo "All scenarios complete!"
EOF

chmod +x run_multi_scenario.sh
./run_multi_scenario.sh
```

## Tips for Production Training

### 1. Balance Population Size and Games

- **Small populations** (4-8 agents): Use 32-64 parallel games
- **Medium populations** (8-16 agents): Use 64-128 parallel games
- **Large populations** (16-32 agents): Use 128-256 parallel games

More games = faster experience collection, but more memory usage.

### 2. Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi
```

Each learner process should use ~10-20% of one GPU. With 8+ learners, you'll saturate a single GPU.

### 3. Checkpoint Regularly

For long training runs (>100K episodes), save checkpoints every 1-10K episodes:

```bash
--checkpoint-interval 5000
```

### 4. Use Larger Networks for Complex Scenarios

- **Trivial scenarios**: 128-256 hidden size
- **Medium scenarios**: 512 hidden size
- **Complex scenarios**: 1024-2048 hidden size

### 5. Training Time Estimates

Approximate wall-clock time (single GPU, 16 agents):

- **10K episodes**: ~5-10 minutes
- **100K episodes**: ~1-2 hours
- **1M episodes**: ~10-20 hours

## Troubleshooting

### "Population size must be >= num_agents_per_game"

**Fix**: Increase `--population-size` to at least match `--num-agents-per-game`:

```bash
--population-size 4 --num-agents-per-game 4  # ✅ Valid
--population-size 2 --num-agents-per-game 4  # ❌ Invalid
```

### Training hangs or deadlocks

**Symptoms**: No progress after initialization

**Fixes**:
1. Check GPU availability: `nvidia-smi`
2. Reduce `--num-games` if out of memory
3. Use `--device cpu` for testing
4. Check logs for error messages

### Out of Memory (OOM)

**Fixes**:
1. Reduce `--hidden-size` (1024 → 512 → 256)
2. Reduce `--batch-size` (256 → 128 → 64)
3. Reduce `--num-games` (256 → 128 → 64)
4. Reduce `--population-size` if possible

### Slow training

**Diagnostics**:
1. Check GPU utilization: `nvidia-smi` (should be 80-100%)
2. Check CPU utilization: `top` (simulator should use 100-200%)

**Fixes**:
1. Increase `--num-games` for more parallelism
2. Increase `--update-interval` to reduce policy update overhead
3. Use `--device cuda` instead of `cpu`

## See Also

- **Test Coverage**: `tests/test_population_training_integration.py` - Integration tests validating the pipeline
- **Implementation**: `bucket_brigade/training/population_trainer.py` - Core training logic
- **Remote Setup**: `CLAUDE.md` - Remote development workflow
- **Hyperparameter Tuning**: `docs/HYPERPARAMETER_TUNING.md` - Optuna-based tuning guide
