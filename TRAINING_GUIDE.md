# Bucket Brigade - Training Guide

## Overview

This guide explains how to train reinforcement learning policies to play Bucket Brigade using the newly set up training infrastructure.

## Quick Start

### 1. Train a Policy

Train a simple policy using PPO:

```bash
uv run python scripts/train_simple.py \
    --num-steps 100000 \
    --num-opponents 3 \
    --batch-size 2048 \
    --hidden-size 128 \
    --save-path models/my_policy.pt
```

**Key Parameters:**
- `--num-steps`: Total training steps (more = better learning but slower)
- `--num-opponents`: Number of opponent agents (2-9)
- `--batch-size`: Batch size for PPO updates (larger = more stable but slower)
- `--hidden-size`: Neural network size (64-256 typical)
- `--lr`: Learning rate (default: 3e-4)

### 2. Evaluate a Trained Policy

Test your trained policy:

```bash
uv run python scripts/evaluate_simple.py \
    --model-path models/my_policy.pt \
    --num-episodes 100 \
    --num-opponents 3
```

## What We Set Up

### Dependencies

Added reinforcement learning dependencies to `pyproject.toml`:
- **PyTorch 2.9.0** - Deep learning framework
- **Gymnasium 0.29.1** - RL environment interface
- **PufferLib 1.0.1** - Multi-agent RL library
- **TensorBoard 2.20.0** - Training visualization

Install with: `uv pip install -e ".[rl]"`

### Training Infrastructure

1. **Environment Wrapper** (`bucket_brigade/envs/puffer_env.py`)
   - Gymnasium-compatible wrapper
   - Trains one agent against N opponents
   - Opponents use fixed policies (random, expert, etc.)
   - Observation space: 32-36 dim vector
   - Action space: MultiDiscrete [house (0-9), mode (0-1)]

2. **Training Script** (`scripts/train_simple.py`)
   - Vanilla PPO implementation
   - Policy network with separate action heads
   - GAE (Generalized Advantage Estimation)
   - Gradient clipping and entropy regularization
   - Automatic model checkpointing

3. **Evaluation Script** (`scripts/evaluate_simple.py`)
   - Load and test trained policies
   - Multi-episode evaluation
   - Performance statistics

## Training Results

Initial training run (50K steps, 3 opponents):
- Training time: ~30 seconds
- Mean reward: -79.82 ± 177.70
- Best episode: +276.00
- Worst episode: -455.00

**Interpretation:**
- Negative rewards indicate work cost outweighs house bonuses
- High variance is normal for short training
- Positive episodes show the policy is learning cooperation

## Next Steps

### Improve Performance

1. **Longer Training**
   ```bash
   uv run python scripts/train_simple.py --num-steps 1000000
   ```

2. **Larger Network**
   ```bash
   uv run python scripts/train_simple.py --hidden-size 256
   ```

3. **Curriculum Learning**
   - Start with 2 opponents, gradually increase
   - Train on easy scenarios first

### Advanced Training

1. **Scenario-Specific Training**
   - Modify environment to use specific scenarios
   - See `bucket_brigade/envs/scenarios.py` for 10+ scenarios

2. **Hyperparameter Tuning**
   - Adjust learning rate (`--lr`)
   - Change batch size and epochs
   - Tune PPO coefficients (entropy, value, clip)

3. **Multi-Agent Training**
   - Train multiple agents simultaneously
   - Self-play training
   - Population-based training

### Visualization

Add TensorBoard logging to track training progress:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs/{run_name}')
writer.add_scalar('train/reward', avg_reward, global_step)
```

View with: `tensorboard --logdir runs/`

## Environment Details

### Observation Space (32-36 dims)

Concatenation of:
- **Houses (10)**: State of each house [0=Safe, 1=Burning, 2=Ruined]
- **Signals (N)**: What each agent signaled [0=Rest, 1=Work]
- **Locations (N)**: Which house each agent is at [0-9]
- **Last Actions (N×2)**: Previous [house, mode] for each agent
- **Scenario Info (10)**: Game parameters (β, κ, rewards, etc.)

Where N = num_opponents + 1 (total agents)

### Action Space

**MultiDiscrete([10, 2])**:
- **House Index [0-9]**: Which house to go to
- **Mode [0-1]**: 0=Rest, 1=Fight fires

### Reward Structure

Per-night rewards include:
- **Work cost**: -5 per agent who fights fires
- **Individual bonus**: +30 per owned safe house (end of game)
- **Team bonus**: Fraction of total safe houses (end of game)

The policy must learn to:
- Balance work cost vs house preservation
- Coordinate with teammates
- Identify which houses are worth saving

## Troubleshooting

### Training is Slow
- Reduce `--batch-size`
- Reduce `--num-opponents`
- Use Rust backend: `cd bucket-brigade-core && pip install -e .`

### Policy Not Learning
- Increase `--num-steps` (try 500K-1M)
- Increase `--hidden-size` (try 256)
- Adjust learning rate
- Check reward statistics during training

### Out of Memory
- Reduce `--batch-size`
- Reduce `--hidden-size`
- Use CPU instead of GPU (automatic if no CUDA)

## File Structure

```
bucket-brigade/
├── scripts/
│   ├── train_simple.py          # Main training script
│   ├── evaluate_simple.py       # Evaluation script
│   ├── train_policy.py          # (Original, requires newer PufferLib)
│   └── train_curriculum.py      # (Original, requires newer PufferLib)
├── bucket_brigade/
│   ├── envs/
│   │   ├── puffer_env.py        # Gymnasium wrapper
│   │   ├── bucket_brigade_env.py # Core environment
│   │   └── scenarios.py         # Pre-defined scenarios
│   └── agents/
│       ├── heuristic_agent.py   # Scripted opponents
│       └── scenario_optimal/    # Expert agents
├── models/                      # Saved model checkpoints
│   └── *.pt
└── TRAINING_GUIDE.md           # This file
```

## Citation

If you use this training infrastructure, please cite:

```bibtex
@software{bucket_brigade_training,
  title = {Bucket Brigade: Multi-Agent Cooperation Training},
  author = {Your Team},
  year = {2024},
  url = {https://github.com/yourusername/bucket-brigade}
}
```

## Support

For issues or questions:
1. Check the main README.md
2. Review environment documentation in `API.md`
3. Open an issue on GitHub

## License

Same as main project license.
