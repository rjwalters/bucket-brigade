# Curriculum Learning for Bucket Brigade

This document describes the curriculum learning implementation for training robust Bucket Brigade agents.

## Overview

Curriculum learning trains agents on progressively difficult scenarios, starting with simple cooperation tasks and advancing to complex multi-agent coordination challenges. This approach leads to:

- **Faster learning** - Agents learn foundational skills before tackling complex scenarios
- **Better final performance** - Progressive difficulty produces more robust policies
- **Improved generalization** - Policies trained on curriculum generalize better across scenarios
- **More stable training** - Gradual difficulty increase reduces training variance

## Quick Start

### Basic Usage

Train an agent using the default curriculum:

```bash
python scripts/train_curriculum.py
```

This will train through 5 stages:
1. **trivial_cooperation** (2 opponents, 100K steps) - Learn basic cooperation
2. **early_containment** (3 opponents, 150K steps) - Learn timing and coordination
3. **greedy_neighbor** (3 opponents, 200K steps) - Learn social dilemmas
4. **sparse_heroics** (4 opponents, 250K steps) - Learn resource allocation
5. **chain_reaction** (4 opponents, 300K steps) - Learn distributed coordination

### Custom Training Run

```bash
python scripts/train_curriculum.py \
    --run-name my_curriculum_run \
    --hidden-size 128 \
    --lr 1e-4 \
    --batch-size 4096
```

### Monitor Training

View training progress in TensorBoard:

```bash
tensorboard --logdir runs/
```

## Curriculum Design

### Default Curriculum Stages

The curriculum is designed to progressively increase difficulty along three dimensions:

1. **Number of agents** - Start with few agents (2-3), increase to many (4+)
2. **Scenario complexity** - Start with trivial scenarios, advance to complex coordination
3. **Coordination requirements** - Start with simple cooperation, advance to multi-step plans

| Stage | Scenario | Opponents | Steps | Key Learning Goal |
|-------|----------|-----------|-------|-------------------|
| 1 | trivial_cooperation | 2 | 100K | Basic fire suppression and cooperation |
| 2 | early_containment | 3 | 150K | Timing and early intervention |
| 3 | greedy_neighbor | 3 | 200K | Resource competition and social dilemmas |
| 4 | sparse_heroics | 4 | 250K | Resource allocation and heroism |
| 5 | chain_reaction | 4 | 300K | Distributed coordination and complex chains |

### Adaptive Progression

Each stage has a **progression threshold** - a target mean reward that indicates the agent has learned the skills for that stage. The curriculum:

- Evaluates the policy after each stage
- Warns if performance is below threshold
- Continues to next stage anyway (allows experimentation)
- Logs performance for later analysis

Future versions may implement automatic stage repetition or early advancement based on these thresholds.

## Training Hyperparameters

### Model Architecture

- `--hidden-size 64` - Hidden layer size (default: 64, larger for more capacity)

### PPO Hyperparameters

- `--batch-size 2048` - Number of environment steps per update (default: 2048)
- `--num-epochs 4` - PPO epochs per update (default: 4)
- `--lr 3e-4` - Learning rate (default: 3e-4)
- `--clip-epsilon 0.2` - PPO clip coefficient (default: 0.2)
- `--value-coef 0.5` - Value loss coefficient (default: 0.5)
- `--entropy-coef 0.01` - Entropy coefficient for exploration (default: 0.01)
- `--max-grad-norm 0.5` - Maximum gradient norm for clipping (default: 0.5)

### Other Options

- `--seed 42` - Random seed for reproducibility
- `--run-name NAME` - Custom name for this training run
- `--custom-curriculum` - Use custom curriculum (requires code modification)

## Output and Checkpoints

### Directory Structure

```
models/{run_name}/
├── stage_0_trivial_cooperation.pt   # Checkpoint after stage 1
├── stage_1_early_containment.pt      # Checkpoint after stage 2
├── stage_2_greedy_neighbor.pt        # Checkpoint after stage 3
├── stage_3_sparse_heroics.pt         # Checkpoint after stage 4
├── stage_4_chain_reaction.pt         # Checkpoint after stage 5
└── curriculum_final.pt               # Final trained model

runs/{run_name}/
└── events.out.tfevents.*             # TensorBoard logs
```

### Loading Trained Models

```python
import torch
from scripts.train_curriculum import PolicyNetwork

# Load a checkpoint
checkpoint = torch.load("models/my_run/curriculum_final.pt")

# Recreate policy
policy = PolicyNetwork(
    obs_dim=checkpoint["obs_dim"],
    action_dims=checkpoint["action_dims"],
    hidden_size=checkpoint["hidden_size"]
)
policy.load_state_dict(checkpoint["policy_state_dict"])
policy.eval()

# Use policy for evaluation or further training
```

## TensorBoard Metrics

### Training Metrics (logged every 100 steps)

- `train/policy_loss` - PPO policy loss
- `train/value_loss` - Value function loss
- `train/entropy` - Policy entropy (exploration)
- `train/total_loss` - Combined loss
- `train/grad_norm` - Gradient norm
- `episode/mean_reward` - Mean episode reward (rolling window)
- `episode/max_reward` - Maximum episode reward
- `episode/min_reward` - Minimum episode reward

### Curriculum Metrics (logged per stage)

- `curriculum/stage_{idx}_eval` - Evaluation reward after stage
- `curriculum/stage_{idx}_train` - Training reward during stage
- `final_eval/{scenario_name}` - Final evaluation on each scenario

## Comparison with Baseline Training

To compare curriculum learning with baseline (non-curriculum) training:

### Baseline: Train Directly on Hard Scenario

```bash
python scripts/train_simple.py \
    --scenario chain_reaction \
    --num-opponents 4 \
    --num-steps 1000000 \
    --run-name baseline_chain_reaction
```

### Curriculum: Train with Progressive Difficulty

```bash
python scripts/train_curriculum.py \
    --run-name curriculum_chain_reaction
```

### Compare Results

Both runs log to TensorBoard. Compare:

1. **Sample efficiency** - Steps to reach target performance
2. **Final performance** - Final evaluation rewards
3. **Training stability** - Variance in episode rewards over time
4. **Generalization** - Performance across multiple scenarios

Expected result: Curriculum training should reach better performance faster and with more stable learning.

## Customizing the Curriculum

### Modifying Stages

Edit `scripts/train_curriculum.py` and modify the `curriculum` list in `CurriculumTrainer.__init__()`:

```python
self.curriculum = [
    {
        "name": "trivial_cooperation",
        "num_opponents": 2,
        "steps": 100_000,
        "description": "Learn basic cooperation",
        "progression_threshold": 5.0,
    },
    # Add or modify stages here
]
```

### Available Scenarios

All scenarios from `bucket_brigade.envs.scenarios`:

- `trivial_cooperation` - Very easy cooperation
- `early_containment` - Early fire containment
- `greedy_neighbor` - Resource competition
- `sparse_heroics` - Heroic resource allocation
- `chain_reaction` - Complex fire chains
- `rest_trap` - Deceptive resting penalty
- `deceptive_calm` - Hidden danger
- `overcrowding` - Too many agents
- `mixed_motivation` - Conflicting incentives

### Adaptive Curriculum

For fully adaptive curricula that adjust based on performance:

1. Set `progression_threshold` values appropriately for each stage
2. Monitor `curriculum/stage_{idx}_eval` metrics in TensorBoard
3. Future versions will support automatic stage repetition/advancement

## Best Practices

### Training Tips

1. **Start small** - Test with shorter stages first (e.g., 10K steps each)
2. **Monitor progress** - Use TensorBoard to track learning curves
3. **Save checkpoints** - Stage checkpoints allow resuming from any point
4. **Compare baselines** - Always compare curriculum vs. direct training
5. **Tune hyperparameters** - Learning rate and batch size significantly affect results

### Hyperparameter Tuning

- **Increase `hidden_size`** for more complex scenarios (64 → 128 → 256)
- **Decrease `lr`** if training is unstable (3e-4 → 1e-4)
- **Increase `batch_size`** for more stable gradients (2048 → 4096)
- **Increase `entropy_coef`** if agent gets stuck (0.01 → 0.02)

### Computational Requirements

- **Total training time**: ~30-60 minutes on CPU for default curriculum (1M total steps)
- **GPU recommended**: For larger hidden sizes or longer training
- **Memory**: ~2-4 GB RAM sufficient for default settings

## References

### Academic Papers

- Bengio et al. (2009) "Curriculum Learning" - Original curriculum learning paper
- Narvekar et al. (2020) "Curriculum learning for reinforcement learning domains: A framework and survey" - Comprehensive survey

### Related Scripts

- `scripts/train_simple.py` - Single-scenario PPO training
- `scripts/train_policy.py` - PufferLib-based training (alternate API)

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'bucket_brigade'`
- **Solution**: Run from repository root: `python scripts/train_curriculum.py`

**Issue**: Training is very slow
- **Solution**: Reduce `--batch-size` or number of steps per stage

**Issue**: Policy doesn't improve
- **Solution**: Try lower learning rate (`--lr 1e-4`) or larger model (`--hidden-size 128`)

**Issue**: GPU out of memory
- **Solution**: Reduce `--batch-size` or `--hidden-size`

### Getting Help

- Check TensorBoard logs for anomalies
- Compare training curves with baseline training
- Try simpler scenarios first to verify implementation
- Consult `scripts/train_simple.py` for single-scenario baseline

## Future Enhancements

Potential improvements to the curriculum learning system:

- **Automatic progression** - Advance stages only when threshold reached
- **Dynamic curriculum** - Adjust difficulty based on real-time performance
- **Multi-task learning** - Train on multiple scenarios simultaneously
- **Transfer learning** - Start from pre-trained models
- **Meta-learning** - Learn how to learn across curricula
