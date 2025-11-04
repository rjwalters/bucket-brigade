# Hyperparameter Tuning Guide

This guide explains how to systematically tune PPO hyperparameters for Bucket Brigade training using Optuna.

## Overview

The hyperparameter tuning infrastructure uses [Optuna](https://optuna.org/), a state-of-the-art optimization framework that employs Bayesian optimization (TPE sampler) and automatic pruning to efficiently search the hyperparameter space.

## Quick Start

Run a hyperparameter sweep with default settings:

```bash
# Basic usage - 100 trials on the default scenario
uv run python scripts/tune_hyperparameters.py

# Tune for a specific scenario
uv run python scripts/tune_hyperparameters.py --scenario hard --n-trials 50

# Parallel tuning (faster)
uv run python scripts/tune_hyperparameters.py --n-jobs 4 --n-trials 100
```

## Hyperparameter Search Space

The tuning script explores the following hyperparameters:

### Network Architecture
- **hidden_size**: [64, 128, 256, 512]
  - Size of hidden layers in the policy network
  - Larger sizes can learn more complex patterns but train slower

### PPO Training Parameters
- **learning_rate**: 1e-5 to 1e-3 (log scale)
  - Step size for gradient descent
  - Too high causes instability, too low slows learning

- **batch_size**: [512, 1024, 2048, 4096]
  - Number of environment steps per training update
  - Larger batches are more stable but slower

- **num_epochs**: 3 to 10
  - Number of gradient descent passes per batch
  - More epochs extract more learning but risk overfitting

- **clip_epsilon**: 0.1 to 0.3
  - PPO clipping parameter for policy updates
  - Controls how much the policy can change per update

- **entropy_coef**: 0.001 to 0.05 (log scale)
  - Weight for entropy bonus (encourages exploration)
  - Higher values promote more random actions

- **value_coef**: 0.3 to 0.7
  - Weight for value function loss
  - Balances policy vs value learning

- **gamma**: 0.95 to 0.999
  - Discount factor for future rewards
  - Higher values prioritize long-term rewards

- **gae_lambda**: 0.9 to 0.99
  - GAE (Generalized Advantage Estimation) parameter
  - Balances bias vs variance in advantage estimation

## Command Line Options

### Basic Options
```bash
--scenario SCENARIO        # Scenario to tune (default: default)
--num-opponents N          # Number of opponents (default: 2)
--num-steps N              # Training steps per trial (default: 50000)
--n-trials N               # Number of trials to run (default: 100)
--seed SEED                # Random seed for reproducibility (default: 42)
```

### Performance Options
```bash
--n-jobs N                 # Parallel jobs (-1 for all CPUs, default: 1)
--eval-interval N          # Steps between evaluations (default: 5000)
--pruner {median,none}     # Pruning strategy (default: median)
```

### Storage Options
```bash
--study-name NAME          # Name for the study (default: auto-generated)
--storage URL              # Database URL for persistence (default: in-memory)
```

## Advanced Usage

### Persistent Storage

Save results to a SQLite database for resumable tuning:

```bash
# Create study with database storage
uv run python scripts/tune_hyperparameters.py \
  --study-name my_hard_scenario_tune \
  --storage sqlite:///experiments/optuna_studies.db \
  --n-trials 200

# Resume the study later (runs additional trials)
uv run python scripts/tune_hyperparameters.py \
  --study-name my_hard_scenario_tune \
  --storage sqlite:///experiments/optuna_studies.db \
  --n-trials 50  # Adds 50 more trials
```

### Parallel Tuning

Run multiple trials in parallel for faster results:

```bash
# Use all CPU cores
uv run python scripts/tune_hyperparameters.py --n-jobs -1 --n-trials 200

# Use 4 parallel workers
uv run python scripts/tune_hyperparameters.py --n-jobs 4 --n-trials 100
```

### Distributed Tuning

Use shared storage to distribute tuning across multiple machines:

```bash
# Machine 1
uv run python scripts/tune_hyperparameters.py \
  --study-name distributed_tune \
  --storage postgresql://user:pass@host/db \
  --n-trials 50

# Machine 2 (same study, different trials)
uv run python scripts/tune_hyperparameters.py \
  --study-name distributed_tune \
  --storage postgresql://user:pass@host/db \
  --n-trials 50
```

## Understanding Results

### Output Files

After tuning, results are saved to `experiments/hyperparameter_tuning/`:

- `{study_name}_results.json` - Complete results including all trials
- `{study_name}_history.html` - Optimization history plot
- `{study_name}_importances.html` - Parameter importance plot
- `{study_name}_parallel.html` - Parallel coordinate plot

### Interpreting Results

The script outputs:
1. **Best trial value**: Mean reward achieved by the best configuration
2. **Best parameters**: Optimal hyperparameter values found
3. **Training command**: Ready-to-use command with best hyperparameters

Example output:
```
Best trial:
  Value (mean reward): 71.42
  Params:
    hidden_size: 128
    learning_rate: 2.0513382630874486e-05
    batch_size: 2048
    num_epochs: 8
    ...

🎯 To train with best hyperparameters, run:

uv run python scripts/train_simple.py \
  --scenario easy \
  --hidden-size 128 \
  --lr 2.05e-05 \
  --batch-size 2048 \
  --num-steps 500000
```

## Tuning Strategies

### Quick Exploration (Fast)
Get rough estimates quickly:
```bash
uv run python scripts/tune_hyperparameters.py \
  --num-steps 20000 \
  --n-trials 30 \
  --n-jobs 4
```

### Standard Tuning (Recommended)
Balanced between speed and accuracy:
```bash
uv run python scripts/tune_hyperparameters.py \
  --num-steps 50000 \
  --n-trials 100 \
  --n-jobs 4
```

### Thorough Search (Slow but Accurate)
For final production hyperparameters:
```bash
uv run python scripts/tune_hyperparameters.py \
  --num-steps 100000 \
  --n-trials 200 \
  --n-jobs 8 \
  --storage sqlite:///experiments/production_tune.db
```

### Per-Scenario Tuning
Tune separately for each scenario:
```bash
for scenario in easy default hard chain_reaction; do
  uv run python scripts/tune_hyperparameters.py \
    --scenario $scenario \
    --n-trials 100 \
    --study-name tune_${scenario} \
    --storage sqlite:///experiments/scenario_tunes.db
done
```

## Pruning

The script uses median pruning to automatically stop unpromising trials:

- **When**: After every `--eval-interval` steps
- **How**: Compares trial performance to median of past trials
- **Effect**: Saves compute by stopping bad configurations early

Disable pruning for exhaustive search:
```bash
uv run python scripts/tune_hyperparameters.py --pruner none
```

## Tips and Best Practices

### 1. Start Small
Begin with quick trials to get a sense of the landscape:
```bash
uv run python scripts/tune_hyperparameters.py --num-steps 20000 --n-trials 20
```

### 2. Use Parallel Jobs
Leverage multiple cores for faster results:
```bash
uv run python scripts/tune_hyperparameters.py --n-jobs -1
```

### 3. Tune Per Scenario
Different scenarios may need different hyperparameters:
```bash
# Tune for hard scenarios
uv run python scripts/tune_hyperparameters.py --scenario hard

# Tune for cooperative scenarios
uv run python scripts/tune_hyperparameters.py --scenario trivial_cooperation
```

### 4. Use Persistent Storage
Save studies to resume later or combine results:
```bash
uv run python scripts/tune_hyperparameters.py \
  --storage sqlite:///experiments/tunes.db \
  --study-name my_study
```

### 5. Monitor Progress
Check intermediate results in the study database or logs

### 6. Validate Results
After finding good hyperparameters, run multiple seeds:
```bash
# Run 5 seeds with best hyperparameters
for seed in 0 1 2 3 4; do
  uv run python scripts/train_simple.py \
    --scenario hard \
    --hidden-size 128 \
    --lr 2.05e-05 \
    --batch-size 2048 \
    --seed $seed \
    --num-steps 500000
done
```

## Troubleshooting

### Out of Memory
Reduce batch size or number of parallel jobs:
```bash
uv run python scripts/tune_hyperparameters.py --n-jobs 2
```

### Slow Training
Use fewer training steps per trial:
```bash
uv run python scripts/tune_hyperparameters.py --num-steps 20000
```

### Inconsistent Results
Increase number of trials or training steps:
```bash
uv run python scripts/tune_hyperparameters.py --n-trials 200 --num-steps 100000
```

### Study Already Exists
Either use a new study name or set `load_if_exists=True` (default behavior)

## Example Workflows

### Quick Test
```bash
# 20-minute quick test
uv run python scripts/tune_hyperparameters.py \
  --scenario easy \
  --num-steps 10000 \
  --n-trials 10 \
  --n-jobs 4
```

### Production Tuning
```bash
# 24-hour production run
uv run python scripts/tune_hyperparameters.py \
  --scenario hard \
  --num-steps 100000 \
  --n-trials 500 \
  --n-jobs 16 \
  --study-name production_hard \
  --storage sqlite:///experiments/production.db
```

### Multi-Scenario Campaign
```bash
# Tune all scenarios overnight
for scenario in easy default hard chain_reaction greedy_neighbor; do
  uv run python scripts/tune_hyperparameters.py \
    --scenario $scenario \
    --num-steps 50000 \
    --n-trials 100 \
    --n-jobs 4 \
    --study-name tune_${scenario} \
    --storage sqlite:///experiments/all_scenarios.db &
done
wait
```

## Related Documentation

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - General training guide
- [Optuna Documentation](https://optuna.readthedocs.io/) - Optuna framework docs
- Issue #11 - Original hyperparameter tuning issue

## Future Improvements

Potential enhancements for the tuning infrastructure:

1. **Multi-objective optimization**: Optimize for both reward and training speed
2. **Transfer learning**: Use knowledge from one scenario to speed up tuning another
3. **Conditional search spaces**: Different ranges based on scenario characteristics
4. **Warm starting**: Begin with known good configurations
5. **Ensemble methods**: Combine multiple good configurations
