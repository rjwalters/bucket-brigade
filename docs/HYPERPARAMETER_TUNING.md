# Hyperparameter Tuning Guide

> **Status (2026-05, issues #335 / #340)**: `scripts/tune_hyperparameters.py`
> was removed along with the PufferLib training path. This document is now
> **reference-only**: the search space, parameter ranges, and tuning strategy
> notes below are kept for posterity, but the runnable orchestration script
> no longer exists and the executable code examples have been stripped. If
> hyperparameter tuning becomes a priority again, re-implement the harness
> against `bucket_brigade.training.joint_trainer.JointPPOTrainer` and file a
> new issue.

This guide describes the Optuna-based search space that was used to tune PPO
hyperparameters for Bucket Brigade training. It is preserved as a reference
for the parameter ranges and tuning strategy.

## Overview

The hyperparameter tuning infrastructure used [Optuna](https://optuna.org/), a state-of-the-art optimization framework that employs Bayesian optimization (TPE sampler) and automatic pruning to efficiently search the hyperparameter space.

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

## Command Line Options (historical)

The retired `scripts/tune_hyperparameters.py` accepted the following flag
groups. They are reproduced here as design notes for any future
re-implementation against `JointPPOTrainer`:

- **Basic**: `--scenario`, `--num-opponents`, `--num-steps`, `--n-trials`, `--seed`.
- **Performance**: `--n-jobs`, `--eval-interval`, `--pruner {median,none}`.
- **Storage**: `--study-name`, `--storage` (e.g. SQLite or PostgreSQL URL for
  resumable / distributed studies).

## Advanced Usage (historical notes)

- **Persistent storage** — point Optuna at a SQLite or PostgreSQL backend so
  studies can be paused, resumed, or extended with additional trials.
- **Parallel tuning** — drive multiple trials in parallel with `--n-jobs`
  (use `-1` for all CPU cores).
- **Distributed tuning** — point multiple machines at the same backing store
  via a shared study name to fan trials out across hosts.

## Understanding Results

### Output Files

After tuning, results are saved to `experiments/hyperparameter_tuning/`:

- `{study_name}_results.json` - Complete results including all trials
- `{study_name}_history.html` - Optimization history plot
- `{study_name}_importances.html` - Parameter importance plot
- `{study_name}_parallel.html` - Parallel coordinate plot

### Interpreting Results

The retired script reported:
1. **Best trial value**: Mean reward achieved by the best configuration.
2. **Best parameters**: Optimal hyperparameter values found.
3. **Training command**: A ready-to-use training command pre-filled with the
   best hyperparameters (the script that this command pointed at has also
   been removed; a future re-implementation should target `JointPPOTrainer`).

## Tuning Strategies (historical notes)

- **Quick exploration** — short training budget (~20k steps), small trial
  count (~30), parallelism `4`.
- **Standard tuning** — medium training budget (~50k steps), ~100 trials,
  parallelism `4`.
- **Thorough search** — long training budget (~100k+ steps), ~200 trials,
  high parallelism, persistent storage.
- **Per-scenario tuning** — run a separate study per scenario so that
  scenario-specific optima are not averaged out.

## Pruning

Median pruning was used to terminate unpromising trials:

- **When**: After every `--eval-interval` steps
- **How**: Compares trial performance to median of past trials
- **Effect**: Saves compute by stopping bad configurations early

## Tips and Best Practices

1. **Start small** — short trials first to read the landscape.
2. **Use parallel jobs** — match `--n-jobs` to available cores.
3. **Tune per scenario** — different scenarios may want different
   hyperparameters.
4. **Use persistent storage** — so studies can be resumed or combined.
5. **Monitor progress** — check intermediate results in the study DB / logs.
6. **Validate results** — re-run with multiple seeds before declaring a
   configuration "best".

## Troubleshooting

- **Out of memory** — reduce batch size or parallelism.
- **Slow training** — use fewer training steps per trial.
- **Inconsistent results** — increase trial count and / or steps per trial.
- **Study already exists** — either pick a new study name or rely on
  `load_if_exists=True`.

## Example Workflows (historical notes)

- **Quick test** — ~10k steps × ~10 trials × 4 workers for a ~20-minute smoke
  run.
- **Production tuning** — ~100k steps × ~500 trials × 16 workers backed by a
  SQLite store for a multi-hour run.
- **Multi-scenario campaign** — a `for scenario in ...; do ...; done & wait`
  loop fanning per-scenario studies into a shared SQLite database.

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
