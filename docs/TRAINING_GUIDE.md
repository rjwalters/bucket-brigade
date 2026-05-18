# Bucket Brigade — Training Guide

## Overview

Reinforcement learning training in Bucket Brigade uses
`bucket_brigade.training.joint_trainer.JointPPOTrainer`, a multi-agent PPO
implementation backed by the Rust `bucket_brigade_core` PyO3 extension. The
trainer drives `BucketBrigadeEnv` directly (no PufferLib / Gymnasium wrapper),
which gives a ~100x speedup over the Python rollout path and lets all P3
research variants (LOLA, COMA, HCA, social influence, NHR, dense progress,
macro-actions, BC initialization, PBT, two-phase commitment, etc.) share a
single trainer.

The legacy PufferLib path (`scripts/train_puffer_gpu.py`,
`bucket_brigade.envs.puffer_env_rust`, `CurriculumTrainer`) was removed in
issue #335.

## Quick Start

### Smoke test (1 iteration, 64 rollout steps)

```bash
uv run python -m experiments.p3_specialization.train \
    --num-iterations 1 \
    --rollout-steps 64
```

### Short training run

```bash
uv run python -m experiments.p3_specialization.train \
    --num-iterations 100 \
    --rollout-steps 256
```

### Full P3 experiments

Each P3 specialist trainer lives under `experiments/p3_specialization/` and
shares a CLI shape. To list available trainers:

```bash
ls experiments/p3_specialization/train*.py
```

To run one (example):

```bash
uv run python -m experiments.p3_specialization.train_macro \
    --num-iterations 500 \
    --rollout-steps 256
```

## Installation

Install with the `rl` extra to pull in `torch`, `gymnasium`, `tensorboard`,
`optuna`, and `plotly`:

```bash
uv sync --extra rl
```

The Rust extension is built automatically; see `bucket-brigade-core/build.sh`
for the manual build path if needed.

## Where to look in the code

```
bucket_brigade/
├── envs/
│   ├── bucket_brigade_env.py     # Core env (driven directly by JointPPOTrainer)
│   └── macro_action_env.py       # Sutton-options wrapper for coarse decisions
└── training/
    ├── joint_trainer.py          # The PPO trainer everything uses
    ├── networks.py               # PolicyNetwork, TransformerPolicyNetwork, HCA
    ├── policy_learner.py         # Async learner process
    ├── population_trainer.py     # Population-based variants
    └── observation_utils.py      # Obs flattening helpers

experiments/p3_specialization/
└── train*.py                     # Per-variant entrypoints (LOLA, COMA, HCA, …)
```

## Related docs

- [POPULATION_TRAINING.md](POPULATION_TRAINING.md) — population-based training
- [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md) — Optuna-based tuning
- [PERFORMANCE.md](PERFORMANCE.md) — Rust speedup notes
