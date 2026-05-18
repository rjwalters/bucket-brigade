# Curriculum Learning for Bucket Brigade

> **Status (2026-05)**: The `CurriculumTrainer` class and
> `scripts/train_curriculum.py` were removed in issue #335 along with the
> PufferLib training path. They had no active users — no tests, no
> experiments — and were a transitive blocker on retiring PufferLib.
>
> Curriculum-style training is still entirely feasible on top of
> `bucket_brigade.training.joint_trainer.JointPPOTrainer`: run successive
> training rounds on harder scenarios, carrying weights forward. If you need
> a packaged "curriculum trainer" again, file a new issue and re-implement
> it against `JointPPOTrainer` rather than restoring this module.

## Motivation (preserved for context)

Training agents on progressively difficult scenarios — starting with simple
cooperation tasks and advancing to complex multi-agent coordination
challenges — has the usual curriculum-learning benefits:

- **Faster learning** — agents learn foundational skills first.
- **Better final performance** — progressive difficulty produces more robust
  policies.
- **Improved generalization** — policies generalize better across scenarios.
- **More stable training** — gradual difficulty reduces training variance.

A reasonable default ordering for Bucket Brigade scenarios:

1. `trivial_cooperation` (2 opponents) — basic cooperation
2. `early_containment` (3 opponents) — timing and coordination
3. `greedy_neighbor` (3 opponents) — social dilemmas
4. `sparse_heroics` (4 opponents) — resource allocation
5. `chain_reaction` (4 opponents) — distributed coordination

## Active path

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the current PPO entrypoints
(`experiments/p3_specialization/train*.py`) and
`bucket_brigade.training.joint_trainer.JointPPOTrainer`.
