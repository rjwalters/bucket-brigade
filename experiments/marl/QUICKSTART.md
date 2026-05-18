# GPU / MARL Training Quick Reference

> **Status (2026-05, issue #335)**: The PufferLib-based `train_gpu.py`,
> `train_gpu_vectorized.py`, `train_puffer_hybrid.py`, and `train_pufferlib.py`
> trainers were removed. The active MARL training path is the Rust-backed
> `bucket_brigade.training.joint_trainer.JointPPOTrainer`, driven by the
> trainers under `experiments/p3_specialization/train*.py` and the
> population-training scripts that still live in this directory
> (`train_population.py`, `train_rust_vectorized.py`,
> `train_vectorized_population.py`).

## 🎯 First Time Setup (On GPU Instance)

```bash
# SSH into GPU instance
ssh <gpu-host>

# Clone repo
cd ~ && git clone https://github.com/rjwalters/bucket-brigade.git
cd bucket-brigade

# Install deps and build the Rust extension
uv sync --extra rl
bash bucket-brigade-core/build.sh
```

## 🚀 Start Training

```bash
# P3 specialization (preferred for current research)
uv run python -m experiments.p3_specialization.train \
    --num-iterations 1000 \
    --rollout-steps 256

# Population-based training (still under experiments/marl/)
uv run python experiments/marl/train_population.py --help
uv run python experiments/marl/train_rust_vectorized.py --help
```

## 📊 Monitor from Local Machine

```bash
cd experiments/marl

# Setup monitoring (includes TensorBoard port forwarding)
./monitor.sh

# Open TensorBoard
open http://localhost:6006

# Sync results back
./sync_results.sh
```

## 🎮 Available Scenarios

- `trivial_cooperation` — easy baseline (p_spark=0, low cooperation needed)
- `greedy_neighbor` — high self-interest pressure
- `chain_reaction` — cascading fire dynamics
- `early_containment` — quick response critical
- `sparse_heroics` — low work probability
- `rest_trap` — rest heavily penalized
- `deceptive_calm` — low initial threat
- `mixed_motivation` — balanced parameters
- `overcrowding` — many agents, coordination challenge

## ⚡ Performance Tips

1. **The Bucket Brigade env is CPU-bound** — GPU buys little; parallelism wins.
2. **Use tmux** for long-running training so disconnects don't kill the job.
3. **Download checkpoints periodically** as backup.

## 💾 Key Locations

- Code: `~/bucket-brigade/`
- TensorBoard logs: `~/bucket-brigade/experiments/marl/runs/`
- Model checkpoints: `~/bucket-brigade/experiments/marl/checkpoints/`

## 📚 Full Documentation

- Active training: [../../docs/TRAINING_GUIDE.md](../../docs/TRAINING_GUIDE.md)
- Population training: [POPULATION_TRAINING_DESIGN.md](POPULATION_TRAINING_DESIGN.md)
