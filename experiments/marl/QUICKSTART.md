# GPU Training Quick Reference

## 🎯 First Time Setup (On GPU Instance)

```bash
# SSH into GPU instance (starts in metta directory)
ssh rwalters-sandbox-2

# Clone our repo (recommend: ~/ for convenience)
cd ~
git clone https://github.com/rjwalters/bucket-brigade.git
cd bucket-brigade

# Run setup script (handles everything: deps, Rust build, verification)
./experiments/marl/setup_gpu.sh

# OR: Let setup script clone for you
cd ~
curl -O https://raw.githubusercontent.com/rjwalters/bucket-brigade/main/experiments/marl/setup_gpu.sh
chmod +x setup_gpu.sh
./setup_gpu.sh  # Will clone to ~/bucket-brigade automatically
```

## 🚀 Start Training

```bash
# Full training (10M steps, ~2.5 hours)
uv run python experiments/marl/train_gpu.py \
    --steps 10000000 \
    --scenario trivial_cooperation \
    --run-name baseline_v1

# Quick test (1M steps, ~15 minutes)
uv run python experiments/marl/train_gpu.py \
    --steps 1000000 \
    --scenario greedy_neighbor \
    --run-name quick_test
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

## 🔍 Common Commands

### On GPU Instance

```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Tail training logs
tail -f experiments/marl/runs/*/events.out.tfevents.*

# Check running processes
ps aux | grep train_gpu
```

### On Local Machine

```bash
# Port forward TensorBoard manually
ssh -L 6006:localhost:6006 rwalters-sandbox-2

# SSH and follow logs
ssh rwalters-sandbox-2 'cd bucket-brigade && tail -f experiments/marl/latest_training.log'

# Download specific run
./sync_results.sh rwalters-sandbox-2 baseline_v1
```

## 💾 Key Locations

**On GPU Instance:**
- Code: `~/bucket-brigade/`
- Runs: `~/bucket-brigade/experiments/marl/runs/`
- Checkpoints: `~/bucket-brigade/experiments/marl/checkpoints/`
- Models: `~/bucket-brigade/experiments/marl/*.pt`

**After Sync (Local):**
- Runs: `experiments/marl/runs/`
- Checkpoints: `experiments/marl/checkpoints/`
- Models: `experiments/marl/*.pt`

## 🎮 Available Scenarios

- `trivial_cooperation` - Easy baseline (p_spark=0, low cooperation needed)
- `greedy_neighbor` - High self-interest pressure
- `chain_reaction` - Cascading fire dynamics
- `early_containment` - Quick response critical
- `sparse_heroics` - Low work probability
- `rest_trap` - Rest heavily penalized
- `deceptive_calm` - Low initial threat
- `mixed_motivation` - Balanced parameters
- `overcrowding` - Many agents, coordination challenge

## ⚡ Performance Tips

1. **GPU Utilization**: Should be 80-90% during training
2. **Batch Size**: Increase if GPU memory allows (default 2048)
3. **Checkpoints**: Download periodically as backup
4. **Early Stopping**: Check TensorBoard - if not improving after 2M steps, may have converged

## 💰 Cost Management

- **g6.2xlarge (L4)**: ~$0.98/hour
- **10M step run**: ~$2.50
- **Stop instance when done**: `sky stop rwalters-sandbox-2`

## 🐛 Troubleshooting

**Training not starting:**
```bash
# Check environment
uv run python -c "from bucket_brigade.envs.puffer_env_rust import make_rust_env; print('OK')"

# Rebuild if needed
cd bucket-brigade-core
cargo clean
VIRTUAL_ENV=../.venv PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release --features python
```

**No GPU detected:**
```bash
nvidia-smi  # Should show L4 GPU
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

**Port forwarding issues:**
```bash
# Kill existing forwards
pkill -f 'ssh.*6006'

# Restart
./monitor.sh
```

## 📚 Full Documentation

See `experiments/marl/README.md` for complete documentation.
