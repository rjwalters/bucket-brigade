# Multi-Agent Reinforcement Learning (MARL) Experiments

PPO training for Bucket Brigade using GPU-accelerated Rust-backed environments.

## 🚀 Quick Start

### On GPU Instance (rwalters-sandbox-2)

```bash
# 1. Clone and setup (first time only)
git clone https://github.com/rjwalters/bucket-brigade.git
cd bucket-brigade
chmod +x experiments/marl/setup_gpu.sh
./experiments/marl/setup_gpu.sh

# 2. Start training
uv run python experiments/marl/train_gpu.py \
    --steps 10000000 \
    --scenario trivial_cooperation \
    --run-name baseline_v1
```

### On Local Machine

```bash
# Monitor training (sets up TensorBoard port forwarding)
cd experiments/marl
chmod +x monitor.sh sync_results.sh
./monitor.sh rwalters-sandbox-2

# View TensorBoard
open http://localhost:6006

# Sync results back
./sync_results.sh rwalters-sandbox-2
```

## 📁 Scripts Overview

### Remote Scripts (Run on GPU Instance)

**`setup_gpu.sh`** - One-time setup script
- Installs dependencies (uv, Python packages)
- Builds Rust core with PyO3 bindings
- Verifies CUDA and environment setup
- Tests environment creation

**`train_gpu.py`** - Main training script
- Uses `RustPufferBucketBrigade` environment (100x faster!)
- Implements PPO with proper checkpointing
- Logs to TensorBoard and console
- Saves checkpoints every 100K steps
- Full configurability via CLI arguments

### Local Scripts (Run on Your Machine)

**`monitor.sh`** - Training monitoring helper
- Sets up SSH port forwarding for TensorBoard (port 6006)
- Shows remote training status
- Displays GPU utilization
- Lists recent checkpoints

**`sync_results.sh`** - Results synchronization
- Downloads TensorBoard runs
- Downloads model checkpoints
- Downloads final models
- Supports syncing specific runs or all results

## 🎯 Training Examples

### Basic Training (10M steps, ~2-3 hours on L4)
```bash
uv run python experiments/marl/train_gpu.py \
    --steps 10000000 \
    --scenario trivial_cooperation \
    --run-name baseline_trivial
```

### Quick Test (1M steps, ~15 minutes)
```bash
uv run python experiments/marl/train_gpu.py \
    --steps 1000000 \
    --scenario greedy_neighbor \
    --run-name quick_test
```

### Custom Configuration
```bash
uv run python experiments/marl/train_gpu.py \
    --steps 5000000 \
    --scenario chain_reaction \
    --opponents 5 \
    --batch-size 4096 \
    --lr 1e-4 \
    --run-name custom_config
```

### All Available Options
```bash
uv run python experiments/marl/train_gpu.py --help
```

Options:
- `--steps`: Total training steps (default: 10M)
- `--scenario`: Scenario name (default: trivial_cooperation)
  - Available: trivial_cooperation, greedy_neighbor, chain_reaction, etc.
- `--opponents`: Number of opponent agents (default: 3)
- `--run-name`: Custom name for this run
- `--batch-size`: PPO batch size (default: 2048)
- `--epochs`: PPO epochs per batch (default: 4)
- `--lr`: Learning rate (default: 3e-4)
- `--checkpoint-interval`: Steps between checkpoints (default: 100K)
- `--eval-interval`: Steps between eval logging (default: 10K)
- `--cpu`: Force CPU training (default: auto-detect GPU)

## 📊 Monitoring Training

### Real-Time Monitoring

**TensorBoard** (recommended):
```bash
# On local machine
./monitor.sh rwalters-sandbox-2
open http://localhost:6006
```

**Console Logs**:
```bash
# SSH and tail logs
ssh rwalters-sandbox-2
cd bucket-brigade
tail -f experiments/marl/runs/*/events.out.tfevents.*
```

**GPU Utilization**:
```bash
ssh rwalters-sandbox-2 'watch -n 1 nvidia-smi'
```

### Training Metrics

TensorBoard tracks:
- **episode/reward**: Reward per episode
- **episode/length**: Episode length
- **train/avg_reward_100ep**: Rolling average (100 episodes)
- **train/steps_per_sec**: Training throughput
- **train/total_episodes**: Cumulative episode count

## 💾 Checkpoints & Models

### Checkpoints
- Saved every 100K steps (configurable with `--checkpoint-interval`)
- Location: `experiments/marl/checkpoints/{run_name}/checkpoint_step_{N}.pt`
- Contains:
  - Model weights
  - Optimizer state
  - Training step
  - Episode history

### Final Models
- Saved at end of training
- Location: `experiments/marl/model_{run_name}.pt`
- Contains: Final policy weights only

### Loading a Checkpoint
```python
import torch
from bucket_brigade.training import PolicyNetwork

# Load checkpoint
checkpoint = torch.load('experiments/marl/checkpoints/baseline_v1/checkpoint_step_1000000.pt')

# Create policy and load weights
policy = PolicyNetwork(obs_dim=32, action_dims=[10, 2])
policy.load_state_dict(checkpoint['model_state_dict'])

# Resume training if needed
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_step = checkpoint['step']
```

## 🎮 Environment Details

### RustPufferBucketBrigade
- **100x faster** than Python implementation
- Rust core for simulation
- PufferLib-compatible interface
- GPU-ready (observations/actions on GPU)

### Observation Space
- **Shape**: (32,)
- **Components**:
  - Houses (10): Fire state per house
  - Signals (N): Agent signals
  - Locations (N): Agent locations
  - Last actions (2N): Previous actions
  - Scenario info (10): Game parameters

### Action Space
- **MultiDiscrete**: [10, 2]
  - House: Which house to visit (0-9)
  - Mode: REST (0) or WORK (1)

## 📈 Expected Results

### Baseline Performance
Based on universal Nash equilibrium research:

- **Random opponents**: Expect reward ~50-70
- **Evolved opponents**: Expect reward ~60-65
- **Training time** (10M steps on L4): ~2-3 hours

### Success Criteria
1. **Convergence**: Reward stabilizes after 1-2M steps
2. **Performance**: Matches or exceeds evolved heuristics
3. **Generalization**: Works across multiple scenarios

### Comparison to Universal Equilibrium
Our research found a universal free-riding strategy dominates across all scenarios. Key question: **Can PPO learn to beat it?**

## 🔬 Research Questions

1. **Baseline vs Evolved**: Can neural policies match heuristic evolution?
2. **Generalization**: Do policies transfer across scenarios?
3. **Opponent Diversity**: How does training against different opponents affect performance?
4. **Nash Equilibrium**: Does PPO converge to the free-riding equilibrium?

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check drivers
nvidia-smi
```

### Build Failures
```bash
# Clean rebuild
cd bucket-brigade-core
cargo clean
VIRTUAL_ENV=../. venv PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
    maturin develop --release --features python
```

### Import Errors
```bash
# Verify installation
python -c "from bucket_brigade.envs.puffer_env_rust import make_rust_env; print('OK')"
python -c "import bucket_brigade_core; print('OK')"
```

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Increase batch size if GPU memory available
- Verify using Rust environment (not Python)

## 📚 Related Documentation

- **Main Training Guide**: `../../TRAINING_GUIDE.md`
- **Hyperparameter Tuning**: `../../docs/HYPERPARAMETER_TUNING.md`
- **Research Summary**: `../RESEARCH_SUMMARY.md`
- **Remote Execution**: `../REMOTE_EXECUTION.md`

## 💡 Tips

1. **Start small**: Test with 1M steps before full 10M run
2. **Name your runs**: Use descriptive `--run-name` for organization
3. **Monitor actively**: Check TensorBoard early to catch issues
4. **Sync regularly**: Download checkpoints periodically as backup
5. **Compare scenarios**: Train on multiple scenarios to test generalization

## 🚨 Cost Management

**AWS g6.2xlarge (L4 GPU)**: ~$0.98/hour

**Estimated costs**:
- Quick test (1M steps, 15 min): ~$0.25
- Full training (10M steps, 2.5 hrs): ~$2.50
- Scenario sweep (9 scenarios): ~$22

**Remember to stop instances** when not training!
```bash
sky stop rwalters-sandbox-2
sky down rwalters-sandbox-2  # Terminate completely
```

## 📝 Notes

- Checkpoints use significant disk space (~33MB each)
- TensorBoard logs grow over time
- Clean up old runs periodically
- GPU memory: L4 has 24GB, plenty for batch_size=2048-4096
