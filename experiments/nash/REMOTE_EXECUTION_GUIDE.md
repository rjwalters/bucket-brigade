# Nash V2 Remote Execution Guide

**Purpose**: Run computationally intensive Nash V2 analysis on remote server with better compute resources.

**Date**: 2025-11-05
**Estimated Runtime**: 3-6 hours for all 9 scenarios (depends on simulation count)

---

## Prerequisites

### Remote Server Access

You should have SSH access configured to your remote server (e.g., SkyPilot cluster, GPU server).

Check your `~/.ssh/config`:
```ssh
Host my-cluster
    HostName my-cluster
    User root
    ProxyCommand ssh -W %h:%p jump-host
```

Test connection:
```bash
ssh my-cluster "echo 'Connected successfully'"
```

### Local Setup (One-time)

Ensure local repo is clean and committed:
```bash
git status
git add experiments/scripts/compute_nash_v2.py experiments/scripts/run_nash_v2_all.sh
git commit -m "feat: Add Nash V2 with evolved agents and batch script"
git push
```

---

## Quick Start

### Option A: Full Automated Run (Recommended)

```bash
# 1. Connect to remote
ssh my-cluster

# 2. Navigate to repo (or clone if first time)
cd ~/bucket-brigade || git clone https://github.com/rjwalters/bucket-brigade.git && cd bucket-brigade

# 3. Pull latest code
git pull

# 4. Setup environment (if not already done)
uv venv
source .venv/bin/activate  # Linux
uv sync --extra rl

# 5. Build Rust core
cd bucket-brigade-core
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release --features python
cd ..

# 6. Run full batch (in tmux for persistence)
tmux new -s nash_v2
./experiments/scripts/run_nash_v2_all.sh --full  # 2000 simulations
# Detach: Ctrl+B, D
```

### Option B: Quick Test First

```bash
# Run quick test on one scenario first (200 simulations)
ssh my-cluster
cd ~/bucket-brigade
tmux new -s nash_v2_test
uv run python experiments/scripts/compute_nash_v2.py chain_reaction --simulations 200
```

---

## Running the Batch Script

### Command Options

```bash
# Standard run (1000 simulations per evaluation)
./experiments/scripts/run_nash_v2_all.sh

# Quick test (200 simulations)
./experiments/scripts/run_nash_v2_all.sh --quick

# Full analysis (2000 simulations for high confidence)
./experiments/scripts/run_nash_v2_all.sh --full

# Specify evolved versions
./experiments/scripts/run_nash_v2_all.sh --evolved-versions v4 v5
```

### What It Does

The batch script runs Nash V2 analysis on all 9 standard scenarios:
1. chain_reaction
2. deceptive_calm
3. early_containment
4. greedy_neighbor
5. mixed_motivation
6. overcrowding
7. rest_trap
8. sparse_heroics
9. trivial_cooperation

For each scenario:
- Loads evolved agent(s) from `experiments/scenarios/{scenario}/evolved_v4/`
- Includes 5 predefined archetypes + evolved agents in initial pool
- Runs Double Oracle with Rust evaluator
- Saves results to `experiments/nash/v2_results/{scenario}/`
- Logs to `logs/nash_v2/{scenario}_{timestamp}.log`

---

## Monitoring Progress

### Using tmux

```bash
# Attach to running session
ssh my-cluster -t "tmux attach -t nash_v2"

# Check if session exists
ssh my-cluster "tmux ls"

# View logs without attaching
ssh my-cluster "tail -f ~/bucket-brigade/logs/nash_v2/*.log"
```

### Check Progress

```bash
# See which scenarios completed
ssh my-cluster "ls -lh ~/bucket-brigade/experiments/nash/v2_results/"

# Check last log
ssh my-cluster "tail -50 ~/bucket-brigade/logs/nash_v2/*.log | tail -50"

# Monitor CPU/memory
ssh my-cluster "htop"
```

### Expected Timeline

| Simulations | Time per scenario | Total (9 scenarios) |
|-------------|-------------------|---------------------|
| 200 (quick) | 15-20 min | ~2.5 hours |
| 1000 (standard) | 30-40 min | ~5 hours |
| 2000 (full) | 60-80 min | ~10 hours |

*Times are estimates and depend on server specs*

---

## Retrieving Results

### Option 1: Git (Recommended)

```bash
# On remote server after completion
cd ~/bucket-brigade
git add experiments/nash/v2_results/
git commit -m "results: Nash V2 analysis for all scenarios"
git push

# On local machine
git pull
```

### Option 2: Direct Copy (rsync)

```bash
# From local machine
rsync -avz --progress \
  my-cluster:~/bucket-brigade/experiments/nash/v2_results/ \
  ./experiments/nash/v2_results/

# Copy logs too
rsync -avz --progress \
  my-cluster:~/bucket-brigade/logs/nash_v2/ \
  ./logs/nash_v2/
```

### Option 3: Archive and Download

```bash
# On remote
cd ~/bucket-brigade
tar czf nash_v2_results.tar.gz experiments/nash/v2_results/ logs/nash_v2/

# On local
scp my-cluster:~/bucket-brigade/nash_v2_results.tar.gz ./
tar xzf nash_v2_results.tar.gz
```

---

## Troubleshooting

### Build Failures

**Problem**: Rust core won't build

```bash
# Solution: Ensure Python feature is enabled
cd bucket-brigade-core
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release --features python
cd ..
```

**Problem**: Import errors

```bash
# Solution: Rebuild and reinstall
cd bucket-brigade-core
rm -rf target/
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release --features python
cd ..
```

### Missing Evolved Agents

**Problem**: "No evolved agents found!"

```bash
# Check if V4 results exist
ls experiments/scenarios/*/evolved_v4/best_agent.json

# If missing, need to run evolution first (or copy from local)
rsync -avz ./experiments/scenarios/*/evolved_v4/ \
  my-cluster:~/bucket-brigade/experiments/scenarios/
```

### Script Hangs

**Problem**: Script appears stuck

```bash
# Check if actually computing (CPU should be high)
ssh my-cluster "top -bn1 | head -20"

# Check for Python processes
ssh my-cluster "ps aux | grep python"

# If truly stuck, kill and restart
ssh my-cluster "pkill -f compute_nash_v2.py"
```

### Out of Memory

**Problem**: Process killed due to OOM

```bash
# Check available memory
ssh my-cluster "free -h"

# Solution: Reduce simulation count
./experiments/scripts/run_nash_v2_all.sh --quick  # Use 200 instead of 1000
```

---

## Post-Processing

After retrieving results, analyze locally:

```bash
# View results
ls experiments/nash/v2_results/*/equilibrium_v2.json

# Quick summary
for scenario in experiments/nash/v2_results/*/; do
    echo "=== $(basename $scenario) ==="
    jq '.equilibrium | {payoff, type, evolved_in_equilibrium}' "$scenario/equilibrium_v2.json"
done

# Compare V1 vs V2
python experiments/scripts/analyze_nash_v2_results.py  # To be created
```

---

## What to Look For in Results

### Key Metrics

1. **Evolved agents in equilibrium**
   - Check `equilibrium.evolved_in_equilibrium` field
   - If > 0: Evolved strategy is part of Nash equilibrium!
   - Compare evolved fitness vs Nash payoff

2. **Payoff comparison**
   - Nash V2 payoff should be ~60 for chain_reaction (we validated this)
   - Compare across scenarios

3. **Equilibrium type**
   - Pure vs mixed
   - Support size

4. **Convergence**
   - All should converge within 10-20 iterations
   - Check `convergence.converged` is true

### Success Criteria

✅ All 9 scenarios complete without errors
✅ All equilibria converged
✅ Results validate: Nash V2 payoffs ~match evolution fitness
✅ Cross-validation complete: Rust-Rust consistency confirmed

---

## Remote Server Recommendations

### Minimum Specs
- **CPU**: 8+ cores (for parallel simulation)
- **RAM**: 16+ GB
- **Storage**: 10+ GB free

### Optimal Specs
- **CPU**: 16+ cores
- **RAM**: 32+ GB
- **GPU**: Not required (Nash uses CPU only)

### Server Options

**SkyPilot Cluster** (if you have access):
```bash
sky launch -c nash-cluster --cpus 16 --memory 32
sky ssh nash-cluster
```

**AWS/GCP Instance**:
- Instance type: c5.4xlarge or similar (CPU-optimized)
- Estimated cost: ~$0.50-1.00 per hour

---

## Files Created/Modified

### Scripts
- `experiments/scripts/compute_nash_v2.py` - V2 Nash computation with evolved agents
- `experiments/scripts/run_nash_v2_all.sh` - Batch script for all scenarios

### Infrastructure
- `bucket_brigade/equilibrium/evolved_agents.py` - Utilities for loading evolved agents
- `bucket_brigade/equilibrium/payoff_evaluator_rust.py` - Rust evaluator (updated)
- `bucket_brigade/equilibrium/double_oracle.py` - Rust migration (updated)

### Documentation
- `experiments/nash/PYTHON_VS_RUST_COMPARISON.md` - Environment mismatch analysis
- `experiments/nash/REMOTE_EXECUTION_GUIDE.md` - This file

---

## Next Steps After Completion

1. **Analyze results** - Compare Nash V2 vs Evolution fitness
2. **Document findings** - Update V2_PLAN.md with actual results
3. **Cross-validate** - Verify Rust-Rust consistency across all scenarios
4. **Epsilon-equilibrium** - Analyze how close evolved strategies are to Nash
5. **Complete Phase 1** - Nash track finished!

---

## Support

**Questions?** Check these files:
- Nash V2 Plan: `experiments/nash/V2_PLAN.md`
- Python/Rust comparison: `experiments/nash/PYTHON_VS_RUST_COMPARISON.md`
- General remote guide: `CLAUDE.md` (Remote Development section)

**Issues?** Common problems and solutions are in the Troubleshooting section above.

---

**Status**: Ready to run
**Recommended**: Start with `--quick` test, then run `--full` batch
**Estimated cost**: ~$1-5 depending on server choice and runtime
