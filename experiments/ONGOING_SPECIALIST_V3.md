# Specialist Evolution v3 - Currently Running ⏳

**Status**: Running on remote machine (rwalters-sandbox-1)
**Started**: [Check logs for exact timestamp]
**Expected completion**: ~10 hours from start

## What's Running

**9 specialist agents** evolving in parallel, one per scenario:
1. chain_reaction
2. deceptive_calm
3. early_containment
4. greedy_neighbor
5. mixed_motivation
6. overcrowding
7. rest_trap
8. sparse_heroics
9. trivial_cooperation

## Configuration

```bash
POPULATION=200
GENERATIONS=2500  # Optimized for ~10 hour runtime
GAMES=50
```

**Key improvements over v2**:
- ✅ Fixed fitness metric (using scenario payoff, not individual rewards)
- ✅ Using RustFitnessEvaluator (100x faster than Python)
- ✅ 2500 generations for full convergence
- ✅ Proper tmux session management

## Runtime Details

**tmux session**: `specialist-evolution`
**Remote machine**: `rwalters-sandbox-1`
**Script**: `scripts/launch_specialists_overnight.sh`

**Each window** (0-8) runs:
```bash
python experiments/scripts/run_evolution.py $scenario \
  --population 200 \
  --generations 2500 \
  --games 50 \
  --output-dir experiments/scenarios/$scenario/evolved_v3
```

**Logs location**: `~/bucket-brigade/logs/evolution/*_v3_*.log`

## Checking Status

### Check if jobs are still running

```bash
ssh rwalters-sandbox-1 "tmux ls"
# Should show: specialist-evolution: 9 windows ...
```

### Check progress of specific scenario

```bash
ssh rwalters-sandbox-1 "tail -20 ~/bucket-brigade/logs/evolution/chain_reaction_v3_*.log"
```

Look for lines like:
```
Generation 1234/2500 | Fitness: 70.00 | Diversity: 0.45 | Time: 12.3s
```

### Check all scenarios at once

```bash
ssh rwalters-sandbox-1 'for scenario in chain_reaction deceptive_calm early_containment greedy_neighbor mixed_motivation overcrowding rest_trap sparse_heroics trivial_cooperation; do echo "=== $scenario ==="; tail -3 ~/bucket-brigade/logs/evolution/${scenario}_v3_*.log 2>/dev/null | grep "Generation" | tail -1 || echo "No progress yet"; done'
```

### Estimate time remaining

If generation X/2500 is shown:
- Each generation: ~12-13 seconds
- Remaining: (2500 - X) × 13 seconds
- Hours left: (2500 - X) × 13 / 3600

Example: At generation 1000/2500
- Remaining: 1500 × 13 = 19,500 seconds = 5.4 hours

## Expected Results

Each scenario will have a new `evolved_v3/` directory:

```
experiments/scenarios/{scenario}/evolved_v3/
├── best_agent.json          # Best evolved agent parameters
├── evolution_trace.json     # Generation-by-generation history
└── generation_*/            # Checkpoints every N generations
    └── snapshot.json
```

## Retrieving Results

### Option 1: Git Commit and Push (Recommended)

On remote machine after completion:
```bash
ssh rwalters-sandbox-1
cd ~/bucket-brigade

# Check results
ls experiments/scenarios/*/evolved_v3/best_agent.json

# Verify all 9 scenarios completed
count=$(ls experiments/scenarios/*/evolved_v3/best_agent.json 2>/dev/null | wc -l)
echo "Completed: $count/9 scenarios"

# Commit results
git add experiments/scenarios/*/evolved_v3/
git add logs/evolution/*_v3_*.log
git commit -m "feat: Specialist evolution v3 results (2500 gen, fixed fitness)"
git push
```

On local machine:
```bash
git pull
```

### Option 2: rsync Direct Copy

From local machine:
```bash
rsync -avz --progress \
  rwalters-sandbox-1:~/bucket-brigade/experiments/scenarios/*/evolved_v3/ \
  ./experiments/scenarios/
```

### Option 3: Archive and Download

On remote:
```bash
ssh rwalters-sandbox-1 "cd ~/bucket-brigade && tar czf specialist_v3_results_$(date +%Y%m%d).tar.gz experiments/scenarios/*/evolved_v3/ logs/evolution/*_v3_*.log"
```

On local:
```bash
scp rwalters-sandbox-1:~/bucket-brigade/specialist_v3_results_*.tar.gz .
tar xzf specialist_v3_results_*.tar.gz
```

## Analysis After Completion

### 1. Verify All Scenarios Completed

```bash
# Should show 9 files
ls experiments/scenarios/*/evolved_v3/best_agent.json
```

### 2. Compare Fitness to v2

Check final fitness values in `evolution_trace.json`:
```bash
for scenario in experiments/scenarios/*/evolved_v3; do
    echo "=== $(basename $(dirname $scenario)) ==="
    python -c "import json; data=json.load(open('$scenario/evolution_trace.json')); print(f'Final fitness: {data[\"generations\"][-1][\"best_fitness\"]:.2f}')"
done
```

### 3. Run Tournament Comparisons

Compare v3 agents against v2 and baselines:
```bash
for scenario in chain_reaction deceptive_calm early_containment greedy_neighbor mixed_motivation overcrowding rest_trap sparse_heroics trivial_cooperation; do
    echo "Testing $scenario..."
    python experiments/scripts/run_comparison.py $scenario \
      --games 100 \
      --output experiments/scenarios/$scenario/comparison_v3/
done
```

### 4. Update Website Data

Sync new results to web frontend:
```bash
# Copy best agents
for scenario in experiments/scenarios/*; do
    scenario_name=$(basename $scenario)
    cp $scenario/evolved_v3/best_agent.json \
       web/public/research/scenarios/$scenario_name/evolved_v3_agent.json
done

# Rebuild web to pick up new data
cd web
pnpm run build
```

## Success Criteria

The v3 run is successful if:
- ✅ All 9 scenarios complete full 2500 generations
- ✅ Fitness values are in tournament scale (±100 range, not ±10)
- ✅ Best fitness improves from v2
- ✅ Diversity stays > 0.1 throughout evolution
- ✅ Tournament performance beats baselines

## Cleanup After Retrieval

On remote machine:
```bash
# Kill tmux session
ssh rwalters-sandbox-1 "tmux kill-session -t specialist-evolution"

# Optional: Archive old logs
ssh rwalters-sandbox-1 "cd ~/bucket-brigade && tar czf logs_archive_$(date +%Y%m%d).tar.gz logs/evolution/ && rm -rf logs/evolution/*"
```

## Next Steps After v3

1. **Analyze v3 vs v2 improvements**
   - Document in INTENSIVE_EVOLUTION_RESULTS.md
   - Update tournament comparisons
   - Identify which scenarios improved most

2. **Consider generalist evolution** (See issue #120)
   - Cross-scenario training
   - Single agent for all scenarios
   - 150-180 generations for 8-10 hour run

3. **Update research insights**
   - Regenerate insights for all scenarios
   - Sync to website

## Troubleshooting

### Jobs stopped prematurely

Check logs for errors:
```bash
ssh rwalters-sandbox-1 "grep -i error ~/bucket-brigade/logs/evolution/*_v3_*.log"
```

Common issues:
- Out of memory: Reduce population or games
- Python version mismatch: Must use Python 3.9-3.13
- Rust module missing: Run `./bucket-brigade-core/build.sh`

### Need to restart a scenario

```bash
# On remote machine
cd ~/bucket-brigade
source .venv/bin/activate

# Restart specific scenario
python experiments/scripts/run_evolution.py greedy_neighbor \
  --population 200 \
  --generations 2500 \
  --games 50 \
  --output-dir experiments/scenarios/greedy_neighbor/evolved_v3 \
  2>&1 | tee logs/evolution/greedy_neighbor_v3_$(date +%Y%m%d_%H%M%S).log
```

---

**Document created**: 2025-11-04
**Last updated**: 2025-11-04
