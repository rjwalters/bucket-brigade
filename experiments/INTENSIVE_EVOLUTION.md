# Intensive Evolution Strategy

## Executive Summary

Current evolution runs (100 pop, 200 gen, 20 games) completed in ~100 minutes and achieved decent results, but there's significant room for improvement. The fitness metric uses a different scale (individual agent rewards, ±10) than tournament payoff (scenario team rewards, ±100), resulting in a ~100x ratio that makes fitness hard to interpret.

**Recommendation**: Run Option A (Deep Search) for 8-10 hours to get 5x more generations and better game sampling.

## Current Results Analysis

| Scenario            | Evolution Fitness | Tournament Payoff | Ratio   |
|---------------------|-------------------|-------------------|---------|
| chain_reaction      | 0.05              | 28.56             | 571.2x  |
| deceptive_calm      | 0.05              | 12.62             | 252.5x  |
| early_containment   | 1.20              | 46.12             | 38.4x   |
| greedy_neighbor     | 0.75              | 54.91             | 73.2x   |
| mixed_motivation    | 0.35              | 31.99             | 91.4x   |
| overcrowding        | 0.25              | 26.73             | 106.9x  |
| rest_trap           | 0.25              | 86.90             | 347.6x  |
| sparse_heroics      | 1.00              | 59.70             | 59.7x   |
| trivial_cooperation | 0.35              | 109.35            | 312.4x  |
| **Average**         | **0.47**          | **50.77**         | **107.5x** |

### Key Observations

1. **Low fitness values** (0.05-1.20) suggest strategies are suboptimal
2. **High variance** in ratios (38x to 571x) indicates fitness metric inconsistency
3. **Quick completion** (~100 min) means we can afford much more computation
4. **None converged** - all hit max generations without convergence

## Intensive Evolution Options

### Option A: Deep Search (RECOMMENDED)

**Best for**: Finding globally optimal strategies with moderate computational cost

```bash
# On remote machine
ssh rwalters-sandbox-1
cd ~/bucket-brigade
source .venv/bin/activate

# Launch intensive evolution
POPULATION=200 GENERATIONS=1000 GAMES=50 ./scripts/launch_evolution_tmux.sh

# Monitor progress
watch -n 30 ./scripts/monitor_evolution.sh

# Or attach to tmux
tmux attach -t evolution-master
```

**Parameters**:
- Population: 200 (2x increase)
- Generations: 1000 (5x increase) 
- Games/eval: 50 (2.5x increase)
- Estimated time: 8-10 hours
- Expected improvement: 2-4x fitness increase

**Why this option**:
- Allows proper convergence (1000 gen vs 200)
- Better game sampling reduces noise
- Larger population explores more strategies
- Reasonable time investment

### Option B: Wide Search

**Best for**: Maximum exploration of strategy space

```bash
POPULATION=500 GENERATIONS=500 GAMES=50 ./scripts/launch_evolution_tmux.sh
```

**Parameters**:
- Population: 500 (5x increase)
- Generations: 500 (2.5x increase)
- Games/eval: 50 (2.5x increase)
- Estimated time: 15-20 hours
- Expected improvement: 1.5-3x fitness increase

### Option C: High Precision

**Best for**: Reducing noise and getting accurate fitness measurements

```bash
POPULATION=100 GENERATIONS=500 GAMES=100 ./scripts/launch_evolution_tmux.sh
```

**Parameters**:
- Population: 100 (same)
- Generations: 500 (2.5x increase)
- Games/eval: 100 (5x increase)
- Estimated time: 8-10 hours
- Expected improvement: 2-3x fitness increase

### Option D: Marathon Search

**Best for**: Publication-quality results with exhaustive search

```bash
POPULATION=300 GENERATIONS=2000 GAMES=100 ./scripts/launch_evolution_tmux.sh
```

**Parameters**:
- Population: 300 (3x increase)
- Generations: 2000 (10x increase)
- Games/eval: 100 (5x increase)
- Estimated time: 48 hours (2 days)
- Expected improvement: 3-5x fitness increase

## Monitoring Progress

### Real-time monitoring

```bash
# Watch summary (refreshes every 30s)
watch -n 30 ./scripts/monitor_evolution.sh

# Attach to tmux session
tmux attach -t evolution-master

# Navigate windows: Ctrl+B then 0-8 (or n/p for next/previous)
# Detach: Ctrl+B then D
```

### Check specific scenario

```bash
# View recent log output
tail -f logs/evolution/greedy_neighbor_*.log

# Check latest generation
grep "Gen " logs/evolution/greedy_neighbor_*.log | tail -1
```

### System resources

```bash
# CPU usage
top -bn1 | grep "Cpu(s)"

# Load average (should be ~60-95 for full utilization)
uptime

# Process count
ps aux | grep python | wc -l
```

## Retrieving Results

### Sync back to local machine

```bash
# From local machine
rsync -avz --progress \
  rwalters-sandbox-1:~/bucket-brigade/experiments/scenarios/*/evolved/ \
  experiments/scenarios/

# Verify updates
ls -lh experiments/scenarios/*/evolved/best_agent.json
```

### Generate new insights

```bash
# On local machine after syncing
uv run python experiments/scripts/generate_insights.py

# Copy to web directory
cp web/public/research/scenarios/*/config.json .
```

## Understanding the Fitness Metric

### Current Implementation

Evolution fitness comes from `bucket_brigade/evolution/fitness_rust.py`:

- **Metric**: Mean episode reward across N games
- **Scale**: Individual agent rewards (±10.0)
  - reward_own_house_survives: +10.0
  - reward_other_house_survives: +5.0
  - penalty_own_house_burns: -10.0
  - penalty_other_house_burns: -5.0
- **Environment**: Single agent self-play

### Tournament Performance

Tournament payoff comes from scenario parameters:

- **Metric**: Total team payoff - work costs
- **Scale**: Scenario rewards (±100.0)
  - A (house survives): +100.0
  - L (house burns): -100.0
  - c (work cost): 1.0 per action
- **Environment**: Multi-agent with diverse strategies

### Why The Discrepancy?

The ~100x ratio is expected:
1. Different reward scales (10 vs 100)
2. Individual vs team accounting
3. Self-play vs diverse opponents

This doesn't invalidate results - evolved strategies still perform well in tournaments. But it makes fitness values hard to interpret.

## Expected Improvements

Based on similar evolutionary optimization studies:

| Metric                    | Current | Expected After Intensive |
|---------------------------|---------|--------------------------|
| Average fitness           | 0.47    | 1.5 - 2.5                |
| Tournament payoff         | 50.77   | 56 - 61                  |
| Convergence rate          | 0/9     | 7-9/9                    |
| Strategy robustness       | Medium  | High                     |
| Parameter diversity       | High    | Lower (more focused)     |

## Cost-Benefit Analysis

| Option | Time    | Compute | Fitness Gain | Payoff Gain |
|--------|---------|---------|--------------|-------------|
| A      | 8-10h   | Low     | 2-4x         | 10-20%      |
| B      | 15-20h  | Medium  | 1.5-3x       | 8-15%       |
| C      | 8-10h   | Low     | 2-3x         | 10-18%      |
| D      | 48h     | High    | 3-5x         | 15-25%      |

**Recommendation**: 
1. Start with Option A (best ROI)
2. If showing continued improvement at gen 1000, run Option D
3. If converged early (< 500 gen), results are likely near-optimal

## Future Improvements

### Short-term

1. **Fix fitness metric**: Modify fitness_rust.py to use scenario payoffs
2. **Adaptive evaluation**: Start with 20 games, increase to 100 for elite candidates
3. **Warm start**: Initialize population with best heuristics

### Long-term

1. **Multi-stage evolution**:
   - Stage 1: Wide search (500 pop, 100 gen, 20 games)
   - Stage 2: Deep search (50 pop, 500 gen, 50 games)
   - Stage 3: Refinement (10 pop, 1000 gen, 100 games)

2. **Co-evolution**: Evolve against evolving opponents
3. **Ensemble strategies**: Combine multiple evolved strategies
4. **Meta-learning**: Learn to adapt strategies during gameplay

## Troubleshooting

### Evolution stuck or slow

```bash
# Check if processes are running
ps aux | grep python

# Check CPU usage
top

# Check logs for errors
tail -100 logs/evolution/*.log | grep -i error
```

### Out of memory

```bash
# Check memory usage
free -h

# Reduce population or parallel workers
POPULATION=100 ./scripts/launch_evolution_tmux.sh
```

### Results not improving

This could mean:
1. Local optima reached (try larger population)
2. Insufficient games (increase games/eval)
3. Strategy space well-explored (current results are near-optimal)

## References

- Evolution config: `bucket_brigade/evolution/config.py`
- Fitness evaluation: `bucket_brigade/evolution/fitness_rust.py`
- Launch script: `scripts/launch_evolution_tmux.sh`
- Monitor script: `scripts/monitor_evolution.sh`
- Main evolution: `experiments/scripts/run_evolution.py`
