# Evolved Expert Agents

This directory contains expert agents evolved for specific scenarios using genetic algorithms.

## Overview

Expert agents are evolved to perform well in specific scenario conditions by testing robustness across diverse team compositions. Each expert is optimized for a particular scenario's parameters (spread rate, extinguish rate, work cost, etc.).

## Directory Structure

```
experiments/evolved_experts/
├── README.md                           # This file
├── chain_reaction/                     # Expert for chain_reaction scenario
│   ├── best_genome.json               # Best evolved parameters
│   ├── config.json                    # Evolution configuration
│   ├── evolution_log.txt              # Human-readable log
│   ├── fitness_history.json           # Convergence data
│   ├── checkpoint_gen050.json         # Checkpoints (every 50 generations)
│   └── evolve_chain_reaction.log      # Full execution log
├── deceptive_calm/
├── default/
├── early_containment/
├── easy/
├── greedy_neighbor/
├── hard/
├── mixed_motivation/
├── overcrowding/
├── rest_trap/
├── sparse_heroics/
└── trivial_cooperation/
```

## Evolution Process

### Phase 1: Infrastructure (Issue #94)
Created `scripts/evolve_scenario_expert.py` for single-scenario evolution with:
- Robustness fitness function (tests against diverse team types)
- Progress logging and checkpointing
- Command-line configurability

### Phase 2: Parallel Execution (Issue #95)
Launched all 12 scenarios in parallel on remote GPU sandbox:
- 12 parallel tmux sessions (one per scenario)
- 4 workers per scenario = 48 total workers
- ~500 generations per scenario
- Expected runtime: 30-60 minutes

### Phase 3: Analysis (Future)
- Visualization of fitness convergence
- Comparison across scenarios
- Research documentation

## Scripts

### Launch Evolution (Remote Execution)
```bash
# Launch all 12 scenarios in parallel
./scripts/launch_parallel_evolution.sh --host rwalters-sandbox-2

# Custom configuration
./scripts/launch_parallel_evolution.sh \
  --host rwalters-sandbox-2 \
  --generations 1000 \
  --population 100 \
  --workers 6

# Dry run (preview without executing)
./scripts/launch_parallel_evolution.sh --host rwalters-sandbox-2 --dry-run
```

### Monitor Progress
```bash
# Single snapshot
./scripts/monitor_evolution.sh --host rwalters-sandbox-2

# Continuous monitoring (refreshes every 10s)
./scripts/monitor_evolution.sh --host rwalters-sandbox-2 --watch

# Monitor specific scenario with details
./scripts/monitor_evolution.sh --host rwalters-sandbox-2 --scenario easy --detailed
```

### Collect Results
```bash
# Collect all results
./scripts/collect_evolution_results.sh --host rwalters-sandbox-2

# Collect specific scenario
./scripts/collect_evolution_results.sh --host rwalters-sandbox-2 --scenario easy

# Force overwrite existing files
./scripts/collect_evolution_results.sh --host rwalters-sandbox-2 --force
```

### Local Evolution (Single Scenario)
```bash
# Evolve for a single scenario locally
uv run python scripts/evolve_scenario_expert.py \
  --scenario easy \
  --output-dir experiments/evolved_experts/easy \
  --generations 100 \
  --population-size 50 \
  --workers 4
```

## Robustness Fitness Function

Each individual is tested against 4 team composition types:
1. **Random**: Random baseline with high variability
2. **Greedy**: High free-riding behavior (low work, high rest)
3. **Fair**: Cooperative behavior (high work, low rest)
4. **Mixed**: Heterogeneous team (mix of above types)

**Fitness Formula**: `mean_reward - 0.1 * variance`

This rewards agents that perform consistently well across different teammates, rather than specializing for one team type.

## File Formats

### best_genome.json
```json
{
  "scenario": "easy",
  "generation": 487,
  "fitness": 42.35,
  "genome": [0.234, 0.567, ..., 0.891],  // 10 parameters
  "parameters": {
    "honesty": 0.234,
    "work_tendency": 0.567,
    "neighbor_help": 0.123,
    "own_priority": 0.789,
    "risk_aversion": 0.456,
    "coordination": 0.345,
    "exploration": 0.678,
    "fatigue_memory": 0.901,
    "rest_bias": 0.432,
    "altruism": 0.210
  }
}
```

### fitness_history.json
```json
{
  "generations": [
    {
      "generation": 0,
      "best_fitness": 15.23,
      "mean_fitness": 8.45,
      "std_fitness": 3.21,
      "diversity": 0.842
    },
    ...
  ]
}
```

### config.json
```json
{
  "scenario": "easy",
  "population_size": 50,
  "num_generations": 500,
  "elite_size": 5,
  "selection_strategy": "tournament",
  "crossover_strategy": "uniform",
  "mutation_strategy": "gaussian",
  "games_per_individual": 20,
  "workers": 4
}
```

## Using Evolved Experts

### Load and Use in Python
```python
import json
import numpy as np
from bucket_brigade.agents.heuristic_agent import HeuristicAgent

# Load best genome
with open('experiments/evolved_experts/easy/best_genome.json', 'r') as f:
    data = json.load(f)

# Create agent
agent = HeuristicAgent(
    params=np.array(data['genome']),
    agent_id=0,
    name='Evolved Easy Expert'
)

# Use in environment
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios import get_scenario_by_name

scenario = get_scenario_by_name('easy', num_agents=4)
env = BucketBrigadeEnv(scenario=scenario, agents=[agent, ...])

obs = env.reset()
while not env.done:
    action = agent.act(obs)
    obs, rewards, done, info = env.step([action, ...])
```

### Compare Experts Across Scenarios
```python
import json
from pathlib import Path

# Load all experts
experts = {}
for scenario_dir in Path('experiments/evolved_experts').iterdir():
    if scenario_dir.is_dir():
        genome_file = scenario_dir / 'best_genome.json'
        if genome_file.exists():
            with open(genome_file) as f:
                experts[scenario_dir.name] = json.load(f)

# Compare fitness
for name, expert in sorted(experts.items(), key=lambda x: x[1]['fitness'], reverse=True):
    print(f"{name:20s}: {expert['fitness']:6.2f}")
```

## Performance Expectations

### With Parallel Evaluation (PR #89)
- **Population**: 50 individuals
- **Games per individual**: 20
- **Evaluations per generation**: 1000 episodes
- **Workers**: 48 (on remote sandbox)
- **Expected speedup**: ~40x over sequential
- **Runtime per scenario**: 2.5-5 minutes
- **Total for 12 scenarios**: 30-60 minutes

### Optimization Opportunities
1. **Rust game loop** (5-10x additional speedup) - similar to Nash optimization
2. **Adaptive games-per-individual** (30% fewer evaluations)
3. **Aggressive early stopping** (10-20% fewer generations)

## Related Issues

- **#84**: Parent issue - Evolved expert agent pipeline
- **#94**: Phase 1 - Evolution script infrastructure
- **#95**: Phase 2 - Parallel remote execution (this implementation)
- **#89**: Parallel fitness evaluation (multiprocessing)
- **#88, #91**: Nash equilibrium optimization (Rust speedup reference)

## Scenarios

The 12 available scenarios represent different cooperation challenges:

1. **trivial_cooperation**: Easy cooperation, low fire spread
2. **easy**: Balanced parameters, moderate difficulty
3. **default**: Standard scenario, baseline difficulty
4. **greedy_neighbor**: High temptation to free-ride
5. **early_containment**: Fast spread, requires quick action
6. **chain_reaction**: Cascading fires, coordination critical
7. **deceptive_calm**: Slow start, sudden escalation
8. **rest_trap**: High work cost, temptation to rest
9. **overcrowding**: Many agents, coordination overhead
10. **sparse_heroics**: Few agents, individual impact high
11. **mixed_motivation**: Heterogeneous incentives
12. **hard**: Extreme difficulty, requires perfect cooperation

Each expert agent is optimized for its specific scenario's dynamics.

## Notes

- Evolution uses genetic algorithm with NEAT-inspired principles
- Parallel evaluation uses Python multiprocessing (40x speedup)
- Robustness testing ensures generalization across team types
- Checkpoints saved every 50 generations for recovery
- Early stopping detects convergence (saves compute)

## Future Work

- **Phase 3**: Visualization, analysis, and research documentation
- **Rust optimization**: Port game loop for 5-10x additional speedup
- **Transfer learning**: Use experts as seed populations for related scenarios
- **Ensemble methods**: Combine multiple experts for robust agents
- **Meta-learning**: Learn to adapt strategy based on observed teammate behavior
