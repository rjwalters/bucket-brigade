# Evolution Research Framework

## Current Status (Quick Runs)

**Goal**: Find evolved agents that outperform hand-designed heuristics

**Results** (9 scenarios):
- **Evolved Wins**: 1/9 (greedy_neighbor only)
- **Average Performance Gap**: Heuristic leads by ~25 points on average
- **Issue**: Quick runs (10 gen, pop 20) insufficient for optimization

## Problem Analysis

### Current Limitations

1. **Short Evolution Time**
   - Only 10 generations in quick mode
   - Population of 20 is small for 10-dimensional search space
   - Mutation scale (0.1) may be too conservative

2. **Fitness Function Mismatch**
   - Evolving against random/self-play partners
   - Should evolve against best_heuristic specifically
   - Tournament with varying team sizes (3-9 agents)

3. **No Targeted Optimization**
   - Same hyperparameters for all scenarios
   - No scenario-specific seeding or initialization
   - Missing head-to-head comparison during evolution

## Improved Evolution Strategy

### 1. Extended Evolution Runs

**Production Configuration**:
```python
{
    "population_size": 100,      # vs 20 in quick mode
    "num_generations": 200,      # vs 10 in quick mode
    "elite_size": 10,            # preserve best
    "mutation_scale": 0.15,      # slightly higher exploration
    "games_per_individual": 50,  # vs 5 in quick mode
    "snapshot_interval": 20      # save progress
}
```

### 2. Competitive Co-Evolution

**Strategy**: Evolve specifically to beat best_heuristic

```python
def competitive_fitness(individual, scenario, best_heuristic_params):
    """
    Fitness function focused on beating the best heuristic.

    Tournament composition:
    - 50% games: evolved agent + 3 random partners vs fires
    - 30% games: evolved agent + 3 best_heuristic partners (cooperative)
    - 20% games: evolved agent + 2 random + 1 best_heuristic (mixed)

    This trains the agent to:
    1. Work with random partners (robustness)
    2. Complement the best heuristic (cooperation)
    3. Handle mixed scenarios (adaptability)
    """
    pass
```

### 3. Warm Start from Best Heuristic

**Seeding Strategy**:
- Initialize 20% of population with best_heuristic parameters
- Initialize 20% with slight mutations of best_heuristic
- Initialize 60% randomly (exploration)

This gives evolution a "head start" near known good solutions.

### 4. Multi-Objective Fitness

**Fitness Components**:
```python
fitness = (
    0.6 * mean_reward +           # Primary: total reward
    0.2 * win_rate +               # Secondary: games with all houses saved
    0.1 * robustness +             # Std dev of rewards (lower = more consistent)
    0.1 * heuristic_advantage      # Direct comparison vs best_heuristic
)
```

## Research Scripts

### Script 1: `run_extended_evolution.py`

Run full evolution with production hyperparameters:

```bash
# Single scenario, full evolution
python experiments/scripts/run_extended_evolution.py greedy_neighbor

# With competitive fitness against best heuristic
python experiments/scripts/run_extended_evolution.py greedy_neighbor --competitive

# Warm start from heuristic
python experiments/scripts/run_extended_evolution.py greedy_neighbor --warm-start

# All scenarios (batch)
python experiments/scripts/run_extended_evolution.py --all
```

### Script 2: `compare_evolution_strategies.py`

Compare different evolution approaches:

```bash
# Compare: baseline, competitive, warm-start, multi-objective
python experiments/scripts/compare_evolution_strategies.py greedy_neighbor

# Generates report comparing all 4 strategies
```

### Script 3: `evolution_head_to_head.py`

Direct tournament: evolved agent vs best_heuristic

```bash
# Run 100 games: evolved vs best_heuristic
python experiments/scripts/evolution_head_to_head.py greedy_neighbor --games 100

# Test across team sizes (3-9 agents)
python experiments/scripts/evolution_head_to_head.py greedy_neighbor --vary-team-size
```

### Script 4: `evolution_analysis.py`

Analyze evolution dynamics:

```bash
# Visualize convergence, diversity, fitness landscapes
python experiments/scripts/evolution_analysis.py greedy_neighbor

# Compare parameter evolution over generations
python experiments/scripts/evolution_analysis.py greedy_neighbor --parameter-evolution
```

## Expected Outcomes

### Success Criteria

**Minimal**: Evolved agent matches best_heuristic (within 5% performance)
**Target**: Evolved agent beats best_heuristic by 10%+
**Stretch**: Evolved agent discovers novel strategies not in heuristic set

### Timeline

- **Phase 1** (1-2 hours): Extended evolution runs for all 9 scenarios
- **Phase 2** (2-4 hours): Competitive co-evolution experiments
- **Phase 3** (1-2 hours): Analysis and documentation

## Data Organization

```
experiments/scenarios/{scenario}/
├── evolved/
│   ├── quick/                    # Quick runs (10 gen, existing)
│   │   ├── best_agent.json
│   │   └── evolution_trace.json
│   ├── extended/                 # Extended runs (200 gen, new)
│   │   ├── best_agent.json
│   │   ├── evolution_trace.json
│   │   └── snapshots/
│   ├── competitive/              # Competitive co-evolution
│   │   └── ...
│   ├── warm_start/               # Warm start from heuristic
│   │   └── ...
│   └── comparison.json           # Compare all strategies
```

## Research Questions

1. **Can evolution beat hand-designed heuristics with more compute?**
   - Hypothesis: Yes, but needs 100+ generations

2. **Does competitive co-evolution improve results?**
   - Hypothesis: Yes, by 15-30% over baseline

3. **Is warm-starting from heuristics beneficial?**
   - Hypothesis: Faster convergence, but may get stuck in local optima

4. **What novel strategies emerge from evolution?**
   - Look for parameter combinations not in our heuristic set
   - Analyze "surprising" high-performing evolved agents

5. **How does performance scale with team size?**
   - Test evolved agents with 3, 5, 7, 9 partners
   - Measure robustness across scenarios

## Implementation Priority

1. ✅ Current state: Quick evolution baseline established
2. 🔄 Next: `run_extended_evolution.py` - Full evolution runs
3. 📋 Then: `evolution_head_to_head.py` - Direct comparison
4. 📋 Then: Competitive co-evolution variant
5. 📋 Finally: Analysis and reporting

## Notes

- Evolution is computationally expensive (~1-2 hours per scenario with 200 generations)
- Consider using Rust backend for faster simulation if needed
- Save checkpoints frequently to resume interrupted runs
- Use proper random seeds for reproducibility
