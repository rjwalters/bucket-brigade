# Baseline Evolution Run - Quick Test

**Date**: 2025-11-04
**Machine**: rwalters-sandbox-1 (64 CPUs, 124GB RAM)
**Worktree**: bucket-brigade-evolution

## Configuration

- Generations: 500 (with early stopping)
- Population: 50
- Elite size: 5
- No explicit worker parallelization in this version
- Early stopping: Default (enabled in code)

## Results Summary

All 12 scenarios converged extremely quickly (4-10 generations):

| Scenario | Convergence Gen | Best Fitness |
|----------|----------------|--------------|
| chain_reaction | 6 | -2.6750 |
| deceptive_calm | 5 | -2.8250 |
| default | 6 | -2.7000 |
| early_containment | 7 | -3.2250 |
| easy | 6 | -2.8250 |
| greedy_neighbor | 4 | -3.2500 |
| hard | 10 | -1.8750 |
| mixed_motivation | 4 | -2.9000 |
| overcrowding | 4 | -2.8750 |
| rest_trap | 7 | -2.8500 |
| sparse_heroics | 7 | -2.9000 |
| trivial_cooperation | 4 | -3.1750 |

**Total Runtime**: < 5 minutes for all 12 scenarios

## Observations

1. **Very Fast Convergence**: All scenarios converged in 4-10 generations
2. **Resource Underutilization**: Only used a fraction of available 64 CPUs
3. **Negative Fitness**: All fitness values negative (need to investigate fitness function)
4. **Consistent Performance**: Similar fitness across most scenarios

## Next Steps

Launch massive overnight run with:
- 10 random seeds per scenario (120 total runs)
- Higher population size (500) 
- More generations (5000)
- Full CPU utilization
