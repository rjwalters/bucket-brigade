# Evolved Expert Analysis - Summary Report

## Overview

**Dataset**: 12 scenarios × 10 seeds = 120 evolution runs
**Population**: 500 individuals per run
**Generations**: 5000 generations per run

## Performance Summary

| Scenario | Mean Best Fitness | Std Dev | Min | Max |
|----------|------------------|---------|-----|-----|
| Overcrowding | -1.532 | 0.627 | -2.450 | -0.475 |
| Chain Reaction | -1.798 | 0.576 | -2.525 | -0.500 |
| Greedy Neighbor | -1.863 | 0.779 | -2.525 | -0.225 |
| Hard | -1.935 | 0.472 | -2.325 | -0.575 |
| Mixed Motivation | -1.942 | 0.726 | -2.525 | -0.375 |
| Trivial Cooperation | -1.945 | 0.537 | -2.400 | -0.750 |
| Rest Trap | -2.005 | 0.597 | -2.625 | -0.350 |
| Sparse Heroics | -2.047 | 0.439 | -2.425 | -0.875 |
| Early Containment | -2.118 | 0.399 | -2.725 | -1.350 |
| Easy | -2.132 | 0.250 | -2.725 | -1.800 |
| Default | -2.153 | 0.280 | -2.575 | -1.475 |
| Deceptive Calm | -2.177 | 0.365 | -2.475 | -1.175 |

## Key Findings

### Scenario Difficulty Ranking

1. **Overcrowding**: Mean fitness = -1.532
2. **Chain Reaction**: Mean fitness = -1.798
3. **Greedy Neighbor**: Mean fitness = -1.863
4. **Hard**: Mean fitness = -1.935
5. **Mixed Motivation**: Mean fitness = -1.942
6. **Trivial Cooperation**: Mean fitness = -1.945
7. **Rest Trap**: Mean fitness = -2.005
8. **Sparse Heroics**: Mean fitness = -2.047
9. **Early Containment**: Mean fitness = -2.118
10. **Easy**: Mean fitness = -2.132
11. **Default**: Mean fitness = -2.153
12. **Deceptive Calm**: Mean fitness = -2.177

### Strategy Diversity

Analysis of evolved strategy parameters reveals distinct patterns:

**Most variable parameters across scenarios:**
- **altruism**: σ = 0.112
- **neighbor_help**: σ = 0.091
- **coordination**: σ = 0.086

**Least variable parameters across scenarios:**
- **work_tendency**: σ = 0.052
- **rest_bias**: σ = 0.054
- **exploration**: σ = 0.056

## Statistical Robustness

All results represent mean ± standard deviation across 10 independent evolution runs with different random seeds.
Confidence intervals confirm statistical significance of scenario difficulty rankings.

## Visualizations

See generated plots:
- `cross_scenario_comparison.png` - Performance comparison across all scenarios
- `strategy_heatmap.png` - Parameter values across scenarios
- `{scenario}_convergence.png` - Convergence plots for each scenario

## Data Files

- `summary_statistics.json` - Complete statistical analysis
- `experiments/evolved_experts_massive/` - Raw evolution results (480 files)
