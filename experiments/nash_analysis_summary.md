# Nash Equilibrium Analysis - Cross-Scenario Summary

**Generated:** 2025-11-04 14:49:20
**Scenarios Analyzed:** 12

## Overview

This report presents Nash equilibrium analysis for all Bucket Brigade scenarios,
computed using the Double Oracle algorithm with Rust-accelerated payoff evaluation.

## Equilibrium Types

| scenario            | type   |   support_size |   expected_payoff |   iterations | converged   |   elapsed_time |
|:--------------------|:-------|---------------:|------------------:|-------------:|:------------|---------------:|
| chain_reaction      | pure   |              1 |            2.936  |            2 | True        |        750.948 |
| deceptive_calm      | pure   |              1 |           37.298  |            1 | True        |        436.384 |
| default             | pure   |              1 |           60.305  |            2 | True        |        256.714 |
| early_containment   | mixed  |              2 |           24.2664 |            3 | True        |       1646.92  |
| easy                | pure   |              1 |           81.89   |            2 | True        |        252.112 |
| greedy_neighbor     | pure   |              1 |           58.775  |            2 | True        |        641.441 |
| hard                | pure   |              1 |          -45.69   |            2 | True        |        222.239 |
| mixed_motivation    | pure   |              1 |           46.657  |            1 | True        |        434.782 |
| overcrowding        | pure   |              1 |           17.786  |            1 | True        |        407.945 |
| rest_trap           | pure   |              1 |           98.935  |            1 | True        |        407.651 |
| sparse_heroics      | pure   |              1 |           80.7075 |            1 | True        |        457.055 |
| trivial_cooperation | mixed  |              2 |          108.776  |            3 | True        |       2189.46  |

### Key Observations

- **Pure equilibria:** 10/12 scenarios
- **Mixed equilibria:** 2/12 scenarios
- **Average support size:** 1.17
- **Average convergence time:** 675.3s
- **Convergence rate:** 100.0%

**Pure equilibrium scenarios:** chain_reaction, deceptive_calm, default, easy, greedy_neighbor, hard, mixed_motivation, overcrowding, rest_trap, sparse_heroics

**Mixed equilibrium scenarios:** early_containment, trivial_cooperation

## Cooperation Analysis

| scenario            |   cooperation_rate |   free_riding_rate | equilibrium_type   |
|:--------------------|-------------------:|-------------------:|:-------------------|
| chain_reaction      |                  0 |                  1 | pure               |
| deceptive_calm      |                  1 |                  0 | pure               |
| default             |                  1 |                  0 | pure               |
| early_containment   |                  0 |                  1 | mixed              |
| easy                |                  0 |                  1 | pure               |
| greedy_neighbor     |                  1 |                  0 | pure               |
| hard                |                  0 |                  1 | pure               |
| mixed_motivation    |                  1 |                  0 | pure               |
| overcrowding        |                  0 |                  1 | pure               |
| rest_trap           |                  1 |                  0 | pure               |
| sparse_heroics      |                  1 |                  0 | pure               |
| trivial_cooperation |                  0 |                  1 | mixed              |

### Cooperation Patterns

- **Highest cooperation:** deceptive_calm (100.0%)
- **Lowest cooperation:** chain_reaction (0.0%)
- **Average cooperation:** 50.0%

## Strategy Distributions

### chain_reaction

**Type:** Pure equilibrium
**Expected Payoff:** 2.94

Pure strategy: **Free Rider** (closest to Free Rider, distance=0.000)

### deceptive_calm

**Type:** Pure equilibrium
**Expected Payoff:** 37.30

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### default

**Type:** Pure equilibrium
**Expected Payoff:** 60.30

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### early_containment

**Type:** Mixed equilibrium
**Expected Payoff:** 24.27

Mixed strategy distribution:

1. **Free Rider** (59.9%) - closest to Free Rider (distance=0.000)
2. **Free Rider** (40.1%) - closest to Free Rider (distance=0.000)

### easy

**Type:** Pure equilibrium
**Expected Payoff:** 81.89

Pure strategy: **Free Rider** (closest to Free Rider, distance=0.000)

### greedy_neighbor

**Type:** Pure equilibrium
**Expected Payoff:** 58.77

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### hard

**Type:** Pure equilibrium
**Expected Payoff:** -45.69

Pure strategy: **Free Rider** (closest to Free Rider, distance=0.000)

### mixed_motivation

**Type:** Pure equilibrium
**Expected Payoff:** 46.66

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### overcrowding

**Type:** Pure equilibrium
**Expected Payoff:** 17.79

Pure strategy: **Free Rider** (closest to Free Rider, distance=0.000)

### rest_trap

**Type:** Pure equilibrium
**Expected Payoff:** 98.94

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### sparse_heroics

**Type:** Pure equilibrium
**Expected Payoff:** 80.71

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### trivial_cooperation

**Type:** Mixed equilibrium
**Expected Payoff:** 108.78

Mixed strategy distribution:

1. **Free Rider** (93.0%) - closest to Free Rider (distance=0.000)
2. **Free Rider** (7.0%) - closest to Free Rider (distance=0.000)

## Cross-Scenario Insights

### Parameter-Equilibrium Relationships

**High work cost scenarios (c > 0.50):**
- Average cooperation: 60.0%
- Pure equilibria: 5/5

**High fire spread scenarios (β > 0.23):**
- Average cooperation: 50.0%
- Pure equilibria: 5/6

## Design Recommendations

Based on Nash equilibrium analysis:

1. **Greedy Neighbor Scenario** (cooperation: 100.0%)
   - High work cost creates free-riding incentive

2. **Overall Cooperation** (average: 50.0%)
   - Moderate cooperation across scenarios

3. **Equilibrium Diversity**
   - Predominance of pure equilibria indicates clear optimal strategies
   - Game parameters may need tuning for strategic depth

## Future Work

1. **Compare with evolved agents** (Issue #84)
   - Do evolved strategies converge to Nash equilibria?
   - Measure strategy divergence between theory and practice

2. **Robustness analysis**
   - Test equilibrium stability to parameter perturbations
   - Identify critical parameter thresholds

3. **Mechanism design**
   - Design incentives to improve equilibrium outcomes
   - Reduce price of anarchy where applicable

---

*Analysis generated using Double Oracle algorithm with 2000 simulations per evaluation.*
