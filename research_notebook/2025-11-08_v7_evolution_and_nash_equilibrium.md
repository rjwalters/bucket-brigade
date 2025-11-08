---
title: "V7 Evolution & Nash Equilibrium Analysis"
date: "2025-11-08"
author: "Research Team"
tags: ["evolution", "nash-equilibrium", "analysis", "v7"]
status: "Complete"
---

# V7 Evolution & Nash Equilibrium Analysis

## Executive Summary

We completed two major experiments across 12 game scenarios:
1. **V7 Evolution**: Evolved optimal agents using heterogeneous tournament fitness
2. **Nash Equilibrium V5**: Computed game-theoretic equilibria for 9 challenge scenarios

**Key Finding**: Hero strategies dominate Nash equilibria (8/9 scenarios), but evolved agents discovered much more diverse and effective strategies.

---

## Nash Equilibrium Analysis

### Results Summary

| Scenario | Equilibrium Type | Expected Payoff | Optimal Strategy |
|----------|------------------|-----------------|------------------|
| chain_reaction | Pure Nash | 803.87 | Hero (100%) |
| deceptive_calm | Pure Nash | 403.40 | Hero (100%) |
| early_containment | Pure Nash | 946.88 | Hero (100%) |
| greedy_neighbor | Pure Nash | 710.54 | Hero (100%) |
| mixed_motivation | Pure Nash | 663.44 | Hero (100%) |
| **overcrowding** | **Mixed Nash** | 115.84 | Hero (99%) + other (1%) |
| rest_trap | Pure Nash | 1007.23 | Hero (100%) |
| sparse_heroics | Pure Nash | 792.98 | Hero (100%) |
| **trivial_cooperation** | **Pure Nash** | 1026.50 | **Free Rider (100%)** |

### Key Insights

1. **Hero Dominance**: 8 out of 9 scenarios have Hero as the Nash equilibrium
   - Hero archetype: `{honesty: 1.0, work: 1.0, neighbor_help: 1.0, risk: 0.1}`
   - Maximizes prosocial behavior in homogeneous populations

2. **One Exception**: `trivial_cooperation` rewards Free Rider strategy
   - Suggests cooperation is easily exploitable in this scenario
   - Free riding becomes the rational strategy when everyone else cooperates

3. **Near-Pure Equilibria**: Only `overcrowding` has mixed strategy (99% Hero)
   - Still effectively pure in practice
   - Suggests game structure strongly favors dominant strategies

4. **Payoff Variation**: Expected payoffs range from 115.84 to 1026.50
   - Lower payoffs in `overcrowding` (resource scarcity)
   - Higher payoffs in `trivial_cooperation` and `rest_trap`

---

## V7 Evolution Analysis

### Evolved Agent Performance

| Scenario | Fitness | Honesty | Work | Neighbor Help | Risk Aversion |
|----------|---------|---------|------|---------------|---------------|
| **deceptive_calm** | **48.80** | 0.65 | 0.21 | 0.74 | 0.11 |
| chain_reaction | 43.67 | 0.42 | 0.00 | **1.00** | 0.90 |
| hard | 43.21 | 0.64 | 0.05 | 0.00 | 0.88 |
| mixed_motivation | 44.68 | 0.70 | 0.13 | 0.10 | 0.07 |
| overcrowding | 41.76 | 0.17 | 0.02 | 0.27 | 0.70 |
| sparse_heroics | 41.74 | 0.32 | 0.04 | 0.49 | 0.09 |
| greedy_neighbor | 38.12 | **0.90** | 0.00 | 0.12 | **0.01** |
| default | 34.81 | 0.37 | 0.07 | 0.00 | **0.98** |
| early_containment | 32.90 | 0.15 | 0.00 | 0.06 | 0.27 |
| rest_trap | 21.82 | 0.57 | 0.17 | 0.02 | 0.77 |
| easy | 17.11 | 0.46 | 0.00 | 0.75 | 0.60 |
| trivial_cooperation | 6.50 | 0.49 | 0.00 | 0.61 | 0.68 |

### Behavioral Patterns

#### 1. **Low Work Tendency Across All Scenarios**
- Mean work_tendency: **0.06** (range: 0.00-0.21)
- **Interpretation**: In heterogeneous tournaments, direct work is less effective than strategic behavior
- Agents evolved to rely on coordination and helping rather than solo work

#### 2. **Honesty Variation** (0.15 to 0.90)
- High honesty: `greedy_neighbor` (0.90), `mixed_motivation` (0.70)
- Low honesty: `early_containment` (0.15), `overcrowding` (0.17)
- **Insight**: Honesty is context-dependent; some scenarios reward deception

#### 3. **Neighbor Help Strategies**
- **High cooperation**: `chain_reaction` (1.00), `easy` (0.75), `deceptive_calm` (0.74)
- **Low cooperation**: `default` (0.00), `hard` (0.00), `rest_trap` (0.02)
- **Pattern**: Scenarios with cascading fires reward helping; others favor selfishness

#### 4. **Risk Aversion Split**
- **Risk-seeking**: `greedy_neighbor` (0.01), `mixed_motivation` (0.07)
- **Risk-averse**: `default` (0.98), `chain_reaction` (0.90), `hard` (0.88)
- **Insight**: Fire spread mechanics influence optimal risk-taking

---

## Comparing Nash vs Evolved Strategies

### The Paradox

**Nash Equilibrium** (theoretical): Hero strategy dominates (8/9 scenarios)

**V7 Evolution** (empirical): Diverse strategies with low work tendency

### Why the Difference?

1. **Evaluation Context**:
   - Nash: Computed for symmetric, homogeneous populations
   - Evolution: Evaluated in heterogeneous tournaments (mixed agent types)

2. **Fitness Function**:
   - Nash: Maximizes payoff against same strategy
   - Evolution: Maximizes performance against diverse opponents

3. **Strategy Space**:
   - Nash: Considers all possible pure strategies
   - Evolution: Searches continuous parameter space

### What This Means

**Nash equilibria identify stable homogeneous strategies**, but **evolved agents optimize for heterogeneous environments**.

In real-world scenarios with diverse populations, evolved strategies may outperform Nash equilibria.

---

## Notable Findings

### 🏆 Best Performer: `deceptive_calm`
- **Fitness**: 48.80 (highest)
- **Strategy**: Balanced approach with moderate honesty (0.65), good cooperation (0.74), low risk aversion (0.11)
- **Why it works**: Scenario rewards calculated risk-taking and strategic cooperation

### 📉 Worst Performer: `trivial_cooperation`
- **Fitness**: 6.50 (lowest)
- **Observation**: Low fitness despite Nash equilibrium predicting high payoff (1026.50)
- **Explanation**: Free Rider Nash equilibrium dominates in homogeneous play, but performs poorly in heterogeneous tournaments

### 🎯 Most Cooperative: `chain_reaction`
- **Neighbor help**: 1.00 (maximum)
- **Reason**: Fires spread rapidly; helping neighbors prevents cascades
- **Also evolved**: High risk aversion (0.90) to avoid dangerous situations

### 😈 Most Deceptive: `early_containment`
- **Honesty**: 0.15 (lowest)
- **Strategy**: Lie about fire states to manipulate other agents
- **Works because**: Early containment means trust is valuable but exploitable

### 🎲 Most Risk-Seeking: `greedy_neighbor`
- **Risk aversion**: 0.01 (minimum)
- **High honesty**: 0.90 (maximum)
- **Pattern**: Honest but aggressive strategy works in neighbor-greedy scenarios

---

## Evolution Process Details

### Training Configuration
- **Population size**: 200 agents per scenario
- **Generations**: 200
- **Mutation rate**: 0.15
- **Fitness evaluation**: Heterogeneous tournament against 5 archetypes
- **Genome length**: 10 parameters (5 behavioral + 5 thresholds)

### Convergence
- All scenarios ran to completion (200 generations)
- Population diversity maintained: 0.89 mean diversity score
- Mean fitness improved: final best agents significantly outperform population mean

---

## Future Research Questions

### 1. **Cross-Scenario Generalization**
Can an agent evolved for one scenario perform well in others?
- Hypothesis: High-cooperation agents (chain_reaction) will struggle in low-cooperation scenarios (hard)

### 2. **V5 vs V6 vs V7 Comparison**
How much did each evolution iteration improve?
- Run tournaments: evolved_v5 vs evolved_v6 vs evolved_v7

### 3. **Nash Equilibrium Robustness**
Are Nash equilibria stable in heterogeneous populations?
- Test: Introduce evolved agents into Nash equilibrium populations
- Measure: Do they disrupt the equilibrium?

### 4. **Team Composition Optimization**
What mix of agent types maximizes team performance?
- Test: Various combinations of evolved + archetype agents

### 5. **Parameter Sensitivity**
Which parameters matter most for each scenario?
- Method: Ablation studies, parameter perturbation analysis

---

## Data & Reproducibility

### Available Artifacts
- **V7 Best Agents**: `experiments/scenarios/*/evolved_v7/best_agent.json`
- **Evolution Logs**: `experiments/scenarios/*/evolved_v7/evolution_log.txt`
- **Checkpoints**: Every 20 generations saved
- **Nash Equilibria**: `experiments/nash/v2_results_v5/*/equilibrium_v2.json`

### Reproduction
```bash
# Re-run v7 evolution (200 gens, 200 pop)
uv run python experiments/scripts/run_evolution.py <scenario> \
  --population 200 --generations 200 --version v7

# Recompute Nash equilibrium
uv run python experiments/scripts/compute_nash_v2.py <scenario> \
  --evolved-versions v5 --simulations 2000 --max-iterations 20
```

---

## Conclusions

1. **Evolution discovers diverse strategies** optimized for heterogeneous environments
2. **Nash equilibria favor heroic behavior** in homogeneous populations
3. **Context matters**: Optimal strategies vary dramatically by scenario
4. **Work isn't everything**: Low work_tendency agents can succeed through coordination
5. **Honesty is strategic**: Some scenarios reward deception

**Next**: Analyze evolution trajectories to understand how these strategies emerged over 200 generations.

