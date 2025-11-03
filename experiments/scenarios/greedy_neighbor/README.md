# Greedy Neighbor Scenario

## Overview

The **Greedy Neighbor** scenario creates a social dilemma through high work cost combined with moderate fire dynamics. This scenario is designed to explore the tension between individual self-interest and collective benefit.

## Scenario Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `beta` (spread) | 0.15 | Low fire spread probability |
| `kappa` (extinguish) | 0.4 | Moderate extinguish efficiency |
| `c` (work cost) | **1.0** | **High work cost** ← Creates free-riding incentive |
| `A` (reward) | 100.0 | Reward per saved house |
| `L` (penalty) | 100.0 | Penalty per ruined house |
| `rho_ignite` | 0.2 | Initial burning fraction |
| `N_min` | 12 | Minimum nights |
| `p_spark` | 0.02 | Spontaneous ignition probability |

## The Story

Fires spread slowly in this neighborhood, and a single firefighter has only moderate success at extinguishing them. The catch? Working is **expensive** (c=1.0, double the typical cost).

This creates a classic social dilemma:
- **If everyone works**: Fires get controlled, but individuals pay high costs
- **If everyone rests**: Fires spread, total payoff collapses
- **If some work, some rest**: Free riders exploit cooperators' efforts

## Research Questions

1. **What cooperation rate emerges in Nash equilibrium?**
   - Theory predicts a mixed strategy equilibrium
   - We expect cooperation rate between 50-70%

2. **Can evolution find the Nash mixed strategy?**
   - Will genetic algorithms converge to game-theoretic prediction?
   - Or will evolution discover something different?

3. **At what work cost does cooperation collapse?**
   - Sensitivity analysis: vary c from 0.5 to 2.0
   - Identify critical threshold for cooperation breakdown

4. **How do free riders perform in mixed populations?**
   - Compare pure cooperator teams vs mixed teams
   - Measure exploitation vs total welfare

## Expected Results

### Theoretical Prediction (Nash Equilibrium)

**Type**: Mixed Strategy Equilibrium

**Cooperation Rate**: ~60-65%

**Intuition**:
- Too many cooperators → free-riding becomes profitable
- Too few cooperators → fires spread, everyone suffers
- Equilibrium balances these forces

### Baseline Heuristics

| Agent Type | Expected Performance |
|------------|---------------------|
| Firefighter | Moderate (exploited by free riders) |
| Free Rider | **High** (benefits from others' work) |
| Hero | Low (pays high costs, overworks) |
| Coordinator | Moderate-High (balances work/rest) |

### Evolution Hypothesis

Evolution should discover a strategy similar to "smart free rider":
- Work only when absolutely necessary
- Prioritize own house
- Low coordination (don't trust others' signals)
- High rest bias

## Directory Contents

- `config.json` - Scenario configuration and research parameters
- `heuristics/` - Hand-tuned baseline agent results
- `evolved/` - Genetic algorithm optimization results
- `nash/` - Nash equilibrium analysis
- `comparison/` - Cross-method comparison and synthesis

## Running Analysis

```bash
# Full analysis pipeline
python experiments/scripts/run_scenario_research.py greedy_neighbor

# Individual components
python experiments/scripts/analyze_heuristics.py greedy_neighbor
python experiments/scripts/run_evolution.py greedy_neighbor
python experiments/scripts/compute_nash.py greedy_neighbor
```

## Related Scenarios

- **trivial_cooperation** - Low cost version (c=0.5), easier cooperation
- **sparse_heroics** - Similar dilemma but with very low fire spread
- **mixed_motivation** - Adds ownership incentives on top of work cost

---

*Scenario designed based on SCENARIO_BRAINSTORM.md*
*Last updated: 2025-11-03*
