# Removed Scenarios Archive

**Date**: 2025-11-06
**Reason**: Focus reduction - keeping core set of ~10 tested scenarios

This document archives 20 exploratory scenarios that were removed from the main scenario registry to maintain focus on a manageable exploration space.

---

## Why These Were Removed

During the 2025-11-06 audit, we identified that the scenario registry had grown to 32 scenarios, but only 12 had been tested with experiments. The 20 untested scenarios represented exploratory ideas that:

1. Were never run through the full research pipeline (Nash V1, Nash V2, evolution)
2. Didn't have experiment directories with results
3. Cluttered the registry and made it harder to focus on core research

**Decision**: Remove these 20 scenarios to maintain a focused set of ~10 core scenarios that cover the cooperation dynamics space.

---

## Removed Scenarios by Category

### Phase 2A: Extreme Scenarios (9 removed)

These tested boundary conditions with extreme parameter values.

#### glacial_spread
- **Parameters**: β=0.02 (extremely low), κ=0.5, c=0.5
- **Purpose**: Fires barely spread - tests minimal threat scenarios
- **Expected**: Strong free-riding incentive

#### explosive_spread
- **Parameters**: β=0.60 (extremely high), κ=0.5, c=0.5
- **Purpose**: Fires spread very aggressively - tests crisis response
- **Expected**: Requires immediate, coordinated action

#### wildfire
- **Parameters**: β=0.75 (near-maximum), κ=0.5, c=0.5
- **Purpose**: Near-uncontrollable spread - tests extreme crisis
- **Expected**: May be unwinnable without perfect cooperation

#### free_work
- **Parameters**: β=0.20, κ=0.5, c=0.05 (extremely low)
- **Purpose**: Work costs almost nothing
- **Expected**: Should eliminate free-riding incentive

#### cheap_work
- **Parameters**: β=0.20, κ=0.5, c=0.10 (very low)
- **Purpose**: Very affordable work
- **Expected**: Strong cooperation incentive

#### expensive_work
- **Parameters**: β=0.20, κ=0.5, c=2.0 (high)
- **Purpose**: Work is costly
- **Expected**: Increased free-riding pressure

#### prohibitive_work
- **Parameters**: β=0.20, κ=0.5, c=5.0 (extremely high)
- **Purpose**: Work is extremely expensive
- **Expected**: Rational agents should minimize work

#### crisis_cheap
- **Parameters**: β=0.60 (very high), κ=0.5, c=0.10 (very low)
- **Purpose**: Fast spread but affordable work
- **Expected**: High-volume cooperation should emerge

#### calm_expensive
- **Parameters**: β=0.02 (extremely low), κ=0.5, c=5.0 (extremely high)
- **Purpose**: Slow spread but costly work
- **Expected**: Minimal cooperation, strategic rest

**Why removed**: These represent parameter sweeps that would be better done programmatically in experiments rather than hardcoded as scenario definitions. The core 9 scenarios already cover the meaningful cooperation dynamics.

---

### Phase 2A.1: Kappa and Spark Sweep (7 removed)

These investigated the `trivial_cooperation` scenario by varying κ (extinguish rate) and p_spark (ongoing fires).

#### easy_kappa_60
- **Parameters**: β=0.15, κ=0.60 (moderate-high), c=0.5, p_spark=0.0
- **Purpose**: Lower bound for easy scenarios

#### easy_kappa_70
- **Parameters**: β=0.15, κ=0.70 (high), c=0.5, p_spark=0.0
- **Purpose**: Mid-range extinguish rate

#### easy_kappa_80
- **Parameters**: β=0.15, κ=0.80 (very high), c=0.5, p_spark=0.0
- **Purpose**: High extinguish rate

#### easy_kappa_90
- **Parameters**: β=0.15, κ=0.90 (extremely high), c=0.5, p_spark=0.0
- **Purpose**: Upper bound, matches trivial_cooperation baseline

#### easy_spark_01
- **Parameters**: β=0.15, κ=0.90, c=0.5, p_spark=0.01 (minimal)
- **Purpose**: Minimal ongoing fires

#### easy_spark_02
- **Parameters**: β=0.15, κ=0.90, c=0.5, p_spark=0.02 (moderate)
- **Purpose**: Moderate ongoing fires

#### easy_spark_05
- **Parameters**: β=0.15, κ=0.90, c=0.5, p_spark=0.05 (high)
- **Purpose**: High ongoing fires

**Why removed**: These are essentially hyperparameter sweeps of a single scenario (`trivial_cooperation`). This type of investigation belongs in experiment scripts, not as separate scenario definitions. If needed, these parameter ranges could be explored programmatically.

---

### Phase 2D: Mechanism Design (4 removed)

These tested scenario designs attempting to induce cooperation and break the free-riding equilibrium.

#### nearly_free_work
- **Parameters**: β=0.30, κ=0.60, c=0.01 (nearly free), ρ=0.15, p_spark=0.02
- **Purpose**: Work cost approaching zero
- **Hypothesis**: Free work should induce cooperation by removing cost barrier
- **Mechanism**: Eliminates work disincentive

#### front_loaded_crisis
- **Parameters**: β=0.70 (very fast), κ=0.40, c=0.30, ρ=0.40 (very high), p_spark=0.0, N_min=8
- **Purpose**: Overwhelming initial fires requiring immediate response
- **Hypothesis**: One-time crisis may coordinate agents better than sustained threat
- **Mechanism**: Urgency and common knowledge of threat timing

#### sustained_pressure
- **Parameters**: β=0.50 (fast), κ=0.30, c=0.40, ρ=0.20, p_spark=0.10 (very high), N_spark=8
- **Purpose**: Continuous high threat requiring persistent effort
- **Hypothesis**: Overwhelming sustained pressure induces cooperation
- **Mechanism**: Constant danger makes defection obviously costly

#### high_stakes
- **Parameters**: β=0.40, κ=0.50, A=500 (5x), L=500 (5x), c=1.0, ρ=0.20
- **Purpose**: Extreme asset values creating high-variance outcomes
- **Hypothesis**: Increased payoff variance induces risk-averse coordination
- **Mechanism**: Large stakes amplify cost of coordination failure

**Why removed**: While interesting mechanism design experiments, these are "future work" ideas that haven't been validated yet. The core 9 scenarios already demonstrate the fundamental cooperation dynamics. If mechanism design becomes a research focus later, these could be re-added as needed.

---

## If You Need These Scenarios

If future research requires exploring these parameter ranges:

1. **For parameter sweeps**: Write experiment scripts that programmatically vary parameters
   - Example: `experiments/scripts/sweep_work_cost.py` could test c ∈ [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
   - This is more flexible than hardcoded scenarios

2. **For mechanism design**: Add specific scenarios back to the registry when needed
   - Start with hypothesis-driven experiments
   - If a mechanism proves interesting, promote it to a core scenario

3. **Git history**: All scenario definitions remain in git history
   - Last commit with these scenarios: [commit before pruning]
   - Use `git show <commit>:bucket_brigade/envs/scenarios.py` to retrieve

---

## Parameter Ranges Covered by Removed Scenarios

For reference, here are the parameter ranges that were explored:

| Parameter | Core Scenarios Range | Removed Scenarios Range | Extension |
|-----------|---------------------|------------------------|-----------|
| β (spread) | 0.05 - 0.45 | 0.02 - 0.75 | Extended both ends |
| κ (extinguish) | 0.30 - 0.95 | 0.30 - 0.95 | No extension |
| c (cost) | 0.20 - 1.00 | 0.01 - 5.00 | Extended both ends |
| A (reward) | 50 - 100 | 50 - 500 | Extended high end |
| L (penalty) | 100 | 100 - 500 | Extended high end |
| ρ (initial) | 0.10 - 0.30 | 0.10 - 0.40 | Extended high end |
| p_spark | 0.00 - 0.05 | 0.00 - 0.10 | Extended high end |

**Key observation**: The removed scenarios extended parameter ranges but didn't explore fundamentally different cooperation dynamics. The core 9 scenarios already span the meaningful cooperation space.

---

## Future Research Directions

If you're considering adding new scenarios, ask:

1. **Does it test a new cooperation dynamic?**
   - Not covered by existing 12 scenarios
   - Example: scenarios with asymmetric agents, changing parameters over time

2. **Is it hypothesis-driven?**
   - Clear research question
   - Expected to produce distinct equilibrium behavior

3. **Can it be explored programmatically instead?**
   - Parameter sweeps belong in experiment scripts
   - Hardcode only canonical parameter combinations

**Good reasons to add scenarios**:
- Testing heterogeneous agent populations
- Dynamic parameter changes over time
- New game mechanics (e.g., agent communication, reputation)

**Bad reasons to add scenarios**:
- Testing slightly different parameter values
- "What if we made fires spread faster?"
- Curiosity without clear hypothesis

---

## Version History

- **v1.0** (2025-11-06): Initial archive of 20 removed scenarios from registry pruning
