# V5 Tournament Validation Results

**Date**: 2025-11-06
**Tournaments Run**: 3 scenarios x 500 games each = 1,500 total games

## Summary

V5 validation tournaments reveal a **critical gap between Nash equilibrium predictions and actual gameplay performance**.

### Key Finding

**Nash Equilibrium Predicted**: V5 would dominate with +1,162% average payoff improvement (experiments/nash/V2_RESULTS_V5.md)

**Tournament Reality**: V5 **underperforms** V4 by -4.7% across all scenarios

## Performance by Scenario

### Chain Reaction
- **V5**: 43.23 average payoff (379 games)
- **V4**: 41.83 average payoff (414 games)
- **Result**: V5 beats V4 by +3.3%

Agent Rankings:
1. evolved_v5: 43.23
2. evolved_v4: 41.83
3. free_rider: 39.92
4. firefighter: -43.40
5. hero: -52.78

### Trivial Cooperation
- **V5**: 6.50 average payoff (432 games)
- **V4**: 6.50 average payoff (378 games)
- **Result**: Tied at 0.0%

Agent Rankings:
1. evolved_v4: 6.50 (tie)
1. evolved_v5: 6.50 (tie)
3. free_rider: 6.22
4. firefighter: -5.29
5. hero: -6.50

### Deceptive Calm
- **V5**: 41.92 average payoff (396 games)
- **V4**: 42.98 average payoff (411 games)
- **Result**: V5 loses to V4 by -2.5%

Agent Rankings:
1. evolved_v4: 42.98
2. evolved_v5: 41.92
3. free_rider: 40.80
4. firefighter: -28.39
5. hero: -36.09

## Overall Performance (All Scenarios)

**Total Games**: 3,600 agent appearances across 1,500 games

Agent Rankings:
1. evolved_v4: 31.12 (1,203 games)
2. evolved_v5: 29.65 (1,207 games)
3. free_rider: 29.04 (1,184 games)
4. firefighter: -25.45 (1,236 games)
5. hero: -31.93 (1,170 games)

**V5 vs V4**: 29.65 vs 31.12 = **-4.7% worse performance**

## Analysis

### Why the Discrepancy?

1. **Nash Equilibrium Assumes Rational Equilibrium Play**
   - Predicts behavior when all players optimize against each other
   - Assumes convergence to equilibrium strategies
   - V5 may be optimized for this theoretical scenario

2. **Tournaments Test Robustness Against Diverse Strategies**
   - Includes archetype agents (hero, firefighter, free_rider)
   - Diverse team compositions create off-equilibrium gameplay
   - V5's optimization may not generalize well to this heterogeneous environment

3. **Potential Overfitting**
   - V5 evolved against V4 and archetypes
   - May have optimized for specific matchups
   - Struggles when facing unexpected strategy combinations

### Implications

- **Nash equilibrium analysis is valuable** but not sufficient for validating agent performance
- **Tournament validation is critical** for understanding real-world robustness
- **Gap between theory and practice** suggests need for alternative training methods
- **PPO training** (currently running) may produce more robust policies by learning from actual gameplay

## Next Steps

### V6 Development Considerations

1. **Increase Population Diversity**
   - Evolve against more varied opponents
   - Include more archetype variations
   - Use tournament-style fitness evaluation

2. **Multi-Objective Optimization**
   - Optimize for both Nash equilibrium performance AND tournament robustness
   - Balance specialization vs generalization

3. **Hybrid Approach**
   - Combine evolution with PPO training
   - Use evolved agents as curriculum for RL training
   - Fine-tune evolved agents with gradient-based methods

### PPO Training Track

- **Baseline PPO** training currently running on GPU (7% complete, ~6 hours remaining)
- Compare PPO-trained agent vs evolved agents in tournaments
- Evaluate if RL produces more robust policies than evolution

## Conclusion

V5's tournament underperformance (-4.7% vs V4) despite strong Nash equilibrium predictions (+1,162% improvement) reveals a fundamental challenge in multi-agent learning: **optimizing for equilibrium doesn't guarantee robust performance in diverse strategic environments**.

This validates our dual-track research approach: Evolution provides Nash-equilibrium insights while PPO training may deliver practical robustness.
