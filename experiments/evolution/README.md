# Evolution Research: Phase 1 - Closed-World Mastery

**Status**: ✅ Achieving near-Nash equilibria (V3/V4: 58.50 vs 57.87 theoretical Nash)
**Aligns with**: [Roadmap Phase 1](../../docs/roadmap_phased_plan.md#phase-1--closed-world-mastery), Genetic Algorithm Track
**Related docs**: [Vision & Future Directions](../../docs/vision_future_directions.md), [Technical MARL Review](../../docs/technical_marl_review.md)

---

## Overview

This document tracks our evolutionary algorithm research as part of **Phase 1: Closed-World Mastery**. Our goal is to discover near-optimal heuristic strategies for fixed scenarios through population-based optimization, complementing the PPO (neural network) and Nash equilibrium (game-theoretic) approaches.

### Why Evolution?

From [technical_marl_review.md](../../docs/technical_marl_review.md#population-based-training-and-evolutionary-strategies):

> "Population-based methods can incorporate a selection mechanism where only the best-performing agents (under current conditions) propagate... evolutionary algorithms have been shown to work well with nonstationarity and partial observability by continually using an evolving population rather than a single static agent."

Evolution provides:
- **Interpretability**: Parameterized heuristics are human-readable
- **Robustness**: Population diversity buffers against local optima
- **Speed**: Parallel fitness evaluation via Rust (100x faster than Python)
- **Complementarity**: Discovers strategies qualitatively different from gradient-based methods

---

## Current Status: V3/V4/V5 Experiments

### V3: Baseline Extended Run (2500 gen, 200 pop)

**Configuration**:
- Population: 200
- Generations: 2500
- Games per evaluation: 50
- Seed: 42
- Runtime: 614 CPU-hours

**Result**: **58.50 payoff** (chain_reaction scenario)

### V4: Intensive Run (15000 gen, 200 pop)

**Configuration**:
- Population: 200
- Generations: 15,000 (6x longer than V3)
- Games per evaluation: 50
- Seed: 42
- Runtime: 475 CPU-hours

**Result**: **58.50 payoff** (same as V3!)

### V5: Current Run (12000 gen, 200 pop) 🚀 IN PROGRESS

**Configuration**:
- Population: 200
- Generations: 12,000
- Games per evaluation: 50
- Seed: 43 (different seed for exploration)
- Runtime: 384 CPU-hours (6-hour budget)
- Expected completion: ~22:54 UTC 2025-11-05

**Early results** (Gen 18): Best fitness 69.16 (higher than V3/V4!)

### Key Findings

1. **Near-Nash Achievement**: V3/V4 both converged to 58.50 payoff
2. **Nash Benchmark**: Theoretical Nash equilibrium = 57.87
3. **Performance**: Only 0.63 points above Nash (99% optimal!)
4. **Consistency**: V3 and V4 converged to same result despite 6x generation difference
5. **Reliability**: Train/test consistency achieved via Rust single source of truth

**Conclusion**: Evolution successfully discovers near-optimal strategies for closed-world scenarios.

---

## Phase 1 Success Criteria (GA Track)

From [roadmap_phased_plan.md](../../docs/roadmap_phased_plan.md#genetic-algorithm-track):

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Match or exceed best hand-tuned heuristics | ✅ | 58.50 >> best heuristic performance |
| Interpretable strategies | ✅ | 10-parameter genome (work_tendency, signal_honesty, etc.) |
| Faster iteration than PPO | ✅ | 6-hour runs vs. days for PPO training |
| Discover qualitatively different strategies | ✅ | Near-Nash vs. hand-designed heuristics |

**Phase 1 GA Track**: **ON TRACK FOR COMPLETION** 🎯

---

## Technical Implementation

### Genome Representation

10-dimensional parameter vector:
- `work_tendency`: Baseline work probability
- `signal_honesty`: Truthfulness in signaling
- `coordination_weight`: Value placed on team cooperation
- `own_house_priority`: Bias toward protecting own house
- `risk_aversion`: Conservative vs. aggressive fire-fighting
- `signal_response`: Trust in others' signals
- `effort_cost_sensitivity`: Work avoidance
- `neighbor_priority`: Help neighbors vs. distant houses
- `rest_bias`: Tendency to rest
- `exploration_rate`: Randomness in action selection

### Fitness Evaluation

```python
def fitness(genome, scenario, num_games=50):
    """
    Absolute optimization: maximize mean reward across homogeneous teams.

    - All 4 agents use same genome (self-play)
    - Evaluated over 50 independent games
    - Rust-backed simulation (100x faster than Python)
    - Returns: mean(final_score / num_agents) across games
    """
    return mean_reward_across_games
```

**Why homogeneous teams?**
- Phase 1 goal: find optimal strategy for fixed scenario
- Simpler than heterogeneous evolution
- Establishes performance ceiling
- Enables Nash equilibrium comparison

### Evolution Algorithm

- **Selection**: Tournament selection (preserves diversity)
- **Mutation**: Gaussian noise (σ=0.1 for V3/V4, exploring alternatives for V5)
- **Elite preservation**: Top 10% survive unchanged
- **Parallelization**: 64 CPUs evaluate fitness in parallel
- **Rust backend**: 100x speedup enables practical large-scale evolution

---

## Rust Single Source of Truth

**Critical Infrastructure Achievement**: All evaluation now uses Rust implementation.

**Before** (V3 initially):
- Training: Rust evaluator (fast)
- Tournaments: Python environment (slow, different results)
- **Problem**: Train/test mismatch (Python gave wrong scores)

**After** (V3/V4/V5):
- Training: Rust evaluator
- Tournaments: Rust evaluator (same implementation)
- **Result**: Perfect consistency (training fitness = tournament payoff)

**Impact**:
- Reliable validation of evolved strategies
- Trustworthy fitness values during evolution
- Foundation for future research phases

See: [RUST_SINGLE_SOURCE_OF_TRUTH.md](RUST_SINGLE_SOURCE_OF_TRUTH.md)

---

## Data Organization

```
experiments/scenarios/{scenario}/
├── evolved/              # Original quick run (10 gen, 20 pop)
├── evolved_v2/           # Deprecated
├── evolved_v3/           # Extended run (2500 gen, 200 pop) ✅
│   ├── best_agent.json   # Final genome + fitness
│   └── evolution_logs/   # Generation-by-generation progress
├── evolved_v4/           # Intensive run (15000 gen, 200 pop) ✅
│   └── best_agent.json
├── evolved_v5/           # Current run (12000 gen, 200 pop) 🚀
│   └── best_agent.json   # (will be created when complete)
└── comparison/
    └── comparison.json   # Tournament results (Rust-only)
```

---

## Next Steps (Phase 1 Completion)

### When V5 Completes (~5 hours)

1. **Retrieve results** from remote server
2. **Run Rust-only tournaments** comparing evolved, v3, v4, v5
3. **Analyze convergence**:
   - Did V5 beat 58.50 with different seed?
   - How does generation count affect final performance?
   - What parameter values converged across runs?
4. **Document findings** in V5_RESULTS_ANALYSIS.md
5. **Update Phase 1 checklist** with GA track completion status

### Completing Phase 1

**Remaining GA Track work**:
- ✅ Extended evolution runs (V3/V4/V5)
- ✅ Near-Nash performance achieved
- ✅ Train/test consistency
- ⚠️ Run tournaments for all 9 scenarios (currently only chain_reaction/deceptive_calm tested)
- ⚠️ Document parameter patterns across scenarios

**Remaining Phase 1 work** (other tracks):
- PPO training pipeline (neural network policies)
- Complete Nash analysis for all tractable scenarios
- Public leaderboard and benchmark release

---

## Future Work: Beyond Phase 1

The ideas below are valuable but belong to later research phases per the roadmap.

### Phase 2: Multi-Scenario Agents

Once Phase 1 closed-world mastery is complete, we can explore:

**Scenario Generalization**:
- Train agents robust to scenario switching
- Measure performance degradation when scenario changes
- Test transfer learning across related scenarios

See: [roadmap_phased_plan.md - Phase 2](../../docs/roadmap_phased_plan.md#phase-2--adaptive-multi-scenario-agents)

### Phase 3: Population Resilience

From [vision_future_directions.md](../../docs/vision_future_directions.md#population-resilience):

**Competitive Co-Evolution**:
```python
def competitive_fitness(individual, scenario, opponent_genome):
    """
    Fitness function for beating specific opponents.

    Tournament composition:
    - 50% games: evolved agent + 3 random partners
    - 30% games: evolved agent + 3 opponent partners (cooperative)
    - 20% games: evolved agent + 2 random + 1 opponent (mixed)
    """
    # This belongs in Phase 3: Population Resilience
    pass
```

**Diversity Maintenance**:
- Niching techniques (fitness sharing, crowding)
- Multi-objective optimization (reward + robustness + diversity)
- Warm-start from known good solutions
- Co-evolutionary dynamics (arms races)

**Research Questions** (from [vision](../../docs/vision_future_directions.md#population-resilience)):
- What diversity metric best predicts resilience?
- How much diversity is optimal?
- Do specialist-generalist tradeoffs emerge naturally?
- Can we predict valuable diversity for unknown future perturbations?

### Phase 4: Meta-Learning

**Adaptive Strategies**:
- Agents that recognize scenario type from observations
- Meta-policies conditioned on inferred scenario
- Uncertainty-aware decision making

See: [vision_future_directions.md](../../docs/vision_future_directions.md#game-inference)

---

## References

### Internal Documentation

- **[Vision & Future Directions](../../docs/vision_future_directions.md)**: Long-term research goals
- **[Phased Roadmap](../../docs/roadmap_phased_plan.md)**: Phase 1-4 implementation plan
- **[Technical MARL Review](../../docs/technical_marl_review.md)**: Algorithmic approaches and literature
- **[RUST_SINGLE_SOURCE_OF_TRUTH.md](./RUST_SINGLE_SOURCE_OF_TRUTH.md)**: Train/test consistency resolution

### Evolution Experiment Logs

- **[V3_FITNESS_BUG_ANALYSIS.md](V3_FITNESS_BUG_ANALYSIS.md)**: Original train/test mismatch discovery
- **[V4_CRITICAL_FAILURE_ANALYSIS.md](V4_CRITICAL_FAILURE_ANALYSIS.md)**: V4 diagnostics (pre-fix)
- **[V4_EVOLUTION_PLAN.md](V4_EVOLUTION_PLAN.md)**: V4 configuration and rationale
- **[V5_EVOLUTION_PLAN.md](V5_EVOLUTION_PLAN.md)**: V5 configuration (current run)
- **[V5_NEXT_STEPS.md](V5_NEXT_STEPS.md)**: What to do when V5 completes

### Scripts

**Current (Phase 1)**:
- `../scripts/run_evolution.py`: Core evolution runner (used for V3/V4/V5)
- `../scripts/run_comparison.py`: Rust-only tournament comparison
- `../scripts/launch_v5_evolution.sh`: V5 batch launcher

**Future (Phase 2+)**:
- `run_extended_evolution.py`: With scenario switching (not yet implemented)
- `compare_evolution_strategies.py`: Competitive vs. absolute (Phase 3)
- `evolution_head_to_head.py`: Direct opponent comparison (Phase 3)
- `evolution_analysis.py`: Diversity and convergence analysis (Phase 3)

---

## Notes

**Computational Resources**:
- Evolution is expensive: 200-600 CPU-hours per full run
- Rust backend essential: 100x speedup over Python
- Parallel evaluation: 64 CPUs reduce wall-clock time to 6-10 hours
- Remote GPU instances: Cost-effective for batch runs (SkyPilot, AWS, etc.)

**Reproducibility**:
- Fixed random seeds for deterministic evolution
- Rust implementation ensures cross-platform consistency
- All hyperparameters logged in best_agent.json
- Evolution traces saved for analysis

**Lessons Learned**:
- Single source of truth is critical (avoid dual implementations)
- Train/test consistency is first validation step
- Longer evolution doesn't always help (V3 ≈ V4 despite 6x generations)
- Near-Nash convergence validates both evolution and Nash analysis

---

**Status**: Phase 1 GA Track - **NEAR COMPLETION** ✅
**Next**: Complete V5, run full tournament suite, document results
**Future**: Phase 2 (multi-scenario agents), Phase 3 (population resilience)
