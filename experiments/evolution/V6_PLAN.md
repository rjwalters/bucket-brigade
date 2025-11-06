# Evolution V6 Plan

**Created**: 2025-11-06
**Motivation**: V5 tournament validation revealed -4.7% underperformance vs V4 despite +1,162% Nash equilibrium improvement

## Problem Statement

V5 optimized for Nash equilibrium but failed in tournament play:
- **Nash Theory**: V5 predicted to dominate (+1,162% payoff improvement)
- **Tournament Reality**: V5 loses to V4 (-4.7% across 3 scenarios)
- **Root Cause**: Evolved against equilibrium assumptions, not robust gameplay

## V6 Objectives

1. **Tournament-Based Fitness**: Evolve directly against tournament performance metrics
2. **Heterogeneous Opposition**: Train against diverse agent types (archetypes + evolved variants)
3. **Generalization**: Optimize for robust performance across varied team compositions

## Key Changes from V5

### 1. Fitness Function: Tournament Performance
```python
# V5: Nash equilibrium expected payoff
fitness = nash_equilibrium_solver(agent, scenario)

# V6: Actual tournament payoff
fitness = tournament_average_payoff(agent, opponent_pool, num_games=100)
```

### 2. Opponent Pool Diversity
- **V5**: Evolved against 5 archetypes + V4
- **V6**: Evolve against 5 archetypes + V4 + V5 + random mutations
  - 8 total opponent types
  - Ensures robust performance against unexpected strategies

### 3. Multi-Scenario Evolution
- Train on 3 core scenarios simultaneously:
  - chain_reaction (most challenging)
  - trivial_cooperation (baseline)
  - deceptive_calm (strategic complexity)
- Fitness = average performance across all 3

### 4. Population Parameters

**Increased diversity to avoid overfitting:**
```python
POPULATION_SIZE = 200  # (was 100 in V5)
GENERATIONS = 200      # (was 100 in V5)
MUTATION_RATE = 0.15   # (was 0.1 in V5 - more exploration)
TOURNAMENT_SIZE = 5    # (selection pressure)
ELITISM = 10           # Keep top 10 agents
```

### 5. Evaluation Protocol

**Training fitness** (fast, 100 games):
- Sample 100 random team compositions
- Mix of opponent types
- Quick feedback for evolution

**Validation fitness** (slower, 500 games):
- Run every 10 generations
- Full heterogeneous tournament
- Track generalization progress

## Implementation Plan

### Phase 1: Setup (Local)
1. Create V6 evolution script with tournament-based fitness
2. Test on single scenario (chain_reaction) with small population
3. Validate fitness function correlates with tournament performance

### Phase 2: Full Evolution (Remote CPU)
1. Launch 3-scenario parallel evolution on rwalters-sandbox-1
2. Run for 200 generations (~12-24 hours)
3. Save checkpoints every 20 generations

### Phase 3: Validation (Local)
1. Run 500-game validation tournaments
2. Compare V6 vs V5 vs V4 vs archetypes
3. Analyze if tournament fitness solved the robustness problem

## Expected Outcomes

**Success Criteria:**
1. V6 beats V4 in tournaments by >5%
2. V6 beats V5 in tournaments (proving robustness matters)
3. Consistent performance across all 3 scenarios
4. No catastrophic failure modes against any opponent type

**If Successful:**
- Validates tournament-based fitness for multi-agent RL
- Demonstrates evolution can produce robust policies
- V6 becomes new baseline for future research

**If Unsuccessful:**
- Suggests evolution may be fundamentally limited
- Reinforces need for PPO/RL gradient-based methods
- Hybrid approach (evolution + RL fine-tuning) becomes priority

## Resource Estimates

**Compute:**
- Population: 200 agents
- Generations: 200
- Fitness evals: 100 games per agent per generation
- Total games: 200 × 200 × 100 = 4M games
- Time estimate: 12-24 hours on 32-core CPU (rwalters-sandbox-1)

**Storage:**
- Evolution traces: ~50MB per scenario
- Best agents: 3 × 1KB = 3KB
- Checkpoints: 20 × 200 agents × 1KB = 4MB

## Launch Command

```bash
# On rwalters-sandbox-1
cd bucket-brigade

# Launch V6 evolution for all 3 scenarios
./experiments/scripts/launch_v6_evolution.sh \
  --scenarios chain_reaction trivial_cooperation deceptive_calm \
  --population 200 \
  --generations 200 \
  --mutation-rate 0.15 \
  --opponent-pool archetypes v4 v5 mutations \
  --fitness tournament \
  --games-per-eval 100 \
  --output experiments/scenarios/*/evolved_v6/
```

## Fallback Plan

If V6 doesn't improve tournament performance:
1. **Hybrid Evolution + RL**: Use V6 as initialization for PPO training
2. **Direct RL**: Abandon evolution, focus 100% on PPO
3. **Curriculum Learning**: Start with easy opponents, gradually increase difficulty

## Next Steps

1. Commit V5 tournament results and summary
2. Push tournament data and V6 plan to GitHub
3. Create V6 evolution script with tournament fitness
4. Launch V6 on rwalters-sandbox-1
5. Monitor progress and compare vs V5/V4 benchmarks
