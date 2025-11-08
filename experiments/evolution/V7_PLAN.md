# Evolution V7 Plan: True Heterogeneous Tournament Training

**Created**: 2025-11-07
**Status**: ✅ COMPLETE - See [V7_RESULTS_ANALYSIS_PLAN.md](V7_RESULTS_ANALYSIS_PLAN.md)
**Motivation**: V6 was never properly implemented - used homogeneous self-play instead of heterogeneous tournament fitness

---

## ⚠️ PLAN STATUS: COMPLETE

V7 evolution has finished for all 12 scenarios:
- ✅ Population: 200, Generations: 200
- ✅ Heterogeneous tournament fitness (correctly implemented)
- ✅ Correct game mechanics (post-fix)
- ✅ Data downloaded locally (~15MB)

**Next Steps**: See [V7_RESULTS_ANALYSIS_PLAN.md](V7_RESULTS_ANALYSIS_PLAN.md) for comprehensive analysis plan.

**Sample Results**:
- chain_reaction: 43.68 fitness
- deceptive_calm: 48.80 fitness (best)
- easy: 17.11 fitness
- hard: 43.21 fitness

---

## Original Plan (For Reference)

## Executive Summary

V7 will **properly implement the original V6 plan**: evolution with heterogeneous opponent pools and tournament-based fitness. This is the experiment we thought we ran in V6, but actually didn't.

**Key Insight from V6**: Homogeneous self-play training (mean_reward) fails in heterogeneous tournaments because agents never learn to handle defectors like free-rider.

## Core Hypothesis

**If we train agents explicitly against diverse opponents (including free-riders), evolution will discover strategies that are robust to exploitation in heterogeneous teams.**

## V7 Objectives

1. **Implement TRUE heterogeneous tournament fitness** (not just bigger population)
2. **Train against explicit opponent pool** including free-rider archetype
3. **Validate that proper training produces tournament-robust agents**
4. **Test if evolution CAN discover anti-free-rider strategies**

## Critical Implementation Changes from V6

### 1. Fitness Function: Heterogeneous Tournament

**V6 (as run)**:
```python
fitness_type = "mean_reward"  # Homogeneous teams only
# Team: [candidate, candidate, candidate, candidate]
```

**V7 (proper implementation)**:
```python
fitness_type = "heterogeneous_tournament"  # NEW
# Team: [candidate, random_opponent_1, random_opponent_2, random_opponent_3]
```

### 2. Opponent Pool Definition

**Fixed opponent pool**:
```python
OPPONENT_POOL = {
    "firefighter": FIREFIGHTER_PARAMS,      # Hard worker, self-focused
    "free_rider": FREE_RIDER_PARAMS,        # Defector (our nemesis!)
    "hero": HERO_PARAMS,                    # Hard worker, altruistic
    "coordinator": COORDINATOR_PARAMS,      # Balanced cooperator
    "self": candidate_genome                # Self-play component
}
```

**Why each opponent matters**:
- **firefighter**: Tests against other hard workers (should cooperate)
- **free_rider**: Forces learning anti-exploitation strategies (critical!)
- **hero**: Tests against altruists (can we benefit without exploiting?)
- **coordinator**: Tests against balanced strategies
- **self**: Maintains self-play component (important for intrinsic strategy quality)

### 3. Fitness Evaluation Process

```python
def evaluate_heterogeneous_tournament(candidate_genome, scenario, num_games=100):
    """
    Evaluate candidate in heterogeneous tournament.

    Returns mean payoff across games with randomly sampled team compositions.
    """
    total_payoff = 0

    # Create opponent agents
    opponents = {
        "firefighter": create_agent(FIREFIGHTER_PARAMS),
        "free_rider": create_agent(FREE_RIDER_PARAMS),
        "hero": create_agent(HERO_PARAMS),
        "coordinator": create_agent(COORDINATOR_PARAMS),
    }

    for game_idx in range(num_games):
        # Create candidate agent
        candidate = create_agent(candidate_genome)

        # Sample 3 random opponents (ensuring candidate is in team)
        teammate_types = random.choices(
            list(opponents.keys()),
            k=3
        )
        teammates = [opponents[t] for t in teammate_types]

        # Form team: [candidate, opponent1, opponent2, opponent3]
        team = [candidate] + teammates

        # Run game
        env = BucketBrigadeEnv(scenario)
        obs = env.reset(seed=game_idx)

        total_reward = 0
        while not env.done:
            actions = [agent.act(obs) for agent in team]
            obs, rewards, dones, info = env.step(actions)
            total_reward += rewards[0]  # Candidate is agent 0

        total_payoff += total_reward

    return total_payoff / num_games
```

**Key properties**:
- Candidate always in position 0 (gets rewards[0])
- Teammates randomly sampled each game
- Forces robustness to team composition
- Candidate faces free-rider ~25% of games (3/12 teammate slots)

## V7 Configuration

### Hyperparameters

```python
# Population parameters (keep V6 values that worked)
POPULATION_SIZE = 200           # Large population for diversity
NUM_GENERATIONS = 200           # May need more due to complex fitness landscape
ELITE_SIZE = 10                 # Keep top 10 agents

# Selection
SELECTION_STRATEGY = "tournament"
TOURNAMENT_SIZE = 5             # Moderate selection pressure

# Variation operators
CROSSOVER_STRATEGY = "uniform"
CROSSOVER_RATE = 0.7
MUTATION_STRATEGY = "gaussian"
MUTATION_RATE = 0.15            # V6 rate (higher than V5)
MUTATION_SCALE = 0.1

# Fitness evaluation
FITNESS_TYPE = "heterogeneous_tournament"  # NEW! The whole point!
GAMES_PER_INDIVIDUAL = 100      # Balance accuracy vs speed
OPPONENT_POOL = ["firefighter", "free_rider", "hero", "coordinator"]

# Diversity maintenance
MAINTAIN_DIVERSITY = True
MIN_DIVERSITY = 0.1

# Early stopping
EARLY_STOPPING = False          # Run full 200 generations
```

### Why These Parameters

**Population: 200**
- Proven effective in V6 for exploration
- Large enough to maintain diversity with heterogeneous fitness

**Generations: 200**
- Heterogeneous fitness is more complex than homogeneous
- May need more generations than V6 actually needed
- Can extend to 300-500 if 200 proves insufficient

**Mutation rate: 0.15**
- Higher than V5 (0.1) to encourage exploration
- Heterogeneous fitness landscape likely has more local optima

**Games per eval: 100**
- Balance between:
  - Accuracy (more games = better estimate)
  - Speed (fewer games = faster generations)
- With 4 opponent types × 25 appearances ≈ good coverage

## Implementation Plan

### Phase 1: Create Heterogeneous Fitness Evaluator (1-2 hours)

**New Python module**: `bucket_brigade/evolution/heterogeneous_evaluator.py`

```python
class HeterogeneousEvaluator:
    """Evaluates agents in heterogeneous tournament settings."""

    def __init__(self, scenario_name: str, opponent_pool: List[str]):
        self.scenario = get_scenario_by_name(scenario_name)
        self.opponent_pool = self._create_opponent_pool(opponent_pool)

    def _create_opponent_pool(self, opponent_types: List[str]):
        """Create fixed opponent agents."""
        # Returns dict of {type: agent}
        pass

    def evaluate(self, genome: np.ndarray, num_games: int = 100) -> float:
        """Evaluate genome in heterogeneous tournament."""
        # Returns mean payoff across random team compositions
        pass
```

**Integrate with Rust backend** (if possible):
- Use `RustFitnessEvaluator` for speed
- Pass opponent genomes explicitly
- May need Rust changes to support heterogeneous teams

**Fallback**: Pure Python implementation if Rust integration too complex

### Phase 2: Update Evolution Script (1 hour)

**Create**: `scripts/evolve_v7.py`

- Copy from `evolve_v6_simple.py`
- Replace `FitnessEvaluator` with `HeterogeneousEvaluator`
- Add opponent pool configuration
- Document what makes V7 different from V6

### Phase 3: Test on Single Scenario (2-4 hours local)

**Test scenario**: `trivial_cooperation` (V6 won here, good baseline)

```bash
# Local test (small scale)
uv run python scripts/evolve_v7.py \
  --scenario trivial_cooperation \
  --population 50 \
  --generations 50 \
  --output experiments/scenarios/trivial_cooperation/evolved_v7_test/
```

**Validation checks**:
1. Verify heterogeneous teams are actually used (add logging)
2. Check fitness values are reasonable
3. Confirm convergence behavior differs from V6
4. Ensure no crashes or numerical issues

### Phase 4: Full Evolution on Remote (12-24 hours)

**Run on**: `rwalters-sandbox-1` (32-core CPU)

```bash
# Launch all 12 scenarios in parallel
for scenario in chain_reaction deceptive_calm default early_containment \
                easy greedy_neighbor hard mixed_motivation \
                overcrowding rest_trap sparse_heroics trivial_cooperation; do

  tmux new-session -d -s "v7_$scenario" \
    "cd bucket-brigade && \
     uv run python scripts/evolve_v7.py \
       --scenario $scenario \
       --population 200 \
       --generations 200 \
       --workers 4 \
       --output experiments/scenarios/$scenario/evolved_v7/ \
       2>&1 | tee logs/evolution/v7_$scenario.log"
done
```

**Monitor**:
- Check initial generations converge properly
- Verify fitness values make sense
- Watch for unexpected behavior

### Phase 5: Tournament Validation (2-3 hours)

Same as V6 validation:

```bash
# Run heterogeneous tournaments for key scenarios
for scenario in easy trivial_cooperation hard chain_reaction; do
  uv run python experiments/scripts/run_heterogeneous_tournament.py \
    --agents evolved_v7 evolved_v6 firefighter hero free_rider coordinator \
    --scenarios $scenario \
    --num-games 100 \
    --output experiments/scenarios/$scenario/tournament_v7_results.csv

  uv run python experiments/scripts/fit_ranking_model.py \
    --data experiments/scenarios/$scenario/tournament_v7_results.csv \
    --output experiments/scenarios/$scenario/comparison/ranking_v7.json
done
```

## Expected Outcomes

### Success Criteria

**Minimum success** (validates approach):
1. V7 beats free-rider in ≥2/4 test scenarios
2. V7 ranks ≥#2 (ahead of most archetypes)
3. No catastrophic failures (fitness doesn't collapse)

**Strong success** (evolution works!):
1. V7 beats free-rider in 3/4 or 4/4 scenarios
2. V7 ranks #1 in ≥2 scenarios
3. V7 significantly beats V6 (>10% skill advantage)

**Exceptional success** (beyond expectations):
1. V7 dominates free-rider across all scenarios
2. V7 discovers novel anti-exploitation strategies
3. V7 generalizes across scenario types

### What V7 Results Will Tell Us

**If V7 succeeds** ✅:
- Confirms heterogeneous training is necessary and sufficient
- Evolution CAN discover robust multi-agent strategies
- Phase 1 complete: we know how to train agents that perform well
- Continue to Phase 2 (advanced multi-agent topics)

**If V7 partially succeeds** ⚠️:
- Some scenarios solvable, others need different approach
- May need scenario-specific training or curriculum
- Consider hybrid: evolution + RL fine-tuning
- Or: multi-scenario fitness (average across scenarios)

**If V7 fails** ❌:
- Heterogeneous tournament fitness insufficient
- May need:
  - Different fitness objectives (minimax, co-evolution)
  - More sophisticated evolutionary operators
  - Curriculum learning (start easy, increase difficulty)
  - Or pivot to PPO/MAPPO gradient-based RL

## Comparison Framework

### V6 vs V7 Key Differences

| Aspect | V6 (as run) | V7 (planned) |
|--------|-------------|--------------|
| **Fitness** | mean_reward | heterogeneous_tournament |
| **Opponents** | Self-clones only | 4 archetypes + self |
| **Training teams** | Homogeneous | Heterogeneous |
| **Selection pressure** | Simple landscape | Complex landscape |
| **Expected convergence** | Fast (gen 20) | Gradual (gen 100-150?) |
| **Free-rider exposure** | Never | ~25% of games |
| **Tournament robustness** | Poor (1/4 wins) | Unknown (testing!) |

### Success Metrics

Track for each scenario:

1. **Training fitness progression**
   - Compare V7 vs V6 convergence curves
   - V7 should converge slower (more complex)

2. **Tournament rankings**
   - V7 vs free-rider head-to-head
   - V7 vs V6 comparison
   - V7 vs all archetypes

3. **Strategy profiles**
   - Does V7 discover different parameters than V6?
   - Anti-free-rider characteristics (lower work_tendency? higher self-priority?)

4. **Robustness metrics**
   - Performance variance across team compositions
   - Worst-case team composition payoff
   - Exploitation resistance score

## Risk Mitigation

### Risk 1: Implementation Complexity

**Risk**: Heterogeneous evaluator is complex, may have bugs

**Mitigation**:
- Unit tests for opponent sampling
- Validate fitness values match hand-calculated examples
- Test on trivial_cooperation first (known V6 baseline)
- Log team compositions in early generations

### Risk 2: Computational Cost

**Risk**: Heterogeneous eval may be slower than homogeneous

**Mitigation**:
- Use Rust backend if possible (100x speedup)
- Parallel evaluation across cores
- If too slow: reduce games_per_individual to 50
- Monitor first scenario, adjust if needed

### Risk 3: Convergence Failure

**Risk**: Complex fitness landscape may prevent convergence

**Mitigation**:
- Start with proven V6 hyperparameters
- Increase generations to 300-500 if needed
- Try higher mutation rate (0.2) if stuck
- Add fitness shaping if raw payoff too noisy

### Risk 4: Free-Rider Still Wins

**Risk**: Even with training, free-rider may be unbeatable

**Mitigation**:
- This is valuable negative result!
- Reveals fundamental game-theoretic properties
- Pivot to mechanism design (change game rules)
- Or accept and focus on other research questions

## Alternative Approaches (If V7 Fails)

### Plan B: Multi-Scenario Fitness

Instead of per-scenario evolution, train one agent for all scenarios:

```python
fitness = mean([
    heterogeneous_tournament(agent, "easy"),
    heterogeneous_tournament(agent, "trivial_cooperation"),
    heterogeneous_tournament(agent, "hard")
])
```

**Pros**: Forces generalization
**Cons**: May be mediocre at all instead of great at any

### Plan C: Curriculum Learning

Start with cooperative opponents, gradually introduce free-riders:

```
Gen 0-50:   opponent_pool = [firefighter, hero, coordinator]
Gen 51-100: Add free_rider at 10% frequency
Gen 101-150: free_rider at 25% frequency
Gen 151-200: free_rider at 50% frequency
```

**Pros**: Gives agents time to learn cooperation first
**Cons**: May not develop robust anti-exploitation

### Plan D: Co-Evolution

Evolve two populations simultaneously:
- Population A: Cooperative agents
- Population B: Exploitative agents

**Pros**: Arms race drives innovation
**Cons**: Much more complex, may be unstable

### Plan E: Pivot to PPO

If evolution fundamentally limited:
- Implement PPO/MAPPO training
- Use V7 best agent as initialization
- Compare gradient-based RL vs evolution

## Timeline

**Total estimated time**: 3-5 days

- **Day 1** (8 hours):
  - Implement HeterogeneousEvaluator
  - Create evolve_v7.py script
  - Test on trivial_cooperation locally

- **Day 2** (2 hours + overnight):
  - Fix any bugs from testing
  - Launch full evolution on remote (runs overnight)

- **Day 3** (4 hours):
  - Collect results
  - Run tournament validation
  - Analyze convergence patterns

- **Day 4** (4 hours):
  - Compare V7 vs V6 vs archetypes
  - Generate visualizations
  - Write results analysis

- **Day 5** (4 hours):
  - Update web dashboard
  - Document findings
  - Plan next iteration (V8 or Phase 2)

## Success Indicators (Quick Check)

After first scenario completes, check:

1. ✅ **Fitness values reasonable**: Should be lower than V6 initially (harder task)
2. ✅ **Gradual convergence**: Should NOT converge by gen 20 like V6
3. ✅ **Strategy differs from V6**: Different genome parameters
4. ✅ **Tournament validation**: Ranks better than V6 vs free-rider

If 3+ indicators positive → Continue full run
If 2+ indicators negative → Debug before full run

## Documentation Requirements

### During Evolution

- `experiments/scenarios/{scenario}/evolved_v7/`
  - `best_agent.json` - Best evolved agent
  - `evolution_results.json` - Full trace
  - `evolution_log.txt` - Human-readable summary
  - `checkpoint_gen*.json` - Every 20 generations

### After Completion

- `experiments/evolution/V7_RESULTS_ANALYSIS.md`
  - Fitness progression vs V6
  - Strategy characterization
  - Tournament performance
  - Success/failure analysis

- `experiments/evolution/V7_TOURNAMENT_SUMMARY.md`
  - Head-to-head: V7 vs free-rider
  - V7 vs V6 comparison
  - Ranking across scenarios
  - Key insights and implications

## Phase 1 Completion Criteria

V7 must achieve for Phase 1 success:

1. ✅ Reproducible training with heterogeneous opponents
2. ✅ Tournament performance matches or predicts training fitness
3. ✅ Beats or competes with free-rider baseline
4. ✅ Generalizes reasonably across ≥50% of scenarios
5. ✅ Clear understanding of what makes strategies robust

**If V7 succeeds**: Phase 1 complete, document methodology, move to Phase 2
**If V7 fails**: Iterate with Plan B/C/D or pivot to PPO

## Key Takeaways

1. **V7 is the experiment we should have run as V6**
2. **This tests the core hypothesis**: Heterogeneous training → robust agents
3. **Results will definitively answer**: Can evolution handle multi-agent exploitation?
4. **Clear success criteria**: Beat free-rider or understand why not
5. **Well-defined fallbacks**: Multiple paths forward regardless of outcome

---

**Status**: Ready to implement
**Confidence**: High (fixing known bug, not exploring new idea)
**Risk**: Low (worst case: learn evolution insufficient, pivot to RL)
**Expected outcome**: V7 beats free-rider in ≥2/4 scenarios, validates approach

**Next step**: Create `HeterogeneousEvaluator` class and test on single scenario
