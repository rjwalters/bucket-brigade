# V4 Evolution Plan - Learning from V3 Failure

## Overview

Plan for evolution v4 with corrected Rust evaluator and lessons learned from v3 failure.

**Goal**: Attempt to beat the successful "evolved" agents (92.06 payoff on chain_reaction) with proper multi-agent training.

## Prerequisites

### 1. Fix Rust Evaluator (BLOCKING)

**File**: `bucket_brigade/evolution/fitness_rust.py`

**Current (broken)**:
```python
def _run_rust_game(args):
    # Only simulates agent 0
    obs = game.get_observation(0)
    action = _heuristic_action(genome, obs_dict, 0, rng)
    rewards, done, info = game.step([action])  # Single action
```

**Fixed (required)**:
```python
def _run_rust_game(args):
    num_agents = python_scenario.num_agents  # Should be 4

    while not done:
        actions = []
        for agent_id in range(num_agents):
            obs = game.get_observation(agent_id)
            obs_dict = {
                "houses": obs.houses,
                "signals": obs.signals,
                "locations": obs.locations,
            }
            action = _heuristic_action(genome, obs_dict, agent_id, rng)
            actions.append(action)

        rewards, done, info = game.step(actions)  # All 4 actions

    return result.final_score
```

### 2. Add Integration Tests

**File**: `tests/test_fitness_alignment.py` (new)

```python
def test_rust_python_fitness_alignment():
    """Verify Rust and Python evaluators produce similar fitness."""
    scenario = get_scenario_by_name("chain_reaction", num_agents=4)
    genome = np.random.rand(10)

    # Rust fitness
    rust_eval = RustFitnessEvaluator(scenario, games_per_individual=10, seed=42)
    rust_fitness = rust_eval.evaluate_individual(Individual(genome))

    # Python fitness
    python_eval = PythonFitnessEvaluator(scenario, games_per_individual=10, seed=42)
    python_fitness = python_eval.evaluate_individual(Individual(genome))

    # Should be within 10% (allowing for implementation differences)
    assert abs(rust_fitness - python_fitness) < 0.1 * abs(python_fitness)

def test_evolution_tournament_alignment():
    """Verify evolution fitness correlates with tournament performance."""
    # Run mini evolution (10 pop, 10 gen)
    # Get best agent
    # Run tournament
    # Verify: higher evolution fitness → higher tournament payoff
    pass
```

### 3. Validation Test Suite

Before running v4, validate:
- ✅ Rust evaluator simulates all 4 agents
- ✅ Integration tests pass
- ✅ Smoke test: Evolve for 10 generations, tournament performance improves

## V4 Configuration

### Conservative Approach (Recommended)

Match the successful "evolved" run configuration:

```python
EvolutionConfig(
    population_size=100,          # Original successful size
    num_generations=1000,         # More than original's 200
    games_per_individual=50,      # More games for stability
    elite_size=5,
    selection_strategy="tournament",
    tournament_size=3,
    crossover_rate=0.7,
    mutation_rate=0.1,
    mutation_scale=0.1,
    maintain_diversity=True,
    min_diversity=0.1,
    early_stopping=False,         # Let it run full course
    seed=42,                      # For reproducibility
)
```

**Rationale**:
- Original "evolved" agents worked well with pop=100
- Increase generations (1000 vs 200) for better convergence
- Use fixed Rust evaluator for 100x speedup
- Conservative approach minimizes risk

**Estimated Runtime**:
- Per scenario: ~20-30 minutes (faster than v3 due to smaller population)
- Total: 9 scenarios × 25 min = **3.75 CPU-hours** (wall-clock: ~25 min parallel)

### Aggressive Approach (Alternative)

If conservative approach works, try:

```python
EvolutionConfig(
    population_size=200,          # More exploration
    num_generations=2000,         # Longer convergence
    games_per_individual=100,     # Maximum stability
    # ... rest same as v3 config
)
```

**Estimated Runtime**: ~2 hours per scenario, 18 CPU-hours total

## Execution Plan

### Phase 1: Validation (Local)

1. **Fix Rust evaluator**
   - Update `fitness_rust.py`
   - Test manually with chain_reaction

2. **Add tests**
   - Create `test_fitness_alignment.py`
   - Run: `pytest tests/test_fitness_alignment.py -v`
   - All tests must pass ✅

3. **Smoke test**
   ```bash
   python experiments/scripts/run_evolution.py chain_reaction \
     --population 10 \
     --generations 10 \
     --games 20 \
     --output-dir experiments/scenarios/chain_reaction/smoke_test \
     --seed 42
   ```

   Verify:
   - Fitness improves over 10 generations
   - Tournament performance better than random
   - No work_tendency=0 agents

### Phase 2: Single Scenario Test (Remote)

Run conservative config on one scenario:

```bash
ssh rwalters-sandbox-1
cd ~/bucket-brigade
python experiments/scripts/run_evolution.py chain_reaction \
  --population 100 \
  --generations 1000 \
  --games 50 \
  --output-dir experiments/scenarios/chain_reaction/evolved_v4 \
  --seed 42
```

**Success Criteria**:
- Final fitness > 0 (not negative like v3)
- work_tendency > 0.3 (not free-riding)
- Tournament payoff > 50 (better than random, ideally > 92.06)

**If successful**: Proceed to Phase 3
**If failed**: Debug, adjust config, retry

### Phase 3: Full Run (Remote)

Run all 9 scenarios in parallel:

```bash
ssh rwalters-sandbox-1
cd ~/bucket-brigade
./scripts/launch_specialists_v4.sh
```

Script should:
- Use fixed conservative config
- Launch 9 tmux windows
- Save to `evolved_v4/` directories
- Log to `logs/evolution/*_v4_*.log`

### Phase 4: Validation

After completion:

```bash
# Retrieve results
rsync -avz rwalters-sandbox-1:~/bucket-brigade/experiments/scenarios/*/evolved_v4/ \
  ./experiments/scenarios/

# Run tournaments
for scenario in chain_reaction deceptive_calm early_containment greedy_neighbor \
                mixed_motivation overcrowding rest_trap sparse_heroics trivial_cooperation; do
    python experiments/scripts/run_comparison.py $scenario \
      --evolution-versions evolved evolved_v4 \
      --num-games 100
done
```

**Success Criteria**:
- All 9 scenarios complete without errors
- Fitness values positive and reasonable
- Tournament performance ≥ original "evolved" agents
- No degenerate strategies (work_tendency=0, etc.)

## Risk Mitigation

### Risk 1: Rust Fix Introduces New Bugs

**Mitigation**:
- Comprehensive integration tests
- Compare with Python evaluator
- Smoke test before full run

### Risk 2: V4 Still Performs Worse

**Possible Causes**:
- Original "evolved" used Python evaluator (different behavior)
- Rust evaluator has other subtle bugs
- Configuration needs tuning

**Mitigation**:
- Start with conservative config (match original)
- A/B test: One scenario with Python, one with Rust
- If Rust still fails, may need to use Python evaluator (100x slower but correct)

### Risk 3: Compute Resource Constraints

**Mitigation**:
- Conservative approach uses less compute
- Can run sequentially if parallel unavailable
- Checkpoint every 100 generations for recovery

## Fallback Options

### Option A: Python Evaluator

If Rust evaluator continues to have issues:

```python
EvolutionConfig(
    # ... same config ...
    use_rust=False,  # Force Python evaluator
)
```

**Tradeoff**: 100x slower, but guaranteed correct
**Runtime**: ~50 hours total (acceptable for overnight run)

### Option B: Hybrid Approach

- Use Rust for initial exploration (0-500 gen)
- Switch to Python for refinement (500-1000 gen)
- Best of both worlds: speed + correctness

### Option C: Accept Current Results

If v4 doesn't beat "evolved":
- Document that "evolved" agents remain state-of-the-art
- Focus research on understanding why they work so well
- Use v4 results for comparative analysis

## Success Metrics

### Minimum Viable Success

- ✅ All 9 scenarios complete without errors
- ✅ Agents are not degenerate (no free-riding)
- ✅ Tournament performance > random baseline

### Target Success

- ✅ Average tournament performance ≥ original "evolved" (+0% regression)
- ✅ At least 5/9 scenarios beat original
- ✅ Fitness values align with tournament performance

### Stretch Success

- ✅ Average tournament performance > original (+10% improvement)
- ✅ All 9 scenarios beat or match original
- ✅ New insights about optimal strategies

## Timeline

### Week 1: Preparation
- Day 1-2: Fix Rust evaluator
- Day 3: Add integration tests
- Day 4: Smoke test and validation
- Day 5: Review and adjust

### Week 2: Execution
- Day 1: Single scenario test (remote)
- Day 2: Analyze results, adjust if needed
- Day 3: Launch full run (remote, overnight)
- Day 4: Retrieve and validate results
- Day 5: Tournament comparisons and documentation

### Week 3: Analysis
- Document findings
- Compare to original "evolved"
- Update website if results are positive
- Plan next steps

## Documentation

After v4 completion, update:

1. **INTENSIVE_EVOLUTION_RESULTS.md**
   - Add v4 results section
   - Compare v3 (failed) vs v4 (fixed)
   - Document whether v4 beat original

2. **V4_RESULTS_ANALYSIS.md** (new)
   - Detailed performance comparison
   - Parameter analysis
   - Lessons learned

3. **README or main docs**
   - Update with current best agents
   - Link to v4 results

## Open Questions

1. **Why did Python evaluator work better originally?**
   - Was it just the multi-agent simulation?
   - Or are there other differences in game logic?

2. **Is 100x Rust speedup worth complexity?**
   - If Rust keeps having bugs, maybe stick with Python
   - Python is slower but battle-tested

3. **Should we try co-evolution?**
   - Evolve against evolving opponents
   - Might prevent free-riding
   - More complex to implement

4. **What about ensemble strategies?**
   - Combine multiple evolved agents
   - Diversity might be beneficial
   - Worth exploring post-v4

## References

- Original success: `experiments/scenarios/*/evolved/`
- V3 failure analysis: `experiments/V3_FITNESS_BUG_ANALYSIS.md`
- Comparison script: `experiments/scripts/run_comparison.py`
- Evolution script: `experiments/scripts/run_evolution.py`

---

**Status**: Plan complete, awaiting Rust evaluator fix
**Priority**: High - original "evolved" agents are good but could be better
**Next Action**: Fix `fitness_rust.py` to simulate all 4 agents
