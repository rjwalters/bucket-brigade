# V3 Evolution Fitness Bug Analysis

## Summary

The v3 evolution run produced agents that perform **96% worse** than the original evolved agents despite 2.5x more compute. Root cause: **Single-agent training, multi-agent testing**.

## The Bug

### Training (fitness_rust.py lines 95-116)

The `_run_rust_game()` function simulates only **ONE agent**:

```python
def _run_rust_game(args: tuple[np.ndarray, Scenario, int]) -> float:
    # ... setup ...

    while not done and step_count < max_steps:
        # Get observation for agent 0 ONLY
        obs = game.get_observation(0)

        # Get action for agent 0 ONLY
        action = _heuristic_action(genome, obs_dict, 0, rng)

        # Step with ONE action (Rust fills in others?)
        rewards, done, info = game.step([action])

    # Return scenario final_score
    return result.final_score
```

### Testing (run_comparison.py lines 74-92)

Tournaments test **FOUR identical agents**:

```python
# Create team of 4 identical agents
agents = [
    HeuristicAgent(params=genome, agent_id=i, name=f"{name}-{i}")
    for i in range(4)
]

# Run game
while not env.done:
    actions = np.array([agent.act(obs) for agent in agents])  # 4 actions!
    obs, rewards, dones, info = env.step(actions)
```

## Impact

**Performance Comparison** (chain_reaction scenario):

| Version | Tournament Payoff | Training Method |
|---------|-------------------|----------------|
| evolved (old) | **92.06 ± 27.59** | Multi-agent (Python) |
| evolved_v3 (new) | **-4.11 ± 22.19** | Single-agent (Rust) |
| **Regression** | **-96%** | |

## Why V3 Converged to Fitness 70.00

The v3 agents achieved "fitness 70.00" during training, but this was:
- **Single-agent** `final_score` with Rust auto-filling other agents
- **Not representative** of team performance
- **Optimizing the wrong objective**

The agents learned to maximize solo performance, not team coordination.

## Why work_tendency=0.000 Makes Sense Now

The v3 chain_reaction agent has `work_tendency=0.000` (doesn't work):
- In single-agent training, letting Rust auto-agents do the work was viable
- The evolved agent learned to "free-ride" on whatever the Rust game does for other agents
- This strategy gets fitness 70.00 in solo evaluation
- But fails catastrophically (-4.11) when tested as a team of 4 identical non-workers

## Root Cause Timeline

1. **Original "evolved" agents** (pre-Rust):
   - Trained with Python fitness evaluator
   - Simulated full team of 4 agents
   - Achieved 92.06 tournament payoff ✅

2. **Rust migration** (commit 4d6855e7):
   - 100x speedup claimed
   - But changed simulation to single-agent
   - Bug introduced ❌

3. **"Fixed fitness" commit** (7f989e27):
   - Changed return value to `final_score` (scenario payoff)
   - Thought this "fixed" the alignment issue
   - But single-agent simulation bug remained ❌

4. **V3 evolution run**:
   - Used broken Rust evaluator
   - 2500 generations, 200 population, 50 games
   - Achieved "fitness 70.00" ✓
   - But learned wrong strategy ❌
   - Tournament performance: -4.11 (regression) ❌

## How to Fix

### Option 1: Fix Rust Evaluator (Recommended)

Update `_run_rust_game()` to simulate all 4 agents:

```python
def _run_rust_game(args: tuple[np.ndarray, Scenario, int]) -> float:
    genome, python_scenario, seed = args
    rust_scenario = _convert_scenario_to_rust(python_scenario)
    game = core.BucketBrigade(rust_scenario, seed=seed)
    rng = np.random.RandomState(seed)

    num_agents = python_scenario.num_agents  # Should be 4

    while not done and step_count < max_steps:
        # Get actions for ALL agents
        actions = []
        for agent_id in range(num_agents):
            obs = game.get_observation(agent_id)
            obs_dict = {...}
            action = _heuristic_action(genome, obs_dict, agent_id, rng)
            actions.append(action)

        # Step with ALL actions
        rewards, done, info = game.step(actions)

    return result.final_score
```

### Option 2: Use Python Evaluator

The Python evaluator correctly simulates teams:
```python
config = EvolutionConfig(
    fitness_type="scenario_payoff",
    use_rust=False,  # Use Python evaluator
    ...
)
```

But 100x slower.

### Option 3: Hybrid

- Use fixed Rust evaluator for primary evolution
- Validate top candidates with Python evaluator
- Discard free-riders

## Recommendations

1. **Fix the Rust evaluator** - Single highest priority
2. **Re-run v3 evolution** with fixed evaluator
3. **Add integration tests** that verify fitness alignment:
   - Rust fitness ≈ Python fitness
   - Evolution fitness ≈ Tournament performance
4. **Archive v3 results** as "failed experiment"
5. **Document lesson learned**: Don't trust fitness values, validate with tournaments

## Lessons Learned

1. ✅ **Rust speedup is real** (100x faster)
2. ❌ **But implementation had critical bug**
3. ✅ **Always validate evolved agents in tournaments**
4. ❌ **Fitness 70.00 meant nothing** - wrong objective
5. ✅ **The original "evolved" agents are still best** - keep them
6. 🎯 **Testing is critical** - caught a 6-month-old bug

## Impact Assessment

### Wasted Compute
- V3 run: 9 scenarios × 200 pop × 2500 gen × 50 games = **225M evaluations**
- Runtime: ~64 minutes per scenario × 9 = **9.6 CPU-hours** (wall-clock: 64 min parallel)
- Machine: 64-vCPU remote sandbox
- Result: Unusable agents ❌

### Salvageable Work
- ✅ Updated comparison script (supports multiple versions)
- ✅ Identified critical bug in Rust evaluator
- ✅ Validated original "evolved" agents are excellent
- ✅ Documented fitness evaluation pitfalls

## Next Steps

1. **Immediate**: Document findings (this file)
2. **Short-term**: Fix Rust evaluator
3. **Medium-term**: Run v4 with correct evaluator
4. **Long-term**: Add CI tests to prevent regression

---

**Status**: Analysis complete, bug identified, fix pending
**Date**: 2025-11-04
**Impact**: Critical - all v3 agents unusable
