# V4 Evolution - Critical Failure Analysis

**Date**: 2025-11-05
**Status**: 🚨 CRITICAL FAILURE - V4 performed identically to broken v3

## Executive Summary

**V4 evolution FAILED**: Despite fixing the Rust evaluator bug and running 15000 generations (6x longer than v3), v4 agents perform identically to the broken v3 agents. Tournament results show **catastrophic failure** with v4 ranking last place, identical to v3.

**Key Findings**:
1. ❌ V4 tournament performance: -6.97 (chain_reaction), -3.87 (deceptive_calm)
2. ❌ V3 tournament performance: -7.01 (chain_reaction), -5.40 (deceptive_calm)
3. ✅ Original "evolved" performance: 75.23 (chain_reaction), 121.35 (deceptive_calm)
4. ⚠️  **Critical discrepancy**: Training fitness 68.8 vs tournament -6.97

**Conclusion**: The fix did not work as expected. There is still a fundamental train/test mismatch.

---

## Tournament Results

### Chain Reaction

| Strategy | Mean Payoff | Std Dev | Rank |
|----------|-------------|---------|------|
| **evolved (original)** | **75.23** | 39.08 | 🥇 #1 |
| best_heuristic | 35.10 | 37.83 | #2 |
| nash_strategy_1 | 0.36 | 34.31 | #3 |
| **evolved_v4** | **-6.97** | 33.13 | #4 |
| **evolved_v3 (broken)** | **-7.01** | 34.66 | #5 |

**Result**: V4 and v3 are IDENTICAL (both around -7)

### Deceptive Calm

| Strategy | Mean Payoff | Std Dev | Rank |
|----------|-------------|---------|------|
| **evolved (original)** | **121.35** | 54.56 | 🥇 #1 |
| nash_strategy_1 | 54.04 | 55.50 | #2 |
| best_heuristic | 47.79 | 56.83 | #3 |
| **evolved_v4** | **-3.87** | 57.81 | #4 |
| **evolved_v3 (broken)** | **-5.40** | 56.94 | #5 |

**Result**: V4 marginally better than v3, but still catastrophically bad

---

## Critical Discrepancy

### Training vs Tournament Metrics

**Training (Rust evaluator)**:
- Fitness metric: `result.final_score` (sum of all agent scores)
- Chain reaction v4 fitness: **68.8**

**Tournament (Python environment)**:
- Fitness metric: `np.mean(total_rewards)` (mean of agent scores)
- Expected if equivalent: 68.8 / 4 = **17.2**
- Actual v4 payoff: **-6.97**

**Gap**: 17.2 vs -6.97 = **24.17 point difference!**

This massive discrepancy indicates:
1. Rust and Python environments behave differently
2. OR: The fix is not actually being applied during training
3. OR: There's another critical bug we haven't found

---

## Investigation Timeline

### Phase 1: Bug Discovery (v3)
- **Finding**: V3 agents performed -96% worse than original "evolved"
- **Root cause**: Rust evaluator only simulated agent 0, not all 4 agents
- **Result**: Agents learned free-riding (work_tendency=0.000)

### Phase 2: Fix Implementation
- **File**: `bucket_brigade/evolution/fitness_rust.py` (lines 99-129)
- **Change**: Loop through all `num_agents` and collect all actions
- **Commit**: `95878ba3` - "fix: Rust evaluator now simulates all 4 agents"
- **Test**: Verified on remote server (fitness: 60.64 - positive!)

### Phase 3: V4 Launch
- **Config**: 200 pop, 15000 gen, 50 games, seed=42
- **Runtime**: 7h 53min (all 9 scenarios completed)
- **Training fitness**: 68.8 (positive - looked good!)

### Phase 4: Tournament Testing
- **Result**: CATASTROPHIC FAILURE
- **V4 performance**: Same as broken v3
- **Discrepancy**: Training fitness 68.8 vs tournament -6.97

---

## Code Audit

### Fix Verification (fitness_rust.py:99-129)

**CONFIRMED**: Remote server has the fix applied.

```python
# Lines 99-122 (verified on remote)
num_agents = python_scenario.num_agents

while not done and step_count < max_steps:
    # Get actions for ALL agents (not just agent 0)
    actions = []
    for agent_id in range(num_agents):  # ✅ Loops through all agents
        obs = game.get_observation(agent_id)
        obs_dict = {
            "houses": obs.houses,
            "signals": obs.signals,
            "locations": obs.locations,
        }

        action = _heuristic_action(genome, obs_dict, agent_id, rng)
        actions.append(action)

    # Step with ALL 4 actions
    rewards, done, info = game.step(actions)  # ✅ All actions provided
    step_count += 1

# Line 128
return result.final_score  # Returns sum of agent scores
```

**Status**: ✅ Fix is correctly applied on remote

### Rust vs Python Environment

**Rust (training)**:
- Environment: `bucket_brigade_core::BucketBrigadeGame`
- Returns: `result.final_score` = sum(agent_scores)
- File: `bucket-brigade-core/src/engine/observation.rs:60`

**Python (tournament)**:
- Environment: `bucket_brigade.envs.BucketBrigadeEnv`
- Returns: `np.mean(total_rewards)` = mean(agent_scores)
- File: `experiments/scripts/run_comparison.py:122`

**Key difference**: Sum vs Mean (should be equivalent after scaling by 4)

---

## Hypotheses for Failure

### Hypothesis 1: Rust Module Not Rebuilt
- **Likelihood**: Low
- **Reason**: Fix is Python-only wrapper code, no Rust compilation needed
- **Test**: Verify Rust bindings are actually loaded

### Hypothesis 2: Rust/Python Environment Mismatch
- **Likelihood**: HIGH
- **Reason**: Training uses Rust core, tournaments use Python env
- **Evidence**: Massive fitness discrepancy (68.8 vs -6.97)
- **Test**: Run same agent through both environments and compare

### Hypothesis 3: Fix Not Applied During Evolution
- **Likelihood**: Medium
- **Reason**: Maybe evaluator is created before fix takes effect?
- **Evidence**: V4 performs identical to v3
- **Test**: Add logging to confirm fix is being executed

### Hypothesis 4: Additional Bug Not Found
- **Likelihood**: High
- **Reason**: We only fixed action collection, not reward accumulation
- **Evidence**: Fix looks correct but results are identical to broken v3
- **Test**: Audit reward calculation in Rust evaluator

---

## Next Steps

### Immediate Actions

1. **Verify Rust/Python equivalence**:
   ```bash
   # Test same agent in both environments
   python experiments/scripts/test_rust_python_parity.py chain_reaction evolved
   ```

2. **Add debug logging to fitness_rust.py**:
   ```python
   # In _run_rust_game(), add:
   print(f"DEBUG: num_agents={num_agents}, actions={len(actions)}")
   print(f"DEBUG: rewards={rewards}, final_score={result.final_score}")
   ```

3. **Run single generation with verbose output**:
   ```bash
   python experiments/scripts/run_evolution.py chain_reaction \
     --population 4 --generations 1 --games 5 \
     --output-dir /tmp/debug_v4
   ```

### Investigation Paths

**Path A: Environment Mismatch** (RECOMMENDED)
- Compare Rust `BucketBrigadeGame` with Python `BucketBrigadeEnv`
- Check if reward calculation differs
- Verify step() behavior is identical

**Path B: Fallback to Python Evaluator**
- Use slower Python evaluator for v5
- Guarantees train/test consistency
- Tradeoff: 100x slower (but correct)

**Path C: Direct Rust Environment Testing**
- Create minimal test script using Rust evaluator
- Run tournaments using Rust instead of Python
- Compare tournament results

---

## Resource Investment

### V4 Evolution Cost
- **Total runtime**: 7h 53min wall-clock
- **Total compute**: 475 CPU-hours (7.88hr × 64 vCPUs × 9 scenarios)
- **Config**: 200 pop, 15000 gen, 50 games per eval
- **Cost**: Wasted - v4 is no better than v3

### Cumulative Waste
- **V3**: 64 min × 64 vCPUs × 9 scenarios = 614 CPU-hours
- **V4**: 475 CPU-hours
- **Total**: 1089 CPU-hours wasted on broken evaluator

---

## Lessons Learned

1. **Always test fix thoroughly before long runs**: Should have run longer smoke tests
2. **Verify train/test equivalence early**: Fitness discrepancy is a red flag
3. **Python fallback has value**: Slow but guaranteed correct
4. **Add extensive logging**: Debug output would have caught this immediately
5. **Cross-validate with multiple scenarios**: Test on 2-3 scenarios before running all 9

---

## Decision Points

### Option 1: Investigate Further (Recommended)
- **Time**: 2-4 hours debugging
- **Outcome**: Understand root cause, potentially fix for v5
- **Risk**: May not find issue quickly

### Option 2: Use Python Evaluator for V5
- **Time**: Immediate (no investigation needed)
- **Outcome**: Guaranteed correct but 100x slower
- **Tradeoff**: ~30-40 hours for full run vs debugging time

### Option 3: Abandon Evolution, Use Original "Evolved"
- **Time**: Immediate
- **Outcome**: Original agents are excellent (75+ payoff)
- **Tradeoff**: Miss opportunity to improve further

---

## Recommended Path Forward

1. **Run environment parity test** (30 min)
   - Compare Rust vs Python on same agent
   - Identify if environments behave differently

2. **If mismatch found**: Fix environment inconsistency
3. **If no mismatch**: Add extensive logging and rerun smoke test
4. **If debugging exceeds 4 hours**: Fall back to Python evaluator

5. **Before any v5 run**:
   - Smoke test: 100 pop, 100 gen, 20 games
   - Verify tournament performance improves over generations
   - Only proceed to full run if smoke test succeeds

---

**Status**: V4 FAILED - Immediate investigation required
**Priority**: HIGH - Evolution pipeline is broken
**Blocker**: Cannot proceed with further evolution until root cause identified

**Next Agent Instructions**: Read this document first, then proceed with investigation per "Next Steps" section.
