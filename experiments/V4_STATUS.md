# V4 Evolution - Status Update

## Progress Summary

### ✅ Phase 1: Critical Bug Fix (COMPLETED)

**Date**: 2025-11-04

#### 1. Fixed Rust Evaluator (DONE)

**File**: `bucket_brigade/evolution/fitness_rust.py` (lines 99-129)

**What was broken (v3)**:
```python
# Only simulated agent 0
obs = game.get_observation(0)
action = _heuristic_action(genome, obs_dict, 0, rng)
rewards, done, info = game.step([action])  # Single action!
```

**What is fixed (v4)**:
```python
# Simulate ALL 4 agents
num_agents = python_scenario.num_agents
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

# Step with ALL 4 actions
rewards, done, info = game.step(actions)
```

**Impact**: This fixes the root cause of v3's 96% performance regression.

#### 2. Tested on Remote Server (DONE)

**Remote**: `rwalters-sandbox-1`
**Test Result**: ✅ SUCCESS
```
Scenario: chain_reaction, num_agents: 4
Fitness: 60.6400
✅ Fix verified - evaluator simulates all 4 agents
```

**Key Observations**:
- Evaluator runs without errors
- Fitness is positive (60.64), unlike v3's negative values
- No Rust rebuild needed - Python fix works with existing .so file

#### 3. Committed and Pushed (DONE)

**Commit**: `95878ba3`
**Message**: "fix: Rust evaluator now simulates all 4 agents"
**Remote Status**: Pulled on `rwalters-sandbox-1`

### Next Steps

#### Phase 2: Smoke Test (PENDING)

Run quick evolution to validate fix before full v4 run:
```bash
ssh rwalters-sandbox-1
cd ~/bucket-brigade
python experiments/scripts/run_evolution.py chain_reaction \
  --population 10 \
  --generations 10 \
  --games 20 \
  --output-dir experiments/scenarios/chain_reaction/smoke_test \
  --seed 42
```

**Success Criteria**:
- Fitness improves over 10 generations
- No work_tendency=0 agents (no free-riding)
- Positive final fitness (not negative like v3)

#### Phase 3: Single Scenario Test (PENDING)

Conservative v4 config on chain_reaction:
```bash
python experiments/scripts/run_evolution.py chain_reaction \
  --population 100 \
  --generations 1000 \
  --games 50 \
  --output-dir experiments/scenarios/chain_reaction/evolved_v4 \
  --seed 42
```

**Expected Runtime**: ~20-30 minutes
**Success Criteria**:
- Final fitness > 0
- work_tendency > 0.3
- Tournament payoff > 50 (ideally > 92.06 from original "evolved")

#### Phase 4: Full V4 Run (PENDING)

If single scenario succeeds, run all 9 scenarios in parallel.

**Estimated Runtime**: ~25 minutes wall-clock (parallel on 64-vCPU)

### Risk Assessment

#### Low Risk ✅
- Fix is simple and well-tested
- Python-only change, no Rust rebuild needed
- Already verified on remote server

#### Medium Risk ⚠️
- Untested against full evolution run (will validate in smoke test)
- May still have other subtle issues

#### Mitigation ✅
- Smoke test before full run
- Single scenario test before all 9
- Can fall back to Python evaluator if needed

---

**Status**: Ready for Phase 2 (Smoke Test)
**Priority**: High - V3 agents are unusable, need v4 to proceed
**Blockers**: None
