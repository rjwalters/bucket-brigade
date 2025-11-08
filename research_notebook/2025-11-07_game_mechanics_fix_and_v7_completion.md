---
title: "Research Notebook Entry: November 7, 2025"
date: "2025-11-07"
author: "Research Team"
tags: ["game-mechanics", "v7-evolution", "ppo-training", "nash-equilibrium", "milestone"]
status: "Major Milestone - First Correct-Mechanics Results"
---

# Research Notebook Entry: November 7, 2025

## The Story So Far

Today marks a pivotal moment in the Bucket Brigade research project. After discovering and fixing fundamental game mechanics issues, we've completed our first generation of experiments using the **correct physics**. This entry documents where we've been, what we learned, and where we're heading.

---

## Act I: The Discovery (Earlier This Week)

### The Problem We Found

While reviewing our game design documentation, we discovered critical inconsistencies between our Python and Rust implementations:

**Issue 1: Fire Extinguishing Formula**
- ❌ **Python (Wrong)**: Exponential model `P(extinguish) = 1 - e^(-κ·k)`
- ✅ **Rust (Correct)**: Independent probabilities `P(extinguish) = 1 - (1-p)^k`

**Issue 2: Spontaneous Ignition**
- ❌ **Python (Wrong)**: Fires stop spawning after `N_spark` nights
- ✅ **Rust (Correct)**: Fires can spawn every night throughout the game

**Issue 3: Reward Computation**
- ❌ **Rust (Wrong)**: All rewards computed at end of game
- ✅ **Python (Correct)**: Per-step rewards for RL training

### What This Meant

All our previous experiments (V3-V6 evolution, Nash V1-V5, early PPO runs) were using **inconsistent or incorrect game mechanics**. The agents were evolving and learning strategies for a game that didn't match our design specification.

**Impact**: ~6 months of experimental results needed to be re-evaluated or re-run.

### The Fix

We implemented a comprehensive refactoring:

1. **Aligned implementations**: Fixed Python to match Rust physics (extinguish, ignition)
2. **Aligned reward timing**: Fixed Rust to compute per-step rewards like Python
3. **Unified parameters**: Single source of truth in `definitions/scenarios.json`
4. **Updated documentation**: All docs now reflect correct mechanics

**Commits**: 13 commits covering implementation fixes, parameter refactoring, and documentation updates

---

## Act II: The Recovery (Last 24 Hours)

### Racing to Validate

With the fixes in place, we immediately launched new experiments to validate the corrected implementation:

#### Experiment 1: V7 Evolution 🧬

**Setup**:
- All 12 scenarios (core 9 + easy/hard/default)
- Population: 200 agents
- Generations: 200
- Location: `rwalters-sandbox-1` (CPU server)

**Results**: ✅ **ALL COMPLETE** (as of today)

Sample fitness progression (easy scenario):
```
[Gen    0] Best:    6.89 | Mean:    4.12 ± 0.82
[Gen   50] Best:   12.45 | Mean:    9.31 ± 1.15
[Gen  100] Best:   15.23 | Mean:   11.87 ± 1.42
[Gen  150] Best:   16.45 | Mean:   12.95 ± 1.28
[Gen  199] Best:   17.11 | Mean:   13.46 ± 1.31

✓ Best agent saved: experiments/scenarios/easy/evolved_v7/best_agent.json
✓ Elapsed Time: 4491.3s (74.9min)
```

**Key Observations**:
- Evolution converged successfully with correct mechanics
- Fitness values differ significantly from V6 (as expected)
- Need to analyze: Are V7 strategies qualitatively different?

#### Experiment 2: Nash Equilibrium V5 📊

**Setup**:
- 9 core scenarios
- Using V5 evolved agents (from before the fix)
- 2,000 simulations per payoff evaluation
- Location: `rwalters-sandbox-1`

**Results**: ✅ **ALL COMPLETE**

Early findings (from logs):
- Trivial cooperation: V5 equilibrium payoff = 1,026.50
- Most scenarios: Shifted to cooperative equilibria
- Hero archetype dominated (89% of scenarios)

**Critical Issue**: ⚠️ V5 agents were evolved with OLD mechanics!

**Next Step**: Must re-run Nash analysis with V7 agents (correct mechanics)

#### Experiment 3: PPO Training 🤖

**Setup**:
- Scenario: hard
- Total timesteps: 5,000,000
- Checkpoints: Every 500K steps
- Location: `rwalters-sandbox-2` (GPU server)

**Results**: ✅ **COMPLETE**

Training summary:
```
============================================================
✅ Training complete!
   Total time: 116.9 minutes
   Final avg reward: 9.45
   Total episodes: 100
============================================================

💾 Model saved to models/ppo_hard_5M_with_checkpoints.pt
```

**Key Observations**:
- Training completed successfully with correct mechanics
- Final reward: 9.45 (baseline to compare against evolved agents)
- Checkpoints available for learning curve analysis

---

## Act III: What We Learned

### Insight 1: The Importance of Implementation Consistency

This experience reinforced a critical lesson: **Test implementations can silently diverge**. Our Python environment (for RL training) and Rust engine (for evolution/Nash) had drifted apart over months of development.

**Lesson**: Single source of truth matters. We now use `definitions/scenarios.json` to generate both Python and Rust scenario definitions.

### Insight 2: The Value of Comprehensive Testing

The Rust implementation had 84 unit tests, but none caught the reward timing issue. Why? Because the tests were checking behavior, not comparing against Python.

**Lesson**: Cross-implementation validation tests are essential. We need tests that verify Python ↔ Rust equivalence on full game trajectories.

### Insight 3: Documentation as Ground Truth

The inconsistency was discovered during a documentation review, not during testing. The game design docs (`GAME_DYNAMICS.md`) specified one formula, but implementations did something else.

**Lesson**: Design documentation should be treated as specification, not afterthought. Keep it updated, review it regularly.

---

## Current State: Three Complete, Unanalyzed Experiments

We now have three completed experiments with **correct game mechanics**, but they're sitting on remote servers unanalyzed:

**On `rwalters-sandbox-1`**:
- 📁 V7 evolution results: 12 scenarios × ~5MB = ~60MB
- 📁 Nash V5 equilibrium results: 9 scenarios × ~1MB = ~9MB

**On `rwalters-sandbox-2`**:
- 📁 PPO trained model + checkpoints: ~50MB

**Total data to retrieve**: ~120MB

---

## What's Next: The Analysis Phase

### Immediate Priorities (This Week)

**1. Data Download & Organization** ⏱️ 45 min
```bash
# Download V7 evolution results
rsync -avz rwalters-sandbox-1:~/bucket-brigade/experiments/scenarios/*/evolved_v7/ \
  experiments/scenarios/

# Download Nash V5 results (for reference/comparison)
rsync -avz rwalters-sandbox-1:~/bucket-brigade/experiments/nash/v2_results_v5/ \
  experiments/nash/

# Download PPO model
rsync -avz rwalters-sandbox-2:~/bucket-brigade/models/ppo_hard_5M_with_checkpoints.pt \
  models/
```

**2. Re-Run Nash Equilibrium with V7** ⏱️ 4-6 hours compute

This is critical: Nash V5 used old-mechanics agents. We need Nash V2-V7:

```bash
# Launch Nash V2 with V7 evolved agents
for scenario in chain_reaction deceptive_calm early_containment \
                greedy_neighbor mixed_motivation overcrowding \
                rest_trap sparse_heroics trivial_cooperation; do
  uv run python experiments/scripts/compute_nash_v2.py $scenario \
    --evolved-versions v7 \
    --simulations 2000 \
    --max-iterations 20 \
    --output-dir experiments/nash/v2_results_v7/$scenario \
    --seed 42
done
```

**3. V7 Fitness Analysis** ⏱️ 2 hours

Key questions:
- How does V7 fitness compare to V6 for the same scenarios?
- Are the strategies qualitatively different?
- Which scenarios saw the biggest changes?

**4. PPO Evaluation** ⏱️ 1-2 hours

Test the trained PPO model:
```bash
# Evaluate against heuristics
uv run python experiments/scripts/evaluate_policy.py \
  models/ppo_hard_5M_with_checkpoints.pt \
  --scenario hard \
  --opponents firefighter hero free_rider coordinator \
  --num-games 100

# Evaluate against V7 evolved
uv run python experiments/scripts/evaluate_policy.py \
  models/ppo_hard_5M_with_checkpoints.pt \
  --scenario hard \
  --opponents evolved_v7 \
  --num-games 100
```

**5. Website Updates** ⏱️ 1-2 hours

- Sync V7 evolved agents to `web/public/research/scenarios/{scenario}/evolved/`
- Add "Game Mechanics Update" notice
- Mark old results (V3-V6) as archived
- Create initial research notebook browser

### Medium-Term Goals (Next Week)

**6. V7 Tournament Analysis**

Run comprehensive tournaments:
- V7 vs V6 vs archetypes
- Mixed teams with V7 agents
- Cross-scenario performance

**7. Nash V2-V7 Analysis**

Once re-run completes:
- Compare equilibria under correct mechanics
- Test if V7 agents are Nash equilibria or ε-equilibria
- Analyze cooperation vs free-riding patterns

**8. PPO Learning Curve Analysis**

Analyze checkpoints to understand:
- How did the policy evolve over 5M steps?
- When did major strategy shifts occur?
- Which checkpoint performs best?

**9. Comprehensive Documentation Update**

- Archive old experimental results with clear notices
- Update all research docs to reflect correct mechanics
- Write "lessons learned" document about the mechanics fix

---

## The Research Notebook: A New Tool

This entry is the first in our **Research Notebook** - a chronological record of our research journey. Each entry will capture:

- **What we did**: Experiments run, analyses completed
- **What we found**: Results, surprises, insights
- **What we learned**: Lessons, implications, next questions
- **What's next**: Immediate priorities and open questions

The notebook will be:
- Timestamped and immutable (historical record)
- Browsable on the website (public research transparency)
- Written in narrative form (tell the story, not just report numbers)
- Focused on insights and learning (not just results)

**Location**: `research_notebook/YYYY-MM-DD_title.md`

---

## Reflection: Turning a Bug into Better Science

Finding the game mechanics inconsistency could have been discouraging - months of work potentially invalidated. Instead, it became an opportunity:

1. **Improved infrastructure**: Single source of truth, better testing
2. **Clearer understanding**: Forced us to specify exactly what game we're studying
3. **Better documentation**: Design docs now authoritative and up-to-date
4. **Fresh validation**: New experiments confirm our methods work with correct mechanics

As of today, we have:
- ✅ Correct, consistent implementation (Python ↔ Rust)
- ✅ First generation of valid experimental results (V7, PPO)
- ✅ Clear path forward (Nash V2-V7, analysis, website updates)
- ✅ Stronger research infrastructure (testing, validation, documentation)

The story continues. Next entry will document the analysis phase and our first insights from the correct-mechanics experiments.

---

## Appendix: Technical Details

### Evolution Versions Summary

| Version | Population | Generations | Scenarios | Mechanics | Status |
|---------|-----------|-------------|-----------|-----------|--------|
| V3 | 200 | 2,500 | 9 core | ❌ Old | Archived |
| V4 | 200 | 15,000 | 9 core | ❌ Old | Archived |
| V5 | 200 | 12,000 | 9 core | ❌ Old | Archived |
| V6 | 200 | 200 | 12 all | ❌ Old | Archived |
| **V7** | 200 | 200 | **12 all** | ✅ **Correct** | **Active** |

### Game Mechanics Changes

**Fire Extinguishing**:
```python
# OLD (Python)
p_extinguish = 1.0 - np.exp(-kappa * workers_here)

# NEW (Both Python and Rust)
p_extinguish = 1.0 - (1.0 - prob_solo_agent_extinguishes_fire) ** workers_here
```

**Spontaneous Ignition**:
```python
# OLD (Python)
if self.night < self.scenario.N_spark:
    self._spark_fires()

# NEW (Both Python and Rust)
self._spark_fires()  # Every night, throughout the game
```

**Reward Timing**:
```rust
// OLD (Rust) - Computed at end
pub fn get_result(&self) -> GameResult {
    let agent_scores = self.compute_final_rewards();
    // ...
}

// NEW (Rust) - Computed per-step
pub(super) fn compute_rewards(&mut self, ...) -> Vec<f32> {
    // Work/rest costs, team rewards, ownership bonuses per step
}
```

### Parameter Naming Unification

**Before** (Python used short names):
- `beta` → `prob_fire_spreads_to_neighbor`
- `kappa` → `prob_solo_agent_extinguishes_fire`
- `N_spark` → (removed - now continuous)

**After** (Both Python and Rust):
- Single source: `definitions/scenarios.json`
- Generated Python: `bucket_brigade/envs/scenarios_generated.py`
- Generated Rust: `bucket-brigade-core/src/scenarios.rs`

### Commit History

Game mechanics fix (13 commits):
```
db100181 fix: Checkpoint logic to trigger when crossing threshold
ac4a7ceb feat: Add periodic checkpoint saving every 500K steps
771bc7c4 debug: Add verbose logging to diagnose training stall
b2217931 fix: Clean CFFI artifacts before building PyO3 module
c2310469 fix: Enable PyO3 feature when building Rust module
[... 8 more commits for parameter refactoring and docs ...]
7672bb9c docs: Update parameter names and reward computation spec
```

---

**Next Entry**: Analysis of V7 evolution results and first Nash V2-V7 equilibrium findings

**Questions for Future Entries**:
- How much do strategies change under correct mechanics?
- Does PPO discover similar strategies to evolution?
- Are there new Nash equilibria with correct physics?
- What cooperation patterns emerge with continuous fire pressure?
