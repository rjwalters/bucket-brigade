# V6 Critical Finding: Plan vs Implementation Mismatch

**Date**: 2025-11-07
**Severity**: 🔴 **CRITICAL**

## TL;DR

**V6 PLAN** said: Train with heterogeneous opponent pool (archetypes + evolved variants)
**V6 ACTUAL**: Trained with homogeneous self-play only (mean_reward fitness)

**This explains why V6 loses to free-rider in tournaments!**

## The Smoking Gun

### What the Plan Said (V6_PLAN.md)

```python
# V6: Actual tournament payoff
fitness = tournament_average_payoff(agent, opponent_pool, num_games=100)

# Opponent Pool Diversity
- V5: Evolved against 5 archetypes + V4
- V6: Evolve against 5 archetypes + V4 + V5 + random mutations
  - 8 total opponent types
  - Ensures robust performance against unexpected strategies
```

### What Actually Ran (evolve_v6_simple.py:229)

```python
config = EvolutionConfig(
    ...
    fitness_type="mean_reward",  # ← Standard fitness with scenario
    games_per_individual=100,
    ...
)
```

### What mean_reward Does

From the FitnessEvaluator code:
- Creates team of 4 IDENTICAL agents (all clones of candidate being evaluated)
- Runs games with homogeneous teams
- Returns mean reward across games
- **NO heterogeneous opponents involved!**

## Evidence from Evolution Results

### Convergence Pattern Confirms Homogeneous Training

Looking at `experiments/scenarios/easy/evolved_v6/evolution_results.json`:

```json
{
  "config": {
    "fitness_type": "mean_reward",  ← NOT tournament
    "games_per_individual": 100
  }
}
```

### Early Convergence Makes Sense Now

**Observation**: Most scenarios converged by generation 20 (95% of final fitness)

**Explanation**:
- Homogeneous self-play has simple optimization landscape
- No pressure to handle diverse opponents
- Population quickly finds local optimum for self-play
- No reason to explore anti-free-rider strategies

## Why This Caused Tournament Failure

### Training Environment (Homogeneous)
```
Team: [evolved_v6, evolved_v6, evolved_v6, evolved_v6]
```
- All agents cooperate (they're identical)
- High work_tendency pays off
- Free-riding doesn't exist
- Evolution optimizes for: "Work hard with cooperative teammates"

### Tournament Environment (Heterogeneous)
```
Team: [evolved_v6, free_rider, firefighter, coordinator]
```
- Mixed agent types
- evolved_v6 works hard (learned in training)
- free_rider exploits the workers
- evolved_v6 never learned to handle defectors

**Result**: Free-rider dominates because V6 subsidizes it!

## The V6 Plan Was Correct, Implementation Was Wrong

### Why This Happened

Looking at the script header comment:
```python
"""
V6 Evolution: Simplified single-scenario evolution with V6 parameters.

This script uses the V6 strategy (larger population, more exploration) but
evolves per-scenario like V4/V5 for compatibility with existing framework.
"""
```

**Root cause**:
- V6 plan required NEW fitness function (tournament-based)
- Implementation took shortcut: reused existing framework
- Used V4/V5 approach (mean_reward) with V6 hyperparameters
- **Only adopted population/mutation changes, NOT fitness function change!**

## Comparison to What Should Have Been

### Actual V6 (evolve_v6_simple.py)
✅ Population: 200 (correct)
✅ Generations: 200 (correct)
✅ Mutation rate: 0.15 (correct)
❌ **Fitness: mean_reward (WRONG - should be tournament)**
❌ **Opponents: homogeneous (WRONG - should be heterogeneous pool)**

### Planned V6 (V6_PLAN.md)
✅ Population: 200
✅ Generations: 200
✅ Mutation rate: 0.15
✅ Fitness: tournament_average_payoff
✅ Opponents: {archetypes, v4, v5, mutations}

**V6 was only 60% implemented!**

## This Validates Our Tournament Findings

### Tournament Results Now Make Perfect Sense

**trivial_cooperation** (V6 wins):
- Simple scenario, homogeneous self-play training sufficient
- No persistent threats requiring adaptation
- V6's work-heavy strategy happens to work

**easy, chain_reaction, hard** (free_rider wins):
- Complex scenarios with heterogeneous teams
- V6 never trained against free-riders
- V6's strategy optimized for cooperative teams only
- Free-rider exploits V6's naivety

### This Is Actually Good News!

**Why**: We now know EXACTLY what to fix:
1. V6 didn't fail because evolution is bad
2. V6 didn't fail because scenarios are too hard
3. **V6 failed because it wasn't actually V6 - it was "V5.5 with bigger population"**

**The real V6 (with heterogeneous opponents) has never been tested!**

## Corrected Understanding of Evolution Versions

| Version | Population | Fitness Type | Opponents | Result |
|---------|------------|--------------|-----------|--------|
| V3 | 100 | mean_reward | Homogeneous | Nash free-rider |
| V4 | 200 | mean_reward | Homogeneous | Near-Nash |
| V5 | 100 | nash_equilibrium | Theoretical | Failed in practice |
| **V6 (as run)** | 200 | **mean_reward** | **Homogeneous** | Loses to free-rider |
| **V6 (as planned)** | 200 | **tournament** | **Heterogeneous** | **NEVER RUN!** |

## Implications

### V7 Must Actually Implement V6 Plan

V7 should be:
- ✅ Population: 200
- ✅ Generations: 200 (or more if needed)
- ✅ Mutation rate: 0.15
- ✅ **Fitness: Heterogeneous tournament payoff**
- ✅ **Opponent pool: {firefighter, free_rider, hero, coordinator, evolved variants}**

### Expected V7 Outcomes

If V6 plan was correct (and we believe it was):
- V7 should learn anti-free-rider strategies
- V7 should win tournaments or at least be competitive
- V7 should show more gradual convergence (more complex fitness landscape)

### If V7 Still Fails

Then we know:
- Problem is NOT implementation bug
- Heterogeneous tournament training was tested properly
- May need:
  - Different fitness objectives (minimax, exploitation penalty)
  - Larger population/more generations
  - Different evolutionary operators
  - Or conclude evolution insufficient, move to PPO

## Action Items

### Immediate

1. ✅ Document this critical finding
2. ⏭️ Update V6_RESULTS_ANALYSIS.md with correction
3. ⏭️ Update V6_TOURNAMENT_SUMMARY.md with explanation
4. ⏭️ Create proper V7 implementation script

### V7 Implementation

1. Create `evolve_v7.py` with TRUE heterogeneous tournament fitness:
```python
def evaluate_fitness(candidate_genome):
    """Evaluate in heterogeneous tournaments."""
    opponent_pool = [
        create_agent("firefighter"),
        create_agent("free_rider"),
        create_agent("hero"),
        create_agent("coordinator"),
        create_agent(candidate_genome),  # Self
    ]

    total_payoff = 0
    for _ in range(num_games):
        # Sample random 4-agent team from pool
        team = random.sample(opponent_pool, 4)
        # Ensure candidate is in team
        if candidate not in team:
            team[random.randint(0, 3)] = candidate

        payoff = run_game(team, scenario)
        total_payoff += payoff

    return total_payoff / num_games
```

2. Test on single scenario first (trivial_cooperation)
3. If successful, run all 12 scenarios
4. Compare V7 vs V6 vs archetypes

### Research Implications

**Phase 1 is NOT complete** because:
- V6 was not properly implemented
- Actual tournament-based evolution has never been tested
- Cannot claim "evolution doesn't work" when we didn't actually try it

**This is actually encouraging**: We have a clear hypothesis and test to run!

## Lessons Learned

1. **Always verify implementation matches plan**
   - V6_PLAN.md said one thing
   - evolve_v6_simple.py did another
   - Should have checked before running 24-hour experiment

2. **"Simplified" can mean "fundamentally different"**
   - Script header said "simplified"
   - Actually meant "completely different fitness function"
   - Simplification invalidated the core hypothesis

3. **Test fitness function explicitly**
   - Should have verified heterogeneous opponents were used
   - Could have caught this before full evolution run
   - Unit test: "Does fitness evaluation include diverse opponents?"

4. **Tournament results were the diagnostic**
   - Without tournament validation, we wouldn't have caught this
   - Training fitness looked great (4794 for easy)
   - Only tournaments revealed the problem

## Conclusion

**V6 was a valuable learning experience**, but for the wrong reason:
- We learned we need to verify implementation
- We learned homogeneous training is insufficient
- We learned tournament validation is critical

**The silver lining**:
- V6 plan was sound (heterogeneous opponents are necessary)
- We now know exactly what to implement in V7
- We have a clear hypothesis to test
- The real experiment starts with V7!

---

**Status**: V6 was "Simplified V4 with bigger population", NOT the planned heterogeneous tournament evolution
**Next**: Implement actual V6 plan as V7
**Expectation**: V7 should beat free-rider or reveal deeper issues with evolution approach
