# V6 Tournament Results Summary

**Date**: 2025-11-07
**Tournaments Run**: 4 scenarios (easy, trivial_cooperation, hard, chain_reaction)
**Games per scenario**: 100
**Agents tested**: evolved_v6, firefighter, hero, free_rider, coordinator

## Critical Finding: Free-Rider Dominates Most Scenarios

### Tournament Rankings

#### trivial_cooperation ✅ (evolved_v6 WINS!)
```
Rank 1: evolved_v6     θ=1.56  [1.33, 1.80]
Rank 2: free_rider     θ=1.46  [1.22, 1.69]
Rank 3: coordinator    θ=0.13  [-0.11, 0.36]
Rank 4: firefighter    θ=-1.25 [-1.49, -1.02]
Rank 5: hero           θ=-1.61 [-1.85, -1.37]
```
**Analysis**: evolved_v6 beats free_rider by 0.10 skill points. This is the ONLY scenario where our evolved agent wins!

#### easy ⚠️ (free_rider wins)
```
Rank 1: free_rider     θ=3.66  [1.93, 5.40]
Rank 2: coordinator    θ=-0.36 [-2.10, 1.38]
Rank 3: firefighter    θ=-1.09 [-2.79, 0.62]
Rank 4: evolved_v6     θ=-1.16 [-2.90, 0.59]  ← LOSES
Rank 5: hero           θ=-1.51 [-3.26, 0.24]
```
**Analysis**: Free-rider dominates with θ=3.66, beating evolved_v6 by 4.82 skill points!

#### chain_reaction ⚠️ (free_rider wins)
```
Rank 1: free_rider     θ=8.22  [0.46, 15.98]
Rank 2: coordinator    θ=-2.29 [-10.05, 5.47]
Rank 3: firefighter    θ=-5.36 [-12.97, 2.25]
Rank 4: evolved_v6     θ=-8.31 [-16.11, -0.51]  ← LOSES
Rank 5: hero           θ=-8.48 [-16.29, -0.67]
```
**Analysis**: Free-rider dominates with massive θ=8.22. High variance indicates volatility.

#### hard ⚠️ (free_rider wins)
```
Rank 1: free_rider     θ=11.29 [8.84, 13.74]
Rank 2: coordinator    θ=1.24  [-1.20, 3.69]
Rank 3: firefighter    θ=-8.24 [-10.65, -5.84]
Rank 4: evolved_v6     θ=-9.82 [-12.29, -7.36]  ← LOSES
Rank 5: hero           θ=-10.63 [-13.10, -8.17]
```
**Analysis**: Free-rider completely dominates with θ=11.29. 10+ skill point advantage!

## Summary Statistics

**evolved_v6 Performance**:
- **Wins**: 1/4 scenarios (25%)
- **Losses**: 3/4 scenarios (75%)
- **Best rank**: #1 (trivial_cooperation)
- **Worst rank**: #4 (tied across easy, chain_reaction, hard)

**free_rider Performance**:
- **Wins**: 3/4 scenarios (75%)
- **Rank #2**: 1/4 scenarios (trivial_cooperation)
- **Average skill**: θ = 6.16 (across all scenarios)
- **Dominance**: Wins by large margins (4-11 skill points)

## Why Does Free-Rider Win?

### Hypothesis 1: Tournament Context Rewards Free-Riding
- Heterogeneous teams dilute the cost of free-riding
- When paired with hard workers (firefighter, hero), free-rider gets maximum benefit
- evolved_v6's high work_tendency (0.76-1.0) means it subsidizes free-riders

### Hypothesis 2: V6 Evolved Against Wrong Opponents
- V6 trained with tournament fitness BUT...
- Training population may have been homogeneous (all V6 agents)
- Didn't face free-riders during evolution
- Optimized for working hard in cooperative teams, not mixed teams

### Hypothesis 3: Scenario Parameters Favor Free-Riding
- easy, chain_reaction, hard: All have conditions where individual work doesn't pay off
- High fire spread (β) or low extinguish (κ) makes cooperation futile
- Free-rider correctly exploits the futility

## Contrast with Training Fitness

### Training Claims vs Tournament Reality

| Scenario | V6 Training Fitness | Tournament Rank | Gap |
|----------|---------------------|-----------------|-----|
| trivial_cooperation | 4826.00 | #1 ✅ | Match! |
| easy | 4794.21 | #4 ❌ | MAJOR MISMATCH |
| chain_reaction | 130.00 | #4 ❌ | MAJOR MISMATCH |
| hard | -2739.50 | #4 ❌ | Match (both bad) |

**Critical Insight**: Training fitness does NOT correlate with tournament performance except in trivial scenarios!

## Why trivial_cooperation is Different

**What makes it special**:
- p_spark = 0 (no ongoing fires)
- Simple, transient threats
- Low fire spread (β=0.05)
- High extinguish (κ=0.50)

**Why evolved_v6 wins here**:
- High work tendency (0.85) pays off in simple scenarios
- No persistent threats that make work futile
- Free-rider's advantage minimized when fires are easy to handle

**Why it doesn't generalize**:
- Other scenarios have p_spark > 0 or higher β
- Persistent/spreading fires make individual effort less effective
- Free-riding becomes more viable

## Implications for V7

### What Went Wrong in V6

1. **Homogeneous Training Population** ⚠️
   - Likely evolved against clones of itself
   - Never faced free-riders during training
   - No selection pressure for robustness to exploitation

2. **Fitness Function Mismatch** ⚠️
   - "Tournament fitness" but which opponents?
   - If opponents were all V6 variants, fitness is misleading
   - Need explicit heterogeneous opponent pool

3. **Scenario Bias** ⚠️
   - May have optimized for cooperative equilibrium
   - Assumes all agents work together
   - Doesn't account for defectors/free-riders

### V7 Recommendations

**1. Heterogeneous Training Pool** ✅
```python
opponent_pool = [
    "evolved_v6",      # Self-play
    "free_rider",      # Explicit defector
    "firefighter",     # Hard worker
    "hero",            # Altruist
    "coordinator",     # Cooperator
    "random_mutants"   # Diversity
]
```

**2. Multi-Scenario Fitness** ✅
- Don't optimize for single scenario
- Fitness = average across {easy, trivial_cooperation, hard}
- Forces generalization

**3. Exploitation-Resistant Objectives** ✅
- Penalize strategies that lose to free-riders
- Reward robust performance across team compositions
- Consider minimax: "worst-case team composition"

**4. Longer Evolution with Diversity Maintenance** ✅
- V6 converged too early (gen 20)
- Add fitness sharing or niching
- Maintain population diversity longer

## Comparison to V5

**Note**: Need to run V5 tournaments for full comparison. Based on available data:

**V5 Strategy** (Nash-based training):
- Optimized for theoretical equilibrium
- Failed in heterogeneous tournaments

**V6 Strategy** (Tournament-based training):
- Better than V5 in training fitness (+12.8%)
- But still loses to free-rider in 3/4 tested scenarios
- Only wins in trivial_cooperation (easiest scenario)

**Conclusion**: V6 is better than V5, but still not robust to free-riding exploitation.

## Phase 1 Status Update

**Original Phase 1 Goals**:
1. ✅ Reproducible training
2. ⚠️ Tournament validation - DONE, revealed free-rider problem
3. ❌ Baseline performance - FAILS to beat free-rider archetype
4. ❌ Generalization - Only wins in 1/4 scenarios
5. ⚠️ Understanding - Now understand free-rider exploitation issue

**Recommendation**: V7 is CRITICAL. Must address heterogeneous opponent training before claiming Phase 1 success.

## Next Steps

### Immediate
1. ✅ Document V6 tournament findings
2. ⏭️ Run remaining 8 scenarios (if needed for confirmation)
3. ⏭️ Design V7 with heterogeneous opponent pool
4. ⏭️ Test free-rider explicitly as training opponent

### V7 Design
- Population: 200
- Generations: 200 (with diversity maintenance)
- **Opponent pool**: {evolved_v6, free_rider, firefighter, hero, coordinator, random}
- **Fitness**: Tournament payoff against random opponent teams
- **Validation**: Test against same heterogeneous pool

### Research Questions
1. Can evolution discover anti-free-rider strategies?
2. Is free-riding truly optimal, or can we find counter-strategies?
3. Does multi-scenario training improve generalization?

---

**Status**: V6 provides valuable negative result - reveals free-rider problem
**Action**: V7 must explicitly train against free-riders to develop robust strategies
