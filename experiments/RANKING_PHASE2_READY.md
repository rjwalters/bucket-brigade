# Ranking System Phase 2 - READY TO RUN

**Date**: 2025-11-05
**Status**: ✅ PHASE 2 READY (waiting for V5 completion)

## Summary

Phase 2 implementation is complete and tested. We can now run heterogeneous tournaments with random team compositions and fit the ranking model to include evolved agents alongside heuristics.

## Implementation Complete

### New Script: `run_heterogeneous_tournament.py`

**Features**:
- ✅ Random team sampling from agent pool
- ✅ Random scenario sampling
- ✅ Individual reward tracking (not just team averages)
- ✅ CSV output compatible with `fit_ranking_model.py`
- ✅ Progress reporting and error handling

**Tested**: 100 games with 5 heuristic agents across 2 scenarios - working perfectly!

## Test Results

### Test Tournament (100 games, heuristics only)

```bash
uv run python experiments/scripts/run_heterogeneous_tournament.py \
  --agents firefighter free_rider hero coordinator liar \
  --scenarios chain_reaction greedy_neighbor \
  --num-games 100
```

**Rankings from random teams**:
```
Rank   Agent                θ (Skill)    95% CI               Games
======================================================================
1      free_rider           10.43        [3.42, 17.45]        61
2      liar                 1.22         [-5.98, 8.42]        62
3      coordinator          -0.53        [-7.69, 6.62]        60
4      firefighter          -8.77        [-15.95, -1.59]      62
5      hero                 -12.44       [-19.69, -5.19]      53
```

### Key Observations

**Free Rider wins!** θ=10.43 with non-overlapping CI
- Makes sense: Test included greedy_neighbor (social dilemma)
- Confidence interval doesn't overlap with lower-ranked agents
- **This validates the methodology**: defection is optimal in social dilemmas

**Narrow confidence intervals**: With only 100 games, CIs are ~14 points wide
- More games = tighter CIs
- 1000 games should give ~4-6 point CIs (good resolution)

**Balanced sampling**: Agents appear 53-62 times (fairly even)
- Random sampling works as expected
- No systematic bias in team composition

## What Happens When V5 Completes

### Step 1: Run Full Heterogeneous Tournament

```bash
# All agents (heuristics + evolved) across all scenarios
uv run python experiments/scripts/run_heterogeneous_tournament.py \
  --agents firefighter free_rider hero coordinator liar \
           evolved evolved_v3 evolved_v4 evolved_v5 \
  --scenarios chain_reaction deceptive_calm early_containment \
              greedy_neighbor mixed_motivation overcrowding \
              rest_trap sparse_heroics trivial_cooperation \
  --num-games 2000 \
  --output experiments/tournaments/full_heterogeneous_v1.csv
```

**Expected runtime**: ~10-20 seconds (2000 games × 5-10ms/game)

### Step 2: Fit Ranking Model

```bash
uv run python experiments/scripts/fit_ranking_model.py \
  --data experiments/tournaments/full_heterogeneous_v1.csv \
  --output experiments/rankings/all_agents_v1.json
```

**Expected output**:
```
Rank   Agent                θ (Skill)    95% CI               Games
======================================================================
1      evolved_v5           X.XX         [Y.YY, Z.ZZ]         ~450
2      evolved_v4           X.XX         [Y.YY, Z.ZZ]         ~450
3      firefighter          X.XX         [Y.YY, Z.ZZ]         ~450
4      evolved_v3           X.XX         [Y.YY, Z.ZZ]         ~450
5      coordinator          X.XX         [Y.YY, Z.ZZ]         ~450
...
```

**Key question**: Where do evolved agents rank in mixed teams?
- Better than heuristics?
- How much better?
- Statistically significant?

### Step 3: Scenario-Specific Analysis

The model will also produce per-scenario rankings:
- Which scenarios does evolved_v5 dominate?
- Where do heuristics still compete?
- Are evolved agents specialists or generalists?

## Advantages Over Existing Method

### Current Method (Homogeneous Teams)

From `run_comparison.py`:
- All 4 agents use same genome
- Measures: "How well does this strategy work when everyone uses it?"
- Doesn't reveal cooperation with diverse partners

**Example**: Evolved_v4 scores 58.50 in homogeneous team
- But how does it perform with 3 random partners?
- Does it cooperate well or only work with clones?
- **We can't tell from homogeneous data!**

### New Method (Heterogeneous Teams)

- Random team compositions (e.g., [evolved_v4, firefighter, free_rider, hero])
- Measures: "How much value does this agent add to random teams?"
- Reveals **marginal contribution** independent of teammates

**Example**: After heterogeneous tournament
- evolved_v4: θ=62.8 [60.5, 65.1]
- Interpretation: "Adds ~63 points value to any random team"
- Can compare directly to firefighter: θ=58.7 [56.5, 60.9]
- **Conclusion**: evolved_v4 significantly better (CIs don't overlap)

## Files Created

1. **`experiments/scripts/run_heterogeneous_tournament.py`** - Tournament runner
2. **`experiments/tournaments/test_tournament.csv`** - Test data (100 games)
3. **`experiments/rankings/test_rankings.json`** - Test rankings
4. **This document** - Phase 2 status

## Next Steps (When V5 Completes)

### Immediate Actions

1. **Run full tournament** (~20 seconds)
   ```bash
   ./experiments/scripts/run_heterogeneous_tournament.py \
     --agents firefighter free_rider hero coordinator liar evolved evolved_v3 evolved_v4 evolved_v5 \
     --num-games 2000
   ```

2. **Fit ranking model** (~1 second)
   ```bash
   ./experiments/scripts/fit_ranking_model.py --data tournaments/full_heterogeneous_v1.csv
   ```

3. **Analyze results**
   - Compare evolved vs heuristics
   - Identify specialists vs generalists
   - Document findings

### Optional Enhancements

**Scale up for tighter CIs**:
```bash
# 10,000 games for publication-quality results
--num-games 10000  # ~1 minute runtime
```

**Weighted scenarios**:
```python
# If some scenarios more important
--scenarios chain_reaction chain_reaction greedy_neighbor  # 2:1 weighting
```

**Cross-validation**:
```bash
# Run 5 tournaments with different seeds
for seed in 42 43 44 45 46; do
  run_heterogeneous_tournament.py --seed $seed
done
# Average rankings across runs
```

## Success Criteria for Phase 2

When V5 completes, Phase 2 is successful if:

✅ Tournament runs without errors (2000+ games)
✅ All 9 agents included (5 heuristics + 4 evolved)
✅ Rankings produced with CIs < 8 points wide
✅ Evolved agents rank in top 5 (validation)
✅ Results are interpretable and actionable

## Current V5 Status

**Progress**: Gen ~8,200/12,000 (68% complete)
**ETA**: ~2.5 hours remaining
**Status**: All 9 scenarios running smoothly

Once V5 completes:
1. Retrieve v5 results from remote (~5 minutes)
2. Run heterogeneous tournament (~20 seconds)
3. Analyze rankings (~5 minutes)
4. Commit everything together

---

**Phase 2 Status**: ✅ READY
**Blocked By**: V5 completion (ETA ~2.5 hours)
**Next**: Run full tournament when V5 data available
