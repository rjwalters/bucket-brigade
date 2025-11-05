# V4 Evolution - Post-Completion Guide

**For the next agent/session after v4 evolution completes**

## Current Status

**V4 Evolution Launched**: 2025-11-05 07:52 UTC (restarted with 10hr config)
**Configuration**: 200 pop, 15000 gen, 50 games, seed=42
**Scenarios**: All 9 scenarios running in parallel on `rwalters-sandbox-1`
**Expected Completion**: ~14:22 UTC (6.5 hours from start)
**Estimated Total Runtime**: 6.5 hours wall-clock (10hr window available)

## Step 1: Verify Completion

### Check if all scenarios finished

```bash
ssh rwalters-sandbox-1 "cd ~/bucket-brigade && bash experiments/scripts/check_v4_progress.sh"
```

**What to look for**:
- All scenarios show "Generation 1000" (or close to it)
- OR: All tmux sessions have exited (`tmux ls` shows no v4_ sessions)

### Check final generation in logs

```bash
ssh rwalters-sandbox-1 "tail -5 ~/bucket-brigade/logs/evolution/*_v4_*.log"
```

**Expected output**: Should see "Generation 1000" for all 9 scenarios

### Verify output files exist

```bash
ssh rwalters-sandbox-1 "ls -l ~/bucket-brigade/experiments/scenarios/*/evolved_v4/best_agent.json"
```

**Expected**: 9 files, one per scenario

## Step 2: Retrieve Results

### Pull results to local machine

```bash
# From project root
rsync -avz --progress \
  rwalters-sandbox-1:~/bucket-brigade/experiments/scenarios/*/evolved_v4/ \
  ./experiments/scenarios/
```

**What this retrieves**:
- `best_agent.json` - Best evolved genome
- `fitness_history.json` - Fitness over generations
- `config.json` - Evolution configuration
- `evolution_log.txt` - Final evolution summary

### Commit results to git

```bash
git add experiments/scenarios/*/evolved_v4/
git commit -m "results: Add v4 evolution results (fixed evaluator)

All 9 scenarios completed with corrected multi-agent Rust evaluator.
Config: pop=100, gen=1000, games=50, seed=42

This fixes the v3 single-agent training bug that caused 96% regression."
git push origin main
```

## Step 3: Quick Sanity Check

### Inspect one evolved agent

```bash
cat experiments/scenarios/chain_reaction/evolved_v4/best_agent.json
```

**Critical checks**:
- ✅ `work_tendency` > 0.3 (not free-riding like v3)
- ✅ `fitness` > 0 (positive, unlike v3's negative values)
- ✅ Parameters look reasonable (not all 0s or 1s)

### Check fitness progression

```bash
cat experiments/scenarios/chain_reaction/evolved_v4/fitness_history.json | grep best_fitness | tail -10
```

**What to look for**:
- Fitness should improve or stabilize over time
- Should NOT be consistently negative (like v3)
- Final fitness should be positive (ideally 50-100+)

## Step 4: Run Tournament Comparisons

### Compare v4 against original "evolved" and v3

```bash
# For each scenario, run tournament with all versions
for scenario in chain_reaction deceptive_calm early_containment greedy_neighbor \
                mixed_motivation overcrowding rest_trap sparse_heroics trivial_cooperation; do
    echo "=== Testing $scenario ==="

    uv run python experiments/scripts/run_comparison.py $scenario \
      --evolution-versions all \
      --num-games 100

    echo ""
done
```

**This will**:
- Auto-detect all evolution versions (evolved, evolved_v3, evolved_v4)
- Run 100-game tournaments
- Generate comparison results
- Save to `experiments/scenarios/*/comparison/`

**Expected runtime**: ~2-5 minutes per scenario, ~20-30 minutes total

## Step 5: Analyze Results

### Quick performance check

```bash
# Check tournament rankings for all scenarios
for scenario in chain_reaction deceptive_calm early_containment greedy_neighbor \
                mixed_motivation overcrowding rest_trap sparse_heroics trivial_cooperation; do
    echo "=== $scenario ==="
    cat experiments/scenarios/$scenario/comparison/comparison.json | \
        jq '.ranking | .[] | "\(.name): \(.mean_payoff | round)"'
    echo ""
done
```

**Success criteria**:
1. ✅ **evolved_v4 > evolved_v3** (v4 fixes v3's bug)
2. ✅ **evolved_v4 >= evolved** (v4 matches or beats original)
3. ✅ **evolved_v4 work_tendency > 0.3** (not free-riding)

### Detailed comparison (chain_reaction example)

```bash
cat experiments/scenarios/chain_reaction/comparison/comparison.json | jq '{
  evolved: .tournament.evolved.mean_payoff,
  evolved_v3: .tournament.evolved_v3.mean_payoff,
  evolved_v4: .tournament.evolved_v4.mean_payoff,
  ranking: .ranking
}'
```

**Key metrics to compare**:
- `mean_payoff`: Higher is better
- `std_payoff`: Lower is more consistent
- Ranking order: evolved_v4 should be #1 or #2

## Step 6: Document Findings

### Create results summary

Create `experiments/V4_RESULTS_ANALYSIS.md` with:

1. **Executive Summary**
   - Did v4 fix the v3 bug? (YES/NO)
   - Performance vs original "evolved": Better/Same/Worse
   - Overall assessment: Success/Partial/Failure

2. **Performance Comparison Table**

```markdown
| Scenario              | evolved | evolved_v3 | evolved_v4 | v4 vs evolved | v4 vs v3 |
|-----------------------|---------|------------|------------|---------------|----------|
| chain_reaction        | XX.XX   | XX.XX      | XX.XX      | +X%           | +X%      |
| deceptive_calm        | XX.XX   | XX.XX      | XX.XX      | +X%           | +X%      |
| ...                   | ...     | ...        | ...        | ...           | ...      |
| **Average**           | XX.XX   | XX.XX      | XX.XX      | +X%           | +X%      |
```

3. **Key Observations**
   - Fitness values during training (negative? positive?)
   - Tournament performance (better? worse?)
   - Work tendency values (free-riding?)
   - Any surprising results?

4. **Lessons Learned**
   - Did the fix work as expected?
   - Are there still issues to address?
   - Recommendations for future runs?

### Update V4_STATUS.md

```bash
# Add completion status to experiments/V4_STATUS.md
```

Update with:
- Completion time
- Final results summary
- Link to detailed analysis

## Step 7: Next Actions

### If V4 succeeded (matched or beat "evolved"):

1. ✅ Mark v3 agents as **deprecated** (add warning to `evolved_v3/`)
2. ✅ Use **evolved_v4** as new best agents
3. ✅ Update website with v4 results
4. ✅ Consider running longer evolution (2000+ gen) if time permits

### If V4 failed (worse than "evolved"):

1. ⚠️ **Debug**: Check logs for errors or issues
2. ⚠️ **Analyze**: Why did v4 perform worse?
3. ⚠️ **Options**:
   - Try Python evaluator (slower but guaranteed correct)
   - Adjust hyperparameters
   - Run longer (2000+ generations)
   - Use original "evolved" agents (they're good!)

### If V4 matched "evolved" but not better:

1. ✅ Success - v4 validates the fix works
2. ✅ Keep both versions (evolved and evolved_v4)
3. ✅ Consider hybrid approach or ensemble strategies

## Quick Reference Commands

### Check if running
```bash
ssh rwalters-sandbox-1 "tmux ls | grep v4_"
```

### Monitor progress (any scenario)
```bash
ssh rwalters-sandbox-1 "tail -f ~/bucket-brigade/logs/evolution/chain_reaction_v4_*.log"
```

### Attach to running session
```bash
ssh rwalters-sandbox-1 -t "tmux attach -t v4_chain_reaction"
```

### Kill all v4 sessions (if needed)
```bash
ssh rwalters-sandbox-1 'for s in $(tmux ls | grep v4_ | cut -d: -f1); do tmux kill-session -t $s; done'
```

## Timeline Expectations

- **Start**: 07:52 UTC (restarted with intensive config)
- **Expected completion**: 14:22 UTC (~6.5 hours)
- **Retrieval**: +5 min (rsync results)
- **Tournaments**: +20-30 min (all scenarios)
- **Analysis**: +15 min (document findings)
- **Total**: ~7 hours from start to full analysis

## Failure Scenarios

### What if evolution didn't complete?

1. Check tmux sessions: `ssh rwalters-sandbox-1 "tmux ls"`
2. Check for errors in logs: `ssh rwalters-sandbox-1 "grep -i error ~/bucket-brigade/logs/evolution/*_v4_*.log"`
3. If crashed early: Check first 50 lines of each log
4. Common issues:
   - Out of memory (unlikely with 100 pop)
   - Rust module not found (should be caught in testing)
   - Permission errors (check output directories exist)

### What if some scenarios completed but others didn't?

1. Identify which failed: `check_v4_progress.sh`
2. Retrieve successful results
3. Re-run failed scenarios individually:
   ```bash
   ssh rwalters-sandbox-1
   cd ~/bucket-brigade
   uv run python experiments/scripts/run_evolution.py <scenario> \
     --population 100 --generations 1000 --games 50 \
     --output-dir experiments/scenarios/<scenario>/evolved_v4 \
     --seed 42
   ```

### What if fitness values are still negative?

This would indicate the fix didn't work:

1. **Verify the fix**: Check `bucket_brigade/evolution/fitness_rust.py:99-129`
2. **Check num_agents**: Should be 4 for all scenarios
3. **Inspect logs**: Look for any warnings about single-agent simulation
4. **Fallback**: Use Python evaluator instead of Rust

## Files to Check

### Before declaring success:

- ✅ `experiments/scenarios/*/evolved_v4/best_agent.json` (9 files)
- ✅ `experiments/scenarios/*/evolved_v4/fitness_history.json` (9 files)
- ✅ `experiments/scenarios/*/comparison/comparison.json` (9 files, after tournaments)
- ✅ `experiments/V4_RESULTS_ANALYSIS.md` (create this)
- ✅ `experiments/V4_STATUS.md` (update with completion)

## Success Checklist

- [ ] All 9 scenarios completed 1000 generations
- [ ] All best_agent.json files exist
- [ ] Fitness values are positive (not negative like v3)
- [ ] work_tendency > 0.3 for all agents (not free-riding)
- [ ] Tournament performance >= original "evolved"
- [ ] Tournament performance >> evolved_v3 (huge improvement)
- [ ] Results committed to git
- [ ] Analysis document created
- [ ] V4_STATUS.md updated

---

**Document Status**: Ready for post-v4 completion
**Created**: 2025-11-05
**V4 Launch Time**: 07:50 UTC
**Estimated Completion**: 08:20 UTC
