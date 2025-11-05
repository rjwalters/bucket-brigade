# V5 Evolution Plan

**Date**: 2025-11-05
**Status**: 🚀 READY TO LAUNCH

## Motivation

V3 and V4 both achieved excellent results (58.50 payoff, near-Nash at 57.87) after we fixed the train/test consistency issue by using Rust as single source of truth. Now we want to see if we can push beyond 58.50 with another intensive run.

## V3/V4 Results Summary

| Version | Generations | Population | CPU-Hours | Result | Status |
|---------|-------------|------------|-----------|---------|--------|
| V3 | 2500 | 200 | 614 | 58.50 | ✅ Excellent |
| V4 | 15000 | 200 | 475 | 58.50 | ✅ Excellent |
| Nash | - | - | - | 57.87 | Reference |

**Key Finding**: V3 and V4 converged to same result despite 6x difference in generations.

## V5 Configuration

### Resource Budget
- **Wall-clock time**: 6 hours
- **CPUs**: 64 vCPUs
- **Total CPU-hours**: 384

### Evolution Parameters
- **Population**: 200
- **Generations**: 12,000
- **Games per evaluation**: 50
- **Seed**: 43 (different from v4's 42)
- **Scenarios**: All 9 standard scenarios
- **Parallel workers**: 64

### Rationale
- **12,000 generations**: Fits within 6-hour budget (V4 used 475 CPU-hours for 15,000 gen)
- **Population 200**: Proven effective in V3/V4
- **Different seed**: May find different local optima
- **Same other params**: Maintain consistency with v4

## Expected Outcomes

### Success Criteria
1. **Primary**: Achieve ≥ 58.50 payoff (match or beat v3/v4)
2. **Stretch**: Beat 58.50 and get closer to theoretical optimum
3. **Validation**: Training and tournament metrics match (Rust consistency)

### Time Estimates
- **Per scenario**: ~40 minutes (6 hours / 9 scenarios)
- **Total wall-clock**: ~6 hours
- **Total CPU**: 384 CPU-hours

## What's Different from V4

1. **Shorter duration**: 12,000 vs 15,000 generations (budget constraint)
2. **Different seed**: 43 vs 42 (exploration)
3. **Rust-only validation**: All testing uses Rust (no Python mismatch)

## Launch Plan

### 1. Create Launch Script
```bash
experiments/scripts/launch_v5_evolution.sh
```

### 2. Copy to Remote Server
```bash
scp experiments/scripts/launch_v5_evolution.sh rwalters-sandbox-1:~/bucket-brigade/experiments/scripts/
```

### 3. Launch Evolution
```bash
ssh rwalters-sandbox-1 "cd ~/bucket-brigade && bash experiments/scripts/launch_v5_evolution.sh"
```

### 4. Monitor Progress
```bash
# Check all logs
ssh rwalters-sandbox-1 "ls -lh ~/bucket-brigade/logs/evolution/*_v5_*.log"

# Watch specific scenario
ssh rwalters-sandbox-1 "tail -f ~/bucket-brigade/logs/evolution/chain_reaction_v5_*.log"
```

## Post-Completion Steps

1. **Retrieve results**:
   ```bash
   for scenario in chain_reaction deceptive_calm greedy_neighbor signal_chaos \
                   resource_race spatial_puzzle dynamic_zones hidden_penalty \
                   efficiency_trap; do
     scp rwalters-sandbox-1:~/bucket-brigade/experiments/scenarios/$scenario/evolved_v5/best_agent.json \
         experiments/scenarios/$scenario/evolved_v5/
   done
   ```

2. **Run tournaments** (Rust-only):
   ```bash
   uv run python experiments/scripts/run_comparison.py chain_reaction \
     --evolution-versions evolved evolved_v3 evolved_v4 evolved_v5
   ```

3. **Analyze results**:
   - Compare v5 vs v4 vs v3 performance
   - Check if v5 improved beyond 58.50
   - Verify train/test consistency (should be perfect with Rust)

4. **Document findings**:
   - Update `RUST_SINGLE_SOURCE_OF_TRUTH.md` with v5 results
   - Create `V5_RESULTS_ANALYSIS.md` if results differ significantly

## Success Indicators

### During Training
- ✅ Fitness values trending upward
- ✅ Final fitness > 50 (near-Nash territory)
- ✅ No errors or crashes
- ✅ All 9 scenarios complete

### During Tournament
- ✅ Training fitness matches tournament payoff (±1 point)
- ✅ Tournament payoff ≥ 58.50
- ✅ Ranks at or near top vs other strategies

## Risk Mitigation

1. **Shorter than V4**: May not converge fully
   - Mitigation: V3 converged in 2500 gen, so 12,000 should be plenty

2. **Different seed may diverge**: Could get worse results
   - Mitigation: V3 and V4 both got 58.50, suggesting stable convergence

3. **Time overrun**: May exceed 6 hours
   - Mitigation: Conservative estimate based on V4 timing

## Notes

- **Rust consistency**: Train/test mismatch issue is RESOLVED
- **Python deprecated**: All evaluation uses Rust now
- **Evolution working**: V3/V4 prove pipeline is reliable
- **Near-optimal**: 58.50 is only 0.63 points above Nash (57.87)

---

**Status**: Ready to launch
**Next**: Create and run launch script
