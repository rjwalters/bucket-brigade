# V5 Evolution - Next Steps

**Date**: 2025-11-05
**Status**: 🚀 V5 IS RUNNING (started 16:54 UTC)
**Completion**: ~22:54 UTC (6 hours)

## V5 Status

V5 evolution was successfully launched at 16:54 UTC on 2025-11-05.

**Configuration**:
- Population: 200
- Generations: 12,000
- Games per evaluation: 50
- Seed: 43
- Scenarios: All 9 standard scenarios
- Expected runtime: ~6 hours (until ~22:54 UTC)

**Early Progress (Gen 18)**:
- chain_reaction: Best fitness 69.16 (already exceeds v3/v4's 58.50!)
- All 9 scenarios running successfully in tmux sessions

## When V5 Completes

### 1. Verify Completion

Check that all scenarios finished successfully:

```bash
ssh rwalters-sandbox-1 "tmux ls | grep v5_"
```

If any sessions are still running, they're still evolving. Otherwise, proceed.

### 2. Retrieve Results

Copy all v5 results back to local machine:

```bash
# Retrieve all evolved_v5 agents
for scenario in chain_reaction deceptive_calm early_containment greedy_neighbor \
                mixed_motivation overcrowding rest_trap sparse_heroics \
                trivial_cooperation; do
  echo "Retrieving $scenario..."
  scp rwalters-sandbox-1:~/bucket-brigade/experiments/scenarios/$scenario/evolved_v5/best_agent.json \
      experiments/scenarios/$scenario/evolved_v5/ 2>/dev/null || echo "  ⚠️  Not found"
done
```

### 3. Run Rust-Only Tournaments

Test all evolution versions using Rust as single source of truth:

```bash
# Compare all versions for chain_reaction
uv run python experiments/scripts/run_comparison.py chain_reaction \
  --evolution-versions evolved evolved_v3 evolved_v4 evolved_v5 \
  --num-games 20

# Run for all scenarios (takes ~15 minutes)
for scenario in chain_reaction deceptive_calm early_containment greedy_neighbor \
                mixed_motivation overcrowding rest_trap sparse_heroics \
                trivial_cooperation; do
  echo "=== $scenario ==="
  uv run python experiments/scripts/run_comparison.py $scenario \
    --evolution-versions all \
    --num-games 20
done
```

### 4. Analyze Results

Check V5 performance:

```bash
# View chain_reaction tournament results
cat experiments/scenarios/chain_reaction/comparison/comparison.json | jq '.ranking'
```

**Expected Results**:
- ✅ **Best case**: V5 beats V3/V4 (> 58.50 payoff)
- ✅ **Good case**: V5 matches V3/V4 (≈ 58.50 payoff)
- ⚠️ **Unexpected**: V5 worse than V3/V4 (< 58.50 payoff)

**Key Questions**:
1. Did V5 improve on v3/v4's 58.50?
2. Is training fitness consistent with tournament payoff? (Should be ±1 point with Rust)
3. How does different seed (43 vs 42) affect convergence?

### 5. Document Findings

Create `experiments/V5_RESULTS_ANALYSIS.md` with:

```markdown
# V5 Results Analysis

## Tournament Performance

### Chain Reaction
| Version | Training Fitness | Tournament Payoff | Gap | Status |
|---------|------------------|-------------------|-----|--------|
| evolved | X.XX | X.XX | ±X.XX | Reference |
| evolved_v3 | 58.50 | 58.50 | 0.00 | Near-Nash |
| evolved_v4 | 58.50 | 58.50 | 0.00 | Near-Nash |
| **evolved_v5** | **X.XX** | **X.XX** | **±X.XX** | **TBD** |
| Nash | - | 57.87 | - | Theoretical |

### Analysis
- Did v5 beat v3/v4? [YES/NO]
- Train/test consistency? [GOOD/MISMATCH]
- Different seed impact? [BETTER/SAME/WORSE]

## Recommendations
[What should we do next? V6? Different approach?]
```

### 6. Update Main Resolution Document

Add v5 results to `experiments/RUST_SINGLE_SOURCE_OF_TRUTH.md`:

```markdown
## V5 Results (2025-11-05)

**Configuration**: 200 pop, 12,000 gen, seed 43
**Result**: [X.XX payoff in chain_reaction]
**Comparison**:
- V3 (seed 42, 2500 gen): 58.50
- V4 (seed 42, 15000 gen): 58.50
- **V5 (seed 43, 12000 gen): X.XX**

**Conclusion**: [Did we improve? Why or why not?]
```

### 7. Commit Results

```bash
cd /Users/rwalters/GitHub/bucket-brigade

# Add all v5 results
git add experiments/scenarios/*/evolved_v5/
git add experiments/scenarios/*/comparison/

# Commit with results
git commit -m "experiment: Add v5 evolution results (12k gen, seed 43)

V5 Results:
- chain_reaction: X.XX payoff
- All 9 scenarios completed successfully
- Train/test consistency: ✅ (Rust-only evaluation)

V5 vs V3/V4:
- V3/V4: 58.50 (seed 42)
- V5: X.XX (seed 43)
- [Better/Same/Worse]

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

git push
```

## Monitoring Progress (While Running)

### Check Current Generation

```bash
ssh rwalters-sandbox-1 "grep 'Gen ' ~/bucket-brigade/logs/evolution/chain_reaction_v5_*.log | tail -5"
```

### Watch Live Progress

```bash
ssh rwalters-sandbox-1 -t "tmux attach -t v5_chain_reaction"
# Detach: Ctrl+B, D
```

### Check All Scenarios Status

```bash
ssh rwalters-sandbox-1 "for log in ~/bucket-brigade/logs/evolution/*_v5_*.log; do
  echo \"=== \$(basename \$log) ===\"
  tail -3 \$log
  echo
done"
```

### Estimated Progress

- **12,000 generations** total
- **~40 generations per minute** (based on V3/V4)
- **~300 minutes** (5 hours) expected
- **Progress %**: (current_gen / 12000) × 100

## If Something Goes Wrong

### Check for Errors

```bash
ssh rwalters-sandbox-1 "grep -i error ~/bucket-brigade/logs/evolution/*_v5_*.log"
```

### Restart Failed Scenario

```bash
# If, say, greedy_neighbor failed:
ssh rwalters-sandbox-1 "cd ~/bucket-brigade && \
  tmux new-session -d -s v5_greedy_neighbor_retry \
  'uv run python experiments/scripts/run_evolution.py greedy_neighbor \
    --population 200 --generations 12000 --games 50 \
    --output-dir experiments/scenarios/greedy_neighbor/evolved_v5 \
    --seed 43 2>&1 | tee logs/evolution/greedy_neighbor_v5_retry.log'"
```

### Check Remote Server Resources

```bash
ssh rwalters-sandbox-1 "top -b -n 1 | head -20"
ssh rwalters-sandbox-1 "df -h"
```

## Success Criteria

1. ✅ All 9 scenarios complete without errors
2. ✅ Training fitness > 50 (near-Nash territory)
3. ✅ Training and tournament metrics match (±1 point)
4. ✅ Tournament payoff ≥ 58.50 (match or beat v3/v4)

## Context for Next Agent

**Background**:
- V3 and V4 both achieved 58.50 payoff (near-Nash at 57.87)
- Evolution was working perfectly all along
- Train/test mismatch was due to Python environment giving wrong scores
- Fixed by using Rust as single source of truth
- V5 uses different seed (43 vs 42) to explore different convergence paths

**Files Modified**:
- `experiments/scripts/run_comparison.py`: Now uses Rust-only evaluation
- `experiments/scripts/diagnose_v4_failure.py`: Now uses Rust-only evaluation
- `experiments/scripts/test_rust_python_parity.py`: Now tests Rust determinism

**Documentation**:
- `experiments/RUST_SINGLE_SOURCE_OF_TRUTH.md`: Main resolution document
- `experiments/V5_EVOLUTION_PLAN.md`: V5 configuration and rationale
- `experiments/V5_NEXT_STEPS.md`: This document

---

**Status**: V5 running successfully
**Next Agent**: Follow steps 1-7 above when V5 completes
**Expected Completion**: ~22:54 UTC (6 hours from 16:54 UTC launch)
