# Nash Research Consolidation

**Date**: 2025-11-05
**Status**: Discovered Python/Rust mismatch (same as evolution research)

---

## Problem Discovered

**V1 Nash results used Python `BucketBrigadeEnv`**, not Rust! 🔍

### Evidence

**Scattered results location**:
```
experiments/scenarios/*/nash/equilibrium.json  (12 scenarios)
```

**Python evaluator used** (`payoff_evaluator.py`):
```python
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv  # Python!
```

**Rust evaluator available** (`payoff_evaluator_rust.py`):
```python
import bucket_brigade_core as core  # Rust!
```

### V1 Configuration Issues

1. **Python environment**: Used old Python simulation (slow, may have bugs)
2. **Low simulation count**: Only 200 simulations (V1 README claims 2000!)
3. **No Rust validation**: Results never validated with Rust single source of truth

**This is the same issue we had with evolution V3/V4!**

---

## Comparison: Evolution vs Nash

| Aspect | Evolution | Nash |
|--------|-----------|------|
| **V1 Evaluator** | Python `BucketBrigadeEnv` | Python `BucketBrigadeEnv` |
| **Issue** | Train/test mismatch | Same risk |
| **Rust Version** | ✅ `fitness_rust.py` | ✅ `payoff_evaluator_rust.py` |
| **Resolution** | Used Rust for V3/V4/V5 | Need to use Rust for V2 |
| **Speedup** | 100x faster | 10-100x faster |
| **Consistency** | Perfect (train=test) | TBD (need to validate) |

---

## V1 Results Summary

### What We Have

**Location**: `experiments/scenarios/*/nash/equilibrium.json` (12 scenarios)

**Key results**:
- chain_reaction: 2.94 payoff (Free Rider equilibrium)
- greedy_neighbor: 58.77 payoff (Coordinator equilibrium)
- 10 pure equilibria, 2 mixed equilibria
- Only used predefined archetypes

**Issues**:
- ❌ Used Python evaluation (may differ from Rust)
- ❌ Only 200 simulations per evaluation (low confidence)
- ❌ Results not validated with Rust
- ❌ No comparison with evolved strategies

### V1 Status: **QUESTIONABLE**

Until we validate with Rust, we cannot trust these results completely.

---

## Consolidation Plan

### Step 1: Organize V1 Results ✅

Move existing results to proper location:

```bash
# Create V1 results directory
mkdir -p experiments/nash/v1_results_python_deprecated

# Move all scenario Nash results
for scenario in experiments/scenarios/*/nash; do
    scenario_name=$(basename $(dirname $scenario))
    mkdir -p experiments/nash/v1_results_python_deprecated/$scenario_name
    cp $scenario/equilibrium.json experiments/nash/v1_results_python_deprecated/$scenario_name/
done

# Mark as deprecated
echo "DEPRECATED: These results used Python BucketBrigadeEnv" > experiments/nash/v1_results_python_deprecated/README.md
```

### Step 2: Update V2 Plan

**V2 must use Rust** for all evaluations:
- Switch `double_oracle.py` to use `PayoffEvaluatorRust`
- Recompute Nash equilibria with Rust (10-100x faster anyway!)
- Use 2000 simulations (not 200) for higher confidence
- Compare Python vs Rust results

### Step 3: Document Differences

Create `experiments/nash/PYTHON_VS_RUST.md`:
- Compare V1 (Python) vs V2 (Rust) results
- Identify any payoff differences
- Similar to evolution's `RUST_SINGLE_SOURCE_OF_TRUTH.md`

---

## Impact on V2 Plan

### Original V2 Plan

1. Load evolved agents (V3/V4/V5)
2. Rerun Double Oracle with expanded strategy pool
3. Compare with V1 Nash results
4. Explain evolution-Nash gap

### Updated V2 Plan

1. **Switch to Rust evaluator** (critical!)
2. Load evolved agents (V3/V4/V5)
3. Run Double Oracle with:
   - Rust `PayoffEvaluatorRust` (not Python)
   - 2000 simulations (not 200)
   - Expanded strategy pool (archetypes + evolved)
4. Compare Python V1 vs Rust V2 results
5. Explain evolution-Nash gap

### Additional Questions

**Python vs Rust comparison**:
- Do Nash results change when using Rust?
- Is chain_reaction still 2.94 with Rust?
- Or does Rust find different equilibria?

This is exciting because it might explain the evolution-Nash gap!

---

## Hypothesis: Python/Rust May Explain The Gap

### The 20× Mystery

```
Nash V1 (Python):    2.94 payoff  (Free Rider)
Evolution (Rust):   58.50 payoff  (Near-optimal)
Gap:               +55.56 points
```

**Possible explanation**: Python Nash used wrong simulation dynamics!

If Python environment differs from Rust:
- Nash computed equilibrium for "wrong game"
- Evolution optimized for "correct game" (Rust)
- Gap is due to environment mismatch, not Nash theory!

**V2 test**: Recompute Nash with Rust. Does it now find 58.50 equilibrium?

---

## Implementation Priority

### Must Do (V2 Critical Path)

1. ✅ **Switch Double Oracle to Rust**
   - Update `double_oracle.py` to import `payoff_evaluator_rust`
   - Test on one scenario (chain_reaction)
   - Compare Python vs Rust results

2. ✅ **Recompute V1 with Rust**
   - Run Nash analysis on all 9 standard scenarios
   - Use 2000 simulations (higher confidence)
   - Document any differences from Python V1

3. ✅ **Move V1 results to deprecated folder**
   - Mark as Python-only
   - Reference in V2 documentation

### Then Do (V2 Evolution Integration)

4. Load evolved agents
5. Expand Double Oracle strategy pool
6. Compare with Rust Nash results
7. Complete gap analysis

---

## Success Criteria (Updated)

### Phase 1 Completion Requirements

1. ✅ **Rust single source of truth**
   - All Nash evaluations use Rust
   - Python version deprecated
   - Consistency validated

2. ✅ **V1 vs V2 comparison**
   - Python V1 results documented
   - Rust V2 results computed
   - Differences explained

3. ✅ **Evolution integration**
   - Evolved strategies in equilibrium analysis
   - Gap explained (Python/Rust or strategic divergence)
   - Cross-validation complete

---

## Files to Update

### Create/Move

- [x] `experiments/nash/CONSOLIDATION.md` (this file)
- [ ] `experiments/nash/v1_results_python_deprecated/` (directory)
- [ ] `experiments/nash/PYTHON_VS_RUST.md` (comparison doc)
- [ ] Move all equilibrium.json files to deprecated folder

### Update

- [ ] `experiments/nash/V2_PLAN.md` - Add Python/Rust comparison
- [ ] `experiments/nash/README.md` - Note V1 used Python (deprecated)
- [ ] `bucket_brigade/equilibrium/double_oracle.py` - Switch to Rust evaluator
- [ ] `experiments/scripts/compute_nash.py` - Use Rust evaluator

---

## Lessons from Evolution Research

The evolution team solved this exact problem:

**Problem**: Python environment gave different results than Rust
**Solution**: Use Rust as single source of truth for everything
**Result**: Perfect train/test consistency

**We should do the same for Nash!**

See: `experiments/evolution/RUST_SINGLE_SOURCE_OF_TRUTH.md`

---

## Timeline Impact

**Original V2 estimate**: 18 hours (2-3 days)

**Updated V2 estimate**: 24 hours (3-4 days)
- +6 hours: Python→Rust migration and validation

**But**: Rust is 10-100x faster, so actual runtime may be faster!

---

## Next Steps

1. **Immediate**: Move V1 results to deprecated folder
2. **Next session**: Switch Double Oracle to Rust
3. **Then**: Recompute V1 with Rust (validate baseline)
4. **Finally**: Add evolved agents and complete V2

---

**Status**: Consolidation plan complete, ready for execution
**Priority**: HIGH - blocks V2 Nash analysis
**Aligns with**: Evolution research Rust single source of truth
