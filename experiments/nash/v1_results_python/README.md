# Nash V1 Results (Python - Deprecated)

**⚠️ DEPRECATED**: These results used Python `BucketBrigadeEnv`, not Rust.

**Date Generated**: 2025-11-04
**Evaluator**: `bucket_brigade.equilibrium.payoff_evaluator` (Python)
**Status**: Questionable - needs Rust validation

---

## Why Deprecated?

These Nash equilibrium results were computed using the **Python simulation environment**, which:
1. **May differ from Rust**: Same issue as evolution V3/V4 train/test mismatch
2. **Low simulation count**: Only 200 simulations (not 2000 as claimed)
3. **Not validated**: Results never cross-checked with Rust

**Recommendation**: Use V2 results (Rust-backed) once available.

---

## V1 Results Summary

### Scenarios Analyzed

12 scenarios using Double Oracle algorithm:
- chain_reaction, deceptive_calm, default
- early_containment, easy, greedy_neighbor
- hard, mixed_motivation, overcrowding
- rest_trap, sparse_heroics, trivial_cooperation

### Key Findings

| Scenario | Type | Payoff | Strategy |
|----------|------|--------|----------|
| chain_reaction | Pure | 2.94 | Free Rider |
| greedy_neighbor | Pure | 58.77 | Coordinator |
| trivial_cooperation | Mixed | 108.78 | Free Rider mix |

**Equilibrium types**: 10 pure, 2 mixed
**Average cooperation**: 50%

### Critical Gap

```
Nash V1 (Python):    2.94 payoff  (chain_reaction)
Evolution (Rust):   58.50 payoff  (chain_reaction)
Gap:               +55.56 points  (20× difference!)
```

**Hypothesis**: Python/Rust environment mismatch may explain the gap.

---

## Files in This Directory

Each subdirectory contains `equilibrium.json` with:
- Scenario parameters
- Algorithm configuration (200 simulations, Double Oracle)
- Equilibrium strategy distribution
- Convergence info

**Structure**:
```
v1_results_python/
├── chain_reaction/equilibrium.json
├── deceptive_calm/equilibrium.json
└── ... (12 scenarios total)
```

---

## What's Next?

**V2 will use Rust** (`payoff_evaluator_rust.py`):
1. Recompute all Nash equilibria with Rust
2. Use 2000 simulations (higher confidence)
3. Compare Python V1 vs Rust V2 results
4. Integrate with evolved strategies (V3/V4/V5)

See: [V2_PLAN.md](../V2_PLAN.md)

---

## Related

- **Evolution Rust resolution**: [../evolution/RUST_SINGLE_SOURCE_OF_TRUTH.md](../../evolution/RUST_SINGLE_SOURCE_OF_TRUTH.md)
- **Nash consolidation**: [../CONSOLIDATION.md](../CONSOLIDATION.md)
- **V1 summary** (based on these results): [../README.md](../README.md)

---

**Status**: Archived for reference
**Use for**: Historical comparison only
**Do not cite**: Results may be incorrect due to Python environment
