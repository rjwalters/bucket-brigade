# Nash Equilibrium V1 - Cross-Scenario Summary

**Version**: V1 (Baseline - Predefined Archetypes Only)
**Generated:** 2025-11-04 14:49:20
**Scenarios Analyzed:** 12
**Next**: [V2 Plan](./V2_PLAN.md) - Integration with evolved strategies

## Overview

This report presents Nash equilibrium analysis for all Bucket Brigade scenarios,
computed using the Double Oracle algorithm with Rust-accelerated payoff evaluation.

## Equilibrium Types

| scenario            | type   |   support_size |   expected_payoff |   iterations | converged   |   elapsed_time |
|:--------------------|:-------|---------------:|------------------:|-------------:|:------------|---------------:|
| chain_reaction      | pure   |              1 |            2.936  |            2 | True        |        750.948 |
| deceptive_calm      | pure   |              1 |           37.298  |            1 | True        |        436.384 |
| default             | pure   |              1 |           60.305  |            2 | True        |        256.714 |
| early_containment   | mixed  |              2 |           24.2664 |            3 | True        |       1646.92  |
| easy                | pure   |              1 |           81.89   |            2 | True        |        252.112 |
| greedy_neighbor     | pure   |              1 |           58.775  |            2 | True        |        641.441 |
| hard                | pure   |              1 |          -45.69   |            2 | True        |        222.239 |
| mixed_motivation    | pure   |              1 |           46.657  |            1 | True        |        434.782 |
| overcrowding        | pure   |              1 |           17.786  |            1 | True        |        407.945 |
| rest_trap           | pure   |              1 |           98.935  |            1 | True        |        407.651 |
| sparse_heroics      | pure   |              1 |           80.7075 |            1 | True        |        457.055 |
| trivial_cooperation | mixed  |              2 |          108.776  |            3 | True        |       2189.46  |

### Key Observations

- **Pure equilibria:** 10/12 scenarios
- **Mixed equilibria:** 2/12 scenarios
- **Average support size:** 1.17
- **Average convergence time:** 675.3s
- **Convergence rate:** 100.0%

**Pure equilibrium scenarios:** chain_reaction, deceptive_calm, default, easy, greedy_neighbor, hard, mixed_motivation, overcrowding, rest_trap, sparse_heroics

**Mixed equilibrium scenarios:** early_containment, trivial_cooperation

## Cooperation Analysis

| scenario            |   cooperation_rate |   free_riding_rate | equilibrium_type   |
|:--------------------|-------------------:|-------------------:|:-------------------|
| chain_reaction      |                  0 |                  1 | pure               |
| deceptive_calm      |                  1 |                  0 | pure               |
| default             |                  1 |                  0 | pure               |
| early_containment   |                  0 |                  1 | mixed              |
| easy                |                  0 |                  1 | pure               |
| greedy_neighbor     |                  1 |                  0 | pure               |
| hard                |                  0 |                  1 | pure               |
| mixed_motivation    |                  1 |                  0 | pure               |
| overcrowding        |                  0 |                  1 | pure               |
| rest_trap           |                  1 |                  0 | pure               |
| sparse_heroics      |                  1 |                  0 | pure               |
| trivial_cooperation |                  0 |                  1 | mixed              |

### Cooperation Patterns

- **Highest cooperation:** deceptive_calm (100.0%)
- **Lowest cooperation:** chain_reaction (0.0%)
- **Average cooperation:** 50.0%

## Strategy Distributions

### chain_reaction

**Type:** Pure equilibrium
**Expected Payoff:** 2.94

Pure strategy: **Free Rider** (closest to Free Rider, distance=0.000)

### deceptive_calm

**Type:** Pure equilibrium
**Expected Payoff:** 37.30

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### default

**Type:** Pure equilibrium
**Expected Payoff:** 60.30

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### early_containment

**Type:** Mixed equilibrium
**Expected Payoff:** 24.27

Mixed strategy distribution:

1. **Free Rider** (59.9%) - closest to Free Rider (distance=0.000)
2. **Free Rider** (40.1%) - closest to Free Rider (distance=0.000)

### easy

**Type:** Pure equilibrium
**Expected Payoff:** 81.89

Pure strategy: **Free Rider** (closest to Free Rider, distance=0.000)

### greedy_neighbor

**Type:** Pure equilibrium
**Expected Payoff:** 58.77

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### hard

**Type:** Pure equilibrium
**Expected Payoff:** -45.69

Pure strategy: **Free Rider** (closest to Free Rider, distance=0.000)

### mixed_motivation

**Type:** Pure equilibrium
**Expected Payoff:** 46.66

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### overcrowding

**Type:** Pure equilibrium
**Expected Payoff:** 17.79

Pure strategy: **Free Rider** (closest to Free Rider, distance=0.000)

### rest_trap

**Type:** Pure equilibrium
**Expected Payoff:** 98.94

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### sparse_heroics

**Type:** Pure equilibrium
**Expected Payoff:** 80.71

Pure strategy: **Coordinator** (closest to Coordinator, distance=0.000)

### trivial_cooperation

**Type:** Mixed equilibrium
**Expected Payoff:** 108.78

Mixed strategy distribution:

1. **Free Rider** (93.0%) - closest to Free Rider (distance=0.000)
2. **Free Rider** (7.0%) - closest to Free Rider (distance=0.000)

## Cross-Scenario Insights

### Parameter-Equilibrium Relationships

**High work cost scenarios (c > 0.50):**
- Average cooperation: 60.0%
- Pure equilibria: 5/5

**High fire spread scenarios (β > 0.23):**
- Average cooperation: 50.0%
- Pure equilibria: 5/6

## Design Recommendations

Based on Nash equilibrium analysis:

1. **Greedy Neighbor Scenario** (cooperation: 100.0%)
   - High work cost creates free-riding incentive

2. **Overall Cooperation** (average: 50.0%)
   - Moderate cooperation across scenarios

3. **Equilibrium Diversity**
   - Predominance of pure equilibria indicates clear optimal strategies
   - Game parameters may need tuning for strategic depth

## V1 Limitations and Next Steps

### Key Limitation

**V1 used only predefined archetypes** (Coordinator, Free Rider, Hero, Liar). This missed:
- Evolved strategies from genetic algorithm research
- Potentially better equilibria outside the archetype space
- Connection between theory (Nash) and practice (evolution)

### Critical Gap: chain_reaction

| Method | Result | Gap |
|--------|--------|-----|
| **V1 Nash (Free Rider)** | 2.94 payoff | Baseline |
| **Evolution V3/V4** | 58.50 payoff | **+55.56** 🔍 |

**Why does evolution achieve 20× better than Nash?** This is the central question for V2.

### V2 Plans

See **[V2_PLAN.md](./V2_PLAN.md)** for:
- Integration with evolved strategies (V3/V4/V5)
- Evolution-Nash comparison and gap analysis
- Epsilon-equilibrium and robustness testing
- Cross-validation between theory and practice

**V2 Goals**:
1. ✅ Add evolved agents to Double Oracle strategy pool
2. ✅ Explain the 2.94 → 58.50 gap in chain_reaction
3. ✅ Validate if evolved strategies are Nash equilibria
4. ✅ Complete Phase 1 Nash track with evolution cross-validation

---

## Phase 2 Goals: Closing the Prediction Gap

**See**: [Phase 2 Research Agenda](../../docs/PHASE_2_RESEARCH_AGENDA.md#nash-track-closing-the-prediction-gap)

### Core Research Questions

1. **Why does evolution beat Nash by 20×?** (chain_reaction: 2.94 → 58.50)
   - Are evolved strategies outside the archetype space?
   - Is the Nash prediction wrong or incomplete?
   - What does the true strategy space look like?

2. **Can Double Oracle find evolved-quality strategies?**
   - Integrate evolved agents (v3/v4/v5) into strategy pool
   - Compute equilibria over expanded space
   - Compare: Nash V1 (archetypes) vs. Nash V2 (with evolution)

3. **Are evolved strategies actually equilibria?**
   - Test if evolved strategies are best-responses to themselves
   - Compute epsilon-equilibrium bounds
   - Validate robustness to perturbations

### Phase 2 Experiments

| Experiment | Description | Status |
|------------|-------------|--------|
| **V2 Nash Computation** | Run Double Oracle with evolved agents as initial strategies | 🚀 Planned |
| **Gap Analysis** | Explain 2.94 → 58.50 discrepancy via strategy space coverage | 🚀 Planned |
| **Epsilon-Equilibrium Testing** | Measure how close evolved strategies are to true equilibria | 🚀 Planned |
| **Cross-Validation** | Do Nash predictions match evolution in expanded space? | 🚀 Planned |
| **Heterogeneous Equilibria** | Mixed agent types, team composition optimization | 📋 Q2 |
| **Robustness Analysis** | Parameter perturbations, stability testing | 📋 Q3 |

### Success Criteria

✅ **Close the gap**: Explain and resolve 2.94 → 58.50 discrepancy
✅ **Formal characterization**: When does Nash V1 fail?
✅ **Integration framework**: Add external strategies to Double Oracle
✅ **All scenarios**: V2 Nash for all 12 scenarios

### Deliverables

- `experiments/nash/V2_RESULTS.md`: Nash computation with evolved strategies
- `experiments/nash/GAP_ANALYSIS.md`: Detailed explanation of chain_reaction gap
- Updated equilibrium tables: V1 (archetypes) vs. V2 (evolved)
- Paper draft: "From Archetypes to Evolution: Expanding Nash Equilibrium Strategy Spaces"

### Timeline

**Q1 (Months 1-3)**:
- ✅ Implement evolved agent integration into Double Oracle
- ✅ Run V2 Nash on chain_reaction with evolved strategies
- ✅ Document gap analysis (2.94 → 58.50 explanation)

**Q2 (Months 4-6)**:
- ✅ V2 Nash for all 12 scenarios
- ✅ Epsilon-equilibrium analysis
- ✅ Heterogeneous team equilibria

**Q3 (Months 7-9)**:
- ✅ Robustness testing (parameter perturbations)
- ✅ Final V2 cross-scenario report
- 🎯 Paper draft

---

## Future Work (Phase 3+)

1. **Mechanism design** (Phase 3+)
   - Design incentives to improve equilibrium outcomes
   - Reduce price of anarchy where applicable

2. **Adaptive equilibria** (Phase 3+)
   - Equilibria for scenario-switching environments
   - Meta-strategies that infer and respond to context

---

**Version**: V1 (Baseline)
**Status**: Complete - provides foundation for V2
**Next**: [V2 Plan](./V2_PLAN.md) - Evolution integration
**Phase 2**: [Research Agenda](../../docs/PHASE_2_RESEARCH_AGENDA.md) - Understanding effectiveness

*Analysis generated using Double Oracle algorithm with 2000 simulations per evaluation.*
