---
title: "Phase 2.5: Maximum Value Extraction from V7 & Nash V5"
date: "2025-11-08"
author: "Research Team"
tags: ["analysis", "planning", "v7", "nash-v5", "phase-2"]
status: "Planning"
---

# Phase 2.5: Extract Maximum Value from Existing Data

**Guiding Principle**: *Start Concrete, Iterate Empirically* → Analyze before experimenting

---

## The Central Insight: Fitness Function Shapes Discovery

### Phase 1.5 (Universal Equilibrium)
- **Fitness**: Self-play (homogeneous teams)
- **Result**: ALL agents converged to identical strategy
- **Interpretation**: Dominant free-riding strategy

### V7 (Diverse Strategies)
- **Fitness**: Heterogeneous tournament (mixed agent types)
- **Result**: 12 distinct strategies with varied cooperation
- **Interpretation**: Context-dependent optima

**Key Question**: Why does the fitness function change what evolution discovers?

---

## Analysis Priorities (Ordered by Insight/Effort Ratio)

### Priority 1: Zero-Compute Analyses ⚡

**A. Nash vs Evolution Parameter Comparison**
- Extract Hero parameters from Nash V5 results
- Compare to v7 evolved agents side-by-side
- Quantify systematic differences (work, honesty, cooperation)

**B. V7 Parameter Patterns**
- Already extracted (see notebook entry)
- Group scenarios by cooperation level
- Identify behavioral archetypes

**C. Hypothesis Generation**
- Document specific testable predictions
- Design minimal experiments to validate
- Prioritize by falsifiability

**Time**: 2-3 hours | **Value**: High (guides all future work)

---

### Priority 2: Evolution Trajectory Visualization 📊

**Question**: How did strategies emerge over 200 generations?

**Data Available**:
- Checkpoints every 20 generations (all 12 scenarios)
- Best/mean/worst fitness at each checkpoint
- Full population genomes

**Visualizations to Create**:
1. **Fitness Curves**: Line plots (best/mean/worst) for all scenarios
2. **Parameter Evolution**: How honesty, cooperation, work change over time
3. **Diversity Metrics**: Population variance throughout evolution
4. **Critical Transitions**: Identify generation where strategies lock in

**Tools**: matplotlib, seaborn (pure visualization, no compute)

**Time**: 4-6 hours | **Value**: Medium-High (understanding emergence)

---

### Priority 3: Nash V5 Deep Dive 🎯

**Questions**:
1. Are Nash equilibria actually Hero in heterogeneous tournaments?
2. Would v7 agents beat Nash predictions in mixed games?
3. Are v7 strategies epsilon-Nash equilibria?

**Zero-Compute Analysis**:
```bash
# Extract all Nash V5 equilibrium strategies
for scenario in experiments/nash/v2_results_v5/*/; do
  jq '.equilibrium | {type, payoff, strategy:.strategy_pool[0]}' \
    $scenario/equilibrium_v2.json
done
```

**Low-Compute Test** (~30 min):
```python
# Evaluate Hero in heterogeneous tournament
hero_params = {honesty: 1.0, work: 1.0, neighbor_help: 1.0, risk: 0.1}
fitness = evaluate_heterogeneous_tournament(hero_params, "chain_reaction")
print(f"Hero heterogeneous fitness: {fitness} vs v7: 43.67")
```

**Time**: 2-3 hours analysis + 30min compute | **Value**: High (resolves paradox)

---

### Priority 4: Scenario Clustering 🗂️

**Goal**: Group scenarios by strategic similarity

**Method**:
1. V7 parameter vectors for all 12 scenarios
2. Compute pairwise distances (L2 norm)
3. Hierarchical clustering
4. Visualize dendrogram

**Expected**: 
- Cooperative cluster: chain_reaction, easy, deceptive_calm
- Selfish cluster: default, hard, rest_trap
- Mixed: others

**Value**: Predictive power for new scenarios

**Time**: 2 hours | **Value**: Medium (taxonomy building)

---

### Priority 5: Cross-Scenario Transfer Matrix 🔀

**Question**: Can agents evolved for scenario A perform well in scenario B?

**Method** (~2 hours compute on sandbox-1):
```python
# 12×12 matrix: [trained_on, tested_on]
for train_scenario in scenarios:
    agent = load_v7(train_scenario)
    for test_scenario in scenarios:
        payoff = evaluate(agent, test_scenario, n_games=100)
        matrix[train, test] = payoff
```

**Analysis**:
- Diagonal: In-scenario performance
- Off-diagonal: Transfer performance
- Clustering: Do cooperative agents transfer to cooperative scenarios?

**Time**: 2 hours compute + 2 hours analysis | **Value**: High (generalization)

---

### Priority 6: Parameter Sensitivity Analysis 🔬

**Question**: Which parameters actually matter?

**Method** (~4 hours compute on sandbox-1):
- For 3 representative scenarios (easy, hard, chain_reaction)
- Perturb each parameter: ±10%, ±25%, ±50%
- Measure fitness change
- Rank parameters by sensitivity

**Expected**:
- **chain_reaction**: neighbor_help most sensitive (max cooperation)
- **hard**: risk_aversion most sensitive (survival focus)
- **trivial_cooperation**: all parameters insensitive (low baseline)

**Time**: 4 hours compute + 3 hours analysis | **Value**: Medium (optimization insights)

---

## Deliverables & Timeline

### Week 1: Foundation (Zero-Compute)
- ✅ V7 parameter analysis (DONE - in notebook)
- 🔲 Nash vs Evolution comparison
- 🔲 Hypothesis generation document
- 🔲 Research notebook entry: "Nash vs Evolution Paradox"

### Week 2: Visualization (Low-Compute)
- 🔲 Evolution trajectory plots (all 12 scenarios)
- 🔲 Parameter drift heatmaps
- 🔲 Scenario clustering dendrogram
- 🔲 Research notebook entry: "Evolution Dynamics"

### Week 3: Empirical Tests (Moderate-Compute)
- 🔲 Hero heterogeneous tournament test
- 🔲 Cross-scenario transfer matrix (12×12)
- 🔲 Research notebook entry: "Generalization & Transfer"

### Week 4: Synthesis
- 🔲 Parameter sensitivity analysis (3 scenarios)
- 🔲 Integration document: "Phase 2.5 Insights"
- 🔲 Phase 2.6 experimental design
- 🔲 Updated research agenda

---

## Decision Tree: Phase 2.6 Design

Based on Phase 2.5 findings, we'll pursue:

**If v7 agents are NOT Nash equilibria in heterogeneous games**:
→ Compute Nash V6 with v7 agents in strategy pool
→ Test: Do evolved strategies disrupt theoretical equilibria?

**If cooperation scenarios cluster strongly**:
→ Meta-evolution: Train on scenario distribution
→ Test: Can single agent handle cooperative scenario family?

**If parameter sensitivity reveals critical dimensions**:
→ Reduced-parameter evolution (3-5 params instead of 10)
→ Test: Does constraining search improve interpretability?

**If cross-scenario transfer is poor**:
→ Multi-task learning or meta-learning approaches
→ Test: Can agents learn to infer scenario type?

---

## Success Metrics

✅ **Understand** why v7 differs from Phase 1.5 universal equilibrium  
✅ **Explain** Nash vs Evolution discrepancy  
✅ **Predict** which scenarios favor cooperation vs selfishness  
✅ **Design** Phase 2.6 experiments with clear motivation  

**Alignment with Principles**:
- ✅ Stay Interpretable: All analyses use small, transparent models
- ✅ Iterate Empirically: Data guides next experiments
- ✅ Embrace Uncertainty: Measure robustness, not just performance
- ✅ Start Concrete: Master current data before new experiments

---

## Next Action

**Start with Priority 1A**: Extract Nash V5 equilibrium parameters and compare to v7 evolved agents. This zero-compute analysis will immediately clarify the Nash vs Evolution paradox.

```bash
# Run this analysis now:
uv run python scripts/compare_nash_vs_v7.py
```

Then proceed through priorities 2-6 based on emerging insights.

