# Phase 1.5: Cross-Scenario Generalization Analysis

**Date**: 2025-11-05
**Status**: Planning
**Prerequisites**: ✅ Nash V2 Complete, ✅ Evolution V4 Complete
**Related**: [Nash V2 Results](../nash/V2_RESULTS.md), [Phase 1 Roadmap](../../docs/roadmap_phased_plan.md)

---

## Executive Summary

Phase 1.5 bridges the gap between Phase 1 (Closed-World Mastery) and Phase 2 (Adaptive Multi-Scenario Agents) by systematically analyzing how well evolved strategies generalize across different scenarios.

**Key Question**: Are our evolved agents specialists (scenario-specific) or generalists (robust across scenarios)?

**Why This Matters**:
- Informs Phase 2 design (Should we train multi-scenario agents?)
- Validates evolutionary approach (Do agents learn general principles or overfit?)
- Identifies robust strategies (Which agents transfer best?)
- Guides mechanism design (What scenario features drive specialization?)

---

## Motivation

### What We Know (From Nash V2)

✅ **Evolution finds Nash equilibria**:
- 8/9 scenarios: Exact Nash equilibrium
- 1/9 scenarios: ε-Nash (ε ≈ 10⁻⁵)
- Cross-validation confirms game-theoretic soundness

✅ **Per-scenario optimization works**:
- Each evolved agent is optimal for its training scenario
- Self-play produces robust equilibrium strategies
- Convergence in 15,000 generations

### What We Don't Know

❓ **Do strategies generalize?**
- Does `chain_reaction` agent work well in `greedy_neighbor`?
- Which scenarios share effective strategies?
- What makes a strategy transferable?

❓ **Specialist vs Generalist?**
- Are agents narrowly optimized or broadly competent?
- Is there a performance cliff when scenarios change?
- Can we identify "universal" strategies?

❓ **Scenario similarity?**
- Which scenarios are strategically similar?
- What parameters drive strategic divergence?
- Can we cluster scenarios by strategic structure?

---

## Phase 1.5 Objectives

### Primary Goals

1. **Cross-Scenario Performance Matrix**
   - Test all 9 evolved agents in all 9 scenarios
   - Create 9×9 performance matrix (81 evaluations)
   - Identify transfer learning patterns

2. **Specialist vs Generalist Classification**
   - Measure generalization gap: (best_payoff - off_scenario_payoff)
   - Identify scenarios requiring specialization
   - Find robust multi-scenario strategies

3. **Scenario Similarity Analysis**
   - Cluster scenarios by strategic compatibility
   - Identify parameter-driven similarity
   - Map scenario space structure

4. **Strategic Feature Analysis**
   - Which strategy parameters transfer across scenarios?
   - What features predict generalization?
   - Connection to scenario parameters (β, c, κ)

### Success Criteria

**Must Have**:
- ✅ Complete 9×9 performance matrix
- ✅ Classify each agent as specialist or generalist
- ✅ Identify best-generalizing agent(s)
- ✅ Document findings in `GENERALIZATION_RESULTS.md`

**Nice to Have**:
- ⚠️ Scenario clustering/similarity metrics
- ⚠️ Parameter sensitivity analysis
- ⚠️ Strategic feature correlation study

---

## Technical Approach

### Phase 1: Data Collection

**Experiment Design**:
```python
# For each evolved agent (9 total)
for agent_scenario in SCENARIOS:
    agent = load_evolved_agent(agent_scenario, version="v4")

    # Test in all scenarios (including training scenario)
    for test_scenario in SCENARIOS:
        payoff = evaluate_agent(
            agent=agent,
            scenario=test_scenario,
            num_simulations=2000,  # Same as Nash V2
            evaluator="RustPayoffEvaluator"
        )

        results[agent_scenario][test_scenario] = payoff
```

**Computational Cost**:
- 9 agents × 9 scenarios = 81 evaluations
- 2000 simulations per evaluation
- ~30-60 seconds per evaluation (Rust evaluator)
- **Total runtime: ~40-80 minutes** (can parallelize)

**Output**: `experiments/generalization/performance_matrix.json`

### Phase 2: Performance Analysis

**Metrics**:

1. **Generalization Score**:
   ```python
   def generalization_score(agent_scenario: str) -> float:
       """Average performance across all scenarios."""
       payoffs = results[agent_scenario].values()
       return np.mean(payoffs)
   ```

2. **Specialization Gap**:
   ```python
   def specialization_gap(agent_scenario: str, test_scenario: str) -> float:
       """Performance drop when off training scenario."""
       on_scenario = results[agent_scenario][agent_scenario]
       off_scenario = results[agent_scenario][test_scenario]
       return on_scenario - off_scenario
   ```

3. **Transfer Efficiency**:
   ```python
   def transfer_efficiency(agent_scenario: str, test_scenario: str) -> float:
       """Performance relative to test scenario's Nash equilibrium."""
       agent_payoff = results[agent_scenario][test_scenario]
       nash_payoff = load_nash_payoff(test_scenario)
       return agent_payoff / nash_payoff
   ```

**Classification**:
- **Specialist**: Generalization score < 80% of on-scenario performance
- **Generalist**: Generalization score ≥ 80% of on-scenario performance
- **Universal**: Transfer efficiency ≥ 95% in all scenarios

### Phase 3: Scenario Similarity

**Clustering Method**:
```python
def scenario_similarity(scenario_a: str, scenario_b: str) -> float:
    """
    Measure strategic similarity between scenarios.

    Method: Average bidirectional transfer efficiency.
    """
    # How well does scenario_a agent work in scenario_b?
    forward = transfer_efficiency(scenario_a, scenario_b)

    # How well does scenario_b agent work in scenario_a?
    backward = transfer_efficiency(scenario_b, scenario_a)

    return (forward + backward) / 2
```

**Visualization**:
- Heatmap: 9×9 similarity matrix
- Dendrogram: Hierarchical clustering of scenarios
- Network graph: Connected scenarios with similarity > 0.9

### Phase 4: Strategic Feature Analysis

**Approach**:
```python
def analyze_strategic_features(performance_matrix):
    """
    Identify which strategy parameters predict generalization.

    Method: Regression analysis
    - X: Strategy parameters (10-dim genome)
    - Y: Generalization score
    - Model: Linear regression or random forest
    """

    # Extract features
    agents = []
    scores = []
    for agent_scenario in SCENARIOS:
        genome = load_agent_genome(agent_scenario)
        score = generalization_score(agent_scenario)
        agents.append(genome)
        scores.append(score)

    # Fit model
    model = RandomForestRegressor()
    model.fit(agents, scores)

    # Feature importance
    importance = model.feature_importances_
    # Which parameters matter most for generalization?

    return importance
```

---

## Expected Outcomes

### Hypothesis 1: Most Agents Are Specialists

**Prediction**: Agents optimized for specific scenarios won't transfer well due to:
- Differing fire spread rates (β: 0.10 to 0.30)
- Varying work costs (c: 0.30 to 0.80)
- Distinct strategic structures

**Test**: If true, we'll see:
- Diagonal dominance in performance matrix
- Large specialization gaps (>20% payoff drop)
- Low transfer efficiency (<70%)

**Implication**: Phase 2 multi-scenario training is necessary

### Hypothesis 2: Some Scenarios Cluster

**Prediction**: Scenarios with similar parameters will share effective strategies:
- (early_containment, greedy_neighbor, overcrowding, rest_trap): All have c=0.50, β=0.20
- (chain_reaction, mixed_motivation): Both have β=0.25, c=0.60

**Test**: Similarity scores >0.9 within clusters

**Implication**: Can reduce Phase 2 training to representative scenarios

### Hypothesis 3: Strategic Features Drive Generalization

**Prediction**: Certain strategy parameters predict generalization:
- `honesty` ≈ 0.31: Low dishonesty → consistent signaling
- `work_tendency` ≈ 0.06: Low work tendency (free riding)
- `risk_aversion` ≈ 0.95: High risk aversion (conservative play)

**Test**: Feature importance analysis shows these parameters have high weights

**Implication**: Can design generalist agents by constraining parameters

---

## Implementation Plan

### Step 1: Setup (1 hour)

- [ ] Create experiment directory structure:
  ```
  experiments/generalization/
  ├── PHASE_1.5_PLAN.md (this file)
  ├── GENERALIZATION_RESULTS.md (to be created)
  ├── performance_matrix.json
  ├── similarity_matrix.json
  ├── feature_importance.json
  └── analysis_plots/
      ├── performance_heatmap.png
      ├── similarity_dendrogram.png
      └── feature_importance.png
  ```

- [ ] Create evaluation script:
  ```bash
  experiments/scripts/evaluate_cross_scenario.py
  ```

- [ ] Create batch runner:
  ```bash
  experiments/scripts/run_generalization_all.sh
  ```

### Step 2: Run Experiments (2-3 hours)

**Option A: Local (Sequential)**:
```bash
cd experiments
./scripts/run_generalization_all.sh

# Runtime: ~80 minutes (81 evaluations × ~1 min each)
```

**Option B: Remote (Parallel)**:
```bash
# On remote server with 16 cores
ssh rwalters-sandbox-1
cd bucket-brigade

# Run 9 agents in parallel (9 × 9min = ~9min total)
./scripts/run_generalization_all.sh --parallel
```

**Output**:
- `performance_matrix.json`: Raw results
- `logs/generalization_*.log`: Execution logs

### Step 3: Analysis (4 hours)

- [ ] Compute performance metrics (generalization scores, specialization gaps)
- [ ] Generate performance heatmap (9×9 matrix visualization)
- [ ] Compute scenario similarity matrix
- [ ] Create similarity dendrogram (hierarchical clustering)
- [ ] Run feature importance analysis (which parameters predict generalization?)
- [ ] Identify specialists vs generalists
- [ ] Find best-generalizing agent

**Script**:
```bash
experiments/scripts/analyze_generalization.py
```

### Step 4: Documentation (3 hours)

- [ ] Create `GENERALIZATION_RESULTS.md` with:
  - Performance matrix (table + heatmap)
  - Specialist vs generalist classification
  - Scenario clustering results
  - Feature importance findings
  - Strategic insights
  - Recommendations for Phase 2

- [ ] Update main README with Phase 1.5 summary
- [ ] Link from roadmap document
- [ ] Commit all results and visualizations

---

## Timeline

**Assuming remote execution** (parallelized):

| Task | Duration | Dependencies |
|------|----------|--------------|
| Setup scripts & structure | 1 hour | - |
| Run experiments (remote) | 10-15 minutes | Setup complete |
| Download results | 5 minutes | Experiments complete |
| Performance analysis | 2 hours | Results local |
| Similarity analysis | 1 hour | Performance analysis |
| Feature analysis | 1 hour | - |
| Visualization | 2 hours | All analyses |
| Documentation | 3 hours | - |
| **Total** | **~11 hours** | **(1.5 days)** |

**Critical path**: Experiments (can run overnight) → Analysis → Documentation

---

## Phase 1.5 Research Questions

### Answered by This Work

1. **Do evolved agents generalize across scenarios?**
   - Quantified by generalization score and transfer efficiency

2. **Which scenarios are strategically similar?**
   - Similarity matrix and clustering analysis

3. **What makes a strategy transferable?**
   - Feature importance reveals key parameters

4. **Should we pursue Phase 2 multi-scenario training?**
   - Decision based on specialization vs generalization findings

### Deferred to Phase 2

5. **Can we train a single agent for all scenarios?**
   - Requires multi-scenario evolutionary training

6. **What is the optimal training curriculum?**
   - Requires curriculum learning experiments

7. **How do agents adapt to new scenarios?**
   - Requires online adaptation experiments

---

## Decision Framework: After Phase 1.5

Based on Phase 1.5 results, we'll choose next steps:

### Outcome A: Strong Generalization

**Observed**: Generalization scores >80%, high transfer efficiency

**Interpretation**: Scenarios share strategic structure, agents learn general principles

**Next Step**:
- **Skip to Phase 2**: Multi-scenario training likely unnecessary
- **Focus on PPO**: Leverage transferable strategies for RL baseline

### Outcome B: Strong Specialization

**Observed**: Generalization scores <60%, large specialization gaps

**Interpretation**: Scenarios require tailored strategies, limited transfer

**Next Step**:
- **Pursue Phase 2**: Multi-scenario training essential
- **Curriculum learning**: Train on diverse scenarios progressively

### Outcome C: Mixed (Most Likely)

**Observed**: Some agents generalize (>70%), others specialize (<60%)

**Interpretation**: Scenario clusters exist, some universal principles

**Next Step**:
- **Phase 1.5B**: Focus on generalizing agents, understand why they transfer
- **Then Phase 2**: Multi-scenario training on cluster representatives
- **Parallel PPO**: Baseline with best-generalizing agent

---

## Success Metrics

### Quantitative

1. **Coverage**: 81/81 evaluations completed (100%)
2. **Consistency**: Evaluation variance <5% (reliable measurements)
3. **Nash alignment**: On-scenario performance matches Nash V2 (±5%)

### Qualitative

4. **Insights**: Clear specialist vs generalist classification
5. **Actionability**: Concrete recommendations for Phase 2
6. **Visualization**: Informative heatmaps and dendrograms

---

## Risks and Mitigations

### Risk 1: Evaluation Noise

**Problem**: Stochastic simulations may produce noisy results

**Mitigation**:
- Use 2000 simulations (same as Nash V2)
- Run 3 trials per evaluation, average results
- Report confidence intervals

### Risk 2: Environment Mismatch

**Problem**: Python/Rust discrepancy (learned from Nash V1)

**Mitigation**:
- **Use RustPayoffEvaluator exclusively**
- Validate against Nash V2 on-scenario payoffs
- Cross-check diagonal entries (should match Nash equilibria)

### Risk 3: Incomplete V4 Data

**Problem**: Some scenarios may lack V4 evolved agents

**Mitigation**:
- Verified all 9 scenarios have V4 agents (per git status)
- Fallback to V3 if needed (but V4 preferred)

### Risk 4: Interpretation Ambiguity

**Problem**: Unclear threshold for "generalist" vs "specialist"

**Mitigation**:
- Use multiple metrics (generalization score, transfer efficiency, specialization gap)
- Report full distribution, not just binary classification
- Compare to baselines (random agent, Nash equilibria)

---

## Outputs and Deliverables

### Data Files

1. **`performance_matrix.json`**: 9×9 payoff matrix
2. **`similarity_matrix.json`**: Scenario similarity scores
3. **`feature_importance.json`**: Parameter weights for generalization
4. **`classification.json`**: Specialist vs generalist labels

### Visualizations

5. **`performance_heatmap.png`**: 9×9 heatmap with annotations
6. **`similarity_dendrogram.png`**: Hierarchical clustering tree
7. **`feature_importance.png`**: Bar chart of parameter importance
8. **`transfer_efficiency_boxplot.png`**: Distribution by scenario

### Documentation

9. **`GENERALIZATION_RESULTS.md`**: Comprehensive analysis report
10. **`PHASE_1.5_SUMMARY.md`**: Executive summary for roadmap

---

## Connection to Broader Research

### Phase 1 (Closed-World Mastery)

**What Phase 1.5 Completes**:
- Final validation of single-scenario optimization
- Understanding limits of per-scenario training
- Foundation for Phase 2 decision

**What Remains in Phase 1**:
- PPO baseline (deferred, may use Phase 1.5 insights)
- Infrastructure hardening (ongoing)

### Phase 2 (Adaptive Multi-Scenario Agents)

**How Phase 1.5 Informs Phase 2**:
- If specialists: Phase 2 essential (heterogeneous training needed)
- If generalists: Phase 2 simplified (leverage transferable strategies)
- Scenario clusters: Reduce Phase 2 training set

**Phase 2 Design Implications**:
- Training curriculum: Start with similar scenarios (clusters)
- Evaluation: Test on unseen scenarios from same cluster
- Generalization baseline: Use Phase 1.5 best-generalizing agent

### Long-Term Research

**Publications**:
- "Evolutionary Strategies Discover Nash Equilibria in Multi-Agent Firefighting" (Nash V2)
- "Cross-Scenario Generalization in Cooperative Multi-Agent Systems" (Phase 1.5)

**Applications**:
- Transfer learning for emergency response
- Robustness analysis for policy deployment
- Scenario design for training diversity

---

## References

**Prerequisites**:
- [Nash V2 Results](../nash/V2_RESULTS.md): Game-theoretic validation of evolved agents
- [Nash V2 Plan](../nash/V2_PLAN.md): V2 experimental design
- [Evolution V4](../scenarios/*/evolved_v4/): Evolved agent genomes

**Related Work**:
- [Phase 1 Roadmap](../../docs/roadmap_phased_plan.md): Overall project structure
- [Technical Review](../../docs/technical_marl_review.md): MARL background

**Future Directions**:
- Phase 2 Plan (to be created after Phase 1.5 results)
- PPO Baseline Plan (deferred)

---

**Status**: Planning Complete, Ready for Implementation
**Next**: User approval → Setup scripts → Run experiments → Analysis

