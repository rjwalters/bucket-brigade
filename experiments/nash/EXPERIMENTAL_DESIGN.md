# Nash Equilibrium V2: Experimental Design & Roadmap

**Phase**: 2 (Understanding Agent Effectiveness)
**Track**: Game-Theoretic Analysis
**Goal**: Close the 2.94 → 58.50 gap by integrating evolved strategies into Nash computation

---

## Overview

This document provides a step-by-step experimental protocol for Phase 2 Nash equilibrium research. The core question: **Why does evolution achieve 20× better payoff than Nash predictions?**

### Hypothesis

The Nash V1 gap exists because:
1. Predefined archetypes (Coordinator, Free Rider, Hero, Liar) don't cover the strategy space
2. Evolved strategies represent novel behaviors outside this space
3. Double Oracle can find better equilibria if given access to evolved strategies

### Validation Approach

Integrate evolved agents into Double Oracle and recompute equilibria to test if:
- Nash V2 payoffs approach evolved performance (58.50)
- Evolved strategies are part of equilibrium support
- Gap closes across all scenarios

---

## Experiment 1: Nash V2 with Evolved Strategies (chain_reaction)

**Goal**: Compute Nash equilibrium with evolved agents in the strategy pool.

**Priority**: 🔴 Critical (Q1)

### Setup

**Prerequisites**:
- ✅ Rust evaluator working (single source of truth)
- ✅ Evolved agents available (v3/v4/v5 in `experiments/scenarios/chain_reaction/`)
- ✅ Double Oracle implementation (`bucket_brigade/equilibrium/`)

**Input strategies**:
1. **Archetypes** (from V1):
   - Coordinator (high cooperation)
   - Free Rider (low cooperation)
   - Hero (high work, own house priority)
   - Liar (dishonest signaling)

2. **Evolved strategies** (new):
   - V3 best agent (58.50 payoff)
   - V4 best agent (58.50 payoff)
   - V5 best agent (TBD, ~69.16 early)

### Protocol

**Step 1: Extract evolved agent genomes**

```bash
cd experiments/nash

# V3 agent
python -c "
import json
with open('../scenarios/chain_reaction/evolved_v3/best_agent.json') as f:
    agent = json.load(f)
print('V3 genome:', agent['genome'])
print('V3 fitness:', agent['fitness'])
"

# Repeat for V4, V5
```

**Step 2: Run Nash V2 computation**

```bash
uv run python experiments/scripts/compute_nash_v2.py \
  chain_reaction \
  --evolved-versions v3 v4 v5 \
  --simulations 2000 \
  --output-dir experiments/nash/v2_results/chain_reaction \
  --seed 42
```

**Parameters**:
- `--evolved-versions`: Which evolved agents to include
- `--simulations`: Games per payoff evaluation (2000 for accuracy)
- `--output-dir`: Where to save results
- `--seed`: Reproducibility

**Expected runtime**: 1-2 hours (depending on Double Oracle iterations)

**Step 3: Analyze results**

Compare V1 vs. V2:

```bash
# Generate comparison report
uv run python experiments/scripts/compare_nash_versions.py \
  --v1-dir experiments/nash/v1_results/chain_reaction \
  --v2-dir experiments/nash/v2_results/chain_reaction \
  --output experiments/nash/GAP_ANALYSIS_chain_reaction.md
```

**Key metrics to extract**:
- Expected payoff (V1 vs. V2)
- Equilibrium type (pure vs. mixed)
- Support strategies (which agents in equilibrium?)
- Gap closure: `(V2_payoff - V1_payoff) / (evolved_payoff - V1_payoff)`

### Expected Outcomes

**Scenario A: Gap closes**
- V2 payoff ≈ 58.50
- Evolved agents in equilibrium support
- **Conclusion**: Nash V1 failed due to limited strategy space

**Scenario B: Gap remains**
- V2 payoff ≈ V1 payoff (still ~2.94)
- Evolved agents not selected by Double Oracle
- **Conclusion**: Evolved strategies are not equilibria (further investigation needed)

**Scenario C: Partial closure**
- V2 payoff > V1 but < evolved (e.g., 30.00)
- Mixed equilibrium with some evolved agents
- **Conclusion**: Strategy space expansion helps but not sufficient

### Deliverables

1. **Results file**: `experiments/nash/v2_results/chain_reaction/equilibrium.json`
   - Equilibrium strategies and probabilities
   - Expected payoff
   - Convergence history

2. **Gap analysis**: `experiments/nash/GAP_ANALYSIS_chain_reaction.md`
   - V1 vs. V2 comparison
   - Explanation of gap closure (or lack thereof)
   - Hypothesis for remaining gap

3. **Visualization**: `experiments/nash/v2_results/chain_reaction/strategy_space.png`
   - Plot showing archetype strategies vs. evolved strategies
   - Equilibrium support highlighted

---

## Experiment 2: Epsilon-Equilibrium Testing

**Goal**: Measure how close evolved strategies are to true Nash equilibria.

**Priority**: 🟡 High (Q1)

### Background

An **epsilon-equilibrium** allows strategies that are "nearly" best responses (within ε payoff of optimal). If evolved strategies are ε-equilibria with small ε, they're "close enough" to Nash.

### Protocol

**Step 1: Define test scenarios**

For each evolved agent (V3, V4, V5):
1. All 4 agents play evolved strategy (homogeneous team)
2. Measure baseline payoff: `U_evolved`

**Step 2: Test deviations**

For each archetype strategy (and other evolved agents):
1. Replace one agent with alternative strategy
2. Measure deviant's payoff: `U_deviant`
3. Compute incentive to deviate: `ε = max(U_deviant - U_evolved, 0)`

**Step 3: Run epsilon test**

```bash
uv run python experiments/scripts/test_epsilon_equilibrium.py \
  chain_reaction \
  --focal-strategy experiments/scenarios/chain_reaction/evolved_v3/best_agent.json \
  --alternative-strategies Coordinator FreRider Hero Liar \
    experiments/scenarios/chain_reaction/evolved_v4/best_agent.json \
    experiments/scenarios/chain_reaction/evolved_v5/best_agent.json \
  --num-games 1000 \
  --output experiments/nash/epsilon_results/chain_reaction_v3.json
```

**Repeat for V4, V5 as focal strategies.**

### Expected Outcomes

**Small ε (< 5 payoff units)**:
- Evolved strategies are approximate equilibria
- No significant incentive to deviate
- **Conclusion**: Evolution finds near-Nash strategies

**Large ε (> 10 payoff units)**:
- Evolved strategies are not equilibria
- Significant incentive to deviate exists
- **Conclusion**: Evolved agents exploit homogeneous teams (not robust)

### Deliverables

1. **Epsilon report**: `experiments/nash/EPSILON_EQUILIBRIUM_chain_reaction.md`
   - Table of ε values for each (focal, deviant) pair
   - Threshold analysis: What ε is acceptable?
   - Interpretation of results

2. **Robustness matrix**: `experiments/nash/epsilon_results/robustness_matrix.csv`
   ```
   Focal,Deviant,Baseline_Payoff,Deviant_Payoff,Epsilon
   V3,Coordinator,58.50,45.20,0.00
   V3,FreRider,58.50,62.30,3.80
   ...
   ```

---

## Experiment 3: Cross-Scenario Nash V2

**Goal**: Compute Nash V2 for all 12 scenarios to understand generalization.

**Priority**: 🟢 Medium (Q2)

### Protocol

**Step 1: Prepare evolved agents for all scenarios**

Check which scenarios have evolved agents:
```bash
for scenario in chain_reaction deceptive_calm early_containment easy \
                greedy_neighbor mixed_motivation overcrowding rest_trap \
                sparse_heroics trivial_cooperation default hard; do
  echo "=== $scenario ==="
  ls -l experiments/scenarios/$scenario/evolved_v4/ 2>/dev/null || echo "Missing"
done
```

**Step 2: Run evolution for missing scenarios**

For scenarios without evolved agents, run V4 evolution:
```bash
# Example: Run evolution for greedy_neighbor if missing
uv run python experiments/scripts/run_evolution.py \
  --scenario greedy_neighbor \
  --population 200 \
  --generations 15000 \
  --games-per-eval 50 \
  --output experiments/scenarios/greedy_neighbor/evolved_v4
```

**Step 3: Batch Nash V2 computation**

```bash
# Create batch script
cat > experiments/nash/run_all_v2.sh <<'EOF'
#!/bin/bash
for scenario in chain_reaction deceptive_calm easy greedy_neighbor \
                mixed_motivation overcrowding rest_trap sparse_heroics \
                trivial_cooperation; do
  echo "Computing Nash V2 for $scenario"
  uv run python experiments/scripts/compute_nash_v2.py \
    $scenario \
    --evolved-versions v4 \
    --simulations 2000 \
    --output-dir experiments/nash/v2_results/$scenario \
    --seed 42
done
EOF

chmod +x experiments/nash/run_all_v2.sh
./experiments/nash/run_all_v2.sh
```

**Expected runtime**: 12-24 hours for all scenarios (parallel if CPU resources available)

### Analysis

**Step 1: Generate cross-scenario summary**

```bash
uv run python experiments/scripts/summarize_nash_v2.py \
  --v1-results experiments/nash/ \
  --v2-results experiments/nash/v2_results/ \
  --output experiments/nash/V2_CROSS_SCENARIO_SUMMARY.md
```

**Step 2: Compare V1 vs. V2 across scenarios**

Key questions:
1. Which scenarios show largest gap closure?
2. Are there patterns by scenario type (cooperation vs. free-riding)?
3. Do evolved agents consistently enter equilibrium support?

**Step 3: Scenario clustering**

Group scenarios by Nash V2 characteristics:
- **Type A**: Gap fully closed (V2 ≈ evolved)
- **Type B**: Gap partially closed (V1 < V2 < evolved)
- **Type C**: No gap (V1 ≈ V2, evolved ≈ V1)
- **Type D**: Gap opened (V2 < V1, unexpected)

### Deliverables

1. **Cross-scenario report**: `experiments/nash/V2_CROSS_SCENARIO_SUMMARY.md`
   - Table comparing all scenarios
   - Gap closure analysis
   - Scenario taxonomy

2. **Updated README**: Add V2 results table to `experiments/nash/README.md`

---

## Experiment 4: Heterogeneous Equilibria

**Goal**: Compute Nash equilibria with diverse agent types (not homogeneous teams).

**Priority**: 🟢 Medium (Q2)

### Background

Phase 1/2 assume **homogeneous teams** (all agents same strategy). Real scenarios may have **heterogeneous teams** (specialists, complementary roles).

**Question**: Do Nash equilibria exist where agents play different strategies in equilibrium?

### Protocol

**Step 1: Define heterogeneous strategy spaces**

For a 4-agent game:
- Agent 1, 2, 3, 4 can each choose from: `{Coordinator, FreRider, Hero, Liar, Evolved}`
- Total combinations: 5^4 = 625 (too many for exhaustive search)

**Step 2: Use Double Oracle with heterogeneous best-response**

Modify `compute_nash_v2.py` to:
1. Allow per-agent strategy assignments
2. Compute best-response per position (agent 1, 2, 3, 4)
3. Add position-specific strategies to oracle pool

```bash
uv run python experiments/scripts/compute_nash_v2_heterogeneous.py \
  chain_reaction \
  --evolved-versions v4 \
  --allow-heterogeneous \
  --simulations 2000 \
  --output-dir experiments/nash/heterogeneous_results/chain_reaction
```

**Step 3: Analyze equilibria**

Check if equilibria have:
- All agents same strategy (homogeneous)
- Mixed strategies with specialization (heterogeneous)

### Expected Outcomes

**Homogeneous equilibrium**:
- Nash selects one strategy for all agents
- No benefit to specialization
- **Conclusion**: Scenario doesn't incentivize diversity

**Heterogeneous equilibrium**:
- Nash assigns different strategies to agents
- Example: 3 Coordinators + 1 Free Rider
- **Conclusion**: Optimal teams have role diversity

### Deliverables

1. **Heterogeneous results**: `experiments/nash/HETEROGENEOUS_EQUILIBRIA.md`
   - Equilibria for each scenario
   - Comparison to homogeneous Nash
   - Implications for team composition

---

## Experiment 5: Robustness to Scenario Perturbations

**Goal**: Test how equilibria change when scenario parameters are perturbed.

**Priority**: 🟣 Low (Q3)

### Protocol

**Step 1: Define perturbations**

For each scenario parameter:
- Fire spread rate (β): ±10%, ±20%
- Work cost (c): ±10%, ±20%
- Rest benefit (r): ±10%, ±20%

**Step 2: Compute Nash for perturbed scenarios**

```bash
# Example: chain_reaction with +10% fire spread
uv run python experiments/scripts/compute_nash_v2.py \
  chain_reaction \
  --perturb fire_spread_rate +0.10 \
  --evolved-versions v4 \
  --output-dir experiments/nash/robustness/chain_reaction_fire+10
```

**Step 3: Measure sensitivity**

For each perturbation:
- Compare equilibrium strategies (same or different?)
- Compare expected payoffs (how much change?)
- Identify critical parameters (largest impact)

### Deliverables

1. **Robustness report**: `experiments/nash/ROBUSTNESS_ANALYSIS.md`
   - Sensitivity matrix (scenario × parameter)
   - Critical thresholds for equilibrium shifts
   - Recommendations for robust strategy design

---

## Implementation Checklist

### Scripts to Create

- [x] `experiments/scripts/compute_nash_v2.py` - Nash with evolved agents
- [ ] `experiments/scripts/compare_nash_versions.py` - V1 vs. V2 comparison
- [ ] `experiments/scripts/test_epsilon_equilibrium.py` - ε-equilibrium testing
- [ ] `experiments/scripts/summarize_nash_v2.py` - Cross-scenario summary
- [ ] `experiments/scripts/compute_nash_v2_heterogeneous.py` - Heterogeneous equilibria

### Analysis Scripts

- [ ] `experiments/scripts/visualize_strategy_space.py` - Plot strategies in behavior space
- [ ] `experiments/scripts/gap_analysis.py` - Compute gap closure metrics
- [ ] `experiments/scripts/sensitivity_analysis.py` - Parameter perturbation effects

### Documentation to Create

- [ ] `experiments/nash/GAP_ANALYSIS_chain_reaction.md` - Initial gap analysis (Exp 1)
- [ ] `experiments/nash/EPSILON_EQUILIBRIUM_chain_reaction.md` - ε-equilibrium results (Exp 2)
- [ ] `experiments/nash/V2_CROSS_SCENARIO_SUMMARY.md` - All scenarios (Exp 3)
- [ ] `experiments/nash/HETEROGENEOUS_EQUILIBRIA.md` - Diverse teams (Exp 4)
- [ ] `experiments/nash/ROBUSTNESS_ANALYSIS.md` - Parameter sensitivity (Exp 5)

---

## Timeline

### Q1 (Months 1-3)

**Week 1-2**: Experiment 1 (chain_reaction V2)
- Run Nash V2 with evolved agents
- Analyze gap closure
- Document findings

**Week 3-4**: Experiment 2 (epsilon-equilibrium)
- Test evolved strategies as ε-equilibria
- Quantify robustness
- Prepare for Q1 checkpoint

**Week 5-12**: Begin Experiment 3 (cross-scenario)
- Run evolution for missing scenarios (as needed)
- Start Nash V2 batch computation

### Q2 (Months 4-6)

**Week 1-4**: Complete Experiment 3
- Finish Nash V2 for all scenarios
- Cross-scenario analysis
- Scenario taxonomy

**Week 5-8**: Experiment 4 (heterogeneous)
- Implement heterogeneous Double Oracle
- Compute diverse-team equilibria
- Compare to homogeneous results

### Q3 (Months 7-9)

**Week 1-4**: Experiment 5 (robustness)
- Parameter perturbations
- Sensitivity analysis
- Critical threshold identification

**Week 5-8**: Synthesis and writing
- Integrate findings across experiments
- Draft paper: "From Archetypes to Evolution: Expanding Nash Equilibrium Strategy Spaces"
- Prepare final report

---

## Success Criteria

### Quantitative

✅ **Gap closure**: V2 payoff within 10% of evolved (chain_reaction)
✅ **Coverage**: Nash V2 computed for all 12 scenarios
✅ **Robustness**: ε-equilibrium bounds < 5 payoff units for evolved strategies
✅ **Generalization**: Consistent patterns across scenarios

### Qualitative

✅ **Understanding**: Clear explanation of V1 → V2 gap
✅ **Integration**: Evolved strategies successfully integrated into Double Oracle
✅ **Prediction**: Formal model relating scenario parameters to equilibrium types
✅ **Validation**: Cross-validation with evolution and MARL tracks

---

## Risk Mitigation

### Risk: Nash V2 doesn't close gap

**Likelihood**: Medium
**Impact**: High (challenges core hypothesis)

**Mitigation**:
1. Test multiple evolved versions (v3, v4, v5)
2. Try larger strategy pools (more archetypes)
3. Consider approximate equilibrium concepts (ε-Nash, correlated equilibrium)
4. Investigate: Are evolved strategies exploiting homogeneous assumption?

### Risk: Double Oracle doesn't converge with evolved agents

**Likelihood**: Low
**Impact**: High (can't compute V2)

**Mitigation**:
1. Increase simulation budget (2000 → 5000 games)
2. Reduce strategy pool size (fewer agents)
3. Use warm-start from V1 equilibrium
4. Debug: Check payoff evaluation consistency

### Risk: Evolution not complete for all scenarios

**Likelihood**: Medium (depends on compute availability)
**Impact**: Medium (delays cross-scenario analysis)

**Mitigation**:
1. Prioritize scenarios with largest V1 gaps
2. Use existing v3 agents if v4 unavailable
3. Parallelize evolution runs across scenarios
4. Accept partial coverage for initial analysis

---

## Resources

### Computational

- **CPU**: Moderate (10-50 core-hours per scenario)
- **GPU**: Not required (Double Oracle is CPU-bound)
- **Memory**: Low (<4GB per run)
- **Storage**: ~100MB per scenario (results + logs)

### Human

- **Primary researcher**: 20-30 hours/month
- **Code review**: 2-4 hours/month
- **Advisor meetings**: 1-2 hours/month

---

## References

### Internal

- **Phase 2 Agenda**: `docs/PHASE_2_RESEARCH_AGENDA.md`
- **Nash V1 Results**: `experiments/nash/README.md`
- **V2 Plan**: `experiments/nash/V2_PLAN.md`
- **Double Oracle Implementation**: `bucket_brigade/equilibrium/double_oracle.py`

### External

- McMahan et al. (2003): "Planning in the Presence of Cost Functions Controlled by an Adversary"
- Lanctot et al. (2017): "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning"
- Balduzzi et al. (2019): "Open-ended Learning in Symmetric Zero-sum Games"

---

**Status**: 🚀 Ready to Execute
**Owner**: Nash Track Lead
**Next Steps**: Begin Experiment 1 (chain_reaction V2)
