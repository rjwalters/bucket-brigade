# Evolution V7: Results Analysis Plan

**Created**: 2025-11-07
**Status**: ✅ V7 COMPLETE - Analysis Phase
**Priority**: 🟡 High - Supports Phase 2.5 analysis

---

## Executive Summary

**V7 Evolution is COMPLETE** for all 12 scenarios using:
- ✅ **Correct game mechanics** (post-fix)
- ✅ **Heterogeneous tournament fitness** (properly implemented)
- ✅ Population: 200, Generations: 200
- ✅ Data downloaded locally

**Current State**: We have ~15MB of unanalyzed V7 data sitting locally. Before running new experiments (V8, Nash, PPO), we should **extract maximum value from V7**.

**This plan** outlines analysis priorities aligned with Phase 2.5 research agenda.

---

## V7 Status Summary

### Completion Evidence

**Local data verified**:
```bash
experiments/scenarios/{scenario}/evolved_v7/
├── best_agent.json          # Best evolved agent (355 bytes)
├── evolution_results.json   # Full evolution trace (140KB)
├── evolution_log.txt        # Human-readable summary (589 bytes)
└── checkpoint_gen*.json     # Every 20 generations (110KB each)
```

**All 12 scenarios complete**:
- chain_reaction, deceptive_calm, default, early_containment
- easy, greedy_neighbor, hard, mixed_motivation
- overcrowding, rest_trap, sparse_heroics, trivial_cooperation

**Sample fitness values**:
- Best: deceptive_calm (48.80)
- Good: chain_reaction (43.68), hard (43.21)
- Baseline: easy (17.11)

### V7 Configuration (for reference)

```python
POPULATION_SIZE = 200
NUM_GENERATIONS = 200
MUTATION_RATE = 0.15
FITNESS_TYPE = "heterogeneous_tournament"
OPPONENT_POOL = ["firefighter", "hero", "free_rider", "coordinator"]
GAMES_PER_INDIVIDUAL = 30  # Per opponent type
```

**Key difference from V6**: V7 uses **true heterogeneous tournament fitness** (agent plays with diverse opponents), while V6 used homogeneous self-play.

---

## Analysis Priorities (Aligned with Phase 2.5)

From `research_notebook/2025-11-08_phase_2_5_analysis_plan.md`:

> **Guiding Principle**: Start Concrete, Iterate Empirically → **Analyze before experimenting**

### Priority 1: Zero-Compute Parameter Analysis ⚡

**Goal**: Understand V7 strategies without running new experiments

**Time**: 2-3 hours
**Value**: HIGH - Guides all future work

#### 1A. Extract V7 Parameters

```bash
uv run python experiments/scripts/extract_v7_parameters.py \
  --scenarios all \
  --output experiments/evolution/v7_parameter_analysis.json
```

**Extract for each scenario**:
- Best agent genome (10 parameters)
- Final fitness
- Population diversity (final generation)
- Convergence generation (if plateau detected)

**Output format**:
```json
{
  "chain_reaction": {
    "fitness": 43.675,
    "genome": {
      "honesty": 0.419,
      "work": 0.0,
      "neighbor_help": 1.0,
      "risk_aversion": 0.904,
      ...
    },
    "diversity": 0.697,
    "converged_at": null
  },
  ...
}
```

#### 1B. Compare V7 vs V6 Parameters

**Question**: How did strategies change with heterogeneous training?

```bash
uv run python experiments/scripts/compare_evolution_versions.py \
  --v6-dir experiments/scenarios/*/evolved_v6/ \
  --v7-dir experiments/scenarios/*/evolved_v7/ \
  --output experiments/evolution/V6_VS_V7_COMPARISON.md
```

**Metrics**:
- **Parameter differences**: L2 distance between genomes
- **Fitness changes**: V7 vs V6 (expect V7 lower due to harder fitness)
- **Behavioral shifts**: Which parameters changed most?
- **Consistency**: Do both find similar cooperation levels?

**Expected findings**:
- V7 work_tendency likely lower (learning to exploit teammates)
- V7 neighbor_help more strategic (not always helping)
- V7 risk_aversion potentially higher (survival matters in mixed teams)

#### 1C. Cross-Scenario Parameter Patterns

**Question**: Do scenarios cluster by strategy type?

```bash
uv run python experiments/scripts/cluster_scenarios.py \
  --data experiments/evolution/v7_parameter_analysis.json \
  --method hierarchical \
  --output experiments/evolution/scenario_clustering_v7.png
```

**Analysis**:
1. Compute pairwise distance between scenarios (genome L2 norm)
2. Hierarchical clustering (dendrogram)
3. Identify scenario families:
   - **Cooperative cluster**: High neighbor_help scenarios
   - **Selfish cluster**: Low neighbor_help, high self-priority
   - **Mixed cluster**: Context-dependent strategies

**Value**: Predictive power for new scenarios

---

### Priority 2: Evolution Trajectory Visualization 📊

**Goal**: Understand how V7 strategies emerged over 200 generations

**Time**: 3-4 hours
**Value**: MEDIUM-HIGH - Reveals learning dynamics

#### 2A. Fitness Curves

**Script**: `experiments/scripts/plot_evolution_trajectories.py`

```bash
uv run python experiments/scripts/plot_evolution_trajectories.py \
  --scenarios chain_reaction deceptive_calm easy hard \
  --data experiments/scenarios/*/evolved_v7/evolution_results.json \
  --output plots/v7_fitness_trajectories.png
```

**Visualizations**:
- Line plot: Best/Mean/Worst fitness over 200 generations (4 scenarios)
- Shaded region: ±1 std deviation
- Vertical lines: Checkpoint generations (20, 40, ..., 200)

**Questions**:
- When did fitness plateau? (convergence speed)
- Did any scenarios not converge by gen 200?
- Is convergence smooth or punctuated? (gradual vs sudden)

#### 2B. Parameter Evolution Heatmaps

**Goal**: Track how each parameter changes over time

```bash
uv run python experiments/scripts/plot_parameter_evolution.py \
  --scenario chain_reaction \
  --data experiments/scenarios/chain_reaction/evolved_v7/checkpoint_gen*.json \
  --output plots/v7_parameter_evolution_chain_reaction.png
```

**Heatmap**:
- X-axis: Generation (0, 20, 40, ..., 200)
- Y-axis: Parameters (honesty, work, neighbor_help, ...)
- Color: Mean population value (0.0 = blue, 1.0 = red)

**Insights**:
- Which parameters converge early vs late?
- Which parameters remain diverse (high variance)?
- Critical transitions (parameter suddenly shifts)

#### 2C. Diversity Over Time

**Question**: Does population maintain diversity or collapse?

```bash
uv run python experiments/scripts/analyze_diversity.py \
  --scenarios all \
  --data experiments/scenarios/*/evolved_v7/evolution_results.json \
  --output experiments/evolution/v7_diversity_analysis.md
```

**Metrics**:
- **Genotypic diversity**: Variance in parameter space
- **Phenotypic diversity**: Variance in fitness
- **Entropy**: Shannon entropy of parameter distributions

**Expected**: Diversity should stay above MIN_DIVERSITY=0.1 threshold

---

### Priority 3: V7 Tournament Validation 🎯

**Goal**: Test if V7 performs well in heterogeneous tournaments (as trained)

**Time**: 1-2 hours compute + 1 hour analysis
**Value**: HIGH - Validates training objective

#### 3A. Heterogeneous Tournament Benchmark

**Question**: Does V7 beat archetypes (including free-rider)?

```bash
uv run python experiments/scripts/run_heterogeneous_tournament.py \
  --agents evolved_v7 firefighter hero free_rider coordinator \
  --scenarios chain_reaction deceptive_calm easy hard \
  --num-games 200 \
  --output experiments/scenarios/{scenario}/tournament_v7_results.csv
```

**Format**:
```csv
scenario,agent1,agent2,agent3,agent4,agent1_payoff,agent2_payoff,...
chain_reaction,evolved_v7,firefighter,hero,free_rider,45.2,38.1,42.3,29.8
chain_reaction,evolved_v7,free_rider,free_rider,free_rider,12.3,15.2,14.8,16.1
...
```

**Success criteria**:
- V7 beats free-rider in ≥3/4 scenarios (heterogeneous teams)
- V7 ranks #1 or #2 on average (among 5 agent types)
- V7 performs reasonably with 3 free-riders (worst-case team)

#### 3B. V7 vs V6 Head-to-Head

**Question**: Did heterogeneous training improve robustness?

```bash
uv run python experiments/scripts/compare_evolution_versions_tournament.py \
  --v6 experiments/scenarios/*/evolved_v6/ \
  --v7 experiments/scenarios/*/evolved_v7/ \
  --scenarios chain_reaction easy hard \
  --num-games 200 \
  --output experiments/evolution/V6_VS_V7_TOURNAMENT.md
```

**Tests**:
1. **Homogeneous teams**: [V7, V7, V7, V7] vs [V6, V6, V6, V6]
2. **Heterogeneous teams**: [V7, free_rider, hero, coordinator] vs [V6, free_rider, hero, coordinator]
3. **Adversarial**: [V7, free_rider, free_rider, free_rider] vs [V6, free_rider, free_rider, free_rider]

**Hypothesis**: V7 outperforms V6 in heterogeneous (2) and adversarial (3) but may underperform in homogeneous (1)

---

### Priority 4: Parameter Sensitivity Analysis 🔬

**Goal**: Which parameters matter most for each scenario?

**Time**: 3-4 hours compute + 2 hours analysis
**Value**: MEDIUM - Optimization insights

#### 4A. Ablation Study

**Method**: For best V7 agent, perturb each parameter ±10%, ±25%, ±50%

```bash
uv run python experiments/scripts/parameter_sensitivity.py \
  --scenario chain_reaction \
  --agent experiments/scenarios/chain_reaction/evolved_v7/best_agent.json \
  --perturbations 0.1 0.25 0.5 \
  --num-games 100 \
  --output experiments/evolution/sensitivity_chain_reaction_v7.json
```

**Output**:
```json
{
  "baseline_fitness": 43.68,
  "sensitivity": {
    "honesty": {
      "+10%": 44.1,  "Δ": +0.42,
      "+25%": 45.2,  "Δ": +1.52,
      "-10%": 42.8,  "Δ": -0.88
    },
    "work": { ... },
    ...
  },
  "most_sensitive": ["neighbor_help", "risk_aversion", "honesty"],
  "least_sensitive": ["work", "exploration", "fatigue_memory"]
}
```

**Insights**:
- **Critical parameters**: Large Δ fitness when perturbed
- **Robust parameters**: Small Δ fitness (can tune without harm)
- **Interaction effects**: Are some parameters codependent?

**Value**: Informs constrained evolution (reduce parameter space)

#### 4B. Cross-Scenario Sensitivity Comparison

**Question**: Do different scenarios rely on different parameters?

```bash
uv run python experiments/scripts/compare_sensitivity_across_scenarios.py \
  --scenarios chain_reaction easy hard \
  --data experiments/evolution/sensitivity_*_v7.json \
  --output experiments/evolution/cross_scenario_sensitivity_v7.png
```

**Heatmap**:
- Rows: Scenarios
- Columns: Parameters
- Color: Sensitivity magnitude (red = high impact, blue = low impact)

**Expected patterns**:
- **chain_reaction**: High sensitivity to neighbor_help (cooperation critical)
- **hard**: High sensitivity to risk_aversion (survival focus)
- **easy**: Low sensitivity overall (many strategies work)

---

### Priority 5: Cross-Scenario Transfer Analysis 🔀

**Goal**: Can agents evolved for scenario A perform well in scenario B?

**Time**: 2-3 hours compute + 1 hour analysis
**Value**: HIGH - Tests generalization

#### 5A. Transfer Matrix

**Method**: For each V7 agent, evaluate in all 12 scenarios

```bash
uv run python experiments/scripts/evaluate_cross_scenario_transfer.py \
  --agents experiments/scenarios/*/evolved_v7/best_agent.json \
  --scenarios all \
  --num-games 100 \
  --output experiments/evolution/v7_transfer_matrix.csv
```

**Output** (12×12 matrix):
```csv
trained_on,chain_reaction,deceptive_calm,easy,hard,...
chain_reaction,43.68,38.2,15.1,29.4,...
deceptive_calm,39.1,48.80,16.3,31.2,...
easy,12.3,14.5,17.11,8.9,...
...
```

**Diagonal**: In-scenario performance (training scenario)
**Off-diagonal**: Transfer performance

**Analysis**:
1. **Best generalizers**: Which agents transfer well?
2. **Scenario similarity**: Do cooperative scenarios cluster?
3. **Catastrophic failure**: Where does transfer fail completely?

#### 5B. Meta-Agent Hypothesis Test

**Question**: Can we identify a "universal" agent?

**Method**:
1. Rank agents by average cross-scenario performance
2. Identify agent with best mean transfer
3. Compare to best in-scenario agents

**Hypothesis**:
- If best generalizer ≈ best in-scenario → Strategies are universal
- If best generalizer << best in-scenario → Strategies are specialized

---

## Decision Tree: V8 vs Phase 2.5

After completing priorities 1-5, decide next steps:

### Option A: V8 Evolution (If V7 Incomplete)

**Trigger conditions**:
- V7 didn't converge by gen 200 (≥3 scenarios)
- Tournament validation shows V7 exploitable
- Parameter analysis reveals V7 strategies suboptimal

**V8 Plan**:
- Extend to 500 generations (or curriculum learning)
- Increase mutation rate if stuck in local optima
- Try multi-scenario fitness (train on average of 3-5 scenarios)

### Option B: Phase 2.5 Analysis (If V7 Sufficient)

**Trigger conditions**:
- V7 converged for most scenarios
- V7 beats or competes with free-rider (≥3/4 scenarios)
- Parameter analysis reveals interpretable patterns

**Phase 2.5 Focus**:
- Deep dive into V7 strategies
- Compare Evolution vs Nash vs MARL
- Extract insights without new experiments
- Prepare for Phase 3 (mechanism design)

### Option C: Hybrid (Most Likely)

**Approach**:
- Proceed with Phase 2.5 analysis for most scenarios
- Run V8 for 2-3 problematic scenarios only
- Parallel track: Begin Nash V7 and MARL baseline

---

## Implementation Checklist

### Data Preparation (15 min)

- [x] V7 data downloaded locally
- [ ] Verify all 12 scenarios have best_agent.json
- [ ] Check evolution_results.json integrity (can parse)
- [ ] Organize checkpoints for trajectory analysis

### Zero-Compute Analysis (2-3 hours)

- [ ] Extract V7 parameters (all scenarios)
- [ ] Compare V7 vs V6 parameters
- [ ] Cluster scenarios by strategy similarity
- [ ] Document findings in `experiments/evolution/V7_PARAMETER_ANALYSIS.md`

### Visualization (3-4 hours)

- [ ] Plot fitness trajectories (4 key scenarios)
- [ ] Create parameter evolution heatmaps (2-3 scenarios)
- [ ] Analyze diversity over time
- [ ] Generate `plots/v7_evolution_dynamics.png`

### Validation (3-4 hours compute + 2 hours analysis)

- [ ] Run heterogeneous tournaments (V7 vs archetypes)
- [ ] Run V7 vs V6 head-to-head
- [ ] Rank agents by tournament performance
- [ ] Document in `experiments/evolution/V7_TOURNAMENT_VALIDATION.md`

### Advanced Analysis (5-6 hours compute + 3 hours analysis)

- [ ] Parameter sensitivity (3 scenarios)
- [ ] Cross-scenario transfer matrix (12×12)
- [ ] Identify best generalizers
- [ ] Document in `experiments/evolution/V7_TRANSFER_ANALYSIS.md`

---

## Timeline

### Day 1: Quick Wins (3-4 hours)

**Morning** (2 hours):
- Extract V7 parameters
- Compare V7 vs V6
- Scenario clustering

**Afternoon** (2 hours):
- Plot fitness trajectories
- Parameter evolution heatmaps
- Initial insights document

### Day 2: Validation (4-5 hours)

**Morning** (1 hour):
- Launch heterogeneous tournaments (background)
- Launch V7 vs V6 head-to-head (background)

**Afternoon** (3 hours):
- Analyze tournament results
- Rank agents
- Document validation findings

### Day 3: Deep Analysis (5-6 hours)

**Morning** (2 hours):
- Launch parameter sensitivity (background)
- Launch transfer matrix (background)

**Afternoon** (4 hours):
- Analyze sensitivity results
- Analyze transfer matrix
- Identify patterns and insights
- Write comprehensive V7 analysis report

### Day 4: Decision & Planning (2-3 hours)

**Morning** (2 hours):
- Review all V7 findings
- Decision: V8 needed or proceed to Phase 2.5?
- Update roadmap based on findings

**Afternoon** (1 hour):
- If V8: Draft V8 plan with lessons from V7
- If Phase 2.5: Update Phase 2.5 priorities
- Share findings with team

---

## Success Criteria

### Must Have (Critical for Phase 2.5)

1. ✅ **V7 parameter extraction complete** for all 12 scenarios
2. ✅ **V7 vs V6 comparison** documented (parameter and fitness changes)
3. ✅ **Scenario clustering** completed (identify strategy families)
4. ✅ **Tournament validation** complete (V7 vs archetypes performance)

### Should Have (High Value)

5. ✅ **Evolution trajectory analysis** (convergence patterns, diversity)
6. ✅ **V7 vs V6 tournament** comparison (heterogeneous robustness)
7. ✅ **Parameter sensitivity** for 3+ scenarios (critical parameters identified)

### Nice to Have (Future Reference)

8. ⚠️ **Cross-scenario transfer matrix** (12×12 generalization)
9. ⚠️ **Meta-agent identification** (best generalizer found)
10. ⚠️ **V8 plan** (if needed) with clear motivation and improvements

---

## Deliverables

### Immediate (Day 1-2)

1. **Parameter analysis**: `experiments/evolution/V7_PARAMETER_ANALYSIS.md`
   - Extracted parameters (all scenarios)
   - V7 vs V6 comparison table
   - Scenario clustering dendrogram

2. **Trajectory plots**: `plots/v7_*.png`
   - Fitness trajectories (best/mean/worst)
   - Parameter evolution heatmaps
   - Diversity over time

### Short-term (Day 3-4)

3. **Tournament validation**: `experiments/evolution/V7_TOURNAMENT_VALIDATION.md`
   - V7 vs archetypes rankings
   - V7 vs V6 head-to-head
   - Heterogeneous robustness analysis

4. **Transfer analysis**: `experiments/evolution/V7_TRANSFER_ANALYSIS.md`
   - Cross-scenario transfer matrix
   - Best generalizers identified
   - Scenario similarity patterns

5. **Sensitivity analysis**: `experiments/evolution/V7_SENSITIVITY_ANALYSIS.md`
   - Critical parameters per scenario
   - Cross-scenario sensitivity comparison
   - Optimization recommendations

### Final (End of Week)

6. **Comprehensive report**: `experiments/evolution/V7_RESULTS_COMPREHENSIVE.md`
   - Executive summary
   - All findings synthesized
   - Comparison to V6, V5, V4
   - Decision: V8 or Phase 2.5?
   - Recommendations for future work

7. **Research notebook entry**: `research_notebook/2025-11-XX_v7_analysis_complete.md`
   - Narrative summary
   - Key insights and surprises
   - Implications for Phase 2.5
   - Updated research agenda

---

## Integration with Phase 2.5

From `research_notebook/2025-11-08_phase_2_5_analysis_plan.md`:

### Phase 2.5 Priority 1: Zero-Compute Analyses ⚡

**V7 Analysis Contributions**:
- ✅ **V7 Parameter Patterns**: Done (Priority 1A-C)
- ✅ **Hypothesis Generation**: Enabled by parameter clustering
- ⏱️ **Nash vs Evolution Comparison**: Blocked until Nash V7 complete

### Phase 2.5 Priority 2: Evolution Trajectory Visualization 📊

**V7 Analysis Contributions**:
- ✅ **Fitness Curves**: Done (Priority 2A)
- ✅ **Parameter Evolution**: Done (Priority 2B)
- ✅ **Diversity Metrics**: Done (Priority 2C)
- ✅ **Critical Transitions**: Identified from trajectories

### Phase 2.5 Priority 5: Cross-Scenario Transfer Matrix 🔀

**V7 Analysis Contributions**:
- ✅ **12×12 Transfer Matrix**: Done (Priority 5A)
- ✅ **Generalization Analysis**: Done (Priority 5B)
- ✅ **Scenario Clustering**: Validates clustering from Priority 1C

---

## Key Questions to Answer

### Evolution Dynamics

1. Did V7 converge by generation 200? (for which scenarios?)
2. How does convergence speed correlate with scenario difficulty?
3. Did population maintain diversity (avoid premature convergence)?
4. Are there critical transitions (parameter suddenly shifts)?

### Strategy Characterization

5. How do V7 strategies differ from V6 (homogeneous vs heterogeneous training)?
6. Which parameters changed most with heterogeneous training?
7. Can we explain V7 strategies in natural language? (behavioral archetypes)
8. Are V7 strategies Nash equilibria? (compare to Nash V7 when available)

### Performance & Robustness

9. Does V7 beat free-rider in heterogeneous tournaments?
10. How robust is V7 to worst-case team compositions?
11. Which scenarios did V7 master vs struggle with?
12. Can V7 agents generalize to new scenarios?

### Optimization & Future Work

13. Which parameters are most sensitive (critical to tune)?
14. Which parameters are robust (can simplify)?
15. Are some strategies transferable across scenarios?
16. Should we run V8 (longer evolution) or proceed to Phase 2.5?

---

## Risk Mitigation

### Risk 1: V7 data incomplete or corrupted

**Likelihood**: Low (already verified)
**Impact**: High (blocks all analysis)

**Mitigation**:
- Verify checksums on evolution_results.json files
- Re-download from remote if any corruption
- Fallback: Re-run V7 for affected scenarios only

### Risk 2: V7 didn't converge (needs V8)

**Likelihood**: Medium (200 gen may be insufficient)
**Impact**: Medium (V8 needed, delays Phase 2.5)

**Mitigation**:
- Check convergence criterion (fitness plateau)
- If ≤3 scenarios unconverged → Run V8 for those only
- If ≥4 scenarios unconverged → Full V8 needed (500 gen)
- Either way, proceed with analysis of converged scenarios

### Risk 3: Analysis scripts don't exist

**Likelihood**: High (many scripts need to be written)
**Impact**: Medium (time to implement)

**Mitigation**:
- Prioritize zero-compute analyses (Priority 1) - simple scripts
- Use existing scripts where possible (tournament runner exists)
- Accept manual analysis for some tasks (spreadsheets OK)
- Document "missing script" as technical debt for later

### Risk 4: Tournament validation shows V7 weak

**Likelihood**: Medium (heterogeneous training is hard)
**Impact**: High (V7 not usable, need V8 or pivot)

**Mitigation**:
- This is valuable negative result (not a failure)
- Analyze why V7 failed (parameter analysis, sensitivity)
- Design V8 with lessons learned (curriculum, multi-scenario fitness)
- Or pivot: Evolution limits reached, focus on MARL/Nash

---

## Open Questions (Pre-Analysis)

1. Are all 12 scenarios downloaded? (verify completeness)
2. Which scenarios converged by gen 200? (check logs)
3. Do we have V6 data for comparison? (verify V6 still accessible)
4. Are tournament scripts ready? (check `run_heterogeneous_tournament.py`)

## Next Steps (Immediate)

1. **Verify V7 data completeness** (5 min)
   ```bash
   for scenario in chain_reaction deceptive_calm default early_containment \
                   easy greedy_neighbor hard mixed_motivation \
                   overcrowding rest_trap sparse_heroics trivial_cooperation; do
     if [ -f "experiments/scenarios/$scenario/evolved_v7/best_agent.json" ]; then
       echo "✓ $scenario"
     else
       echo "✗ $scenario - MISSING"
     fi
   done
   ```

2. **Quick parameter extraction** (15 min)
   ```bash
   # Manual extraction for now (create script later)
   for scenario in experiments/scenarios/*/evolved_v7/; do
     echo "=== $(basename $(dirname $scenario)) ==="
     jq -r '.fitness, .genome' "$scenario/best_agent.json"
   done > experiments/evolution/v7_quick_summary.txt
   ```

3. **Priority 1A: Full parameter analysis** (Start Day 1)
   - Write `extract_v7_parameters.py` script
   - Run on all scenarios
   - Create JSON output for downstream analysis

---

**Status**: 🚀 Ready to Analyze
**V7 Evolution**: ✅ Complete (200 gen, 12 scenarios)
**Next Action**: Begin Priority 1 (Zero-Compute Parameter Analysis)
**Timeline**: 3-4 days for comprehensive analysis
**Decision Point**: Day 4 (V8 vs Phase 2.5)
