# Evolution Research: Experimental Design & Roadmap

**Phase**: 2 (Understanding Agent Effectiveness)
**Track**: Genetic Algorithm / Heuristic Search
**Goal**: Complete scenario-wide understanding of evolved strategies

---

## Overview

This document provides a step-by-step experimental protocol for Phase 2 evolution research. Phase 1 demonstrated near-Nash performance (58.50 vs. 57.87) on chain_reaction. Phase 2 expands to all scenarios and analyzes **what makes strategies effective**.

### Core Questions

1. **Do evolved strategies generalize across scenarios?**
2. **What parameters matter most for each scenario type?**
3. **How does the fitness landscape vary?**

---

## Experiment 1: Complete Scenario Sweep

**Goal**: Run V4 evolution on all 9 remaining scenarios.

**Priority**: 🔴 Critical (Q1-Q2)

### Current Status

**Completed** (V4 evolution):
- ✅ chain_reaction (58.50 payoff)
- ✅ deceptive_calm (tested, results TBD)

**Pending** (9 scenarios):
- greedy_neighbor
- mixed_motivation
- easy
- overcrowding
- rest_trap
- sparse_heroics
- trivial_cooperation
- early_containment
- default

### Protocol

**Step 1: Prioritize scenarios**

Group by strategic characteristics:

**Tier 1 - High Priority** (different strategic profiles):
```
greedy_neighbor      # High work cost (free-riding incentive)
mixed_motivation     # Mixed incentives
easy                 # Simple cooperation (baseline)
```

**Tier 2 - Medium Priority** (coverage):
```
overcrowding         # Resource constraints
rest_trap            # Deceptive incentives
sparse_heroics       # Low agent density
```

**Tier 3 - Completion** (full coverage):
```
trivial_cooperation  # Simplest case
early_containment    # Early intervention
default              # Standard parameters
```

**Step 2: Launch evolution runs**

For each scenario, run V4 configuration (proven successful):

```bash
# Tier 1 - Launch in parallel
for scenario in greedy_neighbor mixed_motivation easy; do
  echo "Starting $scenario evolution..."
  uv run python experiments/scripts/run_evolution.py \
    --scenario $scenario \
    --population 200 \
    --generations 15000 \
    --games-per-eval 50 \
    --output experiments/scenarios/$scenario/evolved_v4 \
    --seed 42 \
    2>&1 | tee logs/evolution_${scenario}_v4.log &
done

# Wait for Tier 1 to complete before Tier 2
wait
```

**Parameters** (V4 configuration):
- Population: 200
- Generations: 15,000
- Games per evaluation: 50
- Seed: 42 (reproducibility)
- Selection: Tournament selection
- Mutation: Gaussian (σ=0.1)
- Elite preservation: 10%

**Expected runtime per scenario**: 6-10 hours (64 CPUs)

**Step 3: Monitor progress**

Check convergence during runs:
```bash
# Watch fitness evolution
tail -f logs/evolution_greedy_neighbor_v4.log | grep "Generation"

# Plot convergence curve
uv run python experiments/scripts/plot_evolution_progress.py \
  experiments/scenarios/greedy_neighbor/evolved_v4/evolution_logs/ \
  --output plots/greedy_neighbor_v4_convergence.png
```

**Step 4: Validate results**

After each run completes:

```bash
scenario="greedy_neighbor"

# Extract best agent
best_genome=$(python -c "import json; print(json.load(open('experiments/scenarios/$scenario/evolved_v4/best_agent.json'))['genome'])")

# Run verification tournament (Rust evaluator)
uv run python experiments/scripts/run_comparison.py $scenario \
  --agents evolved_v4 \
  --num-games 200 \
  --output experiments/scenarios/$scenario/evolved_v4/validation.json

echo "Expected fitness: $(cat experiments/scenarios/$scenario/evolved_v4/best_agent.json | jq '.fitness')"
echo "Validation payoff: $(cat experiments/scenarios/$scenario/evolved_v4/validation.json | jq '.evolved_v4_payoff')"
```

**Success criterion**: Validation payoff within 5% of training fitness (confirms train/test consistency).

### Expected Outcomes

**Convergence patterns**:
- Most scenarios: Converge by 5000-10,000 generations
- Difficult scenarios: May require full 15,000 generations
- Easy scenarios: May converge earlier (could use V3 config)

**Performance vs. Nash**:
- Simple scenarios (easy, trivial_cooperation): Expect near-Nash (within 5%)
- Complex scenarios (greedy_neighbor, mixed_motivation): May exceed Nash if V1 had limited strategy space

### Deliverables

For each scenario:
1. **Best agent**: `experiments/scenarios/{scenario}/evolved_v4/best_agent.json`
2. **Evolution logs**: `experiments/scenarios/{scenario}/evolved_v4/evolution_logs/`
3. **Validation results**: `experiments/scenarios/{scenario}/evolved_v4/validation.json`
4. **Convergence plot**: `plots/{scenario}_v4_convergence.png`

---

## Experiment 2: Parameter Pattern Analysis

**Goal**: Identify which genome parameters matter for each scenario type.

**Priority**: 🟡 High (Q2)

### Background

10-parameter genome:
```python
genome = [
    work_tendency,         # Baseline work probability
    signal_honesty,        # Truthfulness in signaling
    coordination_weight,   # Value placed on team cooperation
    own_house_priority,    # Bias toward protecting own house
    risk_aversion,         # Conservative vs. aggressive
    signal_response,       # Trust in others' signals
    effort_cost_sensitivity,  # Work avoidance
    neighbor_priority,     # Help neighbors vs. distant houses
    rest_bias,             # Tendency to rest
    exploration_rate,      # Randomness in action selection
]
```

**Question**: Which parameters drive performance in different scenarios?

### Protocol

**Step 1: Extract genomes from all scenarios**

```bash
uv run python experiments/scripts/extract_all_genomes.py \
  --scenarios chain_reaction deceptive_calm greedy_neighbor mixed_motivation easy \
              overcrowding rest_trap sparse_heroics trivial_cooperation \
  --version v4 \
  --output experiments/evolution/all_genomes_v4.csv
```

Output format:
```csv
scenario,work_tendency,signal_honesty,coordination_weight,...,fitness
chain_reaction,0.75,0.82,0.91,...,58.50
greedy_neighbor,0.65,0.77,0.88,...,62.30
...
```

**Step 2: Visualize parameter distributions**

```bash
uv run python experiments/scripts/visualize_genome_patterns.py \
  experiments/evolution/all_genomes_v4.csv \
  --output plots/genome_heatmap_v4.png
```

**Analysis**:
- Heatmap: scenario × parameter values
- Clustering: Group scenarios by parameter similarity
- Correlation: Parameter values vs. fitness

**Step 3: Sensitivity analysis**

For each scenario, perturb each parameter individually:

```bash
uv run python experiments/scripts/parameter_sensitivity.py \
  chain_reaction \
  --genome experiments/scenarios/chain_reaction/evolved_v4/best_agent.json \
  --perturbation 0.1 \
  --num-games 100 \
  --output experiments/evolution/sensitivity_chain_reaction.json
```

**Output**:
```json
{
  "work_tendency": {
    "baseline": 58.50,
    "perturbed_+10%": 56.20,  # -2.30 payoff
    "perturbed_-10%": 52.10,  # -6.40 payoff
    "sensitivity": 6.40       # Max impact
  },
  "signal_honesty": {
    "baseline": 58.50,
    "perturbed_+10%": 58.00,  # -0.50 payoff
    "perturbed_-10%": 57.80,  # -0.70 payoff
    "sensitivity": 0.70       # Low impact
  },
  ...
}
```

**Step 4: Identify critical parameters**

Rank parameters by sensitivity (across all scenarios):

```bash
uv run python experiments/scripts/rank_parameter_importance.py \
  experiments/evolution/sensitivity_*.json \
  --output experiments/evolution/PARAMETER_IMPORTANCE.md
```

### Expected Outcomes

**Universal parameters** (high sensitivity across all scenarios):
- `work_tendency`: Fundamental to cooperation
- `coordination_weight`: Team collaboration
- `own_house_priority`: Self-interest vs. altruism

**Scenario-specific parameters**:
- `greedy_neighbor`: `effort_cost_sensitivity` (high work cost)
- `rest_trap`: `risk_aversion` (deceptive fires)
- `sparse_heroics`: `neighbor_priority` (spatial constraints)

**Negligible parameters** (low sensitivity):
- `exploration_rate`: May not matter after convergence
- Parameters that evolved to extremes (0 or 1) across all scenarios

### Deliverables

1. **Genome database**: `experiments/evolution/all_genomes_v4.csv`
2. **Sensitivity analysis**: `experiments/evolution/sensitivity_{scenario}.json` (per scenario)
3. **Parameter importance ranking**: `experiments/evolution/PARAMETER_IMPORTANCE.md`
4. **Visualizations**:
   - Heatmap: `plots/genome_heatmap_v4.png`
   - Clustering dendrogram: `plots/scenario_clustering_v4.png`
   - Sensitivity bar chart: `plots/parameter_sensitivity_v4.png`

---

## Experiment 3: Cross-Scenario Transfer

**Goal**: Test if strategies evolved on one scenario work on others.

**Priority**: 🟡 High (Q1-Q2)

### Protocol

**Step 1: Define transfer matrix**

For N scenarios, create N×N matrix:
- Rows: Training scenario
- Columns: Test scenario
- Cells: Performance (payoff)

**Diagonal** (train=test): Expected high performance
**Off-diagonal** (train≠test): Test generalization

**Step 2: Run transfer experiments**

```bash
# Example: Train on chain_reaction, test on greedy_neighbor
uv run python experiments/scripts/transfer_test.py \
  --train-scenario chain_reaction \
  --test-scenario greedy_neighbor \
  --agent experiments/scenarios/chain_reaction/evolved_v4/best_agent.json \
  --num-games 200 \
  --output experiments/evolution/transfer_chainReaction_to_greedyNeighbor.json
```

**Batch all combinations**:

```bash
cat > experiments/evolution/run_all_transfers.sh <<'EOF'
#!/bin/bash
scenarios="chain_reaction greedy_neighbor mixed_motivation easy trivial_cooperation"

for train_scenario in $scenarios; do
  for test_scenario in $scenarios; do
    echo "Transfer: $train_scenario -> $test_scenario"
    uv run python experiments/scripts/transfer_test.py \
      --train-scenario $train_scenario \
      --test-scenario $test_scenario \
      --agent experiments/scenarios/$train_scenario/evolved_v4/best_agent.json \
      --num-games 200 \
      --output experiments/evolution/transfer_${train_scenario}_to_${test_scenario}.json
  done
done
EOF

chmod +x experiments/evolution/run_all_transfers.sh
./experiments/evolution/run_all_transfers.sh
```

**Step 3: Analyze transfer matrix**

```bash
uv run python experiments/scripts/analyze_transfer_matrix.py \
  experiments/evolution/transfer_*.json \
  --output experiments/evolution/TRANSFER_ANALYSIS.md
```

**Metrics**:
- **Transfer efficiency**: `payoff(train→test) / payoff(test→test)`
- **Generalization**: Average off-diagonal performance
- **Specialization**: Diagonal - off-diagonal (higher = more specialized)

**Example matrix**:
```
               chain_reaction  greedy_neighbor  mixed_motivation  easy
chain_reaction      58.50           45.20            52.10       70.30
greedy_neighbor     40.10           62.30            55.80       68.90
mixed_motivation    43.20           50.50            60.20       72.10
easy                35.60           38.20            44.50       81.90
```

**Interpretation**:
- `chain_reaction → chain_reaction` (58.50): Best on own scenario
- `chain_reaction → easy` (70.30): Transfers well to simpler scenario
- `easy → chain_reaction` (35.60): Simple strategy fails on complex scenario

### Expected Outcomes

**Scenario clusters with high transfer**:
- Within-cluster: >80% transfer efficiency
- Cross-cluster: <60% transfer efficiency
- Example clusters: {easy, trivial_cooperation}, {chain_reaction, greedy_neighbor}

**Universal strategies**:
- Scenarios with high average column performance (test)
- Indicates scenario is "easy" or has dominant strategy

**Specialized strategies**:
- High diagonal, low off-diagonal
- Indicates scenario-specific adaptation

### Deliverables

1. **Transfer results**: `experiments/evolution/transfer_{train}_to_{test}.json` (all pairs)
2. **Transfer matrix**: `experiments/evolution/transfer_matrix.csv`
3. **Analysis report**: `experiments/evolution/TRANSFER_ANALYSIS.md`
4. **Visualizations**:
   - Heatmap: `plots/transfer_matrix_heatmap.png`
   - Clustering: `plots/scenario_clusters_by_transfer.png`

---

## Experiment 4: Fitness Landscape Characterization

**Goal**: Understand optimization difficulty for each scenario.

**Priority**: 🟢 Medium (Q2-Q3)

### Protocol

**Step 1: Compare convergence rates**

```bash
uv run python experiments/scripts/compare_convergence.py \
  --scenarios chain_reaction greedy_neighbor mixed_motivation easy \
  --version v4 \
  --output experiments/evolution/CONVERGENCE_COMPARISON.md
```

**Metrics**:
- **Generations to 90% of final fitness**: Speed of convergence
- **Fitness variance over last 1000 generations**: Stability
- **Plateau detection**: Long periods without improvement

**Step 2: Local optima analysis**

Run multiple evolution runs with different seeds:

```bash
# Run 5 replications for greedy_neighbor
for seed in 42 43 44 45 46; do
  uv run python experiments/scripts/run_evolution.py \
    --scenario greedy_neighbor \
    --population 200 \
    --generations 5000 \
    --games-per-eval 50 \
    --output experiments/scenarios/greedy_neighbor/evolved_v4_seed$seed \
    --seed $seed
done
```

**Analysis**:
```bash
uv run python experiments/scripts/analyze_local_optima.py \
  experiments/scenarios/greedy_neighbor/evolved_v4_seed* \
  --output experiments/evolution/local_optima_greedy_neighbor.json
```

**Metrics**:
- **Fitness variance across seeds**: High variance → many local optima
- **Best fitness spread**: Range of final fitnesses
- **Convergence consistency**: Do all seeds reach similar fitness?

**Step 3: Categorize scenarios by difficulty**

```bash
uv run python experiments/scripts/categorize_difficulty.py \
  experiments/evolution/CONVERGENCE_COMPARISON.md \
  experiments/evolution/local_optima_*.json \
  --output experiments/evolution/FITNESS_LANDSCAPE.md
```

**Categories**:
- **Easy**: Fast convergence, low variance, single optimum
- **Moderate**: Moderate convergence, moderate variance
- **Hard**: Slow convergence, high variance, multiple optima

### Expected Outcomes

**Easy scenarios**:
- Converge by 2000-5000 generations
- All seeds reach similar fitness
- Smooth fitness landscape

**Hard scenarios**:
- Require 10,000+ generations
- High seed variance (different local optima)
- Rugged fitness landscape

### Deliverables

1. **Convergence comparison**: `experiments/evolution/CONVERGENCE_COMPARISON.md`
2. **Local optima analysis**: `experiments/evolution/local_optima_{scenario}.json`
3. **Fitness landscape report**: `experiments/evolution/FITNESS_LANDSCAPE.md`
4. **Visualizations**:
   - Convergence curves overlay: `plots/convergence_comparison_v4.png`
   - Seed variance boxplot: `plots/seed_variance_by_scenario.png`

---

## Experiment 5: Scenario Taxonomy

**Goal**: Create formal classification of scenarios based on strategic requirements.

**Priority**: 🟢 Medium (Q3)

### Protocol

**Step 1: Feature extraction**

For each scenario, compute:

**Scenario parameters**:
- Fire spread rate (β)
- Work cost (c)
- Rest benefit (r)
- Number of houses
- Agent count

**Evolved strategy features**:
- Mean genome parameters
- Cooperation rate (from behavior analysis)
- Risk profile (aggressive vs. conservative)
- Spatial strategy (own house vs. neighbors)

**Performance metrics**:
- Final fitness
- Convergence speed
- Transfer efficiency (to/from other scenarios)

**Step 2: Dimensionality reduction**

```bash
uv run python experiments/scripts/scenario_pca.py \
  experiments/evolution/all_genomes_v4.csv \
  experiments/scenarios/*/config.json \
  --output experiments/evolution/scenario_taxonomy.json
```

**Step 3: Clustering**

```bash
uv run python experiments/scripts/cluster_scenarios.py \
  experiments/evolution/scenario_taxonomy.json \
  --num-clusters 3 \
  --output experiments/evolution/SCENARIO_TAXONOMY.md
```

**Methods**:
- PCA for visualization
- K-means for clustering (try k=2,3,4)
- Hierarchical clustering for dendrogram

### Expected Outcomes

**Example clusters**:

**Cluster 1: Simple Cooperation**
- Scenarios: easy, trivial_cooperation, default
- Characteristics: Low cost, clear incentives
- Optimal strategy: High coordination

**Cluster 2: Free-Riding Traps**
- Scenarios: chain_reaction, overcrowding, hard
- Characteristics: High cost, free-riding incentive
- Optimal strategy: Selective cooperation

**Cluster 3: Mixed Incentives**
- Scenarios: greedy_neighbor, mixed_motivation, rest_trap
- Characteristics: Conflicting objectives
- Optimal strategy: Balanced priorities

### Deliverables

1. **Scenario features**: `experiments/evolution/scenario_features.csv`
2. **Taxonomy report**: `experiments/evolution/SCENARIO_TAXONOMY.md`
3. **Cluster assignments**: `experiments/evolution/scenario_clusters.json`
4. **Visualizations**:
   - PCA plot: `plots/scenario_pca.png`
   - Dendrogram: `plots/scenario_dendrogram.png`
   - Cluster comparison: `plots/cluster_characteristics.png`

---

## Implementation Checklist

### Core Scripts

- [x] `experiments/scripts/run_evolution.py` - Main evolution runner
- [x] `experiments/scripts/run_comparison.py` - Tournament validation
- [ ] `experiments/scripts/extract_all_genomes.py` - Genome database
- [ ] `experiments/scripts/parameter_sensitivity.py` - Parameter perturbation
- [ ] `experiments/scripts/transfer_test.py` - Cross-scenario transfer
- [ ] `experiments/scripts/compare_convergence.py` - Convergence analysis
- [ ] `experiments/scripts/analyze_local_optima.py` - Multi-seed analysis
- [ ] `experiments/scripts/cluster_scenarios.py` - Scenario taxonomy

### Analysis Scripts

- [ ] `experiments/scripts/visualize_genome_patterns.py` - Heatmap visualization
- [ ] `experiments/scripts/rank_parameter_importance.py` - Parameter ranking
- [ ] `experiments/scripts/analyze_transfer_matrix.py` - Transfer analysis
- [ ] `experiments/scripts/categorize_difficulty.py` - Fitness landscape
- [ ] `experiments/scripts/scenario_pca.py` - Dimensionality reduction

### Batch Execution

- [ ] `experiments/evolution/run_all_scenarios.sh` - Batch evolution
- [ ] `experiments/evolution/run_all_transfers.sh` - Transfer matrix
- [ ] `experiments/evolution/run_sensitivity_suite.sh` - Sensitivity analysis

---

## Timeline

### Q1 (Months 1-3)

**Week 1-4**: Experiment 1 - Tier 1 scenarios
- Launch: greedy_neighbor, mixed_motivation, easy
- Monitor convergence
- Validate results

**Week 5-8**: Experiment 1 - Tier 2 scenarios
- Launch: overcrowding, rest_trap, sparse_heroics
- Begin Experiment 3 (transfer tests for completed scenarios)

**Week 9-12**: Experiment 1 - Tier 3 scenarios
- Complete: trivial_cooperation, early_containment, default
- Experiment 3 continued

### Q2 (Months 4-6)

**Week 1-4**: Experiment 2 - Parameter analysis
- Extract all genomes
- Sensitivity analysis
- Parameter importance ranking

**Week 5-8**: Experiment 3 - Transfer matrix
- Complete transfer experiments
- Analyze transfer patterns
- Identify scenario clusters

### Q3 (Months 7-9)

**Week 1-4**: Experiment 4 - Fitness landscape
- Convergence comparison
- Multi-seed analysis
- Difficulty categorization

**Week 5-8**: Experiment 5 - Taxonomy
- Feature extraction
- Clustering analysis
- Final report

**Week 9-12**: Synthesis
- Integrate findings
- Draft paper
- Prepare final deliverables

---

## Success Criteria

### Quantitative

✅ **Coverage**: Evolution results for all 12 scenarios
✅ **Consistency**: Train/test fitness within 5% for all scenarios
✅ **Transfer**: Identify at least 2 scenario clusters with >80% transfer efficiency
✅ **Sensitivity**: Parameter importance ranking covering all 10 parameters

### Qualitative

✅ **Understanding**: Clear explanation of which parameters matter where
✅ **Generalization**: Formal taxonomy grouping scenarios by strategic requirements
✅ **Prediction**: Model predicting scenario difficulty from parameters
✅ **Integration**: Cross-validation with Nash and MARL tracks

---

## Resources

### Computational

- **CPU**: High (200-600 CPU-hours per scenario)
- **Parallelization**: 64 cores per run
- **Memory**: Moderate (4-8GB per run)
- **Storage**: ~500MB per scenario (logs + checkpoints)
- **Total**: ~20-40 GPU-instance-days for full sweep

### Optimization Strategies

**Tier-based execution**:
- Run high-priority scenarios first
- Allows early analysis while remaining runs complete

**Parallel execution**:
- Run 3-4 scenarios simultaneously (if resources allow)
- Reduces wall-clock time

**Compute scheduling**:
- Use spot instances for cost savings
- Checkpoint every 1000 generations for resilience

---

## References

### Internal

- **Phase 2 Agenda**: `docs/PHASE_2_RESEARCH_AGENDA.md`
- **Evolution README**: `experiments/evolution/README.md`
- **Rust Single Source of Truth**: `experiments/evolution/RUST_SINGLE_SOURCE_OF_TRUTH.md`
- **V4 Plan**: `experiments/evolution/V4_EVOLUTION_PLAN.md`

### External

- Eiben & Smith (2015): "Introduction to Evolutionary Computing"
- Pugh et al. (2016): "Quality Diversity: A New Frontier for Evolutionary Computation"
- Colas et al. (2020): "EvoGym: A Large-Scale Benchmark for Co-Optimizing the Design and Control of Soft Robots"

---

**Status**: 🚀 Ready to Execute
**Owner**: Evolution Track Lead
**Next Steps**: Begin Experiment 1 Tier 1 (greedy_neighbor, mixed_motivation, easy)
