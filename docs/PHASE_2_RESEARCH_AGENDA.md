# Phase 2 Research Agenda: Understanding Agent Effectiveness

**Status**: Planning
**Phase 1 Foundation**: Getting Really Good at Training Agents ✅
**Phase 2 Focus**: Understanding What Makes Agents Effective

---

## Overview

Phase 1 established our capability to create effective agents through three complementary approaches:
- **Nash Equilibrium**: Game-theoretic analysis with Double Oracle
- **Genetic Evolution**: Parametric heuristic optimization (58.50 payoff, near-Nash)
- **MARL/PPO**: Neural network gradient-based learning

Phase 2 shifts from **creating agents** to **understanding agent effectiveness**. We ask:
- Why do evolved strategies outperform Nash predictions (2.94 → 58.50)?
- What makes strategies effective across different scenarios?
- How do learning methods compare in their discovery process?
- What generalizable insights can we extract about cooperation under uncertainty?

---

## Research Philosophy

### From Engineering to Science

**Phase 1 mindset**: "Can we build agents that perform well?"
- Focus: Training pipelines, infrastructure, convergence
- Success metric: High reward scores
- Output: Effective agents

**Phase 2 mindset**: "Why do certain strategies work?"
- Focus: Understanding, comparison, generalization
- Success metric: Predictive models of effectiveness
- Output: Transferable insights

### Three-Track Integration

Each track provides a unique lens on the same fundamental question:

**Nash (Game Theory)**:
- **Question**: What should rational agents do?
- **Strength**: Formal guarantees, equilibrium concepts
- **Limitation**: Limited to predefined strategy spaces

**Evolution (Heuristic Search)**:
- **Question**: What works in practice?
- **Strength**: Discovers novel strategies, interpretable parameters
- **Limitation**: No learning dynamics, homogeneous teams

**MARL (Neural Networks)**:
- **Question**: What can agents learn?
- **Strength**: Rich representations, online adaptation
- **Limitation**: Black-box policies, expensive training

**Integration**: By comparing these approaches, we understand the **strategy space landscape**—where different methods excel, fail, and converge.

---

## Phase 2 Goals by Track

### Nash Track: Closing the Prediction Gap

**Phase 1 Achievement**: Computed Nash equilibria for 12 scenarios using predefined archetypes (Coordinator, Free Rider, Hero, Liar).

**Critical Gap Identified**:
| Scenario | Nash (Free Rider) | Evolution | Gap |
|----------|-------------------|-----------|-----|
| chain_reaction | 2.94 | 58.50 | **+55.56** 🔍 |

**Phase 2 Research Questions**:
1. **Why does evolution beat Nash by 20×?**
   - Are evolved strategies outside the archetype space?
   - Is the Nash prediction wrong or just incomplete?
   - What does the "true" strategy space look like?

2. **Can Double Oracle find evolved-quality strategies?**
   - Integrate evolved agents into strategy pool
   - Compute equilibria over expanded space
   - Compare payoffs: Nash V1 vs. Nash V2 (with evolution)

3. **Are evolved strategies actually equilibria?**
   - Test if evolved strategies are best-responses to themselves
   - Compute epsilon-equilibrium bounds
   - Validate robustness to perturbations

**Phase 2 Experiments**:
- ✅ **V2 Nash Computation**: Run Double Oracle with evolved agents (v3/v4/v5) as initial strategies
- ✅ **Gap Analysis**: Explain 2.94 → 58.50 discrepancy via strategy space coverage
- ✅ **Epsilon-Equilibrium Testing**: Measure how close evolved strategies are to true equilibria
- ✅ **Cross-Validation**: Do Nash predictions match evolution results in expanded space?

**Success Criteria**:
- Close the 2.94 → 58.50 gap by finding better equilibria
- Formal characterization of when Nash V1 fails
- Integration framework for adding external strategies to Double Oracle

**Deliverables**:
- `experiments/nash/V2_RESULTS.md`: Nash computation with evolved strategies
- `experiments/nash/GAP_ANALYSIS.md`: Detailed explanation of chain_reaction gap
- Updated equilibrium tables comparing V1 (archetypes) vs. V2 (evolved)

---

### Evolution Track: Scenario-Wide Understanding

**Phase 1 Achievement**: Near-Nash performance (58.50 vs. 57.87) on chain_reaction with interpretable 10-parameter genomes.

**Current Limitation**: Only tested on 2/12 scenarios (chain_reaction, deceptive_calm).

**Phase 2 Research Questions**:
1. **Do evolved strategies generalize across scenarios?**
   - Run V4/V5 evolution on all 9 remaining scenarios
   - Compare parameter values: Are there universal patterns?
   - Test transfer: Can chain_reaction strategies work on greedy_neighbor?

2. **What parameters matter most?**
   - Sensitivity analysis across scenarios
   - Identify critical vs. negligible parameters
   - Discover scenario archetypes by parameter clustering

3. **How does the fitness landscape vary?**
   - Compare convergence rates across scenarios
   - Identify easy vs. hard optimization problems
   - Characterize local optima and plateau regions

**Phase 2 Experiments**:
- ✅ **Scenario Sweep**: Run V4 evolution (15K gen, 200 pop) on all 9 remaining scenarios
- ✅ **Parameter Analysis**: Extract and compare best-agent genomes across scenarios
- ✅ **Transfer Testing**: Evaluate cross-scenario performance (train on A, test on B)
- ✅ **Convergence Study**: Analyze generation-by-generation fitness curves

**Success Criteria**:
- Complete evolution results for all 12 scenarios
- Parameter sensitivity map showing which genes matter where
- Scenario taxonomy based on optimal strategy profiles
- Cross-scenario performance matrix

**Deliverables**:
- `experiments/evolution/SCENARIO_SWEEP_RESULTS.md`: Performance across all scenarios
- `experiments/evolution/PARAMETER_PATTERNS.md`: Analysis of genome convergence
- `experiments/evolution/TRANSFER_ANALYSIS.md`: Cross-scenario generalization study

---

### MARL Track: Learning Dynamics and Baselines

**Phase 1 Achievement**: GPU-accelerated PPO training pipeline with vectorized Rust environments.

**Current Limitation**: No systematic experiments comparing PPO to evolved/Nash baselines.

**Phase 2 Research Questions**:
1. **Can neural networks match evolution?**
   - Train PPO on chain_reaction
   - Compare final performance: PPO vs. evolved (58.50)
   - Analyze learning curves: How fast does PPO reach 58.50?

2. **What do neural policies learn?**
   - Behavioral analysis: Action distributions, cooperation rates
   - Compare to evolved heuristics: Are they learning similar strategies?
   - Ablation studies: Which network components matter?

3. **How does population-based training compare?**
   - Single-agent PPO vs. population training (8-32 agents)
   - Diversity metrics: Do populations discover varied strategies?
   - GPU utilization: Can we achieve 60-95% efficiency?

**Phase 2 Experiments**:
- ✅ **Baseline Establishment**: Train PPO on chain_reaction, compare to 58.50 evolved baseline
- ✅ **Learning Curve Analysis**: Track convergence speed, sample efficiency
- ✅ **Behavioral Comparison**: Extract cooperation rates, action patterns from trained policies
- ✅ **Population Training Pilot**: Test vectorized population training (train_vectorized_population.py)
- ✅ **GPU Optimization Study**: Measure utilization, identify bottlenecks

**Success Criteria**:
- PPO matches or exceeds evolved performance (≥58.50)
- Learning curve documentation showing convergence patterns
- Behavioral analysis showing policy interpretations
- Population training achieving >60% GPU utilization

**Deliverables**:
- `experiments/marl/BASELINE_RESULTS.md`: PPO vs. Evolution comparison
- `experiments/marl/LEARNING_CURVES.md`: Convergence analysis across scenarios
- `experiments/marl/POPULATION_TRAINING_RESULTS.md`: Multi-agent training outcomes
- `experiments/marl/GPU_OPTIMIZATION.md`: Performance tuning findings

---

## Cross-Track Integration

### Comparative Analysis

**Question**: How do the three methods compare in discovering effective strategies?

**Experiments**:
1. **Strategy Space Coverage**:
   - Plot evolved genomes, Nash strategies, PPO policies in behavior space
   - Measure overlap and unique regions discovered by each method
   - Identify "unreachable" strategies for each approach

2. **Convergence Comparison**:
   - Evolution: Generations to 58.50
   - MARL: Training steps to 58.50
   - Nash: Iterations to convergence
   - Compare sample efficiency and computational cost

3. **Robustness Testing**:
   - Perturb scenario parameters (±10% fire spread, work cost, etc.)
   - Measure performance degradation for each method's agents
   - Identify brittleness vs. generalization

**Deliverables**:
- `docs/STRATEGY_SPACE_ANALYSIS.md`: Comparative visualization of discovered strategies
- `docs/METHOD_COMPARISON.md`: Efficiency, performance, and robustness metrics
- `docs/FAILURE_MODE_ANALYSIS.md`: When and why each approach struggles

### Unified Understanding

**Goal**: Develop predictive models of agent effectiveness.

**Research Questions**:
1. Can we predict which scenarios favor which methods?
2. What scenario features correlate with high cooperation vs. free-riding?
3. Can we design scenarios to elicit specific strategic behaviors?

**Experiments**:
- Regression analysis: Scenario parameters → optimal strategy parameters
- Scenario clustering: Group by strategic similarity
- Ablation studies: Modify scenario features, observe strategy changes

**Deliverables**:
- `docs/PREDICTIVE_MODELS.md`: Formal models relating scenarios to strategies
- `docs/SCENARIO_TAXONOMY.md`: Classification of scenarios by strategic requirements

---

## Quarterly Milestones

### Q1: Foundations (Months 1-3)

**Nash Track**:
- ✅ Implement evolved agent integration into Double Oracle
- ✅ Run V2 Nash on chain_reaction with evolved strategies
- ✅ Document gap analysis (2.94 → 58.50 explanation)

**Evolution Track**:
- ✅ Complete V4/V5 evolution for 3 priority scenarios (greedy_neighbor, mixed_motivation, easy)
- ✅ Initial parameter sensitivity analysis
- ⚠️ Transfer testing infrastructure (train on A, test on B)

**MARL Track**:
- ✅ Baseline PPO training on chain_reaction (10M steps)
- ✅ Learning curve documentation
- ✅ Fix VectorEnv GPU integration
- ⚠️ Population training pilot (4-8 agents)

**Integration**:
- 📊 Initial strategy space visualization (Nash, evolved, PPO)
- 📊 Comparative performance table (chain_reaction only)

### Q2: Expansion (Months 4-6)

**Nash Track**:
- ✅ V2 Nash for all 12 scenarios
- ✅ Epsilon-equilibrium analysis
- ✅ Heterogeneous team equilibria (mixed agent types)

**Evolution Track**:
- ✅ Complete scenario sweep (all 12 scenarios)
- ✅ Full parameter analysis report
- ✅ Cross-scenario transfer matrix

**MARL Track**:
- ✅ PPO training on 6 scenarios
- ✅ Behavioral analysis (cooperation rates, action distributions)
- ✅ Population training at scale (16-32 agents)

**Integration**:
- 📊 Full strategy space analysis
- 📊 Method comparison report (convergence, efficiency, robustness)
- 📊 Failure mode documentation

### Q3: Synthesis (Months 7-9)

**Nash Track**:
- ✅ Robustness testing (parameter perturbations)
- ✅ Final V2 cross-scenario report
- 🎯 Paper draft: "From Archetypes to Evolution: Expanding Nash Equilibrium Strategy Spaces"

**Evolution Track**:
- ✅ Final parameter pattern analysis
- ✅ Scenario taxonomy by strategy requirements
- 🎯 Paper draft: "Evolutionary Discovery of Cooperation Strategies in Multi-Agent Fire-Fighting"

**MARL Track**:
- ✅ Complete PPO baseline suite (all 12 scenarios)
- ✅ GPU optimization report
- 🎯 Technical report: "Neural vs. Heuristic: Learning Dynamics in Cooperative MARL"

**Integration**:
- 🎯 Unified technical report: "Three Lenses on Cooperation: Nash, Evolution, and Learning"
- 🎯 Predictive models of scenario-strategy relationships
- 🎯 Public benchmark release with all baselines

---

## Success Metrics

### Research Impact
- **3 papers** (one per track or integrated)
- **Public benchmark** with baselines from all three methods
- **Predictive framework** for scenario-strategy relationships

### Technical Achievements
- ✅ Nash V2 closes 2.94 → 58.50 gap
- ✅ Evolution baselines for all 12 scenarios
- ✅ MARL baselines matching or exceeding evolution
- ✅ Strategy space map showing method coverage

### Understanding
- ✅ Formal explanation of why certain strategies work
- ✅ Scenario taxonomy by strategic requirements
- ✅ Method comparison identifying strengths/weaknesses
- ✅ Transferable insights for multi-agent system design

---

## Resources and Infrastructure

### Computational
- **Nash**: Moderate CPU (hours per scenario, not GPU-dependent)
- **Evolution**: High CPU (200-600 CPU-hours per run, 64 cores)
- **MARL**: High GPU (2-10 GPU-hours per scenario, L4/A10)

### Storage
- Checkpoints: ~500MB per MARL run
- Evolution logs: ~100MB per run
- Nash results: ~10MB per scenario
- Estimated total: ~50-100GB for Phase 2

### Human Resources
- Primary researcher time: 6-9 months
- Periodic reviews with advisors/collaborators
- User testing for benchmark release

---

## Risk Mitigation

### Technical Risks

**Risk**: Nash V2 still doesn't find 58.50 strategies
**Mitigation**: Incremental integration—start with single evolved agent, gradually expand pool

**Risk**: MARL training doesn't converge to evolved performance
**Mitigation**: Hyperparameter tuning, architectural search, population-based training

**Risk**: Evolution doesn't generalize across scenarios
**Mitigation**: Start with related scenarios, use transfer learning, analyze failure modes

### Research Risks

**Risk**: Methods converge to same strategies (no interesting differences)
**Mitigation**: Expand scenario suite to increase diversity, test on more extreme parameters

**Risk**: Gap remains unexplained despite analysis
**Mitigation**: Collaborate with game theory experts, consider approximate solution concepts

**Risk**: Results don't generalize beyond Bucket Brigade
**Mitigation**: Connect findings to broader MARL literature, test on related domains

---

## Transition to Phase 3

Phase 2 establishes **understanding** of agent effectiveness in fixed scenarios. Phase 3 will explore:

1. **Adaptive Multi-Scenario Agents**: Policies that switch strategies based on inferred scenario
2. **Population Resilience**: Heterogeneous teams, competitive co-evolution
3. **Real-World Transfer**: Sim-to-real, robustness to distribution shift

Phase 2 outputs provide the foundation:
- Baseline strategies for comparison
- Understanding of what makes strategies effective
- Methods for rapid agent creation

---

## Summary

**Phase 1**: Created effective agents (Nash, Evolution, MARL)
**Phase 2**: Understand why they're effective
**Phase 3**: Generalize beyond fixed scenarios

**Core Questions**:
- Why does evolution beat Nash predictions?
- What do neural networks learn vs. evolution?
- How do methods compare in discovering effective strategies?
- What makes strategies effective across scenarios?

**Deliverables**:
- Nash V2 with evolved strategy integration
- Evolution baselines for all scenarios
- MARL baselines matching evolution
- Unified cross-method analysis
- Predictive models and public benchmark

**Timeline**: 9 months
**Success**: Transferable insights, published research, public benchmarks
