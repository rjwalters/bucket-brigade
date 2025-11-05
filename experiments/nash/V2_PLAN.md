# Nash Equilibrium V2 Plan

**Date**: 2025-11-05
**Status**: Planning phase
**Related**: Evolution V3/V4/V5 results, [Phase 1 Roadmap](../../docs/roadmap_phased_plan.md)

---

## V1 Summary (Completed)

### What We Have

**Analysis Results** (experiments/nash/README.md):
- ✅ 12 scenarios analyzed using Double Oracle algorithm
- ✅ 10 pure equilibria, 2 mixed equilibria identified
- ✅ Cooperation patterns documented (50% average)
- ✅ Strategy distributions computed using predefined archetypes
- ✅ Rust-accelerated payoff evaluation (100x faster)

**Key V1 Findings**:
- **chain_reaction**: Pure Free Rider equilibrium, 2.94 payoff
- **greedy_neighbor**: Pure Coordinator equilibrium, 58.77 payoff
- **trivial_cooperation**: Mixed equilibrium, 108.78 payoff
- Most scenarios (10/12) have pure equilibria

**V1 Limitations**:
- ❌ Only used predefined archetypes (Coordinator, Free Rider, Hero, Liar)
- ❌ No comparison with evolved strategies
- ❌ No robustness/stability analysis
- ❌ No explanation of evolution-Nash gap

---

## V1 vs Evolution: The Mystery 🔍

### Chain Reaction Scenario

| Approach | Result | Gap |
|----------|--------|-----|
| **V1 Nash (Theoretical)** | 2.94 payoff (Free Rider) | Baseline |
| **V1 Coordinator Archetype** | ~57.87 payoff (estimated) | +54.93 |
| **Evolution V3/V4** | 58.50 payoff (near-optimal) | +55.56 |

**The Big Question**: Why does evolution achieve 58.50 when Nash predicts 2.94?

**Possible Explanations**:
1. **Cooperative Equilibrium**: Evolution found a *different* equilibrium (not Free Rider)
2. **Homogeneous Teams**: Self-play creates cooperation not captured by adversarial Nash
3. **Strategy Space**: Evolved strategies outside the V1 archetype set
4. **Mixed Strategies**: Evolution approximates a mixed strategy Nash missed by V1

**Resolution**: V2 must compare evolved agents directly with Nash analysis.

---

## V2 Goals: Evolution-Nash Integration

### Primary Objective

**Bridge the gap** between evolutionary results and game-theoretic predictions.

### Specific Goals

1. **Compare Evolved Strategies with Nash**
   - Add evolved agents (V3/V4/V5) to Double Oracle strategy pool
   - Recompute Nash equilibria including evolved strategies
   - Measure strategic divergence: how far are evolved agents from Nash?
   - Answer: Is 58.50 a Nash equilibrium we missed?

2. **Explain the Evolution-Nash Gap**
   - Why does evolution achieve 58.50 in chain_reaction vs Nash 2.94?
   - Is there a cooperative equilibrium at 58.50?
   - Test hypothesis: homogeneous self-play ≠ adversarial equilibrium

3. **Expand Strategy Space**
   - V1: Only predefined archetypes
   - V2: Include all evolved agents from V3/V4/V5 (9 scenarios × 3 versions = 27 agents)
   - Discover new equilibria with richer strategy set

4. **Robustness Analysis**
   - Epsilon-equilibrium: how close to Nash are evolved strategies?
   - Parameter sensitivity: stability to scenario perturbations
   - Opponent diversity: performance against heterogeneous teams

---

## V2 Technical Approach

### Phase 1: Load Evolved Strategies

```python
# Load evolved agents from experiments/scenarios/*/evolved_v{3,4,5}/
def load_evolved_agents(scenario: str) -> List[Agent]:
    """Load V3, V4, V5 evolved agents for scenario."""
    agents = []
    for version in ["v3", "v4", "v5"]:
        path = f"experiments/scenarios/{scenario}/evolved_{version}/best_agent.json"
        if exists(path):
            genome = load_genome(path)
            agent = HeuristicAgent(genome)
            agent.name = f"Evolved_{version}"
            agents.append(agent)
    return agents
```

### Phase 2: Expand Double Oracle

```python
# Run Double Oracle with evolved agents in initial pool
def compute_nash_with_evolved(scenario: str):
    """
    Nash equilibrium including evolved strategies.

    Initial pool:
    - Predefined archetypes (Coordinator, Free Rider, Hero, Liar)
    - Evolved agents (V3, V4, V5)
    """
    initial_strategies = (
        get_archetypes() +
        load_evolved_agents(scenario)
    )

    equilibrium = double_oracle(
        scenario=scenario,
        initial_strategies=initial_strategies,
        num_simulations=2000  # Same as V1
    )

    return equilibrium
```

### Phase 3: Compare Results

```python
def compare_nash_versions(scenario: str):
    """
    Compare V1 (archetypes only) vs V2 (with evolved agents).

    Metrics:
    - Equilibrium payoff change
    - Strategy distribution change
    - Support size (pure vs mixed)
    - Cooperation rate
    """
    v1_nash = load_v1_nash(scenario)
    v2_nash = compute_nash_with_evolved(scenario)

    print(f"V1 payoff: {v1_nash.payoff:.2f}")
    print(f"V2 payoff: {v2_nash.payoff:.2f}")
    print(f"Change: {v2_nash.payoff - v1_nash.payoff:+.2f}")

    # Key question: Do evolved strategies appear in equilibrium?
    for strategy, prob in v2_nash.distribution.items():
        if "Evolved" in strategy.name:
            print(f"✅ {strategy.name} in equilibrium ({prob:.1%})")
```

### Phase 4: Robustness Analysis

**Epsilon-Equilibrium**:
```python
def epsilon_equilibrium_distance(agent: Agent, scenario: str) -> float:
    """
    How far is agent from best response?

    Returns:
    - ε = 0: Perfect Nash equilibrium
    - ε > 0: Agent could improve by ε by switching strategy
    """
    current_payoff = evaluate_payoff(agent, agent, scenario)  # Self-play
    best_response = find_best_response([agent], scenario)
    br_payoff = evaluate_payoff(best_response, agent, scenario)

    epsilon = br_payoff - current_payoff
    return max(0, epsilon)  # Non-negative
```

**Parameter Sensitivity**:
```python
def robustness_to_perturbation(agent: Agent, scenario: str):
    """
    Test equilibrium stability to parameter changes.

    Perturb:
    - Fire spread probability (β ± 10%)
    - Work cost (c ± 10%)
    - Initial fires
    """
    baseline_payoff = evaluate_payoff(agent, agent, scenario)

    for param, delta in [("fire_spread", 0.1), ("work_cost", 0.1)]:
        perturbed_scenario = perturb_scenario(scenario, param, delta)
        perturbed_payoff = evaluate_payoff(agent, agent, perturbed_scenario)

        print(f"{param} +{delta:.0%}: {perturbed_payoff:.2f} (Δ{perturbed_payoff - baseline_payoff:+.2f})")
```

---

## V2 Research Questions

### Evolution-Nash Questions

1. **Is 58.50 a Nash equilibrium?**
   - Run Double Oracle with evolved V3/V4 agents
   - Do they appear in the equilibrium support?
   - Or are they exploitable?

2. **Why the gap between 2.94 and 58.50?**
   - V1 Nash found Free Rider equilibrium (2.94)
   - Evolution found cooperative strategy (58.50)
   - Is this because:
     - a) V1 missed a cooperative equilibrium?
     - b) Homogeneous self-play ≠ adversarial Nash?
     - c) Evolved strategy is epsilon-equilibrium, not exact?

3. **Do evolved strategies generalize across scenarios?**
   - Load chain_reaction evolved agent
   - Test in greedy_neighbor, deceptive_calm, etc.
   - Measure cross-scenario performance

### Theoretical Questions

4. **What is the relationship between self-play and Nash?**
   - Self-play (evolution): All agents use same strategy
   - Nash: Best response against opponent distribution
   - Are they equivalent for symmetric games?

5. **How stable are evolved equilibria?**
   - Epsilon-equilibrium distance for V3/V4/V5
   - Robustness to opponent diversity
   - Stability to parameter perturbations

6. **Can we predict equilibrium type from scenario parameters?**
   - Pure vs mixed
   - Cooperative vs free-riding
   - Based on (β, c, initial fires, etc.)

---

## V2 Implementation Plan

### Step 1: Data Preparation
- [x] V3/V4/V5 evolution results available
- [ ] Load evolved agents from JSON
- [ ] Validate genomes work with Nash evaluator
- [ ] Create unified agent registry

### Step 2: Nash Recomputation
- [ ] Run Double Oracle with evolved agents (all 9 scenarios)
- [ ] Compare V1 vs V2 equilibria
- [ ] Identify scenarios where evolved strategies enter equilibrium
- [ ] Document payoff changes

### Step 3: Gap Analysis
- [ ] For chain_reaction: Why 2.94 → 58.50?
- [ ] Test if 58.50 is a Nash equilibrium
- [ ] Compute epsilon-equilibrium distance
- [ ] Identify exploitability (if any)

### Step 4: Robustness Testing
- [ ] Parameter sensitivity analysis
- [ ] Cross-scenario generalization
- [ ] Opponent diversity tests
- [ ] Stability to perturbations

### Step 5: Documentation
- [ ] Create V2_RESULTS.md with findings
- [ ] Update main Nash README with V2 insights
- [ ] Link to evolution research (cross-validation)
- [ ] Publish comparison tables and visualizations

---

## V2 Success Criteria

### Must Have (Phase 1 Completion)

1. ✅ **Evolution-Nash comparison complete**
   - All 9 scenarios analyzed with V1+V2 methods
   - Payoff gaps explained
   - Strategic divergence quantified

2. ✅ **chain_reaction mystery resolved**
   - Understand why evolution achieves 58.50 vs Nash 2.94
   - Identify if 58.50 is a different equilibrium
   - Validate or refute theoretical predictions

3. ✅ **Epsilon-equilibrium analysis**
   - Measure how close evolved strategies are to Nash
   - Quantify exploitability
   - Document stability

### Nice to Have (Phase 1+)

4. ⚠️ **Cross-scenario generalization**
   - Test evolved agents outside training scenarios
   - Identify robust vs specialized strategies
   - Inform Phase 2 (multi-scenario agents)

5. ⚠️ **Parameter sensitivity**
   - Equilibrium stability to perturbations
   - Critical parameter thresholds
   - Inform scenario design

6. ⚠️ **Mechanism design insights**
   - How to incentivize cooperation
   - Reduce price of anarchy
   - Phase 3+ research direction

---

## V2 Timeline Estimate

**Assuming V5 completes** (provides V3/V4/V5 data for all 9 scenarios):

| Task | Time | Dependencies |
|------|------|--------------|
| Data prep & agent loading | 2 hours | V5 completion |
| Nash recomputation (9 scenarios) | 6 hours | Double Oracle runtime |
| Gap analysis & interpretation | 4 hours | Nash results |
| Robustness testing | 3 hours | - |
| Documentation | 3 hours | All analyses complete |
| **Total** | **~18 hours** | **(2-3 days)** |

**Bottleneck**: Double Oracle runtime (can be parallelized on remote server)

---

## V2 vs V1 Comparison Table

| Aspect | V1 (Completed) | V2 (Planned) |
|--------|----------------|--------------|
| **Strategy Pool** | 4 archetypes | 4 archetypes + 27 evolved agents |
| **Scenarios** | 12 (includes test scenarios) | 9 (standard research scenarios) |
| **Analysis** | Pure theory | Theory + evolution comparison |
| **Questions** | "What is Nash?" | "Does evolution find Nash?" |
| **Robustness** | None | Epsilon-equilibrium, stability |
| **Cross-validation** | Independent | Integrated with evolution research |
| **Phase 1 Status** | ✅ Foundations | 🎯 Completion milestone |

---

## Future: V3+ (Phase 2+)

Ideas for later iterations (not Phase 1):

### V3: Heterogeneous Equilibria (Phase 2)
- Nash equilibria with diverse agent types
- Specialist roles (firefighter, coordinator, free rider)
- Team composition optimization

### V4: Meta-Game Analysis (Phase 3)
- Population dynamics
- Evolutionary stability
- Co-evolutionary arms races

### V5: Mechanism Design (Phase 3+)
- Optimal scenario parameters
- Incentive alignment
- Social welfare maximization

---

## References

- **V1 Results**: [experiments/nash/README.md](./README.md)
- **Evolution Research**: [experiments/evolution/README.md](../evolution/README.md)
- **Phase 1 Roadmap**: [docs/roadmap_phased_plan.md](../../docs/roadmap_phased_plan.md)
- **Technical Review**: [docs/technical_marl_review.md](../../docs/technical_marl_review.md)

---

**Status**: V2 planning complete, awaiting V5 results for execution
**Next**: When V5 completes, begin data prep and Nash recomputation
