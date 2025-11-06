# Scale Testing Plan: Testing Universality at N=6, 8, 10

## Objective

Test whether the universal Nash equilibrium discovered for N=4 agents generalizes to larger population sizes.

**Key Question**: Does the universal free-riding strategy remain optimal as the number of agents increases?

## Motivation

Phases 1-2 established that a single strategy is optimal across all scenarios for N=4 agents. However:
- Cooperation problems often change fundamentally with population size
- Free-riding may become less viable at larger N (harder to hide)
- Coordination challenges may increase with N
- Nash equilibria may differ at different population sizes

**Hypothesis**: Universal strategy scales gracefully to N=6, 8, 10

## Experimental Design

### Phase 1: Quick Validation (Test Universal Strategy)

**Approach**: Test N=4 universal strategy on N=6, 8, 10 scenarios

**Scenarios to test**:
- chain_reaction (baseline)
- sparse_heroics (high performance)
- crisis_cheap (extreme scenario)
- easy_spark_02 (optimal p_spark)

**Evaluation**:
- Self-play with N=4 universal genome on N=6, 8, 10 scenarios
- Compare performance to N=4 baseline
- Measure: Absolute payoff, relative degradation

**Expected time**: 30 minutes

**Success criterion**: If degradation < 20%, universal strategy scales well

### Phase 2: Evolution (If Needed)

**Trigger**: If universal strategy degradation > 20% at any N

**Approach**: Evolve new strategies for N=6, 8, 10

**Scenarios**:
- chain_reaction
- sparse_heroics
- crisis_cheap

**Evolution parameters**:
- Population: 200
- Generations: 10,000 (vs 15,000 for N=4)
- Selection: Tournament (k=5)
- Mutation: 0.1
- Evaluations per genome: 100 simulations

**Expected time**:
- N=6: ~3-4 hours per scenario
- N=8: ~4-6 hours per scenario
- N=10: ~6-10 hours per scenario

**Overnight feasibility**: Yes - can run 1-2 scenarios per size

### Phase 3: Comparison (If Evolution Run)

**Compare**:
- N=4 universal genome parameters
- N=6 evolved genome parameters
- N=8 evolved genome parameters
- N=10 evolved genome parameters

**Analysis**:
- Genome L2 distance from universal
- Performance comparison
- Test if new genomes differ significantly

## Scenarios

Using representative scenarios that cover different dynamics:

| Scenario | β | c | p_spark | Why Selected |
|----------|---|---|---------|--------------|
| chain_reaction | 0.45 | 0.70 | 0.03 | Baseline scenario |
| sparse_heroics | 0.10 | 0.80 | 0.02 | High performance |
| crisis_cheap | 0.60 | 0.10 | 0.03 | Extreme parameters |
| easy_spark_02 | 0.15 | 0.50 | 0.02 | Optimal p_spark |

## Success Criteria

### Hypothesis H1: Universal strategy scales gracefully

**Test**: N=4 universal strategy performs well on N=6, 8, 10

**Metrics**:
- Absolute payoff remains > 50
- Relative degradation < 20% vs N=4 performance
- No catastrophic failures

**If TRUE**: Universality extends to larger populations ✓

**If FALSE**: Need population-specific strategies

### Hypothesis H2: Evolved strategies (if needed) match universal

**Test**: If we evolve for N=6, 8, 10, do they match N=4 universal?

**Metrics**:
- Genome L2 distance < 0.01 (effectively identical)
- Performance difference < 5%

**If TRUE**: Universal equilibrium truly universal ✓

**If FALSE**: Population size affects equilibrium

## Implementation

### Quick Test Script

Create `experiments/scripts/test_scale_quick.py`:
- Load N=4 universal genome
- Test on N=6, 8, 10 scenarios (4 scenarios × 3 sizes = 12 tests)
- Report performance metrics
- Flag cases where degradation > 20%

### Evolution Script (If Needed)

Modify `experiments/scripts/run_evolution_v4.py`:
- Add `--num-agents` parameter
- Run for N=6, 8, 10
- Save results in `experiments/scenarios/{scenario}/evolved_N{N}/`

### Evaluation Script

Create `experiments/scripts/evaluate_scale.py`:
- Compare N=4, 6, 8, 10 genomes
- Cross-evaluate (e.g., N=6 genome on N=8 scenario)
- Generate comparison tables

## Expected Outcomes

### Scenario 1: Universal Strategy Scales Perfectly

**Result**: N=4 genome works well on N=6, 8, 10

**Interpretation**:
- Free-riding equilibrium independent of population size
- Symmetric game structure preserved
- No evolution needed

**Next steps**: Document scaling behavior, proceed to paper

### Scenario 2: Universal Strategy Degrades Moderately (10-20%)

**Result**: Performance drops but remains viable

**Interpretation**:
- Some population size effects
- But universal strategy approximately optimal
- Evolution might improve slightly

**Next steps**:
- Document degradation pattern
- Optional: Run evolution to quantify improvement potential
- Proceed to paper with scaling limits noted

### Scenario 3: Universal Strategy Fails at Larger N (>20% degradation)

**Result**: Significant performance drop

**Interpretation**:
- Population size fundamentally changes equilibrium
- Need specialized strategies for different N

**Next steps**:
- Run evolution for N=6, 8, 10
- Analyze how equilibrium changes with N
- Major finding for paper!

## Timeline

**Immediate** (tonight):
1. Create quick test script (30 min)
2. Run quick test (30 min)
3. Analyze results (30 min)
4. Decision point (1.5 hours total)

**If evolution needed** (overnight):
1. Set up evolution runs for 1-2 scenarios per size
2. Let run overnight (6-10 hours)
3. Analyze in morning (1 hour)
4. Document findings (1 hour)

**Total**: 2-12 hours depending on path

## Files to Create

- `experiments/scale_testing/SCALE_TESTING_PLAN.md` - This file
- `experiments/scripts/test_scale_quick.py` - Quick validation script
- `experiments/scale_testing/quick_results.json` - Quick test results
- `experiments/scale_testing/SCALE_TESTING_ANALYSIS.md` - Findings (if interesting)

## Open Questions

1. **How does payoff scale with N?**
   - Linear, sublinear, superlinear?
   - Depends on scenario parameters?

2. **Does free-riding become more/less viable at larger N?**
   - More agents = easier to free-ride (hide in crowd)?
   - Or harder (more workers cover free-riders)?

3. **Does cooperation threshold change with N?**
   - Critical p_spark depend on N?
   - Phase transitions at certain population sizes?

4. **Computational cost scaling**:
   - Evolution time scales roughly O(N²) (more agents × longer episodes)
   - May hit practical limits at N=16, 32

## Risk Assessment

**Low Risk**:
- Quick test takes minimal time
- If universal strategy scales, we're done
- Clean addition to paper

**Medium Risk**:
- Evolution for N=6, 8, 10 takes significant time
- May not finish overnight
- Could run on remote GPU if needed

**High Risk**:
- If results show complex population-size dependencies
- May open new research questions
- Could delay paper submission

**Mitigation**:
- Start with quick test
- Only evolve if necessary
- Can publish N=4 results alone if time-constrained
- Note scaling as future work if complex

## Success Definition

**Minimum success**: Document how N=4 universal strategy performs on N=6, 8, 10

**Full success**: Establish whether universal equilibrium is population-size invariant

**Stretch success**: Complete evolution for all sizes, full characterization of scaling
