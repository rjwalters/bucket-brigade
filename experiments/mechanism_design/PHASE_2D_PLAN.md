# Phase 2D: Mechanism Design for Cooperation - Plan

## Objective

Design scenarios that incentivize cooperation and break the universal free-riding equilibrium.

**Goal**: Achieve work_tendency > 0.5 in at least one scenario (current universal strategy: ~0.06-0.10)

## Motivation

Phases 2A and 2A.1 revealed:
- Universal strategy is remarkably robust across parameter extremes
- Only breaks on transient threats (p_spark=0)
- Achieves this through "lazy free-riding" (low work_tendency)

**Question**: Can we design scenarios where free-riding is suboptimal and cooperation is required?

## Constraints

Working within existing game mechanics (Scenario parameters):
- β (fire spread rate)
- κ (extinguish probability)
- c (work cost)
- A (asset value)
- L (loss from fire)
- ρ_ignite (initial fire probability)
- N_min (minimum neighbors for spread)
- p_spark (ongoing fire probability)
- N_spark (neighbors for spontaneous ignition)

Future work could extend game engine for:
- Coordination bonuses (reward synchronized actions)
- Information asymmetry (resting agents see noisy state)
- Explicit punishment mechanisms
- Temporal reward structure changes

## Design Strategies

### Strategy 1: Front-Loaded Crisis

**Hypothesis**: Force immediate coordinated response by creating overwhelming initial threat.

**Mechanism**:
- Very high ρ_ignite (many initial fires)
- High β (fires spread explosively)
- Low p_spark (no ongoing fires - one-time crisis)
- Moderate c (work is affordable)

**Expected behavior**:
- Universal strategy may fail due to slow response
- Need immediate all-hands-on-deck approach
- Similar to p_spark=0 scenarios but with higher urgency

**Test**: Does crisis intensity overcome transient threat problem?

### Strategy 2: Costly Inaction

**Hypothesis**: Make resting so expensive that working becomes optimal.

**Mechanism**:
- Extremely low c (work is nearly free, c → 0)
- High β (fires spread fast)
- High p_spark (continuous threat)
- High L (high loss from fires)

**Expected behavior**:
- If c ≈ 0, working is costless
- Universal strategy should shift to high work_tendency
- Tests lower bound of c where free-riding breaks

**Test**: Does c → 0 induce cooperation?

### Strategy 3: High Stakes Coordination

**Hypothesis**: Increase payoff variance to make coordination critical.

**Mechanism**:
- Very high A (asset value)
- Very high L (loss value)
- High β (fast spread - all-or-nothing)
- Moderate c (work is meaningful cost)
- High p_spark (persistent threat)

**Expected behavior**:
- High variance in outcomes (total success or total failure)
- Free-riding may lead to catastrophic collective loss
- Coordination becomes critical

**Test**: Does high-variance payoff structure induce cooperation?

### Strategy 4: Sustained Pressure

**Hypothesis**: Continuous high threat requires sustained effort.

**Mechanism**:
- Very high p_spark (many ongoing fires)
- High β (fast spread)
- Low κ (difficult to extinguish)
- Moderate c (work is costly but necessary)

**Expected behavior**:
- Fires accumulate without sustained work
- Free-riding leads to overwhelming fire load
- Requires persistent cooperation

**Test**: Does sustained high pressure break free-riding?

### Strategy 5: Sparse High-Value Targets

**Hypothesis**: Few critical assets require focused protection.

**Mechanism**:
- Low grid density (fewer houses)
- Very high A (each house extremely valuable)
- High β (fires spread to nearby houses fast)
- Low p_spark (limited threat, but critical)

**Expected behavior**:
- Each house loss is catastrophic
- Need focused, coordinated protection
- Free-riding risks high-value loss

**Test**: Does concentrated value induce cooperation?

## Success Criteria

**Quantitative**:
1. Identify at least one scenario where universal strategy achieves work_tendency > 0.5
2. Measure cooperation level (% of timesteps spent working)
3. Identify parameter thresholds where cooperation emerges

**Qualitative**:
1. Understand when free-riding equilibrium breaks down
2. Document design patterns for cooperation-inducing scenarios
3. Identify minimal mechanisms for cooperation

## Evaluation Methodology

For each mechanism scenario:

1. **Universal strategy test**: Evaluate self-play with universal genome
2. **Baseline comparison**: Compare to baseline scenarios (chain_reaction, etc.)
3. **Work tendency analysis**: Measure % of timesteps agents spend working
4. **Payoff analysis**: Compare to theoretical max (all work) and min (all rest)

**Metrics**:
- `work_tendency`: Average work probability from genome
- `actual_work_rate`: Measured % of actions that are "work" in simulation
- `cooperation_level`: actual_work_rate compared to baseline
- `equilibrium_payoff`: Self-play payoff with universal strategy

## Expected Outcomes

**Scenario 1 (Front-loaded crisis)**:
- Likely fails (transient threat, p_spark=0)
- But may provide data on crisis response

**Scenario 2 (Costly inaction, c → 0)**:
- Most likely to induce cooperation
- As c → 0, working becomes free
- Should see work_tendency increase

**Scenario 3 (High stakes)**:
- Uncertain - may just reduce overall payoff
- Variance might not change equilibrium

**Scenario 4 (Sustained pressure)**:
- Moderate chance - similar to existing high-pressure scenarios
- May just reduce payoff without changing strategy

**Scenario 5 (Sparse high-value)**:
- Requires game engine changes (grid size)
- Defer to future work

## Implementation Plan

1. **Phase 2D.1**: Design and register 4-5 mechanism scenarios
2. **Phase 2D.2**: Test universal strategy on mechanism scenarios
3. **Phase 2D.3**: Analyze cooperation levels and identify thresholds
4. **Phase 2D.4**: Document findings and design patterns

## Hypotheses

**H1**: c → 0 will induce cooperation (work_tendency > 0.5)
- Prediction: **LIKELY** - when work is free, no reason not to work

**H2**: Front-loaded crisis will induce cooperation
- Prediction: **UNLIKELY** - transient threats already shown to fail (p_spark=0)

**H3**: Sustained high pressure will induce cooperation
- Prediction: **UNLIKELY** - Phase 2A showed universal strategy handles high β, high p_spark well

**H4**: High-variance payoffs will induce cooperation
- Prediction: **UNCERTAIN** - depends on risk aversion parameters

## Limitations

1. **Game engine constraints**: Cannot implement true mechanism design (coordination bonuses, information asymmetry) without modifying core game
2. **Parameter-based only**: Limited to varying scenario parameters, not game rules
3. **Equilibrium focus**: Testing only universal strategy, not evolving new strategies

## Future Extensions

If Phase 2D shows cooperation can be induced:
- **Evolve specialized strategies** for cooperation-inducing scenarios
- **Compare to universal strategy** to measure adaptation
- **Test heterogeneous populations** (cooperators + free-riders)

If Phase 2D shows cooperation cannot be induced with parameters alone:
- **Propose game engine modifications** (coordination bonuses, etc.)
- **Design theoretical mechanisms** based on findings
- **Document requirements** for cooperation in this domain

## Files to Create

- `experiments/mechanism_design/PHASE_2D_PLAN.md` - This file
- `bucket_brigade/envs/scenarios.py` - Add mechanism scenarios
- `experiments/scripts/test_mechanism_scenarios.py` - Test script
- `experiments/mechanism_design/results.json` - Results data
- `experiments/mechanism_design/PHASE_2D_ANALYSIS.md` - Findings

## Timeline

Estimated: 2-3 hours
- Design: 30 min
- Implementation: 30 min
- Testing: 30-60 min (depends on scenarios)
- Analysis: 30-60 min
