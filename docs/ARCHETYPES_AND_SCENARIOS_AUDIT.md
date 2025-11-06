# Archetypes and Scenarios Audit

**Date**: 2025-11-06
**Purpose**: Comprehensive audit of behavioral archetypes and scenario configurations to ensure consistency, completeness, and proper organization.

---

## Executive Summary

This audit covers:
- **5 behavioral archetypes** defining agent strategies
- **12 scenarios** in the registry (core 9 + 3 basic)
- **All 12 scenarios** have active experiment directories and research results
- **20 exploratory scenarios removed** to maintain focused research scope (see `REMOVED_SCENARIOS_ARCHIVE.md`)

### Key Findings

✅ **Strengths**:
- Clear 10-parameter behavioral model consistently applied across all archetypes
- Focused scenario set of 12 tested configurations covering cooperation dynamics space
- Parameter consistency between Python definitions and experiment configs
- Core 9 scenarios thoroughly tested with Nash V1, Nash V2, and evolution
- **2025-11-06 Update**: Pruned 20 untested scenarios to maintain ~10 scenario target

⚠️ **Remaining Gaps**:
- No archetype for "neutral/random" baseline strategy
- Nash V2 winning evolved agents not yet cataloged as archetypes
- Documentation could benefit from cross-references between related scenarios

---

## Archetypes

Archetypes are predefined 10-dimensional behavioral strategies representing common playstyles in the Bucket Brigade game.

### Parameter Vector Definition

All archetypes use the same 10-parameter structure:

| Index | Parameter | Range | Description |
|-------|-----------|-------|-------------|
| 0 | `honesty_bias` | [0, 1] | Truthfulness in signaling |
| 1 | `work_tendency` | [0, 1] | Propensity to work vs rest |
| 2 | `neighbor_help_bias` | [0, 1] | Willingness to help neighbors |
| 3 | `own_house_priority` | [0, 1] | Self-interest vs collective focus |
| 4 | `risk_aversion` | [0, 1] | Cautiousness in decision-making |
| 5 | `coordination_weight` | [0, 1] | Trust in others' signals |
| 6 | `exploration_rate` | [0, 1] | Randomness in actions |
| 7 | `fatigue_memory` | [0, 1] | Behavioral consistency over time |
| 8 | `rest_reward_bias` | [0, 1] | Preference for resting |
| 9 | `altruism_factor` | [0, 1] | Cooperation motivation |

### Archetype Catalog

#### 1. Firefighter
**Behavioral Profile**: Honest, hard-working, cooperative agent representing professional firefighting mentality.

```python
[1.0,  # honesty_bias - always truthful
 0.9,  # work_tendency - works most nights
 0.5,  # neighbor_help_bias - balanced
 0.8,  # own_house_priority - prioritizes own house
 0.5,  # risk_aversion - moderate
 0.7,  # coordination_weight - trusts signals
 0.1,  # exploration_rate - low randomness
 0.0,  # fatigue_memory - no inertia
 0.0,  # rest_reward_bias - doesn't prefer rest
 0.8]  # altruism_factor - high cooperation
```

**Key Traits**: High honesty (1.0), high work tendency (0.9), high altruism (0.8)
**Use Case**: Baseline cooperative strategy; professional ideal

#### 2. Free Rider
**Behavioral Profile**: Selfish agent that avoids work and free-rides on others' efforts.

```python
[0.7,  # honesty_bias - mostly truthful
 0.2,  # work_tendency - avoids work
 0.0,  # neighbor_help_bias - doesn't help neighbors
 0.9,  # own_house_priority - only cares about own house
 0.0,  # risk_aversion - not concerned with community fires
 0.0,  # coordination_weight - ignores signals
 0.1,  # exploration_rate - low randomness
 0.0,  # fatigue_memory - no inertia
 0.9,  # rest_reward_bias - strongly prefers rest
 0.0]  # altruism_factor - no altruism
```

**Key Traits**: Low work tendency (0.2), zero neighbor help (0.0), zero altruism (0.0)
**Use Case**: Testing worst-case defection; social dilemma baseline

#### 3. Hero
**Behavioral Profile**: Maximum effort and maximum cooperation representing ideal altruistic behavior.

```python
[1.0,  # honesty_bias - always truthful
 1.0,  # work_tendency - always works
 1.0,  # neighbor_help_bias - helps everyone
 0.5,  # own_house_priority - balanced
 0.1,  # risk_aversion - brave
 0.5,  # coordination_weight - moderate trust
 0.0,  # exploration_rate - no randomness
 0.9,  # fatigue_memory - consistent behavior
 0.0,  # rest_reward_bias - never rests
 1.0]  # altruism_factor - maximum altruism
```

**Key Traits**: Maximum work (1.0), maximum neighbor help (1.0), maximum altruism (1.0)
**Use Case**: Upper bound on cooperation; Nash equilibrium in chain_reaction scenario

#### 4. Coordinator
**Behavioral Profile**: Balanced, trust-based strategy emphasizing coordination and communication.

```python
[0.9,  # honesty_bias - mostly truthful
 0.6,  # work_tendency - moderate work
 0.7,  # neighbor_help_bias - cooperative
 0.6,  # own_house_priority - balanced
 0.8,  # risk_aversion - cautious
 1.0,  # coordination_weight - high trust in signals
 0.05, # exploration_rate - very low randomness
 0.0,  # fatigue_memory - no inertia
 0.2,  # rest_reward_bias - slight rest preference
 0.6]  # altruism_factor - moderate altruism
```

**Key Traits**: Maximum coordination weight (1.0), high honesty (0.9), balanced work/altruism (0.6)
**Use Case**: Communication-focused strategy; Nash equilibrium in multiple scenarios

#### 5. Liar
**Behavioral Profile**: Deceptive agent with selfish motives who exploits trust.

```python
[0.1,  # honesty_bias - mostly dishonest
 0.7,  # work_tendency - works when beneficial
 0.0,  # neighbor_help_bias - no neighbor help
 0.9,  # own_house_priority - highly selfish
 0.2,  # risk_aversion - moderate
 0.8,  # coordination_weight - reads signals but lies
 0.3,  # exploration_rate - moderate randomness
 0.0,  # fatigue_memory - no inertia
 0.4,  # rest_reward_bias - moderate rest preference
 0.2]  # altruism_factor - low altruism
```

**Key Traits**: Very low honesty (0.1), high coordination reading (0.8) but deceptive signaling, low altruism (0.2)
**Use Case**: Testing vulnerability to deception; equilibrium in sparse_heroics scenario

### Archetype Usage in Research

| Archetype | Nash V2 Equilibria Count | Example Scenarios |
|-----------|--------------------------|-------------------|
| Coordinator | 0 | - |
| Firefighter | 0 | - |
| Hero | 1 | chain_reaction |
| Free Rider | 1 | sparse_heroics (Liar variant) |
| Liar | 0 | - |
| **Evolved Agent** | **7** | **77.8% of scenarios** |

**Key Insight**: Nash V2 results show that evolved agents are optimal in 7/9 scenarios, while predefined archetypes only win in 2/9 scenarios. This suggests:
1. Genetic algorithms discover superior strategies beyond human-designed archetypes
2. Archetypes serve as useful baselines but don't capture optimal play
3. Future research may benefit from additional archetypes informed by evolved strategies

---

## Scenarios

Scenarios define the game parameters (fire dynamics, rewards, costs) that shape strategic incentives.

### Scenario Parameter Structure

All scenarios share the same parameter structure:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `beta` | float | [0.02, 0.75] | Fire spread probability per neighbor |
| `kappa` | float | [0.3, 0.95] | Solo agent extinguish probability |
| `A` | float | [50, 500] | Team reward per saved house |
| `L` | float | [100, 500] | Team penalty per burned house |
| `c` | float | [0.01, 5.0] | Cost per worker per night |
| `rho_ignite` | float | [0.1, 0.4] | Initial fraction of houses burning |
| `N_min` | int | [8, 20] | Minimum nights before termination |
| `p_spark` | float | [0.0, 0.10] | Spontaneous ignition probability |
| `N_spark` | int | [0, 20] | Nights with sparks active |
| `num_agents` | int | [4] | Number of agents (typically 4) |

### Scenario Categories

Scenarios are organized into 5 research categories:

1. **Basic (3)**: General-purpose scenarios for initial testing
2. **Core (9)**: Primary research scenarios, thoroughly tested
3. **Phase 2A Extreme (9)**: Boundary testing with extreme parameters
4. **Phase 2A.1 Kappa/Spark Sweep (7)**: Investigation of trivial cooperation
5. **Phase 2D Mechanism Design (4)**: Cooperation-inducing scenario designs

---

## Basic Scenarios (3)

### default
**Purpose**: Standard balanced scenario for general testing
**Parameters**: β=0.25, κ=0.5, c=0.5, ρ=0.2
**Status**: ✅ Experiment directory exists
**Story**: Moderate fire spread and extinguish rates with typical cost structure.

### easy
**Purpose**: Low-difficulty scenario with favorable conditions
**Parameters**: β=0.1, κ=0.8, c=0.5, ρ=0.1
**Status**: ✅ Experiment directory exists
**Story**: Low spread, high extinguish rate - cooperation naturally rewarded.

### hard
**Purpose**: High-difficulty scenario with challenging conditions
**Parameters**: β=0.4, κ=0.3, c=0.5, ρ=0.3
**Status**: ✅ Experiment directory exists
**Story**: High spread, low extinguish rate - requires strong coordination.

---

## Core Scenarios (9)

These scenarios form the primary research dataset, with full experiment histories including Nash V1, Nash V2, and genetic evolution results.

### 1. trivial_cooperation
**Status**: ✅ Experiment directory exists
**Nash V2**: Evolved agent optimal (payoff: 26.5)
**Parameters**: β=0.15, κ=0.9, c=0.5, ρ=0.1, p_spark=0.0
**Story**: Fires are rare and extinguish easily. Cooperation should be straightforward.
**Research Question**: Why doesn't universal cooperation emerge?
**Finding**: Free-riding equilibrium persists even with trivial cooperation requirements.

### 2. early_containment
**Status**: ✅ Experiment directory exists
**Nash V2**: Evolved agent optimal (payoff: 64.9)
**Parameters**: β=0.35, κ=0.6, c=0.5, ρ=0.3, p_spark=0.02
**Story**: Fires start aggressive but can be stopped with early teamwork.
**Research Question**: Does urgency induce cooperation?
**Finding**: Evolved strategy balances immediate response with resource conservation.

### 3. greedy_neighbor
**Status**: ✅ Experiment directory exists
**Nash V2**: Evolved agent optimal (payoff: 64.9)
**Parameters**: β=0.15, κ=0.4, c=1.0, ρ=0.2, p_spark=0.02
**Story**: Social dilemma between self-interest and cooperation due to high work cost.
**Research Question**: At what work cost does cooperation collapse?
**Finding**: Even with c=1.0 (double normal), evolved agents maintain cooperation.

### 4. sparse_heroics
**Status**: ✅ Experiment directory exists
**Nash V2**: Liar archetype optimal (payoff: 67.2)
**Parameters**: β=0.1, κ=0.5, c=0.8, ρ=0.15, N_min=20
**Story**: Few workers can make the difference in long games with moderate costs.
**Research Question**: Can strategic work allocation outperform uniform cooperation?
**Finding**: Deceptive free-riding (Liar archetype) slightly outperforms evolved agents.

### 5. rest_trap
**Status**: ✅ Experiment directory exists
**Nash V2**: Evolved agent optimal (payoff: 64.9)
**Parameters**: β=0.05, κ=0.95, c=0.2, ρ=0.1, p_spark=0.02
**Story**: Fires usually extinguish themselves, creating temptation to rest.
**Research Question**: Does high self-extinguish rate induce free-riding?
**Finding**: Despite favorable conditions, evolved agents maintain vigilance.

### 6. chain_reaction
**Status**: ✅ Experiment directory exists
**Nash V2**: Hero archetype optimal (payoff: 803.9) 🌟
**Parameters**: β=0.45, κ=0.6, c=0.7, ρ=0.3, p_spark=0.03
**Story**: High spread requires distributed teams working simultaneously.
**Research Question**: Does crisis force cooperation beyond local optima?
**Finding**: Pure Hero cooperation (803.9) vastly outperforms evolved agents (68.8). This is the most dramatic Nash/Evolution gap, showing genetic algorithms can get stuck at local optima when global coordination is critical.

### 7. deceptive_calm
**Status**: ✅ Experiment directory exists
**Nash V2**: Evolved agent optimal (payoff: 48.6)
**Parameters**: β=0.25, κ=0.6, c=0.4, ρ=0.1, N_min=20, p_spark=0.05
**Story**: Occasional flare-ups reward honest signaling in long games.
**Research Question**: Does unpredictability favor honesty?
**Finding**: Evolved strategy balances vigilance with cost management.

### 8. overcrowding
**Status**: ✅ Experiment directory exists
**Nash V2**: Evolved agent optimal (payoff: 64.9)
**Parameters**: β=0.2, κ=0.3, A=50, c=0.6, ρ=0.1
**Story**: Too many workers reduce efficiency (low κ, low A).
**Research Question**: Does coordination failure emerge from overcrowding?
**Finding**: Evolved agents optimize work allocation despite diminishing returns.

### 9. mixed_motivation
**Status**: ✅ Experiment directory exists
**Nash V2**: Evolved agent optimal (payoff: 61.0)
**Parameters**: β=0.3, κ=0.5, c=0.6, ρ=0.2, N_min=15
**Story**: Ownership creates self-interest conflicts.
**Research Question**: How do individual incentives affect collective action?
**Finding**: Evolved strategy balances self-interest with team coordination.

---

---

## Removed Scenarios

**Note**: On 2025-11-06, we removed 20 exploratory scenarios from the registry to maintain a focused research scope of ~10 scenarios. These included:

- **9 Phase 2A extreme scenarios**: Parameter boundary testing (glacial_spread, explosive_spread, wildfire, etc.)
- **7 Phase 2A.1 scenarios**: Kappa and spark parameter sweeps of trivial_cooperation
- **4 Phase 2D scenarios**: Mechanism design experiments (nearly_free_work, front_loaded_crisis, etc.)

**Rationale**: These scenarios represented parameter sweeps and exploratory ideas that would be better implemented as programmatic experiment scripts rather than hardcoded scenario definitions. The core 12 scenarios already cover the meaningful cooperation dynamics space.

**For details**: See `docs/REMOVED_SCENARIOS_ARCHIVE.md` for complete descriptions and parameters of all removed scenarios.

---

## Consistency Analysis

### Parameter Consistency

✅ **Verified Consistency**:
- Python scenario definitions match experiment `config.json` files
- All archetypes use identical 10-parameter structure
- Parameter naming is consistent across codebase
- Nash V2 equilibrium files correctly reference scenario parameters

### Code Organization

✅ **Well-Structured**:
- `bucket_brigade/agents/archetypes.py` - Archetype definitions (126 lines)
- `bucket_brigade/envs/scenarios.py` - Scenario registry (810 lines)
- `experiments/scenarios/*/config.json` - Per-scenario configuration and research insights

### Documentation Cross-References

⚠️ **Could Be Improved**:
- Scenario docstrings don't reference related scenarios
- No explicit mapping between scenario phases and research goals
- Missing table of scenario parameter ranges for quick reference

---

## Gap Analysis

### Missing Archetypes

**Neutral/Random Baseline**:
- Purpose: Control for testing statistical significance of strategy differences
- Parameters: All 0.5 (uniform randomness)
- Use case: Null hypothesis baseline

**Evolved Archetypes**:
- Purpose: Catalog successful evolved strategies as named archetypes
- Opportunity: Convert the 7 evolved agents that won Nash V2 into archetypes
- Benefit: Create library of empirically validated strategies

### Scenario Coverage

| Category | In Registry | With Experiments | Status |
|----------|-------------|------------------|--------|
| Basic | 3 | 3 | ✅ Complete |
| Core | 9 | 9 | ✅ Complete |
| **Total** | **12** | **12** | **100% Coverage** |

**2025-11-06 Update**: Removed 20 exploratory scenarios to maintain focused scope. All scenarios in the registry now have full experiment coverage.

### Scenario Design Opportunities

**High-Cooperation Test Scenarios**:
- Parameters designed to make cooperation obviously optimal
- Test whether agents can recognize and exploit cooperative opportunities
- Example: Very low β, very high κ, very low c

**Communication Test Scenarios**:
- Parameters that make honesty/coordination critical
- Vary coordination_weight effectiveness
- Test vulnerability to deception

**Dynamic Scenarios**:
- Parameters that change over time (not currently supported)
- Would require environment changes but could test adaptation

---

## Recommendations

### Immediate Actions

1. **Add Neutral Archetype**:
   - Add to `archetypes.py`: All parameters = 0.5
   - Use as statistical baseline in all experiments

2. **Create Evolved Archetypes**:
   - Extract the 7 Nash V2 winning evolved agents
   - Add to archetype library with descriptive names
   - Document their behavioral characteristics

3. **Run Phase 2A.1 Experiments**:
   - Highest priority: Kappa/spark sweeps directly test Nash V2 findings
   - Create experiment directories and run full pipeline
   - Will clarify boundaries of free-riding equilibrium

4. **Run Phase 2D Experiments**:
   - Test mechanism design scenarios
   - Validate whether any scenario breaks universal free-riding
   - Critical for Phase 2 research goals

### Documentation Improvements

1. **Add Cross-Reference Tables**:
   - Scenario similarity matrix (parameter distances)
   - Archetype effectiveness by scenario type
   - Parameter sensitivity analysis summary

2. **Create Scenario Decision Tree**:
   - Help researchers choose appropriate scenarios
   - Based on research questions and parameter sensitivities

3. **Document Parameter Ranges**:
   - Observed min/max for each parameter across all scenarios
   - Typical values vs extreme values
   - Physical interpretations and constraints

### Research Opportunities

1. **Scenario Clustering**:
   - Cluster scenarios by parameter similarity
   - Identify redundancies and gaps in parameter space coverage

2. **Archetype Performance Analysis**:
   - Which archetypes excel in which scenario types?
   - Can we predict optimal archetype from scenario parameters?

3. **Evolved Strategy Analysis**:
   - What patterns distinguish Nash-optimal evolved agents?
   - Can we derive new archetypes from evolved strategies?
   - Are there behavioral motifs that transfer across scenarios?

---

## Appendix: Scenario Registry

Complete list of all 12 scenarios in `SCENARIO_REGISTRY`:

### Basic Scenarios (3)
1. ✅ `default` - Standard balanced scenario
2. ✅ `easy` - Low difficulty with favorable conditions
3. ✅ `hard` - High difficulty with challenging conditions

### Core Research Scenarios (9)
4. ✅ `trivial_cooperation` - Fires rare and easy (Nash V2: Evolved optimal, 26.5)
5. ✅ `early_containment` - Aggressive start, early teamwork (Nash V2: Evolved optimal, 64.9)
6. ✅ `greedy_neighbor` - High work cost social dilemma (Nash V2: Evolved optimal, 64.9)
7. ✅ `sparse_heroics` - Few workers make difference (Nash V2: Liar archetype, 67.2)
8. ✅ `rest_trap` - Self-extinguishing temptation (Nash V2: Evolved optimal, 64.9)
9. ✅ `chain_reaction` - High spread distributed teams 🌟 (Nash V2: Hero archetype, 803.9)
10. ✅ `deceptive_calm` - Occasional flare-ups (Nash V2: Evolved optimal, 48.6)
11. ✅ `overcrowding` - Too many workers (Nash V2: Evolved optimal, 64.9)
12. ✅ `mixed_motivation` - Ownership conflicts (Nash V2: Evolved optimal, 61.0)

**Note**: 20 exploratory scenarios were removed on 2025-11-06. See `docs/REMOVED_SCENARIOS_ARCHIVE.md` for details.

---

## Version History

- **v1.0** (2025-11-06): Initial audit covering all 5 archetypes and 32 scenarios
