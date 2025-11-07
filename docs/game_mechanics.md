# Bucket Brigade: Core Game Mechanics

**Implementation Reference**: `bucket-brigade-core/src/` (Rust) is the canonical source of truth for all game mechanics.

## Overview

Bucket Brigade is a multi-agent cooperation game where 4-10 agents work together to save a town from fires while managing individual energy costs and incentives for free-riding.

## Game Elements

### World State
- **10 houses** arranged in a circle (positions 0-9)
- Each house has three possible states:
  - **Safe** (0): Undamaged house
  - **Burning** (1): House is on fire
  - **Ruined** (2): House is destroyed

### Agents
- **4-10 agents**, each owning exactly one house (assigned in round-robin fashion)
- Agents make decisions simultaneously each night
- Each agent has individual incentives that may conflict with team goals

## Turn Structure

Each night follows this sequence:

1. **Observation Phase**: Agents observe current fire state
2. **Signal Phase**: Agents broadcast intended action (Work or Rest)
3. **Action Phase**: Agents simultaneously choose (house, mode)
4. **Extinguish Phase**: Workers attempt to extinguish fires at their locations
5. **Burn-out Phase**: Unextinguished fires become ruined houses
6. **Spread Phase**: Fires spread to neighboring safe houses
7. **Spontaneous Ignition**: Random fires start on safe houses
8. **Reward Calculation**: Individual and team rewards computed
9. **Termination Check**: Game ends if conditions met

## Key Mechanics

### Fire Dynamics
- **Spread**: Burning houses ignite neighbors with probability `prob_fire_spreads_to_neighbor`
- **Spontaneous ignition**: Safe houses catch fire with probability `prob_house_catches_fire` on **every night** throughout the game
- **Extinguishing**: Uses independent probabilities model
  - Formula: `P(extinguish) = 1 - (1 - prob_solo_agent_extinguishes_fire)^num_workers`
  - Each worker has independent probability `prob_solo_agent_extinguishes_fire` of success
  - Example: With `prob_solo=0.5` and 2 workers: `P = 1 - (0.5)² = 0.75` (75% chance)

### Agent Decisions
- **Signal**: Choose to signal "Work" or "Rest" (cheap talk, may be deceptive)
- **Action**: Choose (house_index ∈ [0,9], mode ∈ {Work, Rest})

### Rewards
- **Team reward**: Based on houses saved/ruined and total work performed
- **Individual reward**: Team share + own house bonus - work costs
- **Social dilemma**: Individual incentives may encourage free-riding

## Termination Conditions

Game ends when:
- `night ≥ min_nights` AND (`all_safe` OR `all_ruined` OR `night ≥ 100`)

## Scenarios

The game includes 12 predefined scenarios with different parameter combinations, each testing different aspects of cooperation:

**Difficulty Levels**:
- `default` - Balanced baseline scenario
- `easy` - Low fire spread, high extinguish efficiency (training)
- `hard` - High fire spread, low extinguish efficiency (challenge)

**Research Scenarios**:
- `trivial_cooperation` - Pure cooperation (easy fires, no social dilemma)
- `early_containment` - Aggressive fires requiring fast coordination
- `greedy_neighbor` - High work cost creates free-rider incentive
- `sparse_heroics` - Few workers can make the difference
- `rest_trap` - Fires usually self-extinguish, but not always
- `chain_reaction` - High spread requires distributed teams
- `deceptive_calm` - Occasional flare-ups reward honest signaling
- `overcrowding` - Too many workers reduce efficiency
- `mixed_motivation` - Ownership creates self-interest conflicts

**Canonical Definition**: See `bucket-brigade-core/src/scenarios.rs` for authoritative parameter values.

---

*This document provides the canonical description of Bucket Brigade game mechanics. The authoritative implementation is in `bucket-brigade-core/src/`. Reference this from other documents to ensure consistency.*
