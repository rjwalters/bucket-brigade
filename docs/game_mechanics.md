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

The reward system has two distinct phases:

#### Per-Night Rewards
Each night, agents receive immediate rewards based on their action choice:
```python
R_night_i = {
    -cost_to_work_one_night  # if agent worked
    +0.5                      # if agent rested
}
```

This creates an immediate energy/effort tradeoff each turn.

#### Final Rewards (Game End)
At game termination, agents receive rewards based on final house states:

**Team Component (Public Goods)**:
```python
team_reward = (saved_houses × team_reward_house_survives)
            - (ruined_houses × team_penalty_house_burns)
```

**Critical**: Each agent receives the FULL team_reward (not divided). This creates a classic public goods dilemma where individual incentives diverge from team incentives.

**Individual Components**:
```python
# Owned house outcome
own_house_reward = {
    +reward_own_house_survives   # if own house saved
    -penalty_own_house_burns     # if own house ruined
    0                             # if own house still burning
}

# Neighbor house outcomes (left and right neighbors)
neighbor_reward = sum([
    +reward_other_house_survives  # if neighbor saved
    -penalty_other_house_burns    # if neighbor ruined
    0                              # if neighbor still burning
] for neighbor in [left_neighbor, right_neighbor])

# Agent's final reward
R_final_i = team_reward + own_house_reward + neighbor_reward
```

**Total Accumulated Reward**:
```python
R_agent_i = sum(R_night_i for all nights) + R_final_i
```

#### Key Properties

1. **Public Goods Structure**: Each agent gets the FULL team reward, not a share. This means:
   - Free-riders benefit from others' work without cost
   - Individual rationality conflicts with collective welfare
   - Classic social dilemma emerges naturally

2. **Spatial Incentives**: Neighbor house rewards create spatial coordination incentives:
   - Agents care more about nearby fires
   - Distributed coverage is rewarded
   - Agents can't ignore distant fires entirely (team reward)

3. **Free-Rider Problem**:
   - Resting gives +0.5 per night
   - Working costs `cost_to_work_one_night` per night
   - Both get same team reward regardless of contribution
   - Optimal individual strategy may be resting (free-riding)

4. **Work/Rest Tradeoff**:
   - Short-term: Resting is better (+0.5 vs. -cost)
   - Long-term: Working improves team outcome (more houses saved)
   - Requires agents to sacrifice immediate reward for future benefit

#### Examples

**Scenario: Easy (default parameters)**
```python
# Parameters
cost_to_work_one_night = 0.15
team_reward_house_survives = 3.0
team_penalty_house_burns = 1.0
reward_own_house_survives = 5.0
penalty_own_house_burns = 10.0
reward_other_house_survives = 1.0
penalty_other_house_burns = 2.0

# Example game: 5 nights, 4 agents
# Agent 0: Worked 3 nights, Rested 2 nights
# Own house: Saved, Left neighbor: Ruined, Right neighbor: Saved
# Team outcome: 7 saved, 3 ruined

# Agent 0's reward:
per_night = (3 × -0.15) + (2 × 0.5) = -0.45 + 1.0 = 0.55
team = (7 × 3.0) - (3 × 1.0) = 21.0 - 3.0 = 18.0
own = 5.0  # saved
neighbors = -2.0 + 1.0 = -1.0  # left ruined, right saved
R_agent_0 = 0.55 + 18.0 + 5.0 - 1.0 = 22.55

# Agent 1: Rested all 5 nights
# Own house: Saved, Both neighbors: Saved
# (Same team outcome - gets full team reward despite free-riding!)

per_night = 5 × 0.5 = 2.5
team = 18.0  # Same as agent 0!
own = 5.0
neighbors = 1.0 + 1.0 = 2.0
R_agent_1 = 2.5 + 18.0 + 5.0 + 2.0 = 27.5

# Agent 1 earned MORE despite contributing nothing!
# This is the free-rider problem in action.
```

#### Implementation Reference

**Source of Truth**: `bucket-brigade-core/src/engine/rewards.rs`
- Lines 12-22: Per-night work/rest rewards
- Lines 30-33: Team reward calculation (public goods)
- Lines 40-41: Each agent receives FULL team reward
- Lines 43-61: Own house and neighbor house rewards

⚠️ **Note**: Python environment (`bucket_brigade/environment.py`) is DEPRECATED for research use. Use Rust implementation via PyO3 bindings. See `experiments/evolution/RUST_SINGLE_SOURCE_OF_TRUTH.md` for migration details.

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
