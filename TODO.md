# Issue #53 - Remaining Implementation Work

## Overview

This document tracks the remaining implementation work for renaming scenario parameters from terse mathematical symbols to self-documenting verbose names with `prob_` prefixes.

**Status**: `scenarios.rs` is complete and committed. The code will not compile until all files below are updated.

**Commit**: f3056f5 - "WIP: Update Scenario struct with improved parameter names"

## Completed ✅

- [x] `bucket-brigade-core/src/scenarios.rs` - All parameter names updated, serde aliases added for backward compatibility
- [x] All 4 predefined scenarios updated with new parameter names and adjusted values

## Remaining Work

### 1. Core Engine Updates ❌

**File**: `bucket-brigade-core/src/engine.rs`

**Required Changes**:

#### A. Update all parameter references (~41 occurrences)

Replace old names with new names throughout the file:
- `beta` → `prob_fire_spreads_to_neighbor`
- `kappa` → `prob_solo_agent_extinguishes_fire`
- `rho_ignite` / `p_spark` → `prob_house_catches_fire`
- `a` → `team_reward_house_survives`
- `l` → `team_penalty_house_burns`
- `c` → `cost_to_work_one_night`
- `n_min` → `min_nights`
- `a_own` → `reward_own_house_survives`
- `a_neighbor` → `reward_other_house_survives`
- Add references to: `penalty_own_house_burns`, `penalty_other_house_burns`

#### B. Change reset() method - Probabilistic initialization

**Old behavior** (deterministic):
```rust
// Lines ~95-102
let num_burning = (self.scenario.rho_ignite * 10.0).round() as usize;
let mut burn_indices = std::collections::HashSet::new();
while burn_indices.len() < num_burning {
    burn_indices.insert(self.rng.randint(0, 10));
}
for idx in burn_indices {
    self.houses[idx] = 1;
}
```

**New behavior** (probabilistic per-house):
```rust
// Each house independently has prob_house_catches_fire chance of starting on fire
for house_idx in 0..10 {
    if self.rng.random() < self.scenario.prob_house_catches_fire {
        self.houses[house_idx] = 1;
    }
}
```

**Rationale**: Simplification - removes special case, makes night 0 consistent with ongoing behavior.

#### C. Change extinguish formula - Independent probabilities

**Old formula** (exponential):
```rust
// Line ~172
let p_extinguish = 1.0 - (-self.scenario.kappa * workers_here as f32).exp();
```

**New formula** (independent probability multiplication):
```rust
let p_extinguish = 1.0 - (1.0 - self.scenario.prob_solo_agent_extinguishes_fire).powi(workers_here as i32);
```

**Rationale**: Makes the parameter meaning clearer - "probability that one agent extinguishes the fire". Multiple agents' efforts are combined as independent probabilities.

**Note**: Scenario values were adjusted to maintain similar game balance:
- `kappa=0.9` → `prob_solo_agent_extinguishes_fire=0.7`
- `kappa=0.6` → `prob_solo_agent_extinguishes_fire=0.45`
- `kappa=0.4` → `prob_solo_agent_extinguishes_fire=0.33`
- `kappa=0.5` → `prob_solo_agent_extinguishes_fire=0.39`

#### D. Remove spontaneous_ignition_nights conditional

**Old behavior** (limited duration):
```rust
// Lines ~245-247
if self.night < self.scenario.spontaneous_ignition_nights {
    self.spark_fires();
}
```

**New behavior** (always applies):
```rust
self.spontaneous_ignition();  // Runs every night
```

**Also rename the method**:
```rust
// Old name
fn spark_fires(&mut self) { ... }

// New name
fn spontaneous_ignition(&mut self) {
    for house_idx in 0..10 {
        if self.houses[house_idx] == 0 && self.rng.random() < self.scenario.prob_house_catches_fire {
            self.houses[house_idx] = 1;
        }
    }
}
```

**Rationale**: Simplification - fires can ignite throughout the game, not just in early rounds.

### 2. Python Bindings ❌

**File**: `bucket-brigade-core/src/python.rs`

**Required Changes**:

#### A. Update PyScenario constructor

```rust
#[pymethods]
impl PyScenario {
    #[new]
    fn new(
        prob_fire_spreads_to_neighbor: f32,
        prob_solo_agent_extinguishes_fire: f32,
        prob_house_catches_fire: f32,
        team_reward_house_survives: f32,
        team_penalty_house_burns: f32,
        cost_to_work_one_night: f32,
        min_nights: u32,
        num_agents: usize,
        reward_own_house_survives: f32,
        reward_other_house_survives: f32,
        penalty_own_house_burns: f32,
        penalty_other_house_burns: f32,
    ) -> Self {
        PyScenario {
            inner: Scenario {
                prob_fire_spreads_to_neighbor,
                prob_solo_agent_extinguishes_fire,
                prob_house_catches_fire,
                team_reward_house_survives,
                team_penalty_house_burns,
                cost_to_work_one_night,
                min_nights,
                num_agents,
                reward_own_house_survives,
                reward_other_house_survives,
                penalty_own_house_burns,
                penalty_other_house_burns,
            }
        }
    }
```

#### B. Update all 12 getter methods

```rust
#[getter]
fn prob_fire_spreads_to_neighbor(&self) -> f32 {
    self.inner.prob_fire_spreads_to_neighbor
}

#[getter]
fn prob_solo_agent_extinguishes_fire(&self) -> f32 {
    self.inner.prob_solo_agent_extinguishes_fire
}

#[getter]
fn prob_house_catches_fire(&self) -> f32 {
    self.inner.prob_house_catches_fire
}

#[getter]
fn team_reward_house_survives(&self) -> f32 {
    self.inner.team_reward_house_survives
}

#[getter]
fn team_penalty_house_burns(&self) -> f32 {
    self.inner.team_penalty_house_burns
}

#[getter]
fn cost_to_work_one_night(&self) -> f32 {
    self.inner.cost_to_work_one_night
}

#[getter]
fn min_nights(&self) -> u32 {
    self.inner.min_nights
}

#[getter]
fn num_agents(&self) -> usize {
    self.inner.num_agents
}

#[getter]
fn reward_own_house_survives(&self) -> f32 {
    self.inner.reward_own_house_survives
}

#[getter]
fn reward_other_house_survives(&self) -> f32 {
    self.inner.reward_other_house_survives
}

#[getter]
fn penalty_own_house_burns(&self) -> f32 {
    self.inner.penalty_own_house_burns
}

#[getter]
fn penalty_other_house_burns(&self) -> f32 {
    self.inner.penalty_other_house_burns
}
```

### 3. WASM Bindings ❌

**File**: `bucket-brigade-core/src/wasm.rs`

**Required Changes**:

Update WasmScenario constructor with new parameter names (similar to python.rs):

```rust
#[wasm_bindgen]
impl WasmScenario {
    #[wasm_bindgen(constructor)]
    pub fn new(
        prob_fire_spreads_to_neighbor: f32,
        prob_solo_agent_extinguishes_fire: f32,
        prob_house_catches_fire: f32,
        team_reward_house_survives: f32,
        team_penalty_house_burns: f32,
        cost_to_work_one_night: f32,
        min_nights: u32,
        num_agents: usize,
        reward_own_house_survives: f32,
        reward_other_house_survives: f32,
        penalty_own_house_burns: f32,
        penalty_other_house_burns: f32,
    ) -> WasmScenario { ... }
}
```

### 4. Python Test Updates ❌

**File**: `tests/test_rust_integration.py`

**Lines 173-177** - Update attribute names:

```python
# Old
assert rust_scenario.fire_spread_prob == python_scenario.beta
assert rust_scenario.extinguish_efficiency == python_scenario.kappa
assert rust_scenario.team_reward_per_house == python_scenario.A
assert rust_scenario.team_penalty_per_house == python_scenario.L

# New
assert rust_scenario.prob_fire_spreads_to_neighbor == python_scenario.beta
assert rust_scenario.prob_solo_agent_extinguishes_fire == python_scenario.kappa
assert rust_scenario.team_reward_house_survives == python_scenario.A
assert rust_scenario.team_penalty_house_burns == python_scenario.L
```

**File**: `test_rust_core.py`

**Line 18** - Update attribute names:

```python
# Old
print(f"✅ Loaded scenario: {scenario.fire_spread_prob}, {scenario.extinguish_efficiency}")

# New
print(f"✅ Loaded scenario: {scenario.prob_fire_spreads_to_neighbor}, {scenario.prob_solo_agent_extinguishes_fire}")
```

### 5. Testing ❌

Once all files are updated:

```bash
# Run Rust tests (52+ tests should pass)
cd bucket-brigade-core
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test

# Build Python extension
cd bucket-brigade-core
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install -e .

# Run Python integration tests
cd ..
python test_rust_core.py
pytest tests/test_rust_integration.py -v

# Run full test suite
make test-python
```

### 6. Final Steps ❌

- [ ] Update PR description to reflect all changes
- [ ] Ensure all tests pass
- [ ] Push changes to origin
- [ ] Request review with `loom:review-requested` label

## Parameter Name Mapping Reference

| Old Name | New Name | Type | Description |
|----------|----------|------|-------------|
| `beta` | `prob_fire_spreads_to_neighbor` | f32 | Probability fire spreads to adjacent house |
| `kappa` | `prob_solo_agent_extinguishes_fire` | f32 | Probability one agent extinguishes fire |
| `rho_ignite`, `p_spark` | `prob_house_catches_fire` | f32 | Probability house catches fire (any night) |
| `a` | `team_reward_house_survives` | f32 | Team reward for each house that survives |
| `l` | `team_penalty_house_burns` | f32 | Team penalty for each house that burns |
| `a_own` | `reward_own_house_survives` | f32 | Individual reward when own house survives |
| `a_neighbor` | `reward_other_house_survives` | f32 | Individual reward when other house survives |
| (new) | `penalty_own_house_burns` | f32 | Individual penalty when own house burns |
| (new) | `penalty_other_house_burns` | f32 | Individual penalty when other house burns |
| `c` | `cost_to_work_one_night` | f32 | Cost incurred when agent chooses to work |
| `n_min` | `min_nights` | u32 | Minimum nights before game can end |
| `n_spark` | (removed) | - | No longer needed with continuous ignition |

## Behavior Changes

1. **Probabilistic Initialization**: Each house independently has `prob_house_catches_fire` chance of starting on fire (night 0)
2. **Independent Probability Extinguish**: Formula changed from `1 - e^(-kappa*n)` to `1 - (1-p)^n`
3. **Continuous Spontaneous Ignition**: Fires can ignite on any night, not just early rounds

## Notes

- All old parameter names maintain backward compatibility via serde aliases
- Scenario values were adjusted to maintain similar game balance with new formula
- The ownership penalty parameters prepare for issue #52 (individual agent scoring)
