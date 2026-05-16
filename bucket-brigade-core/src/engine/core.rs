use super::types::StepResult;
use crate::rng::DeterministicRng;
use crate::scenarios::Scenario;
use crate::{Action, GameNight, HouseState};

/// Core Bucket Brigade game engine
pub struct BucketBrigade {
    pub(super) houses: Vec<HouseState>,
    pub(super) agent_positions: Vec<u8>,
    pub(super) agent_signals: Vec<u8>,
    pub(super) last_actions: Vec<Action>,
    pub(super) night: u32,
    pub(super) done: bool,
    pub(super) rewards: Vec<f32>,
    pub(super) rng: DeterministicRng,
    pub scenario: Scenario,
    pub num_agents: usize,
    /// Owner agent for each of the 10 houses (round-robin assignment).
    /// House `i` is owned by agent `i % num_agents`. Mirrors the Python
    /// `BucketBrigadeEnv.house_owners` array so that the per-house ownership
    /// reward fields (`reward_own_house_survives` etc.) can be applied per-agent.
    pub(super) house_owners: Vec<u8>,
    /// Per-agent "home position" on the 10-house ring (issue #203). Used by
    /// the rewards engine to scale the per-step work cost by the ring-arc
    /// distance from the agent's home to the house it works at. Derived from
    /// `scenario.agent_home_positions` when non-empty, otherwise from the
    /// existing `house_owners` round-robin (the first house each agent owns —
    /// agent `i` -> house `i`). When `scenario.distance_cost_alpha == 0.0`
    /// the cost contribution is zero, so pre-#203 scenarios are bit-exactly
    /// unchanged.
    pub(super) agent_home_positions: Vec<u8>,
    pub(super) trajectory: Vec<GameNight>,
}

/// Derive the per-agent home-position vector from a scenario.
///
/// * If `scenario.agent_home_positions` is non-empty, it is used directly
///   after a length and bounds check.
/// * Otherwise the round-robin `house_owners` assignment is reused (agent
///   `i`'s home is house `i`, since `i % num_agents == i` for `i < num_agents`).
fn derive_agent_home_positions(scenario: &Scenario, num_agents: usize) -> Vec<u8> {
    if scenario.agent_home_positions.is_empty() {
        // Backward-compat fallback: reuse the round-robin ownership anchor.
        (0..num_agents as u8).collect()
    } else {
        assert_eq!(
            scenario.agent_home_positions.len(),
            num_agents,
            "Scenario.agent_home_positions has length {} but num_agents = {}. \
             Per-agent home position vectors must match num_agents.",
            scenario.agent_home_positions.len(),
            num_agents
        );
        for (i, &pos) in scenario.agent_home_positions.iter().enumerate() {
            assert!(
                (pos as usize) < 10,
                "Scenario.agent_home_positions[{}] = {} is out of range [0, 10)",
                i,
                pos
            );
        }
        scenario.agent_home_positions.clone()
    }
}

/// Minimum-arc distance between two indices on a ring of length 10.
///
/// Used by the work-cost calculation in `engine/rewards.rs` (issue #203) and
/// exposed at crate-private scope for direct unit-testing of the spatial
/// cost term.
#[inline]
pub(super) fn ring_dist_10(a: u8, b: u8) -> u8 {
    let a = a as i32;
    let b = b as i32;
    let raw = (a - b).abs();
    raw.min(10 - raw) as u8
}

impl BucketBrigade {
    pub fn new(scenario: Scenario, num_agents: usize, seed: Option<u64>) -> Self {
        // Issue #222: fail fast on allowlisted-but-unrecognized fields.
        // `engine/rewards.rs` unconditionally uses ring-arc geometry, so an
        // unknown `distance_metric` would silently fall back to it. Mirroring
        // the Python `Scenario.__post_init__` validator here keeps every
        // engine-construction path (Rust literal, PyScenario kwargs, WASM JSON)
        // consistent.
        scenario
            .validate()
            .unwrap_or_else(|e| panic!("Invalid Scenario passed to BucketBrigade::new: {e}"));
        let agent_home_positions = derive_agent_home_positions(&scenario, num_agents);
        let mut engine = Self {
            houses: vec![0; 10],
            agent_positions: vec![0; num_agents],
            agent_signals: vec![0; num_agents],
            last_actions: vec![[0, 0, 0]; num_agents],
            night: 0,
            done: false,
            rewards: vec![0.0; num_agents],
            rng: DeterministicRng::new(seed),
            scenario,
            num_agents,
            house_owners: (0..10).map(|i| (i % num_agents) as u8).collect(),
            agent_home_positions,
            trajectory: Vec::new(),
        };
        engine.reset();
        engine
    }

    pub fn reset(&mut self) {
        self.houses = vec![0; 10];
        self.agent_positions = vec![0; self.num_agents];
        self.agent_signals = vec![0; self.num_agents];
        self.last_actions = vec![[0, 0, 0]; self.num_agents];
        self.night = 0;
        self.done = false;
        self.rewards = vec![0.0; self.num_agents];
        // Re-initialize house ownership in case num_agents changed via external mutation.
        self.house_owners = (0..10).map(|i| (i % self.num_agents) as u8).collect();
        // Re-derive home positions for the same reason (and to pick up any
        // scenario mutation).
        self.agent_home_positions = derive_agent_home_positions(&self.scenario, self.num_agents);
        self.trajectory = Vec::new();

        // Initialize fires - probabilistic per-house
        for house_idx in 0..10 {
            if self.rng.random() < self.scenario.prob_house_catches_fire {
                self.houses[house_idx] = 1;
            }
        }

        self.record_night();
    }

    pub fn step(&mut self, actions: &[Action]) -> StepResult {
        if self.done {
            panic!("Game is already finished");
        }

        // Store previous house states for reward calculation
        let prev_houses = self.houses.clone();

        // 1. Signal phase (issue #235): signals are now a first-class action
        // dimension. Each agent broadcasts `action[2]` independently of the
        // work/rest bit `action[1]`, enabling deceptive signaling. Pre-#235
        // this was `action[1]` (the work bit), so signals carried no
        // information beyond the action itself.
        self.agent_signals = actions.iter().map(|action| action[2]).collect();

        // 2. Action phase: update agent positions
        self.last_actions = actions.to_vec();
        self.agent_positions = actions.iter().map(|action| action[0]).collect();

        // 3. Extinguish phase
        // Agents respond to fires visible at start of turn
        self.extinguish_fires(actions);

        // 4. Burn-out phase
        // Unextinguished fires become ruined houses
        self.burn_out_houses();

        // 5. Spread phase
        // Fires spread to neighbors (visible next turn)
        self.spread_fires();

        // 6. Spontaneous ignition phase
        // New fires can ignite on any night (visible next turn)
        self.spontaneous_ignition();

        // 7. Compute rewards
        self.rewards = self.compute_rewards(actions, &prev_houses);

        // 8. Check termination
        self.done = self.check_termination();

        // 9. Record this night
        self.record_night();

        // 10. Advance to next night
        self.night += 1;

        StepResult {
            rewards: self.rewards.clone(),
            done: self.done,
            info: serde_json::json!({}),
        }
    }

    pub(super) fn check_termination(&self) -> bool {
        if self.night < self.scenario.min_nights {
            return false;
        }

        let all_safe = self.houses.iter().all(|&h| h == 0);
        let all_ruined = self.houses.iter().all(|&h| h == 2);

        all_safe || all_ruined
    }
}
