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
    /// Owner agent for each house on the ring (round-robin assignment).
    /// House `i` is owned by agent `i % num_agents`. Mirrors the Python
    /// `BucketBrigadeEnv.house_owners` array so that the per-house ownership
    /// reward fields (`reward_own_house_survives` etc.) can be applied per-agent.
    ///
    /// Length matches `scenario.num_houses` (issue #254). The default for
    /// every pre-#254 scenario is 10; new scenarios like `v2_minimal` use
    /// a smaller ring.
    pub(super) house_owners: Vec<u8>,
    /// Per-agent "home position" on the ring (issue #203). Used by the
    /// rewards engine to scale the per-step work cost by the ring-arc
    /// distance from the agent's home to the house it works at. Derived from
    /// `scenario.agent_home_positions` when non-empty, otherwise from the
    /// existing `house_owners` round-robin (the first house each agent owns —
    /// agent `i` -> house `i`). When `scenario.distance_cost_alpha == 0.0`
    /// the cost contribution is zero, so pre-#203 scenarios are bit-exactly
    /// unchanged.
    ///
    /// Positions are validated against `scenario.num_houses` (issue #254).
    pub(super) agent_home_positions: Vec<u8>,
    /// Per-house accumulated suppression progress (issue #253). Used only
    /// by the `"continuous"` extinguish mode: each work step at a burning
    /// house adds `scenario.suppression_per_worker * workers_here` and the
    /// fire transitions BURNING -> SAFE when the accumulator reaches 1.0.
    /// Zeroed on ignition and burn-out so per-fire progress does not leak
    /// across the same-house fire cycle. In the default `"bernoulli"`
    /// mode the vector is allocated to scenario size but never written —
    /// the extra storage is `4 * num_houses` bytes per env, negligible
    /// against the existing `Vec<HouseState>` and trajectory storage.
    ///
    /// Length matches `scenario.num_houses` (issue #254).
    pub(super) fire_progress: Vec<f32>,
    pub(super) trajectory: Vec<GameNight>,
}

/// Derive the per-agent home-position vector from a scenario.
///
/// * If `scenario.agent_home_positions` is non-empty, it is used directly
///   after a length and bounds check.
/// * Otherwise the round-robin `house_owners` assignment is reused (agent
///   `i`'s home is house `i`, since `i % num_agents == i` for `i < num_agents`).
fn derive_agent_home_positions(scenario: &Scenario, num_agents: usize) -> Vec<u8> {
    let num_houses = scenario.num_houses as usize;
    if scenario.agent_home_positions.is_empty() {
        // Backward-compat fallback: reuse the round-robin ownership anchor.
        // Issue #254: wrap modulo `num_houses` so this stays in-range when
        // `num_agents > num_houses` (e.g. `v2_minimal` with 4 agents on
        // 2 houses gives positions [0, 1, 0, 1]). For pre-#254 scenarios
        // where `num_agents <= num_houses`, `i % num_houses == i` so the
        // result is identical to the pre-#254 `(0..num_agents)` derivation.
        (0..num_agents).map(|i| (i % num_houses) as u8).collect()
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
                (pos as usize) < num_houses,
                "Scenario.agent_home_positions[{}] = {} is out of range [0, {})",
                i,
                pos,
                num_houses
            );
        }
        scenario.agent_home_positions.clone()
    }
}

/// Minimum-arc distance between two indices on a ring of length `n`.
///
/// Used by the work-cost calculation in `engine/rewards.rs` (issue #203) and
/// exposed at crate-private scope for direct unit-testing of the spatial
/// cost term. Issue #254 generalized this from a hardcoded length-10 ring
/// (`ring_dist_10`) to a length parameter so non-10-house scenarios work
/// correctly. The `ring_dist_10` wrapper below is kept for callers that
/// always run on the canonical 10-house ring.
#[inline]
pub(super) fn ring_dist(n: u8, a: u8, b: u8) -> u8 {
    let n = n as i32;
    let a = a as i32;
    let b = b as i32;
    let raw = (a - b).abs();
    raw.min(n - raw) as u8
}

/// Backward-compatible wrapper around [`ring_dist`] pinned to the 10-house
/// ring. Existing call sites that always run on the pre-#254 fixed-10 ring
/// can keep using this; new code should call `ring_dist(n, a, b)` directly.
#[inline]
#[allow(dead_code)]
pub(super) fn ring_dist_10(a: u8, b: u8) -> u8 {
    ring_dist(10, a, b)
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
        let num_houses = scenario.num_houses as usize;
        assert!(
            num_houses >= 2,
            "Scenario.num_houses must be at least 2 (got {}) — the engine \
             spread phase assumes a real ring with distinct neighbors",
            num_houses
        );
        // Note: we intentionally do NOT require `num_agents <= num_houses`.
        // `v2_minimal` (issue #254) has 4 agents on 2 houses — agents 2 and
        // 3 are "unowned-house workers" under the round-robin
        // `house_owners[i] = i % num_agents` semantics. This matches the
        // architect's option-E spec; if the user wants exclusive 1:1
        // ownership they should set `num_agents <= num_houses` themselves.
        let agent_home_positions = derive_agent_home_positions(&scenario, num_agents);
        let mut engine = Self {
            houses: vec![0; num_houses],
            agent_positions: vec![0; num_agents],
            agent_signals: vec![0; num_agents],
            last_actions: vec![[0, 0, 0]; num_agents],
            night: 0,
            done: false,
            rewards: vec![0.0; num_agents],
            rng: DeterministicRng::new(seed),
            scenario,
            num_agents,
            house_owners: (0..num_houses).map(|i| (i % num_agents) as u8).collect(),
            agent_home_positions,
            // Issue #253: zero-initialized; only written by the continuous
            // extinguish branch.
            fire_progress: vec![0.0; num_houses],
            trajectory: Vec::new(),
        };
        engine.reset();
        engine
    }

    pub fn reset(&mut self) {
        let num_houses = self.scenario.num_houses as usize;
        self.houses = vec![0; num_houses];
        self.agent_positions = vec![0; self.num_agents];
        self.agent_signals = vec![0; self.num_agents];
        self.last_actions = vec![[0, 0, 0]; self.num_agents];
        self.night = 0;
        self.done = false;
        self.rewards = vec![0.0; self.num_agents];
        // Re-initialize house ownership in case num_agents changed via external mutation.
        self.house_owners = (0..num_houses)
            .map(|i| (i % self.num_agents) as u8)
            .collect();
        // Re-derive home positions for the same reason (and to pick up any
        // scenario mutation).
        self.agent_home_positions = derive_agent_home_positions(&self.scenario, self.num_agents);
        // Issue #253: zero the per-house suppression accumulator. Done
        // before `_initialize_houses` so any subsequent ignition starts
        // with `fire_progress[i] = 0.0` (the continuous-mode dispatch
        // also zeroes on ignition, but resetting up front keeps the
        // invariant explicit).
        self.fire_progress = vec![0.0; num_houses];
        self.trajectory = Vec::new();

        // Initialize fires - probabilistic per-house
        for house_idx in 0..num_houses {
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

        // 0. Action-validity sanitization (issue #251). When
        //    `action_validity_mode == "adjacent_only"`, rewrite any agent
        //    action whose target house is more than ring-distance 1 from the
        //    agent's home position to that home position (a no-op move into
        //    home). The mode bit and signal bit are preserved. When the mode
        //    is the default `"always_valid"` this is a true bit-exact no-op:
        //    `sanitize_actions` returns a clone of the input slice. Every
        //    downstream phase (extinguish, rewards, observation) consumes the
        //    sanitized actions, not the raw input, so policies can't cheat
        //    the constraint by separately attacking different phases.
        let sanitized = self.sanitize_actions(actions);

        // 1. Signal phase (issue #235): signals are now a first-class action
        // dimension. Each agent broadcasts `action[2]` independently of the
        // work/rest bit `action[1]`, enabling deceptive signaling. Pre-#235
        // this was `action[1]` (the work bit), so signals carried no
        // information beyond the action itself.
        self.agent_signals = sanitized.iter().map(|action| action[2]).collect();

        // 2. Action phase: update agent positions
        self.last_actions = sanitized.clone();
        self.agent_positions = sanitized.iter().map(|action| action[0]).collect();

        // 3. Extinguish phase
        // Agents respond to fires visible at start of turn
        self.extinguish_fires(&sanitized);

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
        self.rewards = self.compute_rewards(&sanitized, &prev_houses);

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

    /// Apply the per-agent position-constrained action mask (issue #251).
    ///
    /// When `scenario.action_validity_mode == "always_valid"` (the default)
    /// this is a clone-and-return — every house index is a valid target for
    /// every agent. Pre-#251 scenarios hit this branch and are bit-exact.
    ///
    /// When the mode is `"adjacent_only"`, agent `i` may target any house
    /// `j` with `ring_dist(num_houses, agent_home_positions[i], j) <= 1`.
    /// Out-of-reach targets are rewritten to `agent_home_positions[i]` (a
    /// no-op move into home that preserves the boundary gradient — the
    /// policy can still choose meaningfully between home and adjacent
    /// reach). The mode bit (`action[1]`) and signal bit (`action[2]`) are
    /// untouched, so a policy that targets an out-of-reach house can still
    /// WORK or REST and can still broadcast any signal — just from home.
    fn sanitize_actions(&self, actions: &[Action]) -> Vec<Action> {
        if self.scenario.action_validity_mode == "always_valid" {
            // Hot path: clone and return. This is the pre-#251 codepath.
            return actions.to_vec();
        }
        // `adjacent_only`: rewrite out-of-reach house indices to home.
        // The allowlist (`scenarios.rs::ALLOWED_ACTION_VALIDITY_MODES`) +
        // `Scenario::validate` chokepoint guarantee no other mode reaches
        // this branch.
        let num_houses = self.scenario.num_houses;
        let mut sanitized = actions.to_vec();
        for (agent_idx, action) in sanitized.iter_mut().enumerate() {
            // Defensive: agents beyond `self.num_agents` shouldn't exist,
            // but guard the array bound rather than panicking.
            if agent_idx >= self.agent_home_positions.len() {
                break;
            }
            let home = self.agent_home_positions[agent_idx];
            if ring_dist(num_houses, home, action[0]) > 1 {
                action[0] = home;
            }
        }
        sanitized
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
