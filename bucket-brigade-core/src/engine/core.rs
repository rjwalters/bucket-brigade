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
    pub(super) trajectory: Vec<GameNight>,
}

impl BucketBrigade {
    pub fn new(scenario: Scenario, seed: Option<u64>) -> Self {
        let mut engine = Self {
            houses: vec![0; 10],
            agent_positions: vec![0; scenario.num_agents],
            agent_signals: vec![0; scenario.num_agents],
            last_actions: vec![[0, 0]; scenario.num_agents],
            night: 0,
            done: false,
            rewards: vec![0.0; scenario.num_agents],
            rng: DeterministicRng::new(seed),
            scenario,
            trajectory: Vec::new(),
        };
        engine.reset();
        engine
    }

    pub fn reset(&mut self) {
        self.houses = vec![0; 10];
        self.agent_positions = vec![0; self.scenario.num_agents];
        self.agent_signals = vec![0; self.scenario.num_agents];
        self.last_actions = vec![[0, 0]; self.scenario.num_agents];
        self.night = 0;
        self.done = false;
        self.rewards = vec![0.0; self.scenario.num_agents];
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

        // 1. Signal phase (signals are implicit in actions for now)
        self.agent_signals = actions.iter().map(|action| action[1]).collect();

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
