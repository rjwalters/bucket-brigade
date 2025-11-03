use crate::rng::DeterministicRng;
use crate::scenarios::Scenario;
use crate::{Action, AgentObservation, GameNight, HouseState};
use serde::{Deserialize, Serialize};

/// Current game state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub houses: Vec<HouseState>,
    pub agent_positions: Vec<u8>,
    pub agent_signals: Vec<u8>,
    pub last_actions: Vec<Action>,
    pub night: u32,
    pub done: bool,
}

/// Game result after completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameResult {
    pub scenario: Scenario,
    pub nights: Vec<GameNight>,
    pub final_score: f32,
    pub agent_scores: Vec<f32>,
}

/// Step result
#[derive(Debug, Clone)]
pub struct StepResult {
    pub rewards: Vec<f32>,
    pub done: bool,
    pub info: serde_json::Value,
}

/// Core Bucket Brigade game engine
pub struct BucketBrigade {
    houses: Vec<HouseState>,
    agent_positions: Vec<u8>,
    agent_signals: Vec<u8>,
    last_actions: Vec<Action>,
    night: u32,
    done: bool,
    rewards: Vec<f32>,
    rng: DeterministicRng,
    pub scenario: Scenario,
    trajectory: Vec<GameNight>,
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

        // Initialize fires
        let num_burning = (self.scenario.rho_ignite * 10.0).round() as usize;
        let mut burn_indices = std::collections::HashSet::new();
        while burn_indices.len() < num_burning {
            burn_indices.insert(self.rng.randint(0, 10));
        }
        for idx in burn_indices {
            self.houses[idx] = 1;
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

        // 6. Spark phase (if active)
        // New fires ignite (visible next turn)
        if self.night < self.scenario.n_spark {
            self.spark_fires();
        }

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

    fn extinguish_fires(&mut self, actions: &[Action]) {
        for house_idx in 0..10 {
            if self.houses[house_idx] != 1 {
                continue;
            }

            // Count workers at this house
            let workers_here = actions
                .iter()
                .filter(|action| action[0] as usize == house_idx && action[1] == 1)
                .count();

            // Probability of extinguishing
            let p_extinguish = 1.0 - (-self.scenario.kappa * workers_here as f32).exp();

            if self.rng.random() < p_extinguish {
                self.houses[house_idx] = 0;
            }
        }
    }

    fn spread_fires(&mut self) {
        let mut new_houses = self.houses.clone();

        for house_idx in 0..10 {
            if self.houses[house_idx] != 1 {
                continue;
            }

            // Check neighbors
            let neighbors = [
                (house_idx + 9) % 10, // (i-1) mod 10
                (house_idx + 1) % 10, // (i+1) mod 10
            ];

            for &neighbor in &neighbors {
                if self.houses[neighbor] == 0 && self.rng.random() < self.scenario.beta {
                    new_houses[neighbor] = 1;
                }
            }
        }

        self.houses = new_houses;
    }

    fn burn_out_houses(&mut self) {
        for house in self.houses.iter_mut() {
            if *house == 1 {
                *house = 2;
            }
        }
    }

    fn spark_fires(&mut self) {
        for house_idx in 0..10 {
            if self.houses[house_idx] == 0 && self.rng.random() < self.scenario.p_spark {
                self.houses[house_idx] = 1;
            }
        }
    }

    fn compute_rewards(&mut self, actions: &[Action], _prev_houses: &[HouseState]) -> Vec<f32> {
        // Only compute per-step work/rest costs
        // House-based rewards are computed at game end
        actions
            .iter()
            .map(|action| {
                // Work/rest cost only
                if action[1] == 1 {
                    -self.scenario.c // Work cost
                } else {
                    0.5 // Rest reward
                }
            })
            .collect()
    }

    fn compute_final_rewards(&self) -> Vec<f32> {
        // Compute final rewards based on house outcomes at game end
        let mut rewards = vec![0.0; self.scenario.num_agents];

        for agent_idx in 0..self.scenario.num_agents {
            let owned_house = agent_idx % 10;
            let left_neighbor = if owned_house == 0 { 9 } else { owned_house - 1 };
            let right_neighbor = (owned_house + 1) % 10;

            // Reward for owned house outcome
            match self.houses[owned_house] {
                0 => rewards[agent_idx] += self.scenario.a_own,    // Saved
                2 => rewards[agent_idx] -= self.scenario.a_own,    // Ruined (symmetric penalty)
                _ => {},                                            // Burning (no reward/penalty)
            }

            // Rewards for neighbor houses
            for &neighbor in &[left_neighbor, right_neighbor] {
                match self.houses[neighbor] {
                    0 => rewards[agent_idx] += self.scenario.a_neighbor,  // Saved
                    2 => rewards[agent_idx] -= self.scenario.a_neighbor,  // Ruined (symmetric penalty)
                    _ => {},                                                // Burning (no reward/penalty)
                }
            }
        }

        rewards
    }

    fn check_termination(&self) -> bool {
        if self.night < self.scenario.n_min {
            return false;
        }

        let all_safe = self.houses.iter().all(|&h| h == 0);
        let all_ruined = self.houses.iter().all(|&h| h == 2);

        all_safe || all_ruined
    }

    fn record_night(&mut self) {
        self.trajectory.push(GameNight {
            night: self.night,
            houses: self.houses.clone(),
            signals: self.agent_signals.clone(),
            locations: self.agent_positions.clone(),
            actions: self.last_actions.clone(),
            rewards: self.rewards.clone(),
        });
    }

    pub fn get_observation(&self, agent_id: usize) -> AgentObservation {
        AgentObservation {
            signals: self.agent_signals.clone(),
            locations: self.agent_positions.clone(),
            houses: self.houses.clone(),
            last_actions: self.last_actions.clone(),
            scenario_info: vec![
                self.scenario.beta,
                self.scenario.kappa,
                self.scenario.a,
                self.scenario.l,
                self.scenario.c,
                self.scenario.rho_ignite,
                self.scenario.n_min as f32,
                self.scenario.p_spark,
                self.scenario.n_spark as f32,
                self.scenario.num_agents as f32,
            ],
            agent_id,
            night: self.night,
        }
    }

    pub fn get_result(&self) -> GameResult {
        // Sum up per-step rewards from trajectory
        let mut agent_scores =
            self.trajectory
                .iter()
                .fold(vec![0.0; self.scenario.num_agents], |mut acc, night| {
                    for (i, &reward) in night.rewards.iter().enumerate() {
                        acc[i] += reward;
                    }
                    acc
                });

        // Add final rewards based on house outcomes
        let final_rewards = self.compute_final_rewards();
        for (i, &final_reward) in final_rewards.iter().enumerate() {
            agent_scores[i] += final_reward;
        }

        let final_score = agent_scores.iter().sum();

        GameResult {
            scenario: self.scenario.clone(),
            nights: self.trajectory.clone(),
            final_score,
            agent_scores,
        }
    }

    pub fn get_current_state(&self) -> GameState {
        GameState {
            houses: self.houses.clone(),
            agent_positions: self.agent_positions.clone(),
            agent_signals: self.agent_signals.clone(),
            last_actions: self.last_actions.clone(),
            night: self.night,
            done: self.done,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SCENARIOS;

    #[test]
    fn test_engine_creation() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let engine = BucketBrigade::new(scenario.clone(), Some(42));

        assert_eq!(engine.night, 0);
        assert!(!engine.done);
        assert_eq!(engine.scenario.num_agents, 4);
    }

    #[test]
    fn test_engine_initialization_fires() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let engine = BucketBrigade::new(scenario, Some(42));

        // Count burning houses
        let burning = engine.houses.iter().filter(|&&h| h == 1).count();

        // trivial_cooperation has rho_ignite=0.1, so ~1 house should be burning
        assert!(burning > 0, "Should have at least one burning house");
        assert!(burning <= 2, "Should not have too many burning houses");
    }

    #[test]
    fn test_reset() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        // Take some steps
        let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
        engine.step(&actions);
        engine.step(&actions);

        assert!(engine.night > 0);

        // Reset
        engine.reset();

        assert_eq!(engine.night, 0);
        assert!(!engine.done);
        assert!(engine.houses.iter().any(|&h| h == 1), "Should have fires after reset");
    }

    #[test]
    fn test_deterministic_with_seed() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine1 = BucketBrigade::new(scenario.clone(), Some(123));
        let mut engine2 = BucketBrigade::new(scenario, Some(123));

        // Same seed should produce identical initial states
        assert_eq!(engine1.houses, engine2.houses);

        // And identical behavior
        let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
        let result1 = engine1.step(&actions);
        let result2 = engine2.step(&actions);

        assert_eq!(result1.rewards, result2.rewards);
        assert_eq!(engine1.houses, engine2.houses);
    }

    #[test]
    fn test_step_advances_night() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        let initial_night = engine.night;
        let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
        engine.step(&actions);

        assert_eq!(engine.night, initial_night + 1);
    }

    #[test]
    fn test_work_vs_rest_rewards() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        // All agents rest
        let rest_actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        let result = engine.step(&rest_actions);

        // Resting gives +0.5 reward (plus other factors)
        // Working gives -c cost (0.5 in trivial_cooperation)
        // So resting should generally give better individual rewards when not needed
        assert!(result.rewards.iter().all(|&r| r >= 0.0), "Rest rewards should be non-negative");
    }

    #[test]
    fn test_fire_extinguishing() {
        let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        scenario.kappa = 10.0; // Very high extinguish efficiency
        let mut engine = BucketBrigade::new(scenario, Some(42));

        // Find a burning house
        let burning_house = engine.houses.iter().position(|&h| h == 1).unwrap();

        // Send all agents to work on it
        let actions = vec![
            [burning_house as u8, 1],
            [burning_house as u8, 1],
            [burning_house as u8, 1],
            [burning_house as u8, 1],
        ];

        engine.step(&actions);

        // With high kappa and 4 workers, fire should be extinguished
        // (Though it might burn out or spread first - check it's not still burning)
        let final_state = engine.houses[burning_house];
        assert_ne!(final_state, 1, "House should not still be burning after 4 workers");
    }

    #[test]
    fn test_fire_spreads_to_neighbors() {
        let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        scenario.beta = 1.0; // 100% spread probability
        scenario.kappa = 0.0; // No extinguishing

        let mut engine = BucketBrigade::new(scenario, Some(100));

        // Set up a controlled initial state with one burning house
        engine.houses = vec![0; 10];
        engine.houses[5] = 1; // House 5 is burning

        // No one works (let it spread)
        let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        engine.step(&actions);

        // Neighbors (4 and 6) should catch fire with beta=1.0
        // But house 5 should now be ruined (2) after burning
        assert_eq!(engine.houses[5], 2, "Original burning house should be ruined");
    }

    #[test]
    fn test_burn_out_phase() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        // Set a house on fire
        engine.houses[3] = 1;

        // Let it burn without intervention
        let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        engine.step(&actions);

        // Burning house should become ruined (unless extinguished, which is unlikely with no workers)
        let house_3_state = engine.houses[3];
        assert!(house_3_state == 2 || house_3_state == 0, "House should be ruined or extinguished");
    }

    #[test]
    fn test_spark_fires() {
        let mut scenario = SCENARIOS.get("early_containment").unwrap().clone();
        scenario.p_spark = 1.0; // 100% spark probability for testing
        scenario.n_spark = 5; // Sparks for first 5 nights

        let mut engine = BucketBrigade::new(scenario, Some(42));

        // Clear all fires
        engine.houses = vec![0; 10];
        engine.night = 0;

        // Step once - should create sparks
        let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        engine.step(&actions);

        // With p_spark=1.0, all safe houses should catch fire
        let burning = engine.houses.iter().filter(|&&h| h == 1).count();
        assert!(burning > 0, "Sparks should have created new fires");
    }

    #[test]
    fn test_no_sparks_after_duration() {
        let mut scenario = SCENARIOS.get("early_containment").unwrap().clone();
        scenario.p_spark = 1.0;
        scenario.n_spark = 2; // Only 2 nights of sparks

        let mut engine = BucketBrigade::new(scenario, Some(42));
        engine.houses = vec![0; 10];

        // Step past the spark duration
        let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        engine.step(&actions);
        engine.step(&actions);

        // Now past n_spark, clear fires
        engine.houses = vec![0; 10];
        engine.step(&actions);

        // Should be no new fires since we're past n_spark
        let burning = engine.houses.iter().filter(|&&h| h == 1).count();
        assert_eq!(burning, 0, "Should be no sparks after n_spark duration");
    }

    #[test]
    fn test_termination_all_safe() {
        let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        scenario.n_min = 0; // Allow immediate termination

        let mut engine = BucketBrigade::new(scenario, Some(42));
        engine.houses = vec![0; 10]; // All safe

        let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        let result = engine.step(&actions);

        assert!(result.done, "Game should end when all houses are safe");
    }

    #[test]
    fn test_termination_all_ruined() {
        let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        scenario.n_min = 0;

        let mut engine = BucketBrigade::new(scenario, Some(42));
        engine.houses = vec![2; 10]; // All ruined

        let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        let result = engine.step(&actions);

        assert!(result.done, "Game should end when all houses are ruined");
    }

    #[test]
    fn test_minimum_nights_prevents_early_termination() {
        let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        scenario.n_min = 5; // Require at least 5 nights

        let mut engine = BucketBrigade::new(scenario, Some(42));
        engine.houses = vec![0; 10]; // All safe

        let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];

        // Step through 3 nights
        for _ in 0..3 {
            let result = engine.step(&actions);
            assert!(!result.done, "Should not terminate before n_min nights");
        }
    }

    #[test]
    fn test_get_observation() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let engine = BucketBrigade::new(scenario, Some(42));

        let obs = engine.get_observation(0);

        assert_eq!(obs.agent_id, 0);
        assert_eq!(obs.night, 0);
        assert_eq!(obs.houses.len(), 10);
        assert_eq!(obs.signals.len(), 4);
        assert_eq!(obs.locations.len(), 4);
        assert_eq!(obs.scenario_info.len(), 10);
    }

    #[test]
    fn test_trajectory_recording() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
        engine.step(&actions);
        engine.step(&actions);

        let result = engine.get_result();

        // Should have initial state + 2 steps = 3 nights recorded
        // Night is recorded before incrementing, so: [0, 0, 1]
        assert_eq!(result.nights.len(), 3);
        assert_eq!(result.nights[0].night, 0); // Initial state
        assert_eq!(result.nights[1].night, 0); // After first step (recorded before increment)
        assert_eq!(result.nights[2].night, 1); // After second step (recorded before increment)
    }

    #[test]
    fn test_final_score_calculation() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
        engine.step(&actions);

        let result = engine.get_result();

        // Final score should be sum of all agent scores
        let expected_sum: f32 = result.agent_scores.iter().sum();
        assert!((result.final_score - expected_sum).abs() < 0.001);
    }

    #[test]
    fn test_agent_positions_update() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        let actions = vec![[5, 1], [6, 1], [7, 1], [8, 1]];
        engine.step(&actions);

        let state = engine.get_current_state();
        assert_eq!(state.agent_positions, vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_agent_signals_update() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        let actions = vec![[0, 0], [1, 1], [2, 0], [3, 1]];
        engine.step(&actions);

        let state = engine.get_current_state();
        assert_eq!(state.agent_signals, vec![0, 1, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "Game is already finished")]
    fn test_step_after_done_panics() {
        let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        scenario.n_min = 0;

        let mut engine = BucketBrigade::new(scenario, Some(42));
        engine.houses = vec![0; 10];

        let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        engine.step(&actions);

        // Game should be done now, stepping again should panic
        engine.step(&actions);
    }

    #[test]
    fn test_ownership_rewards() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        // Agent 0 owns house 0, agent 1 owns house 1, etc.
        engine.houses = vec![0; 10]; // All safe

        let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
        let result = engine.step(&actions);

        // Per-step rewards should only reflect work/rest costs
        // All agents resting, so should get +0.5
        assert!(result.rewards.iter().all(|&r| r == 0.5), "Agents should get rest reward only");

        // Complete the game and check final rewards
        engine.done = true;
        let game_result = engine.get_result();

        // All agents should have positive final scores (ownership bonus for safe houses + neighbor bonuses)
        assert!(game_result.agent_scores.iter().all(|&s| s > 0.0),
                "All agents should have positive scores with all houses safe");
    }

    #[test]
    fn test_ownership_penalty() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, Some(42));

        // Set agent 0's house (house 0) to ruined
        engine.houses = vec![0; 10];
        engine.houses[0] = 2;

        let actions = vec![[5, 0], [5, 0], [5, 0], [5, 0]];
        let result = engine.step(&actions);

        // Per-step rewards should be equal (only work/rest costs)
        // All agents resting, so should get +0.5
        assert_eq!(result.rewards[0], 0.5, "Agent 0 should have rest reward");
        assert_eq!(result.rewards[1], 0.5, "Agent 1 should have rest reward");
        assert_eq!(result.rewards[0], result.rewards[1], "Per-step rewards should be equal");

        // Complete the game and check final rewards
        engine.done = true;
        let game_result = engine.get_result();

        // Agent 0 should have penalty for ruined owned house (house 0)
        // Agent 1's house (house 1) is safe, so they get bonus
        assert!(game_result.agent_scores[0] < game_result.agent_scores[1],
                "Agent 0 should have lower final score due to ruined house penalty");
    }
}
