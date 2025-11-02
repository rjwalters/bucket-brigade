use crate::scenarios::Scenario;
use crate::rng::DeterministicRng;
use crate::{GameNight, AgentObservation, Action, HouseState};
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
        self.extinguish_fires(actions);

        // 4. Spread phase
        self.spread_fires();

        // 5. Burn-out phase
        self.burn_out_houses();

        // 6. Spark phase (if active)
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
            let workers_here = actions.iter()
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
                (house_idx + 9) % 10,  // (i-1) mod 10
                (house_idx + 1) % 10,  // (i+1) mod 10
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

    fn compute_rewards(&mut self, actions: &[Action], prev_houses: &[HouseState]) -> Vec<f32> {
        // Count outcomes
        let saved_houses = self.houses.iter().filter(|&&h| h == 0).count() as f32;
        let ruined_houses = self.houses.iter().filter(|&&h| h == 2).count() as f32;

        // Team reward
        let team_reward = self.scenario.a * (saved_houses / 10.0) - self.scenario.l * (ruined_houses / 10.0);

        let mut rewards = Vec::with_capacity(self.scenario.num_agents);

        for agent_idx in 0..self.scenario.num_agents {
            let mut reward = 0.0;

            // Work/rest cost
            if actions[agent_idx][1] == 1 {
                reward -= self.scenario.c; // Work cost
            } else {
                reward += 0.5; // Rest reward
            }

            // Ownership bonus/penalty
            let owned_house = agent_idx % 10;
            if prev_houses[owned_house] == 0 && self.houses[owned_house] == 0 {
                reward += 1.0; // Bonus for keeping house safe
            }
            if self.houses[owned_house] == 2 {
                reward -= 2.0; // Penalty for ruined house
            }

            // Team reward share
            reward += 0.1 * team_reward;

            rewards.push(reward);
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
        let agent_scores = self.trajectory.iter()
            .fold(vec![0.0; self.scenario.num_agents], |mut acc, night| {
                for (i, &reward) in night.rewards.iter().enumerate() {
                    acc[i] += reward;
                }
                acc
            });

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
