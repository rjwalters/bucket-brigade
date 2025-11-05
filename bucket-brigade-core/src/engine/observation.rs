use super::core::BucketBrigade;
use super::types::{GameResult, GameState};
use crate::{AgentObservation, GameNight};

impl BucketBrigade {
    pub(super) fn record_night(&mut self) {
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
                self.scenario.prob_fire_spreads_to_neighbor,
                self.scenario.prob_solo_agent_extinguishes_fire,
                self.scenario.team_reward_house_survives,
                self.scenario.team_penalty_house_burns,
                self.scenario.cost_to_work_one_night,
                self.scenario.prob_house_catches_fire,
                self.scenario.min_nights as f32,
                self.num_agents as f32,
                self.scenario.reward_own_house_survives,
                self.scenario.reward_other_house_survives,
                self.scenario.penalty_own_house_burns,
                self.scenario.penalty_other_house_burns,
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
                .fold(vec![0.0; self.num_agents], |mut acc, night| {
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
