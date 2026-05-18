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
        // The four ownership reward fields are now per-agent vectors (#198);
        // expose their mean here to preserve the 12-element scenario_info
        // layout. For scenarios with uniform per-agent weights (the legacy
        // scalar-promoted case) the mean equals the original scalar, so
        // existing consumers see identical values.
        fn mean(v: &[f32]) -> f32 {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            }
        }

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
                mean(&self.scenario.reward_own_house_survives),
                mean(&self.scenario.reward_other_house_survives),
                mean(&self.scenario.penalty_own_house_burns),
                mean(&self.scenario.penalty_other_house_burns),
            ],
            agent_id,
            night: self.night,
            // Issue #252: round-1 commitment signals from the most recent
            // signal phase. Zeros in simultaneous mode (no signal phase
            // ever runs). In two-phase mode this carries the round-1
            // signals between the signal-phase write and the next
            // step_two_phase call.
            round1_signals: self.round1_signals.clone(),
        }
    }

    pub fn get_result(&self) -> GameResult {
        // Sum up per-step rewards from trajectory
        // All rewards (work/rest costs, team rewards, ownership bonuses) are computed per-step
        let agent_scores =
            self.trajectory
                .iter()
                .fold(vec![0.0; self.num_agents], |mut acc, night| {
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
