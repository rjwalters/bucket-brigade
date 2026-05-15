use super::core::BucketBrigade;
use crate::{Action, HouseState};

impl BucketBrigade {
    pub(super) fn compute_rewards(
        &mut self,
        actions: &[Action],
        prev_houses: &[HouseState],
    ) -> Vec<f32> {
        // Per-step rewards for RL training (matches Python implementation)
        // Includes: work/rest costs, team rewards, and ownership bonuses

        let mut rewards = vec![0.0; self.num_agents];

        // Count current house states
        let saved_houses = self.houses.iter().filter(|&&h| h == 0).count() as f32;
        let ruined_houses = self.houses.iter().filter(|&&h| h == 2).count() as f32;
        let total_saved_fraction = saved_houses / 10.0;
        let total_burned_fraction = ruined_houses / 10.0;

        // Team reward component (shared by all, public goods)
        let team_reward = self.scenario.team_reward_house_survives * total_saved_fraction
            - self.scenario.team_penalty_house_burns * total_burned_fraction;

        for agent_idx in 0..self.num_agents {
            // 1. Work/rest component
            if actions[agent_idx][1] == 1 {
                rewards[agent_idx] -= self.scenario.cost_to_work_one_night; // Work cost
            } else {
                rewards[agent_idx] += 0.5; // Rest reward
            }

            // 2 & 3. Per-house ownership rewards.
            // For each of the 10 houses, decide whether the agent owns it and
            // apply the appropriate per-house reward field. This wires up the
            // four previously-unused `Scenario` ownership reward fields
            // (`reward_own_house_survives`, `reward_other_house_survives`,
            // `penalty_own_house_burns`, `penalty_other_house_burns`).
            for house_idx in 0..10 {
                let is_own = (self.house_owners[house_idx] as usize) == agent_idx;

                // Save event: BURNING -> SAFE this step.
                if prev_houses[house_idx] != 0 && self.houses[house_idx] == 0 {
                    rewards[agent_idx] += if is_own {
                        self.scenario.reward_own_house_survives
                    } else {
                        self.scenario.reward_other_house_survives
                    };
                }

                // Currently-ruined penalty (applied every step the house is
                // RUINED, matching the Python behavior in
                // `bucket_brigade_env.py::_compute_rewards`). The penalty field
                // stores the magnitude as a positive number; subtract it.
                if self.houses[house_idx] == 2 {
                    rewards[agent_idx] -= if is_own {
                        self.scenario.penalty_own_house_burns
                    } else {
                        self.scenario.penalty_other_house_burns
                    };
                }
            }

            // 4. Team reward component (full public goods incentive)
            rewards[agent_idx] += team_reward;
        }

        rewards
    }
}
