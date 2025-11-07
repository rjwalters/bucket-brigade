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

            // 2. Ownership changes: bonus for owned houses that become safe
            let owned_house = agent_idx % 10;
            if prev_houses[owned_house] != 0 && self.houses[owned_house] == 0 {
                rewards[agent_idx] += 1.0; // Bonus for saving owned house
            }

            // 3. Penalty for owned houses that are currently ruined
            if self.houses[owned_house] == 2 {
                rewards[agent_idx] -= 2.0;
            }

            // 4. Team reward component (full public goods incentive)
            rewards[agent_idx] += team_reward;
        }

        rewards
    }
}
