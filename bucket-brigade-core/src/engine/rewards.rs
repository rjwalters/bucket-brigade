use super::core::BucketBrigade;
use crate::{Action, HouseState};

impl BucketBrigade {
    pub(super) fn compute_rewards(
        &mut self,
        actions: &[Action],
        _prev_houses: &[HouseState],
    ) -> Vec<f32> {
        // Only compute per-step work/rest costs
        // House-based rewards are computed at game end
        actions
            .iter()
            .map(|action| {
                // Work/rest cost only
                if action[1] == 1 {
                    -self.scenario.cost_to_work_one_night // Work cost
                } else {
                    0.5 // Rest reward
                }
            })
            .collect()
    }

    pub(super) fn compute_final_rewards(&self) -> Vec<f32> {
        // Compute final rewards based on house outcomes at game end
        let mut rewards = vec![0.0; self.num_agents];

        // Compute collective team reward (public goods component)
        let saved_houses = self.houses.iter().filter(|&&h| h == 0).count() as f32;
        let ruined_houses = self.houses.iter().filter(|&&h| h == 2).count() as f32;
        let team_reward = saved_houses * self.scenario.team_reward_house_survives
            - ruined_houses * self.scenario.team_penalty_house_burns;

        for (agent_idx, reward) in rewards
            .iter_mut()
            .enumerate()
            .take(self.num_agents)
        {
            // Each agent receives full team reward (public goods incentive)
            *reward += team_reward;

            let owned_house = agent_idx % 10;
            let left_neighbor = if owned_house == 0 { 9 } else { owned_house - 1 };
            let right_neighbor = (owned_house + 1) % 10;

            // Reward for owned house outcome
            match self.houses[owned_house] {
                0 => *reward += self.scenario.reward_own_house_survives, // Saved
                2 => *reward -= self.scenario.penalty_own_house_burns,   // Ruined
                _ => {} // Burning (no reward/penalty)
            }

            // Rewards for neighbor houses
            for &neighbor in &[left_neighbor, right_neighbor] {
                match self.houses[neighbor] {
                    0 => *reward += self.scenario.reward_other_house_survives, // Saved
                    2 => *reward -= self.scenario.penalty_other_house_burns,   // Ruined
                    _ => {} // Burning (no reward/penalty)
                }
            }
        }

        rewards
    }
}
