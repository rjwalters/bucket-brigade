use super::core::{ring_dist, BucketBrigade};
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

        // Issue #254: divide by `scenario.num_houses` (defaults to 10 for
        // every pre-#254 scenario, so the math is unchanged).
        let num_houses = self.scenario.num_houses as usize;
        let num_houses_f = num_houses as f32;

        // Count current house states
        let saved_houses = self.houses.iter().filter(|&&h| h == 0).count() as f32;
        let ruined_houses = self.houses.iter().filter(|&&h| h == 2).count() as f32;
        let total_saved_fraction = saved_houses / num_houses_f;
        let total_burned_fraction = ruined_houses / num_houses_f;

        // Team reward component (shared by all, public goods)
        let team_reward = self.scenario.team_reward_house_survives * total_saved_fraction
            - self.scenario.team_penalty_house_burns * total_burned_fraction;

        for agent_idx in 0..self.num_agents {
            // 1. Work/rest component.
            //
            // Issue #203 (option A): when `scenario.distance_cost_alpha != 0`,
            // the work cost is additively scaled by the ring-arc distance
            // between the agent's home position and the house it works at:
            //     cost = base_cost + alpha * ring_dist_10(home, target).
            // When `alpha == 0` (the implicit default for every pre-#203
            // scenario) this collapses to the unscaled `base_cost`, so
            // existing scenarios are bit-exactly unchanged.
            if actions[agent_idx][1] == 1 {
                let base_cost = self.scenario.cost_to_work_one_night;
                let alpha = self.scenario.distance_cost_alpha;
                let work_cost = if alpha == 0.0 {
                    base_cost
                } else {
                    let home = self.agent_home_positions[agent_idx];
                    let target = actions[agent_idx][0];
                    // Issue #254: ring length now reads from the scenario.
                    let dist = ring_dist(self.scenario.num_houses, home, target) as f32;
                    base_cost + alpha * dist
                };
                rewards[agent_idx] -= work_cost;
            } else {
                rewards[agent_idx] += 0.5; // Rest reward
            }

            // 2 & 3. Per-house ownership rewards.
            // For each of the 10 houses, decide whether the agent owns it and
            // apply the appropriate per-house reward field. This wires up the
            // four `Scenario` ownership reward fields
            // (`reward_own_house_survives`, `reward_other_house_survives`,
            // `penalty_own_house_burns`, `penalty_other_house_burns`).
            //
            // As of issue #198 these fields are per-agent vectors indexed by
            // ``agent_idx``. Scalar JSON inputs are auto-promoted to per-agent
            // vectors by `deserialize_scalar_or_vec` in `scenarios.rs`, so
            // existing scenarios behave identically to the pre-#198 scalar
            // semantics.
            for house_idx in 0..num_houses {
                let is_own = (self.house_owners[house_idx] as usize) == agent_idx;

                // Save event: BURNING -> SAFE this step.
                if prev_houses[house_idx] != 0 && self.houses[house_idx] == 0 {
                    rewards[agent_idx] += if is_own {
                        self.scenario.reward_own_house_survives[agent_idx]
                    } else {
                        self.scenario.reward_other_house_survives[agent_idx]
                    };
                }

                // Currently-ruined penalty (applied every step the house is
                // RUINED, matching the Python behavior in
                // `bucket_brigade_env.py::_compute_rewards`). The penalty field
                // stores the magnitude as a positive number; subtract it.
                if self.houses[house_idx] == 2 {
                    rewards[agent_idx] -= if is_own {
                        self.scenario.penalty_own_house_burns[agent_idx]
                    } else {
                        self.scenario.penalty_other_house_burns[agent_idx]
                    };
                }
            }

            // 4. Team reward component (full public goods incentive)
            rewards[agent_idx] += team_reward;
        }

        rewards
    }
}
