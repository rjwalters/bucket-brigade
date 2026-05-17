use super::core::BucketBrigade;
use crate::Action;

impl BucketBrigade {
    pub(super) fn extinguish_fires(&mut self, actions: &[Action]) {
        let num_houses = self.scenario.num_houses as usize;
        for house_idx in 0..num_houses {
            if self.houses[house_idx] != 1 {
                continue;
            }

            // Count workers at this house
            let workers_here = actions
                .iter()
                .filter(|action| action[0] as usize == house_idx && action[1] == 1)
                .count();

            // Probability of extinguishing - independent probabilities
            let p_extinguish = 1.0
                - (1.0 - self.scenario.prob_solo_agent_extinguishes_fire).powi(workers_here as i32);

            if self.rng.random() < p_extinguish {
                self.houses[house_idx] = 0;
            }
        }
    }

    pub(super) fn spread_fires(&mut self) {
        // Issue #254: ring length is now `scenario.num_houses` rather than a
        // hardcoded 10. The neighbor wraparound modulo also tracks the
        // scenario value so 2-house and other small-ring topologies wrap
        // correctly. Note that on a 2-house ring the (i-1) and (i+1)
        // neighbors are the same house, so spread is single-step but the
        // probability roll still happens twice — matching the way the
        // 10-ring case handled it for the trivial wrap edge.
        let num_houses = self.scenario.num_houses as usize;
        let mut new_houses = self.houses.clone();

        for house_idx in 0..num_houses {
            if self.houses[house_idx] != 1 {
                continue;
            }

            // Check neighbors (wrap modulo num_houses).
            let neighbors = [
                (house_idx + num_houses - 1) % num_houses,
                (house_idx + 1) % num_houses,
            ];

            for &neighbor in &neighbors {
                if self.houses[neighbor] == 0
                    && self.rng.random() < self.scenario.prob_fire_spreads_to_neighbor
                {
                    new_houses[neighbor] = 1;
                }
            }
        }

        self.houses = new_houses;
    }

    pub(super) fn burn_out_houses(&mut self) {
        for house in self.houses.iter_mut() {
            if *house == 1 {
                *house = 2;
            }
        }
    }

    pub(super) fn spontaneous_ignition(&mut self) {
        let num_houses = self.scenario.num_houses as usize;
        for house_idx in 0..num_houses {
            if self.houses[house_idx] == 0
                && self.rng.random() < self.scenario.prob_house_catches_fire
            {
                self.houses[house_idx] = 1;
            }
        }
    }
}
