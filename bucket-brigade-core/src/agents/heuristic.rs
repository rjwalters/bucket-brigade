use crate::{Action, Agent, AgentObservation};
use rand::Rng;

/// Heuristic agent with parameterized strategy
///
/// This agent uses a 10-parameter strategy vector to make decisions:
/// - theta[0]: honesty
/// - theta[1]: work_tendency
/// - theta[2]: neighbor_help
/// - theta[3]: own_house_priority
/// - theta[4]: (reserved)
/// - theta[5]: (reserved)
/// - theta[6]: (reserved)
/// - theta[7]: (reserved)
/// - theta[8]: rest_reward_bias
/// - theta[9]: (reserved)
#[derive(Debug, Clone)]
pub struct HeuristicAgent {
    /// Agent strategy parameters
    pub params: [f64; 10],
    /// Agent ID
    pub id: usize,
    /// Agent name
    pub name: String,
}

impl HeuristicAgent {
    /// Create a new heuristic agent with given parameters
    pub fn new(id: usize, name: &str, params: [f64; 10]) -> Self {
        Self {
            params,
            id,
            name: name.to_string(),
        }
    }

    /// Select action based on heuristic strategy
    ///
    /// This implementation matches the Python `_heuristic_action` function
    /// in bucket_brigade/equilibrium/payoff_evaluator_rust.py:33-69
    pub fn select_action<R: Rng>(&self, obs: &AgentObservation, rng: &mut R) -> Action {
        // Extract key parameters
        let work_tendency = self.params[1];
        let own_house_priority = self.params[3];
        let rest_reward_bias = self.params[8];

        // Decide whether to work or rest
        if rng.gen::<f64>() < work_tendency * (1.0 - rest_reward_bias) {
            // Work - choose which house
            let owned_house = self.id % 10;

            // Check if owned house is burning and prioritize it
            if obs.houses[owned_house] == 1 && rng.gen::<f64>() < own_house_priority {
                // Work on owned house
                [owned_house as u8, 1]
            } else {
                // Choose a burning house
                let burning: Vec<usize> = obs
                    .houses
                    .iter()
                    .enumerate()
                    .filter(|(_, &state)| state == 1)
                    .map(|(idx, _)| idx)
                    .collect();

                let house = if !burning.is_empty() {
                    // Pick random burning house
                    burning[rng.gen_range(0..burning.len())]
                } else {
                    // No burning houses, go to owned house
                    owned_house
                };

                [house as u8, 1]
            }
        } else {
            // Rest at owned house
            let owned_house = self.id % 10;
            [owned_house as u8, 0]
        }
    }
}

impl Agent for HeuristicAgent {
    fn act(&self, _obs: &AgentObservation) -> Action {
        // This implementation requires an RNG, so we can't implement the trait method
        // without a default RNG. For now, we'll panic - users should call select_action directly.
        panic!("HeuristicAgent requires an RNG - use select_action() instead")
    }

    fn reset(&mut self) {
        // Heuristic agent is stateless, nothing to reset
    }

    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn test_heuristic_agent_creation() {
        let params = [0.5; 10];
        let agent = HeuristicAgent::new(0, "TestAgent", params);
        assert_eq!(agent.id(), 0);
        assert_eq!(agent.name(), "TestAgent");
        assert_eq!(agent.params.len(), 10);
    }

    #[test]
    fn test_heuristic_agent_deterministic() {
        let params = [0.0, 0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0];
        let agent = HeuristicAgent::new(0, "TestAgent", params);

        let obs = AgentObservation {
            signals: vec![0; 4],
            locations: vec![0; 4],
            houses: vec![0, 1, 0, 1, 0, 0, 0, 0, 0, 0], // Houses 1 and 3 burning
            last_actions: vec![[0, 0]; 4],
            scenario_info: vec![0.0; 10],
            agent_id: 0,
            night: 0,
        };

        // Same seed should give same action
        let mut rng1 = Pcg64::seed_from_u64(42);
        let action1 = agent.select_action(&obs, &mut rng1);

        let mut rng2 = Pcg64::seed_from_u64(42);
        let action2 = agent.select_action(&obs, &mut rng2);

        assert_eq!(action1, action2);
    }

    #[test]
    fn test_heuristic_agent_high_work_tendency() {
        let mut params = [0.0; 10];
        params[1] = 1.0; // work_tendency = 1.0
        params[3] = 0.0; // own_house_priority = 0.0
        params[8] = 0.0; // rest_reward_bias = 0.0

        let agent = HeuristicAgent::new(0, "TestAgent", params);

        let obs = AgentObservation {
            signals: vec![0; 4],
            locations: vec![0; 4],
            houses: vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0], // House 1 burning
            last_actions: vec![[0, 0]; 4],
            scenario_info: vec![0.0; 10],
            agent_id: 0,
            night: 0,
        };

        let mut rng = Pcg64::seed_from_u64(42);

        // With work_tendency=1.0, should always work (action[1]=1)
        for _ in 0..10 {
            let action = agent.select_action(&obs, &mut rng);
            assert_eq!(action[1], 1, "Should be working");
        }
    }

    #[test]
    fn test_heuristic_agent_low_work_tendency() {
        let mut params = [0.0; 10];
        params[1] = 0.0; // work_tendency = 0.0
        params[8] = 0.0; // rest_reward_bias = 0.0

        let agent = HeuristicAgent::new(0, "TestAgent", params);

        let obs = AgentObservation {
            signals: vec![0; 4],
            locations: vec![0; 4],
            houses: vec![0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            last_actions: vec![[0, 0]; 4],
            scenario_info: vec![0.0; 10],
            agent_id: 0,
            night: 0,
        };

        let mut rng = Pcg64::seed_from_u64(42);

        // With work_tendency=0.0, should always rest (action[1]=0)
        for _ in 0..10 {
            let action = agent.select_action(&obs, &mut rng);
            assert_eq!(action[1], 0, "Should be resting");
            assert_eq!(action[0], 0, "Should rest at owned house (0 % 10 = 0)");
        }
    }

    #[test]
    fn test_heuristic_agent_own_house_priority() {
        let mut params = [0.0; 10];
        params[1] = 1.0; // work_tendency = 1.0 (always work)
        params[3] = 1.0; // own_house_priority = 1.0 (always prioritize owned house)
        params[8] = 0.0; // rest_reward_bias = 0.0

        let agent = HeuristicAgent::new(0, "TestAgent", params);

        let obs = AgentObservation {
            signals: vec![0; 4],
            locations: vec![0; 4],
            houses: vec![1, 1, 1, 1, 0, 0, 0, 0, 0, 0], // Owned house (0) and others burning
            last_actions: vec![[0, 0]; 4],
            scenario_info: vec![0.0; 10],
            agent_id: 0,
            night: 0,
        };

        let mut rng = Pcg64::seed_from_u64(42);

        // With own_house_priority=1.0 and owned house burning, should work on house 0
        for _ in 0..10 {
            let action = agent.select_action(&obs, &mut rng);
            assert_eq!(action[0], 0, "Should prioritize owned house 0");
            assert_eq!(action[1], 1, "Should be working");
        }
    }

    #[test]
    fn test_heuristic_agent_no_burning_houses() {
        let mut params = [0.0; 10];
        params[1] = 1.0; // work_tendency = 1.0
        params[3] = 0.0; // own_house_priority = 0.0
        params[8] = 0.0; // rest_reward_bias = 0.0

        let agent = HeuristicAgent::new(3, "TestAgent", params);

        let obs = AgentObservation {
            signals: vec![0; 4],
            locations: vec![0; 4],
            houses: vec![0; 10], // No burning houses
            last_actions: vec![[0, 0]; 4],
            scenario_info: vec![0.0; 10],
            agent_id: 3,
            night: 0,
        };

        let mut rng = Pcg64::seed_from_u64(42);

        // No burning houses, should work on owned house (3 % 10 = 3)
        let action = agent.select_action(&obs, &mut rng);
        assert_eq!(action[0], 3, "Should go to owned house when no fires");
        assert_eq!(action[1], 1, "Should be working");
    }

    #[test]
    fn test_heuristic_agent_rest_reward_bias() {
        let mut params = [0.0; 10];
        params[1] = 0.5; // work_tendency = 0.5
        params[8] = 1.0; // rest_reward_bias = 1.0 (nullifies work tendency)

        let agent = HeuristicAgent::new(0, "TestAgent", params);

        let obs = AgentObservation {
            signals: vec![0; 4],
            locations: vec![0; 4],
            houses: vec![1; 10], // All houses burning
            last_actions: vec![[0, 0]; 4],
            scenario_info: vec![0.0; 10],
            agent_id: 0,
            night: 0,
        };

        let mut rng = Pcg64::seed_from_u64(42);

        // With rest_reward_bias=1.0, effective work tendency = 0.5 * (1-1) = 0, should always rest
        for _ in 0..10 {
            let action = agent.select_action(&obs, &mut rng);
            assert_eq!(action[1], 0, "Should be resting due to rest_reward_bias");
        }
    }
}
