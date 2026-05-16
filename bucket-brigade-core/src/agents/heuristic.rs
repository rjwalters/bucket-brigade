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
    /// in bucket_brigade/equilibrium/payoff_evaluator_rust.py:40-80.
    ///
    /// Issue #235: returns a 3-element `[house, mode, signal]`.
    ///
    /// Issue #240: the signal channel now honors `honesty_bias` (param 0).
    /// With probability `honesty_bias` the agent broadcasts truthfully
    /// (`signal == mode`); otherwise it broadcasts the opposite bit.
    /// For archetypes with `honesty_bias == 1.0` (firefighter, hero) the
    /// behavior is unchanged. For the Liar archetype (`honesty_bias = 0.1`)
    /// the agent signals deceptively roughly 90% of the time.
    ///
    /// This is a simplified version of the deception logic in the richer
    /// Python `HeuristicAgent.act` (`bucket_brigade/agents/heuristic_agent.py`):
    /// the Python path computes a continuous `work_intent` and then signals
    /// the *thresholded* intent (potentially flipped), whereas this Rust
    /// path signals the *realized mode* (potentially flipped). Both paths
    /// produce the same marginal `P(signal != mode) ≈ 1 - honesty_bias` for
    /// the Liar archetype, which is what the Nash equilibrium analyses
    /// over the archetype space care about.
    pub fn select_action<R: Rng>(&self, obs: &AgentObservation, rng: &mut R) -> Action {
        // Extract key parameters
        let honesty_bias = self.params[0];
        let work_tendency = self.params[1];
        let own_house_priority = self.params[3];
        let rest_reward_bias = self.params[8];

        // Decide whether to work or rest
        let (house, mode): (u8, u8) =
            if rng.gen::<f64>() < work_tendency * (1.0 - rest_reward_bias) {
                // Work - choose which house
                let owned_house = self.id % 10;

                // Check if owned house is burning and prioritize it
                if obs.houses[owned_house] == 1 && rng.gen::<f64>() < own_house_priority {
                    (owned_house as u8, 1)
                } else {
                    let burning: Vec<usize> = obs
                        .houses
                        .iter()
                        .enumerate()
                        .filter(|(_, &state)| state == 1)
                        .map(|(idx, _)| idx)
                        .collect();
                    let house = if !burning.is_empty() {
                        burning[rng.gen_range(0..burning.len())]
                    } else {
                        owned_house
                    };
                    (house as u8, 1)
                }
            } else {
                // Rest at owned house
                let owned_house = self.id % 10;
                (owned_house as u8, 0)
            };

        // Issue #240: thread honesty_bias into the signal channel. With
        // probability `honesty_bias`, broadcast the true mode; otherwise
        // broadcast the opposite bit. Honest archetypes (honesty_bias=1.0)
        // are unchanged; the Liar archetype (honesty_bias=0.1) lies ~90%.
        let signal = if rng.gen::<f64>() < honesty_bias {
            mode
        } else {
            1 - mode
        };

        [house, mode, signal]
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
            last_actions: vec![[0, 0, 0]; 4],
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
            last_actions: vec![[0, 0, 0]; 4],
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
            last_actions: vec![[0, 0, 0]; 4],
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
            last_actions: vec![[0, 0, 0]; 4],
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
            last_actions: vec![[0, 0, 0]; 4],
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
            last_actions: vec![[0, 0, 0]; 4],
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

    /// Issue #240: Verify that `honesty_bias` actually controls the signal
    /// channel. Prior to this fix the Rust path hardcoded `signal == mode`
    /// regardless of `honesty_bias`, which silently made the Rust-backed
    /// Nash evaluator disagree with the Python `HeuristicAgent.act` path
    /// for the Liar archetype. This test pins both ends of the spectrum.
    #[test]
    fn test_heuristic_agent_honesty_bias_signaling() {
        // Honest agent (honesty_bias=1.0): signal must always equal mode.
        let mut honest_params = [0.0; 10];
        honest_params[0] = 1.0; // honesty_bias
        honest_params[1] = 0.5; // work_tendency — yields mix of work/rest
        let honest = HeuristicAgent::new(0, "Honest", honest_params);

        // Liar agent (honesty_bias=0.0): signal must always be flipped.
        let mut liar_params = [0.0; 10];
        liar_params[0] = 0.0; // honesty_bias
        liar_params[1] = 0.5;
        let liar = HeuristicAgent::new(0, "Liar", liar_params);

        let obs = AgentObservation {
            signals: vec![0; 4],
            locations: vec![0; 4],
            houses: vec![0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            last_actions: vec![[0, 0, 0]; 4],
            scenario_info: vec![0.0; 10],
            agent_id: 0,
            night: 0,
        };

        let mut rng = Pcg64::seed_from_u64(12345);
        let mut honest_agreements = 0;
        let mut liar_agreements = 0;
        let n = 200;

        for _ in 0..n {
            let a = honest.select_action(&obs, &mut rng);
            if a[1] == a[2] {
                honest_agreements += 1;
            }
        }
        for _ in 0..n {
            let a = liar.select_action(&obs, &mut rng);
            if a[1] == a[2] {
                liar_agreements += 1;
            }
        }

        assert_eq!(
            honest_agreements, n,
            "honesty_bias=1.0 must always signal honestly (signal == mode)"
        );
        assert_eq!(
            liar_agreements, 0,
            "honesty_bias=0.0 must always signal deceptively (signal != mode)"
        );
    }

    /// Issue #240: Verify the partial-honesty path. With honesty_bias=0.1
    /// (the Liar archetype) the signal should disagree with mode in
    /// roughly 90% of decisions. Use a large sample and a wide tolerance
    /// so the test is robust to RNG variance.
    #[test]
    fn test_heuristic_agent_liar_archetype_lies_most_of_time() {
        // Liar archetype params (matches Python LIAR_PARAMS).
        let params = [0.1, 0.7, 0.0, 0.9, 0.2, 0.8, 0.3, 0.0, 0.4, 0.2];
        let liar = HeuristicAgent::new(0, "Liar", params);

        let obs = AgentObservation {
            signals: vec![0; 4],
            locations: vec![0; 4],
            houses: vec![1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            last_actions: vec![[0, 0, 0]; 4],
            scenario_info: vec![0.0; 10],
            agent_id: 0,
            night: 0,
        };

        let mut rng = Pcg64::seed_from_u64(99);
        let n = 2000;
        let mut disagreements = 0;
        for _ in 0..n {
            let a = liar.select_action(&obs, &mut rng);
            if a[1] != a[2] {
                disagreements += 1;
            }
        }

        let lie_rate = disagreements as f64 / n as f64;
        // Expected ~0.90, allow ±0.05 for RNG noise.
        assert!(
            (0.85..=0.95).contains(&lie_rate),
            "Liar archetype should lie ~90% of the time, got {:.3}",
            lie_rate
        );
    }
}
