pub mod agents;
pub mod engine;
pub mod rng;
pub mod scenarios;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;

use serde::{Deserialize, Serialize};

// Re-export key types for easy access
pub use engine::{BucketBrigade, GameResult, GameState};
pub use scenarios::{Scenario, SCENARIOS};

// Re-export Python module for PyO3
#[cfg(feature = "python")]
pub use python::bucket_brigade_core;

/// House states: 0 = Safe, 1 = Burning, 2 = Ruined
pub type HouseState = u8;

/// Agent actions: [house_index, mode] where mode is 0=REST, 1=WORK
pub type Action = [u8; 2];

/// Represents a single night in the game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameNight {
    pub night: u32,
    pub houses: Vec<HouseState>,
    pub signals: Vec<u8>,
    pub locations: Vec<u8>,
    pub actions: Vec<Action>,
    pub rewards: Vec<f32>,
}

/// Agent observation structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentObservation {
    pub signals: Vec<u8>,
    pub locations: Vec<u8>,
    pub houses: Vec<HouseState>,
    pub last_actions: Vec<Action>,
    pub scenario_info: Vec<f32>,
    pub agent_id: usize,
    pub night: u32,
}

/// Generic agent trait
pub trait Agent {
    fn act(&self, obs: &AgentObservation) -> Action;
    fn reset(&mut self);
    fn id(&self) -> usize;
    fn name(&self) -> &str;
}

/// Simple random agent for testing
pub struct RandomAgent {
    id: usize,
    name: String,
}

impl RandomAgent {
    pub fn new(id: usize, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
        }
    }
}

impl Agent for RandomAgent {
    fn act(&self, _obs: &AgentObservation) -> Action {
        [rand::random::<u8>() % 10, rand::random::<u8>() % 2]
    }

    fn reset(&mut self) {}

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

    #[test]
    fn test_random_agent_creation() {
        let agent = RandomAgent::new(0, "TestAgent");
        assert_eq!(agent.id(), 0);
        assert_eq!(agent.name(), "TestAgent");
    }

    #[test]
    fn test_random_agent_act() {
        let agent = RandomAgent::new(0, "TestAgent");
        let obs = AgentObservation {
            signals: vec![0, 0, 0, 0],
            locations: vec![0, 0, 0, 0],
            houses: vec![0; 10],
            last_actions: vec![[0, 0]; 4],
            scenario_info: vec![0.0; 10],
            agent_id: 0,
            night: 0,
        };

        // Random agent should produce valid actions
        for _ in 0..100 {
            let action = agent.act(&obs);
            assert!(action[0] < 10, "House index should be < 10");
            assert!(action[1] < 2, "Mode should be 0 or 1");
        }
    }

    #[test]
    fn test_random_agent_reset() {
        let mut agent = RandomAgent::new(0, "TestAgent");
        agent.reset(); // Should not panic
        assert_eq!(agent.id(), 0); // Agent should still be valid
    }

    #[test]
    fn test_agent_observation_structure() {
        let obs = AgentObservation {
            signals: vec![0, 1, 0, 1],
            locations: vec![3, 5, 7, 9],
            houses: vec![0, 1, 2, 0, 0, 1, 2, 0, 0, 1],
            last_actions: vec![[0, 0], [1, 1], [2, 0], [3, 1]],
            scenario_info: vec![0.15, 0.9, 100.0, 100.0, 0.5, 0.1, 12.0, 0.0, 12.0, 4.0],
            agent_id: 2,
            night: 5,
        };

        assert_eq!(obs.agent_id, 2);
        assert_eq!(obs.night, 5);
        assert_eq!(obs.signals.len(), 4);
        assert_eq!(obs.locations.len(), 4);
        assert_eq!(obs.houses.len(), 10);
        assert_eq!(obs.scenario_info.len(), 10);
    }

    #[test]
    fn test_agent_observation_serialization() {
        let obs = AgentObservation {
            signals: vec![0, 1],
            locations: vec![3, 5],
            houses: vec![0; 10],
            last_actions: vec![[0, 0], [1, 1]],
            scenario_info: vec![0.15, 0.9, 100.0, 100.0, 0.5, 0.1, 12.0, 0.0, 12.0, 2.0],
            agent_id: 0,
            night: 1,
        };

        let json = serde_json::to_string(&obs).unwrap();
        let deserialized: AgentObservation = serde_json::from_str(&json).unwrap();

        assert_eq!(obs.agent_id, deserialized.agent_id);
        assert_eq!(obs.night, deserialized.night);
        assert_eq!(obs.signals, deserialized.signals);
    }

    #[test]
    fn test_game_night_serialization() {
        let night = GameNight {
            night: 3,
            houses: vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            signals: vec![0, 1, 0, 1],
            locations: vec![2, 3, 4, 5],
            actions: vec![[2, 1], [3, 1], [4, 0], [5, 1]],
            rewards: vec![1.5, 2.0, 0.5, -0.5],
        };

        let json = serde_json::to_string(&night).unwrap();
        let deserialized: GameNight = serde_json::from_str(&json).unwrap();

        assert_eq!(night.night, deserialized.night);
        assert_eq!(night.houses, deserialized.houses);
        assert_eq!(night.rewards, deserialized.rewards);
    }

    #[test]
    fn test_game_night_clone() {
        let night = GameNight {
            night: 1,
            houses: vec![0; 10],
            signals: vec![0; 4],
            locations: vec![0; 4],
            actions: vec![[0, 0]; 4],
            rewards: vec![0.0; 4],
        };

        let cloned = night.clone();
        assert_eq!(night.night, cloned.night);
        assert_eq!(night.houses, cloned.houses);
    }

    #[test]
    fn test_multiple_random_agents() {
        let agents: Vec<RandomAgent> = (0..4)
            .map(|i| RandomAgent::new(i, &format!("Agent{}", i)))
            .collect();

        for (i, agent) in agents.iter().enumerate() {
            assert_eq!(agent.id(), i);
            assert_eq!(agent.name(), format!("Agent{}", i));
        }
    }

    #[test]
    fn test_action_type() {
        let action: Action = [5, 1];
        assert_eq!(action[0], 5); // House index
        assert_eq!(action[1], 1); // Mode (work)
    }

    #[test]
    fn test_house_states() {
        let safe: HouseState = 0;
        let burning: HouseState = 1;
        let ruined: HouseState = 2;

        assert_eq!(safe, 0);
        assert_eq!(burning, 1);
        assert_eq!(ruined, 2);
    }
}
