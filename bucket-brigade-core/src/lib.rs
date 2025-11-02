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
