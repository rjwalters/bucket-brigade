use crate::scenarios::Scenario;
use crate::{Action, GameNight, HouseState};
use serde::{Deserialize, Serialize};

/// Current game state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub houses: Vec<HouseState>,
    pub agent_positions: Vec<u8>,
    pub agent_signals: Vec<u8>,
    pub last_actions: Vec<Action>,
    pub night: u32,
    pub done: bool,
}

/// Game result after completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameResult {
    pub scenario: Scenario,
    pub nights: Vec<GameNight>,
    pub final_score: f32,
    pub agent_scores: Vec<f32>,
}

/// Step result
#[derive(Debug, Clone)]
pub struct StepResult {
    pub rewards: Vec<f32>,
    pub done: bool,
    pub info: serde_json::Value,
}
