use crate::{BucketBrigade, Scenario};
use wasm_bindgen::prelude::*;

/// WASM-compatible Bucket Brigade environment
#[wasm_bindgen]
pub struct WasmBucketBrigade {
    inner: BucketBrigade,
}

#[wasm_bindgen]
impl WasmBucketBrigade {
    #[wasm_bindgen(constructor)]
    pub fn new(scenario_json: &str, num_agents: usize) -> Result<WasmBucketBrigade, JsValue> {
        let scenario: Scenario = serde_json::from_str(scenario_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse scenario: {}", e)))?;

        Ok(Self {
            inner: BucketBrigade::new(scenario, num_agents, None),
        })
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[wasm_bindgen]
    pub fn step(&mut self, actions_json: &str) -> Result<String, JsValue> {
        let actions: Vec<[u8; 2]> = serde_json::from_str(actions_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse actions: {}", e)))?;

        let result = self.inner.step(&actions);

        let response = serde_json::json!({
            "rewards": result.rewards,
            "done": result.done,
            "info": result.info
        });

        Ok(response.to_string())
    }

    #[wasm_bindgen]
    pub fn get_observation(&self, agent_id: usize) -> Result<String, JsValue> {
        let obs = self.inner.get_observation(agent_id);
        serde_json::to_string(&obs)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize observation: {}", e)))
    }

    #[wasm_bindgen]
    pub fn get_current_state(&self) -> Result<String, JsValue> {
        let state = self.inner.get_current_state();
        serde_json::to_string(&state)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize state: {}", e)))
    }

    #[wasm_bindgen]
    pub fn get_result(&self) -> Result<String, JsValue> {
        let result = self.inner.get_result();
        serde_json::to_string(&result)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {}", e)))
    }

    #[wasm_bindgen]
    pub fn is_done(&self) -> bool {
        self.inner.get_current_state().done
    }

    #[wasm_bindgen]
    pub fn get_scenario(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner.scenario)
            .map_err(|_e| JsValue::from_str("Failed to serialize scenario"))
    }
}

/// WASM-compatible Scenario
#[wasm_bindgen]
pub struct WasmScenario {
    inner: Scenario,
}

#[wasm_bindgen]
impl WasmScenario {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prob_fire_spreads_to_neighbor: f32,
        prob_solo_agent_extinguishes_fire: f32,
        prob_house_catches_fire: f32,
        team_reward_house_survives: f32,
        team_penalty_house_burns: f32,
        cost_to_work_one_night: f32,
        min_nights: u32,
        reward_own_house_survives: Option<f32>,
        reward_other_house_survives: Option<f32>,
        penalty_own_house_burns: Option<f32>,
        penalty_other_house_burns: Option<f32>,
    ) -> WasmScenario {
        Self {
            inner: Scenario {
                prob_fire_spreads_to_neighbor,
                prob_solo_agent_extinguishes_fire,
                prob_house_catches_fire,
                team_reward_house_survives,
                team_penalty_house_burns,
                cost_to_work_one_night,
                min_nights,
                reward_own_house_survives: reward_own_house_survives.unwrap_or(100.0),
                reward_other_house_survives: reward_other_house_survives.unwrap_or(50.0),
                penalty_own_house_burns: penalty_own_house_burns.unwrap_or(0.0),
                penalty_other_house_burns: penalty_other_house_burns.unwrap_or(0.0),
            },
        }
    }

    #[wasm_bindgen]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize scenario: {}", e)))
    }
}

/// Get predefined scenario by name
#[wasm_bindgen]
pub fn get_scenario(name: &str) -> Result<String, JsValue> {
    match crate::SCENARIOS.get(name) {
        Some(scenario) => serde_json::to_string(scenario)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize scenario: {}", e))),
        None => Err(JsValue::from_str(&format!("Scenario '{}' not found", name))),
    }
}

/// Get list of available scenario names
#[wasm_bindgen]
pub fn get_scenario_names() -> Vec<String> {
    crate::SCENARIOS.keys().map(|k| k.to_string()).collect()
}
