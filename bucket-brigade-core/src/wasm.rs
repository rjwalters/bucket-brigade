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
    pub fn new(scenario_json: &str) -> Result<WasmBucketBrigade, JsValue> {
        let scenario: Scenario = serde_json::from_str(scenario_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse scenario: {}", e)))?;

        Ok(Self {
            inner: BucketBrigade::new(scenario, None),
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
        fire_spread_prob: f32,
        extinguish_efficiency: f32,
        team_reward_per_house: f32,
        team_penalty_per_house: f32,
        work_cost_per_night: f32,
        initial_fire_fraction: f32,
        min_nights: u32,
        spontaneous_ignition_prob: f32,
        spontaneous_ignition_nights: u32,
        num_agents: usize,
        owned_house_value: Option<f32>,
        neighbor_house_value: Option<f32>,
    ) -> WasmScenario {
        Self {
            inner: Scenario {
                fire_spread_prob,
                extinguish_efficiency,
                team_reward_per_house,
                team_penalty_per_house,
                work_cost_per_night,
                initial_fire_fraction,
                min_nights,
                spontaneous_ignition_prob,
                spontaneous_ignition_nights,
                num_agents,
                owned_house_value: owned_house_value.unwrap_or(100.0),
                neighbor_house_value: neighbor_house_value.unwrap_or(50.0),
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
