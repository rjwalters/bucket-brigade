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

        // Issue #254: the WASM frontend (browser UI in `web/`) is fixed at
        // 10 houses visually and assumes that ring length throughout the
        // TypeScript layer. Approach 3 of the issue's curator analysis
        // explicitly defers the WASM/UI rework to a follow-up issue, so we
        // reject non-10 scenarios here with a clear message rather than
        // letting them render broken or crash deep in the renderer. The
        // Rust core, PyO3 surface, and Python envs all support arbitrary
        // `num_houses`; non-10 scenarios (e.g. `v2_minimal`) are
        // Python-only for now.
        if scenario.num_houses != 10 {
            return Err(JsValue::from_str(&format!(
                "WASM frontend is fixed at 10 houses; scenario has num_houses={}. \
                 Non-10-house scenarios (e.g. v2_minimal, issue #254) are \
                 Python-only — use bucket_brigade.envs.* from Python instead.",
                scenario.num_houses
            )));
        }

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
        // Issue #235: action vector is [house, mode, signal] (length 3).
        // For backward compat with already-deployed frontends we first try
        // to deserialize the post-#235 length-3 form; if that fails, we
        // fall back to the legacy length-2 form and promote it to length-3
        // by copying the mode bit into the signal slot (honest default).
        // JavaScript callers should migrate to the length-3 form so they
        // can emit deceptive signals; the fallback is purely transitional.
        let actions: Vec<[u8; 3]> = match serde_json::from_str::<Vec<[u8; 3]>>(actions_json) {
            Ok(a) => a,
            Err(_) => {
                let legacy: Vec<[u8; 2]> = serde_json::from_str(actions_json)
                    .map_err(|e| JsValue::from_str(&format!("Failed to parse actions: {}", e)))?;
                legacy.into_iter().map(|a| [a[0], a[1], a[1]]).collect()
            }
        };

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
    /// Build a Scenario from JS-land. Per issue #198 the ownership reward
    /// fields are per-agent vectors; for the wasm constructor we still
    /// accept a scalar (auto-promoted to a length-10 vector to match the
    /// 10-house ring / max-agent count). Callers that need explicit
    /// per-agent vectors should build a JSON scenario and round-trip it
    /// through ``serde_json`` instead.
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
    ) -> Result<WasmScenario, JsValue> {
        const DEFAULT_LEN: usize = 10;
        let inner = Scenario {
            prob_fire_spreads_to_neighbor,
            prob_solo_agent_extinguishes_fire,
            prob_house_catches_fire,
            team_reward_house_survives,
            team_penalty_house_burns,
            cost_to_work_one_night,
            min_nights,
            reward_own_house_survives: vec![
                reward_own_house_survives.unwrap_or(100.0);
                DEFAULT_LEN
            ],
            reward_other_house_survives: vec![
                reward_other_house_survives.unwrap_or(50.0);
                DEFAULT_LEN
            ],
            penalty_own_house_burns: vec![penalty_own_house_burns.unwrap_or(0.0); DEFAULT_LEN],
            penalty_other_house_burns: vec![penalty_other_house_burns.unwrap_or(0.0); DEFAULT_LEN],
            // Issue #203 spatial-cost fields. The WASM constructor uses
            // pre-#203 defaults so existing JS/TS callers are unchanged.
            // To consume the `positional_default` scenario from JS, use
            // the JSON serialization path (`Scenario::to_json` /
            // `from_json`) which preserves the new fields.
            agent_home_positions: Vec::new(),
            distance_cost_alpha: 0.0,
            distance_metric: "ring_arc".to_string(),
            // Issue #254: WASM frontend is fixed at 10 houses; hardcode
            // here so the rest of the constructor path can't accidentally
            // produce a non-10 scenario via the WASM surface.
            num_houses: 10,
            // Issue #259: action-conditioned shaping defaults to off so
            // existing JS/TS callers see byte-identical rewards. The
            // browser UI doesn't expose these knobs yet; to consume a
            // scenario with shaping enabled, route through the JSON path
            // (``Scenario::to_json`` / ``from_json``) which preserves all
            // fields.
            action_shaping_alpha: 0.0,
            action_shaping_beta: 0.0,
            // Issue #265: dense progress shaping defaults to off so existing
            // JS/TS callers see byte-identical rewards. Routed through the
            // JSON path (``Scenario::to_json`` / ``from_json``) when non-zero.
            progress_shaping_coef: 0.0,
            // Issue #283: potential-based team-welfare shaping defaults to
            // off so existing JS/TS callers see byte-identical rewards.
            // The browser UI doesn't expose these knobs; to consume a
            // scenario with shaping enabled, route through the JSON path.
            team_welfare_lambda: 0.0,
            team_welfare_gamma: 1.0,
            team_welfare_kind: "none".to_string(),
        };
        // Issue #222: route programmatic construction through the allowlist
        // validator so future kwargs additions can't reintroduce silent
        // ring-arc fallback.
        inner.validate().map_err(|e| JsValue::from_str(&e))?;
        Ok(Self { inner })
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
