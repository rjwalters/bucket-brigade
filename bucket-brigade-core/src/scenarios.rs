use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Maximum number of agents supported. Mirrors the 10-house ring constraint
/// in the engine: at most one owner per house, so num_agents <= 10. Used as
/// the default vector length for the four ownership reward fields.
const MAX_AGENTS: usize = 10;

/// Default for `Scenario::num_houses` (issue #254).
///
/// Every pre-#254 scenario assumed a 10-house ring; this default keeps JSON
/// deserialization, programmatic construction, and the WASM frontend
/// bit-exactly backward-compatible. New scenarios (`v2_minimal`) override it
/// by setting the field explicitly. The WASM surface (`wasm.rs`) panics if
/// it sees any other value — the browser UI is still hard-coded to 10
/// houses visually, so non-10 scenarios are Python-only for now.
fn default_num_houses() -> u8 {
    10
}

/// Deserialize a value that is either a scalar `f32` or an array of `f32`s.
///
/// This allows JSON scenario definitions (and other external serialized
/// scenarios) to keep using scalar values for the per-agent ownership reward
/// fields. A scalar `v` is promoted to `vec![v; MAX_AGENTS]`; an explicit
/// array is taken as-is. Mirrors the auto-promotion behavior in the Python
/// `Scenario.__post_init__` (issue #198).
fn deserialize_scalar_or_vec<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Number(n) => {
            let scalar = n
                .as_f64()
                .ok_or_else(|| D::Error::custom("expected numeric scalar"))?
                as f32;
            Ok(vec![scalar; MAX_AGENTS])
        }
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let n = item
                    .as_f64()
                    .ok_or_else(|| D::Error::custom("expected numeric array element"))?
                    as f32;
                out.push(n);
            }
            Ok(out)
        }
        _ => Err(D::Error::custom(
            "expected scalar or array for per-agent ownership reward field",
        )),
    }
}

/// Default for `Scenario::distance_cost_alpha`. Zero preserves bit-exact
/// backward compatibility with pre-#203 behavior (no spatial cost term).
fn default_distance_cost_alpha() -> f32 {
    0.0
}

/// Default for `Scenario::action_shaping_alpha`. Zero preserves bit-exact
/// backward compatibility with pre-#259 behavior (no action-conditioned
/// reward shaping). When non-zero, each worker that participates in
/// extinguishing a fire receives a credit-shared bonus of
/// `alpha * (1 / workers_at_house_this_step)` (issue #259).
fn default_action_shaping_alpha() -> f32 {
    0.0
}

/// Default for `Scenario::action_shaping_beta`. Zero preserves bit-exact
/// backward compatibility with pre-#259 behavior. When non-zero, each
/// agent working at a house that started the step SAFE and is still SAFE
/// at end-of-step receives a flat `beta` bonus for preventive presence
/// (issue #259).
fn default_action_shaping_beta() -> f32 {
    0.0
}

/// Allowed values for `Scenario::distance_metric`.
///
/// `engine/rewards.rs` currently consumes `distance_cost_alpha + ring_dist_10(...)`
/// unconditionally — it does not branch on `distance_metric`. That makes any
/// unrecognized string a silent ring-arc fallback. To prevent that, the
/// deserializer (`deserialize_distance_metric`) and the programmatic-construction
/// chokepoint (`Scenario::validate`) both reject any value not listed here.
///
/// Keep in sync with the Python validator in
/// `bucket_brigade/envs/scenarios_generated.py::Scenario.__post_init__`.
pub const ALLOWED_DISTANCE_METRICS: &[&str] = &["ring_arc"];

/// Default for `Scenario::distance_metric`. `"ring_arc"` is the only supported
/// value today; the field is future-proofing for alternative geometries.
fn default_distance_metric() -> String {
    "ring_arc".to_string()
}

/// Custom serde deserializer for `Scenario::distance_metric` that enforces the
/// `ALLOWED_DISTANCE_METRICS` allowlist. Without this, the field accepts any
/// string and silently falls back to ring-arc geometry at runtime (since
/// `engine/rewards.rs` never reads the field) — see issue #222.
fn deserialize_distance_metric<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    let s = String::deserialize(deserializer)?;
    if !ALLOWED_DISTANCE_METRICS.contains(&s.as_str()) {
        return Err(D::Error::custom(format!(
            "Scenario.distance_metric={s:?} is not supported; allowed values: {ALLOWED_DISTANCE_METRICS:?}"
        )));
    }
    Ok(s)
}

/// Default for `Scenario::agent_home_positions`. Empty means "fall back to the
/// existing `house_owners` round-robin assignment" so the field is optional in
/// JSON and pre-#203 scenarios behave identically. When set, length must equal
/// `num_agents` at engine instantiation time (see `engine/core.rs`).
fn default_agent_home_positions() -> Vec<u8> {
    Vec::new()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    // Topology (issue #254): number of houses on the ring. Defaults to 10
    // for backward compatibility with every pre-#254 scenario; serde reads
    // the field as optional so existing JSON files don't need updating.
    // The engine, Python envs, and PyO3 surface all read this field instead
    // of the literal `10`. The WASM frontend (`wasm.rs`) panics if it
    // receives a scenario with `num_houses != 10` because the browser UI
    // still assumes a 10-house ring visually.
    #[serde(default = "default_num_houses")]
    pub num_houses: u8,

    // Fire dynamics
    pub prob_fire_spreads_to_neighbor: f32, // Probability fire spreads to adjacent house
    pub prob_solo_agent_extinguishes_fire: f32, // Probability one agent extinguishes fire
    pub prob_house_catches_fire: f32,       // Probability house catches fire each night

    // Team scoring (collective outcome)
    pub team_reward_house_survives: f32, // Team reward for each house that survives
    pub team_penalty_house_burns: f32,   // Team penalty for each house that burns

    // Individual rewards (ownership-based, per-agent vectors).
    //
    // As of issue #198 these four fields are `Vec<f32>` with one entry per
    // agent. Each is indexed by ``agent_id`` in the reward computation
    // (see `engine/rewards.rs`). Deserialization accepts either a scalar
    // (auto-promoted to `vec![scalar; MAX_AGENTS]`) or an explicit array,
    // mirroring the Python ``Scenario.__post_init__`` promotion semantics.
    #[serde(deserialize_with = "deserialize_scalar_or_vec")]
    pub reward_own_house_survives: Vec<f32>, // Per-agent reward when own house survives
    #[serde(deserialize_with = "deserialize_scalar_or_vec")]
    pub reward_other_house_survives: Vec<f32>, // Per-agent reward when other house survives
    #[serde(deserialize_with = "deserialize_scalar_or_vec")]
    pub penalty_own_house_burns: Vec<f32>, // Per-agent penalty when own house burns
    #[serde(deserialize_with = "deserialize_scalar_or_vec")]
    pub penalty_other_house_burns: Vec<f32>, // Per-agent penalty when other house burns

    // Costs and structure
    pub cost_to_work_one_night: f32, // Cost incurred when agent chooses to work
    pub min_nights: u32,             // Minimum nights before game can end

    // Spatial cost asymmetry (issue #203, optional, additive).
    //
    // Per-agent "home position" on the 10-house ring. When non-empty (length
    // must equal `num_agents` at engine init), the work cost for agent `i`
    // working at house `j` is `cost_to_work_one_night + distance_cost_alpha *
    // ring_dist(agent_home_positions[i], j)`. When empty (the default for every
    // pre-#203 scenario) the engine falls back to the existing `house_owners`
    // assignment to derive a home position; behavior is still unchanged because
    // `distance_cost_alpha` defaults to `0.0` (the spatial term collapses to
    // zero). Serde defaults make all three fields optional in JSON.
    #[serde(default = "default_agent_home_positions")]
    pub agent_home_positions: Vec<u8>,
    #[serde(default = "default_distance_cost_alpha")]
    pub distance_cost_alpha: f32,
    #[serde(
        default = "default_distance_metric",
        deserialize_with = "deserialize_distance_metric"
    )]
    pub distance_metric: String,

    // Action-conditioned reward shaping (issue #259, optional, additive).
    //
    // Both default to `0.0` so every pre-#259 scenario is bit-exactly
    // preserved (the engine takes a fast-path skip when both are zero;
    // see `engine/rewards.rs`).
    //
    // `action_shaping_alpha` rewards workers who participated in
    // extinguishing a fire this step. Credit is split among co-extinguishers:
    // each worker at house `h` (where `h` transitioned BURNING -> SAFE this
    // step) receives `alpha * (1 / workers_at_h)`. Strict BURNING(1) -> SAFE(0)
    // transition only — stricter than the save-event check in the existing
    // per-house ownership loop, which fires on any non-SAFE -> SAFE
    // transition (RUINED -> SAFE is impossible in current dynamics but
    // type-wise covered).
    //
    // `action_shaping_beta` rewards agents working at houses that were
    // SAFE at start-of-step and are still SAFE at end-of-step — preventive
    // presence. Flat per-agent bonus; not credit-shared.
    //
    // REST actions (`action[1] == 0`) never receive either bonus.
    #[serde(default = "default_action_shaping_alpha")]
    pub action_shaping_alpha: f32,
    #[serde(default = "default_action_shaping_beta")]
    pub action_shaping_beta: f32,
}

impl Scenario {
    /// Validate fields that have allowlist constraints.
    ///
    /// The serde deserializer (`deserialize_distance_metric`) catches unknown
    /// `distance_metric` values on the JSON path, but programmatic construction
    /// (e.g. building a `Scenario` literal in Rust, the `PyScenario::new`
    /// kwargs path, or the WASM constructor) bypasses serde entirely. This
    /// helper is the single chokepoint for re-checking the allowlist on those
    /// paths; it is called from `BucketBrigade::new` so the engine fails fast
    /// rather than running with a silent ring-arc fallback (issue #222).
    pub fn validate(&self) -> Result<(), String> {
        if !ALLOWED_DISTANCE_METRICS.contains(&self.distance_metric.as_str()) {
            return Err(format!(
                "Scenario.distance_metric={:?} is not supported; allowed values: {:?}",
                self.distance_metric, ALLOWED_DISTANCE_METRICS
            ));
        }
        Ok(())
    }
}

/// Predefined scenarios.
///
/// Originally a `phf::Map`, but the four ownership reward fields now hold
/// `Vec<f32>` values (issue #198), which can't be constructed in a
/// `const` context. We use `std::sync::LazyLock<HashMap>` instead; the
/// API surface (`.get`, `.keys`, `.entries`-equivalent via `.iter`) is the
/// same at call sites.
pub static SCENARIOS: LazyLock<HashMap<&'static str, Scenario>> = LazyLock::new(|| {
    // Helper: build a per-agent reward vector of length `MAX_AGENTS` from a
    // scalar value. Keeps the static scenario data terse while matching the
    // Python scalar->list promotion semantics.
    fn per_agent(v: f32) -> Vec<f32> {
        vec![v; MAX_AGENTS]
    }

    let mut m = HashMap::new();

    // Standard difficulty scenarios
    //
    // NOTE: ownership rewards rebalanced #197 (1.0/2.0 -> 20.0/40.0) to give PPO a
    // per-agent gradient signal. Preserves 1:2 ratio. Team rewards unchanged
    // (preserves Slepian-Wolf protocol's reward scale). #198 generalized the
    // ownership fields to per-agent vectors; the rebalanced scalars are wrapped
    // via the per_agent() helper.
    //
    // NOTE: every entry below uses `agent_home_positions: Vec::new()`,
    // `distance_cost_alpha: 0.0`, and `distance_metric: "ring_arc"` — the
    // pre-#203 defaults. The `positional_default` scenario at the bottom
    // overrides these to enable spatial cost asymmetry.
    m.insert(
        "default",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.25,
            prob_solo_agent_extinguishes_fire: 0.5,
            prob_house_catches_fire: 0.02,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.5,
            min_nights: 12,
            reward_own_house_survives: per_agent(20.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(40.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "easy",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.1,
            prob_solo_agent_extinguishes_fire: 0.8,
            prob_house_catches_fire: 0.01,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.5,
            min_nights: 10,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "hard",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.4,
            prob_solo_agent_extinguishes_fire: 0.3,
            prob_house_catches_fire: 0.05,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.5,
            min_nights: 15,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    // Research scenarios testing specific cooperation dynamics
    m.insert(
        "trivial_cooperation",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.15,
            prob_solo_agent_extinguishes_fire: 0.9,
            prob_house_catches_fire: 0.0, // Note: Python uses rho_ignite=0.1 + no sparks
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.5,
            min_nights: 12,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "early_containment",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.35,
            prob_solo_agent_extinguishes_fire: 0.6,
            prob_house_catches_fire: 0.02,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.5,
            min_nights: 12,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "greedy_neighbor",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.15,
            prob_solo_agent_extinguishes_fire: 0.4,
            prob_house_catches_fire: 0.02,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 1.0, // High work cost creates social dilemma
            min_nights: 12,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "sparse_heroics",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.1,
            prob_solo_agent_extinguishes_fire: 0.5,
            prob_house_catches_fire: 0.02,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.8,
            min_nights: 20, // Longer games
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "rest_trap",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.05,
            prob_solo_agent_extinguishes_fire: 0.95,
            prob_house_catches_fire: 0.02,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.2,
            min_nights: 12,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "chain_reaction",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.45,
            prob_solo_agent_extinguishes_fire: 0.6,
            prob_house_catches_fire: 0.03,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.7,
            min_nights: 15,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "deceptive_calm",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.25,
            prob_solo_agent_extinguishes_fire: 0.6,
            prob_house_catches_fire: 0.05, // Occasional sparks
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.4,
            min_nights: 20, // Long games
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "overcrowding",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.2,
            prob_solo_agent_extinguishes_fire: 0.3, // Low efficiency
            prob_house_catches_fire: 0.02,
            team_reward_house_survives: 50.0, // Lower reward
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.6,
            min_nights: 12,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m.insert(
        "mixed_motivation",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.3,
            prob_solo_agent_extinguishes_fire: 0.5,
            prob_house_catches_fire: 0.03,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.6,
            min_nights: 15,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    // Issue #199 sanity-check scenario. Per-agent ownership signal is dominant
    // (50/100 own vs 0 other) and team signal is reduced (10 vs default 100).
    // The Python-side definition in scenarios.json fixes the vectors at length
    // 4 (canonical num_agents). Here we promote to MAX_AGENTS (10) to match
    // the Rust struct invariant; engine indexing uses `agent_id < num_agents`
    // so the extra entries are inert. Keeps the 50/100/0 magnitudes consistent
    // regardless of how the scenario is materialized.
    m.insert(
        "minimal_specialization",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.25,
            prob_solo_agent_extinguishes_fire: 0.5,
            prob_house_catches_fire: 0.02,
            team_reward_house_survives: 10.0,
            team_penalty_house_burns: 10.0,
            cost_to_work_one_night: 0.5,
            min_nights: 12,
            reward_own_house_survives: per_agent(50.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(100.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    // Issue #203 option A: spatial cost asymmetry on the 10-ring. Same reward
    // magnitudes as `default`, but per-agent work cost is
    // `base_cost + alpha * ring_dist(home, target)`. Home positions
    // [0, 3, 5, 8] give max spread on a 10-ring for 4 agents; alpha = 0.1
    // means working 5 steps from home costs 1.0 (2x the base). This creates
    // a per-agent gradient that doesn't rely on reward magnitude differences.
    m.insert(
        "positional_default",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.25,
            prob_solo_agent_extinguishes_fire: 0.5,
            prob_house_catches_fire: 0.02,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.5,
            min_nights: 12,
            reward_own_house_survives: per_agent(20.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(40.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: vec![0, 3, 5, 8],
            distance_cost_alpha: 0.1,
            distance_metric: "ring_arc".to_string(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    // Issue #254 / option E of architect proposal #234: a minimal 2-house x
    // 4-agent diagnostic for PPO learnability. Joint action space shrinks
    // from `4 * (10 * 2 * 2)` = 1,600 per-agent moves to `4 * (2 * 2 * 2)`
    // = 32 per-agent moves, so PPO has a tractable exploration problem.
    // Ownership pattern is round-robin: houses 0 and 1 are owned by
    // agents 0 and 1 respectively (agents 2 and 3 are unowned-house
    // workers under the pre-#254 `house_owners = i % num_agents`
    // semantics). Ignition rate is bumped to 0.05 because with only
    // 2 houses the default 0.02 leaves most episodes empty of fires.
    m.insert(
        "v2_minimal",
        Scenario {
            prob_fire_spreads_to_neighbor: 0.25,
            prob_solo_agent_extinguishes_fire: 0.5,
            prob_house_catches_fire: 0.05,
            team_reward_house_survives: 10.0,
            team_penalty_house_burns: 10.0,
            cost_to_work_one_night: 0.5,
            min_nights: 8,
            reward_own_house_survives: per_agent(50.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(100.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: default_distance_metric(),
            num_houses: 2,
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        },
    );

    m
});

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper for building uniform per-agent reward vectors in tests.
    fn per_agent(v: f32) -> Vec<f32> {
        vec![v; MAX_AGENTS]
    }

    #[test]
    fn test_all_scenarios_exist() {
        // Standard difficulty scenarios
        assert!(SCENARIOS.get("default").is_some());
        assert!(SCENARIOS.get("easy").is_some());
        assert!(SCENARIOS.get("hard").is_some());

        // Research scenarios
        assert!(SCENARIOS.get("trivial_cooperation").is_some());
        assert!(SCENARIOS.get("early_containment").is_some());
        assert!(SCENARIOS.get("greedy_neighbor").is_some());
        assert!(SCENARIOS.get("sparse_heroics").is_some());
        assert!(SCENARIOS.get("rest_trap").is_some());
        assert!(SCENARIOS.get("chain_reaction").is_some());
        assert!(SCENARIOS.get("deceptive_calm").is_some());
        assert!(SCENARIOS.get("overcrowding").is_some());
        assert!(SCENARIOS.get("mixed_motivation").is_some());
        assert!(SCENARIOS.get("minimal_specialization").is_some());
        assert!(SCENARIOS.get("positional_default").is_some());
        // Issue #254: v2_minimal is the 2x4 PPO learnability diagnostic.
        assert!(SCENARIOS.get("v2_minimal").is_some());
    }

    #[test]
    fn test_scenario_not_found() {
        assert!(SCENARIOS.get("nonexistent").is_none());
        assert!(SCENARIOS.get("").is_none());
    }

    #[test]
    fn test_trivial_cooperation_values() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap();
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.15);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.9);
        assert_eq!(scenario.team_reward_house_survives, 100.0);
        assert_eq!(scenario.team_penalty_house_burns, 100.0);
        assert_eq!(scenario.cost_to_work_one_night, 0.5);
        assert_eq!(scenario.prob_house_catches_fire, 0.0);
        assert_eq!(scenario.min_nights, 12);
    }

    #[test]
    fn test_early_containment_values() {
        let scenario = SCENARIOS.get("early_containment").unwrap();
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.35);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.6);
        assert_eq!(scenario.prob_house_catches_fire, 0.02);
    }

    #[test]
    fn test_greedy_neighbor_values() {
        let scenario = SCENARIOS.get("greedy_neighbor").unwrap();
        assert_eq!(scenario.cost_to_work_one_night, 1.0);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.4);
    }

    #[test]
    fn test_default_values() {
        let scenario = SCENARIOS.get("default").unwrap();
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.25);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.5);
        assert_eq!(scenario.prob_house_catches_fire, 0.02);
        assert_eq!(scenario.team_reward_house_survives, 100.0);
        assert_eq!(scenario.team_penalty_house_burns, 100.0);
    }

    #[test]
    fn test_all_scenarios_valid_probabilities() {
        for (name, scenario) in SCENARIOS.iter() {
            assert!(
                scenario.prob_fire_spreads_to_neighbor >= 0.0
                    && scenario.prob_fire_spreads_to_neighbor <= 1.0,
                "Scenario '{}' has invalid prob_fire_spreads_to_neighbor: {}",
                name,
                scenario.prob_fire_spreads_to_neighbor
            );
            assert!(
                scenario.prob_solo_agent_extinguishes_fire >= 0.0
                    && scenario.prob_solo_agent_extinguishes_fire <= 1.0,
                "Scenario '{}' has invalid prob_solo_agent_extinguishes_fire: {}",
                name,
                scenario.prob_solo_agent_extinguishes_fire
            );
            assert!(
                scenario.prob_house_catches_fire >= 0.0 && scenario.prob_house_catches_fire <= 1.0,
                "Scenario '{}' has invalid prob_house_catches_fire: {}",
                name,
                scenario.prob_house_catches_fire
            );
        }
    }

    #[test]
    fn test_scenario_clone() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.15);
    }

    #[test]
    fn test_scenario_serialization() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap();
        let json = serde_json::to_string(scenario).unwrap();
        assert!(json.contains("\"prob_fire_spreads_to_neighbor\":0.15"));

        let deserialized: Scenario = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.prob_fire_spreads_to_neighbor,
            scenario.prob_fire_spreads_to_neighbor
        );
    }

    /// Per-agent ownership reward fields round-trip through serde as arrays.
    #[test]
    fn test_scenario_ownership_vector_roundtrip() {
        let scenario = SCENARIOS.get("default").unwrap();
        let json = serde_json::to_string(scenario).unwrap();
        // Serialized form is an array (the canonical post-#198 representation).
        // Value is 20.0 post-#197 ownership rebalance on the default scenario.
        assert!(json.contains("\"reward_own_house_survives\":[20.0"));
        let deserialized: Scenario = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.reward_own_house_survives,
            scenario.reward_own_house_survives
        );
    }

    /// Backward compatibility: a scalar JSON value for an ownership reward
    /// field deserializes into a per-agent vector via the
    /// `deserialize_scalar_or_vec` helper. This mirrors the Python
    /// `Scenario.__post_init__` promotion semantics (issue #198).
    #[test]
    fn test_scenario_scalar_backward_compat_deserialize() {
        let json = r#"{
            "prob_fire_spreads_to_neighbor": 0.25,
            "prob_solo_agent_extinguishes_fire": 0.5,
            "prob_house_catches_fire": 0.02,
            "team_reward_house_survives": 100.0,
            "team_penalty_house_burns": 100.0,
            "reward_own_house_survives": 1.0,
            "reward_other_house_survives": 0.0,
            "penalty_own_house_burns": 2.0,
            "penalty_other_house_burns": 0.0,
            "cost_to_work_one_night": 0.5,
            "min_nights": 12
        }"#;
        let scenario: Scenario = serde_json::from_str(json).unwrap();
        // Scalar 1.0 should expand to MAX_AGENTS == 10 entries.
        assert_eq!(scenario.reward_own_house_survives.len(), MAX_AGENTS);
        assert!(scenario.reward_own_house_survives.iter().all(|&v| v == 1.0));
        assert!(scenario.penalty_own_house_burns.iter().all(|&v| v == 2.0));
    }

    #[test]
    fn test_scenario_count() {
        let count = SCENARIOS.keys().count();
        assert_eq!(
            count, 15,
            "Expected 15 predefined scenarios (3 difficulty + 9 research + 1 sanity-check + 1 positional + 1 v2 minimal)"
        );
    }

    /// Issue #199: ``minimal_specialization`` is the algorithm-vs-env disambiguator.
    /// Per-agent ownership signal must dominate the team signal: 50 own-save and
    /// 100 own-burn vs. 10 team. ``reward_other_house_survives`` and
    /// ``penalty_other_house_burns`` must be zero so cross-agent reward streams
    /// fully decorrelate.
    #[test]
    fn test_minimal_specialization_values() {
        let scenario = SCENARIOS.get("minimal_specialization").unwrap();
        assert_eq!(scenario.team_reward_house_survives, 10.0);
        assert_eq!(scenario.team_penalty_house_burns, 10.0);
        assert!(scenario
            .reward_own_house_survives
            .iter()
            .all(|&v| v == 50.0));
        assert!(scenario
            .reward_other_house_survives
            .iter()
            .all(|&v| v == 0.0));
        assert!(scenario.penalty_own_house_burns.iter().all(|&v| v == 100.0));
        assert!(scenario.penalty_other_house_burns.iter().all(|&v| v == 0.0));
        // Other dynamics copied from `default`.
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.25);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.5);
        assert_eq!(scenario.prob_house_catches_fire, 0.02);
        assert_eq!(scenario.cost_to_work_one_night, 0.5);
        assert_eq!(scenario.min_nights, 12);
    }

    #[test]
    fn test_scenarios_have_positive_rewards() {
        for (name, scenario) in SCENARIOS.iter() {
            assert!(
                scenario.team_reward_house_survives > 0.0,
                "Scenario '{}' should have positive reward for saved houses",
                name
            );
            assert!(
                scenario.team_penalty_house_burns > 0.0,
                "Scenario '{}' should have positive penalty for ruined houses",
                name
            );
        }
    }

    // Scenario-specific gameplay characteristic tests
    // These verify that each scenario has the parameters that create its intended strategic dynamics

    #[test]
    fn test_trivial_cooperation_is_easy() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap();
        // Should have high extinguish rate and low spread
        assert!(
            scenario.prob_solo_agent_extinguishes_fire >= 0.8,
            "Trivial cooperation should have high extinguish rate"
        );
        assert!(
            scenario.prob_fire_spreads_to_neighbor <= 0.2,
            "Trivial cooperation should have low spread rate"
        );
        assert!(
            scenario.cost_to_work_one_night <= 0.5,
            "Trivial cooperation should have low work cost"
        );
    }

    #[test]
    fn test_greedy_neighbor_creates_social_dilemma() {
        let scenario = SCENARIOS.get("greedy_neighbor").unwrap();
        // Should have high work cost to create free-riding incentive
        assert!(
            scenario.cost_to_work_one_night >= 0.8,
            "Greedy neighbor should have high work cost to create social dilemma"
        );
    }

    #[test]
    fn test_early_containment_is_aggressive() {
        let scenario = SCENARIOS.get("early_containment").unwrap();
        // Should have high spread rate requiring fast response
        assert!(
            scenario.prob_fire_spreads_to_neighbor >= 0.3,
            "Early containment should have high spread rate"
        );
    }

    #[test]
    fn test_chain_reaction_has_highest_spread() {
        let scenario = SCENARIOS.get("chain_reaction").unwrap();
        // Chain reaction should have the highest spread rate
        assert!(
            scenario.prob_fire_spreads_to_neighbor >= 0.4,
            "Chain reaction should have very high spread rate"
        );
        for (name, other) in SCENARIOS.iter() {
            if name != &"chain_reaction" {
                assert!(
                    scenario.prob_fire_spreads_to_neighbor >= other.prob_fire_spreads_to_neighbor,
                    "Chain reaction should have highest spread rate, but {} has higher",
                    name
                );
            }
        }
    }

    #[test]
    fn test_rest_trap_has_highest_extinguish_rate() {
        let scenario = SCENARIOS.get("rest_trap").unwrap();
        // Rest trap should have very high extinguish rate (fires usually extinguish themselves)
        assert!(
            scenario.prob_solo_agent_extinguishes_fire >= 0.9,
            "Rest trap should have very high extinguish rate"
        );
        assert!(
            scenario.prob_fire_spreads_to_neighbor <= 0.1,
            "Rest trap should have very low spread rate"
        );
        assert!(
            scenario.cost_to_work_one_night <= 0.3,
            "Rest trap should have low work cost"
        );
    }

    #[test]
    fn test_sparse_heroics_has_long_games() {
        let scenario = SCENARIOS.get("sparse_heroics").unwrap();
        // Sparse heroics should have longer minimum game length
        assert!(
            scenario.min_nights >= 15,
            "Sparse heroics should have longer minimum nights"
        );
    }

    #[test]
    fn test_deceptive_calm_has_long_games() {
        let scenario = SCENARIOS.get("deceptive_calm").unwrap();
        // Deceptive calm should have longer games with occasional sparks
        assert!(
            scenario.min_nights >= 15,
            "Deceptive calm should have longer minimum nights"
        );
        assert!(
            scenario.prob_house_catches_fire >= 0.03,
            "Deceptive calm should have occasional sparks"
        );
    }

    #[test]
    fn test_overcrowding_has_low_efficiency() {
        let scenario = SCENARIOS.get("overcrowding").unwrap();
        // Overcrowding should have low extinguish efficiency
        assert!(
            scenario.prob_solo_agent_extinguishes_fire <= 0.4,
            "Overcrowding should have low extinguish efficiency"
        );
        // May also have lower team reward
        assert!(
            scenario.team_reward_house_survives <= 100.0,
            "Overcrowding may have reduced rewards"
        );
    }

    #[test]
    fn test_easy_vs_hard_difficulty_ordering() {
        let easy = SCENARIOS.get("easy").unwrap();
        let default_scenario = SCENARIOS.get("default").unwrap();
        let hard = SCENARIOS.get("hard").unwrap();

        // Easy should have easier parameters than default
        assert!(
            easy.prob_solo_agent_extinguishes_fire
                > default_scenario.prob_solo_agent_extinguishes_fire,
            "Easy should have higher extinguish rate than default"
        );
        assert!(
            easy.prob_fire_spreads_to_neighbor < default_scenario.prob_fire_spreads_to_neighbor,
            "Easy should have lower spread rate than default"
        );

        // Hard should have harder parameters than default
        assert!(
            hard.prob_solo_agent_extinguishes_fire
                < default_scenario.prob_solo_agent_extinguishes_fire,
            "Hard should have lower extinguish rate than default"
        );
        assert!(
            hard.prob_fire_spreads_to_neighbor > default_scenario.prob_fire_spreads_to_neighbor,
            "Hard should have higher spread rate than default"
        );
    }

    #[test]
    fn test_all_scenarios_use_standard_rewards() {
        // Most scenarios should use standard 100/100 rewards. Exceptions:
        //   - "overcrowding" intentionally uses reward=50 to model
        //     diminishing returns when too many agents pile on.
        //   - "minimal_specialization" (issue #199) intentionally uses
        //     reward=penalty=10 so the per-agent ownership signal (50/100)
        //     dominates the shared team signal — that's the entire point
        //     of the sanity-check scenario.
        //   - "v2_minimal" (issue #254) mirrors `minimal_specialization`'s
        //     reward shape on a 2-house ring as a PPO learnability unit
        //     test (option E of architect proposal #234).
        for (name, scenario) in SCENARIOS.iter() {
            if name == &"overcrowding" || name == &"minimal_specialization" || name == &"v2_minimal"
            {
                continue;
            }
            assert_eq!(
                scenario.team_reward_house_survives, 100.0,
                "Scenario '{}' should use standard reward (100)",
                name
            );
            assert_eq!(
                scenario.team_penalty_house_burns, 100.0,
                "Scenario '{}' should use standard penalty (100)",
                name
            );
        }
    }

    /// Suppress the "unused" warning if `per_agent` is only used by some tests.
    #[test]
    fn test_per_agent_helper_compiles() {
        let v = per_agent(3.5);
        assert_eq!(v.len(), MAX_AGENTS);
        assert!(v.iter().all(|&x| x == 3.5));
    }

    /// Issue #203: ``positional_default`` introduces spatial cost asymmetry.
    /// Same reward magnitudes as ``default``, but with four agents anchored at
    /// home positions [0, 3, 5, 8] and a distance cost coefficient of 0.1.
    #[test]
    fn test_positional_default_values() {
        let scenario = SCENARIOS.get("positional_default").unwrap();
        // Reward magnitudes match `default` (so the only difference is spatial).
        assert_eq!(scenario.team_reward_house_survives, 100.0);
        assert_eq!(scenario.team_penalty_house_burns, 100.0);
        assert_eq!(scenario.cost_to_work_one_night, 0.5);
        assert!(scenario
            .reward_own_house_survives
            .iter()
            .all(|&v| v == 20.0));
        assert!(scenario.penalty_own_house_burns.iter().all(|&v| v == 40.0));
        // New spatial knobs.
        assert_eq!(scenario.agent_home_positions, vec![0u8, 3, 5, 8]);
        assert_eq!(scenario.distance_cost_alpha, 0.1);
        assert_eq!(scenario.distance_metric, "ring_arc");
        // Fire dynamics unchanged from `default`.
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.25);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.5);
        assert_eq!(scenario.prob_house_catches_fire, 0.02);
        assert_eq!(scenario.min_nights, 12);
    }

    /// Backward compat: every scenario other than ``positional_default`` must
    /// preserve the pre-#203 spatial defaults so existing scenarios are
    /// bit-exactly identical to today's behavior.
    #[test]
    fn test_non_positional_scenarios_have_zero_distance_cost() {
        for (name, scenario) in SCENARIOS.iter() {
            if name == &"positional_default" {
                continue;
            }
            assert_eq!(
                scenario.distance_cost_alpha, 0.0,
                "Scenario '{}' must keep distance_cost_alpha=0.0 \
                 to preserve pre-#203 behavior",
                name
            );
            assert!(
                scenario.agent_home_positions.is_empty(),
                "Scenario '{}' must keep agent_home_positions empty \
                 (engine then falls back to house_owners round-robin)",
                name
            );
            assert_eq!(
                scenario.distance_metric, "ring_arc",
                "Scenario '{}' must use the default ring_arc distance metric",
                name
            );
        }
    }

    /// Scalar JSON inputs that omit the new #203 fields should round-trip
    /// through serde, yielding the documented defaults
    /// (alpha=0.0, metric="ring_arc", home_positions=[]).
    ///
    /// Issue #254 extension: also confirms `num_houses` defaults to 10
    /// when omitted, so all 14 pre-#254 scenarios remain bit-exact.
    #[test]
    fn test_scenario_pre203_fields_optional_in_json() {
        let json = r#"{
            "prob_fire_spreads_to_neighbor": 0.25,
            "prob_solo_agent_extinguishes_fire": 0.5,
            "prob_house_catches_fire": 0.02,
            "team_reward_house_survives": 100.0,
            "team_penalty_house_burns": 100.0,
            "reward_own_house_survives": 1.0,
            "reward_other_house_survives": 0.0,
            "penalty_own_house_burns": 2.0,
            "penalty_other_house_burns": 0.0,
            "cost_to_work_one_night": 0.5,
            "min_nights": 12
        }"#;
        let scenario: Scenario = serde_json::from_str(json).unwrap();
        assert_eq!(scenario.distance_cost_alpha, 0.0);
        assert_eq!(scenario.distance_metric, "ring_arc");
        assert!(scenario.agent_home_positions.is_empty());
        // Issue #254: num_houses defaults to 10 when omitted.
        assert_eq!(scenario.num_houses, 10);
    }

    /// Issue #222: unknown ``distance_metric`` values must be rejected at
    /// deserialization time. Without this guard, the engine silently falls
    /// back to ring-arc geometry because ``engine/rewards.rs`` never branches
    /// on the field.
    #[test]
    fn test_unknown_distance_metric_rejected_in_deserialize() {
        let json = r#"{
            "prob_fire_spreads_to_neighbor": 0.25,
            "prob_solo_agent_extinguishes_fire": 0.5,
            "prob_house_catches_fire": 0.02,
            "team_reward_house_survives": 100.0,
            "team_penalty_house_burns": 100.0,
            "reward_own_house_survives": 1.0,
            "reward_other_house_survives": 0.0,
            "penalty_own_house_burns": 2.0,
            "penalty_other_house_burns": 0.0,
            "cost_to_work_one_night": 0.5,
            "min_nights": 12,
            "distance_metric": "euclidean"
        }"#;
        let err = serde_json::from_str::<Scenario>(json)
            .expect_err("euclidean should be rejected by the allowlist");
        let msg = err.to_string();
        assert!(
            msg.contains("distance_metric") && msg.contains("euclidean"),
            "Deserialization error should name the offending field and value, got: {msg}"
        );
        assert!(
            msg.contains("ring_arc"),
            "Error should advertise the allowed values, got: {msg}"
        );
    }

    /// Empty-string, mixed-case, and trailing-whitespace variants must all be
    /// rejected — only the exact ``"ring_arc"`` literal is supported.
    #[test]
    fn test_distance_metric_edge_cases_rejected_in_deserialize() {
        for bogus in ["", "Ring_Arc", "ring_arc ", "RING_ARC", " ring_arc"] {
            let json = format!(
                r#"{{
                    "prob_fire_spreads_to_neighbor": 0.25,
                    "prob_solo_agent_extinguishes_fire": 0.5,
                    "prob_house_catches_fire": 0.02,
                    "team_reward_house_survives": 100.0,
                    "team_penalty_house_burns": 100.0,
                    "reward_own_house_survives": 1.0,
                    "reward_other_house_survives": 0.0,
                    "penalty_own_house_burns": 2.0,
                    "penalty_other_house_burns": 0.0,
                    "cost_to_work_one_night": 0.5,
                    "min_nights": 12,
                    "distance_metric": "{bogus}"
                }}"#
            );
            assert!(
                serde_json::from_str::<Scenario>(&json).is_err(),
                "Expected rejection for distance_metric={bogus:?}, but it was accepted"
            );
        }
    }

    /// Programmatic construction path: building a ``Scenario`` literal with a
    /// bogus ``distance_metric`` should fail ``validate()``. This guards every
    /// non-serde construction site (PyScenario, WasmScenario, engine init).
    #[test]
    fn test_validate_rejects_unknown_distance_metric() {
        let scenario = Scenario {
            prob_fire_spreads_to_neighbor: 0.25,
            prob_solo_agent_extinguishes_fire: 0.5,
            prob_house_catches_fire: 0.02,
            team_reward_house_survives: 100.0,
            team_penalty_house_burns: 100.0,
            cost_to_work_one_night: 0.5,
            min_nights: 12,
            reward_own_house_survives: per_agent(1.0),
            reward_other_house_survives: per_agent(0.0),
            penalty_own_house_burns: per_agent(2.0),
            penalty_other_house_burns: per_agent(0.0),
            agent_home_positions: default_agent_home_positions(),
            distance_cost_alpha: default_distance_cost_alpha(),
            distance_metric: "bogus".to_string(),
            num_houses: default_num_houses(),
            action_shaping_alpha: default_action_shaping_alpha(),
            action_shaping_beta: default_action_shaping_beta(),
        };
        let err = scenario
            .validate()
            .expect_err("bogus metric should fail validate()");
        assert!(
            err.contains("distance_metric"),
            "Error should name the field, got: {err}"
        );
        assert!(
            err.contains("bogus"),
            "Error should name the offending value, got: {err}"
        );
        assert!(
            err.contains("ring_arc"),
            "Error should list allowed values, got: {err}"
        );
    }

    /// ``validate()`` should accept every value listed in the allowlist (and
    /// every predefined scenario which uses the canonical default).
    #[test]
    fn test_validate_accepts_known_distance_metrics() {
        for (name, scenario) in SCENARIOS.iter() {
            assert!(
                scenario.validate().is_ok(),
                "Predefined scenario {name:?} should pass validate()"
            );
        }
    }

    /// Allowlist invariant: ``"ring_arc"`` is always allowed and the default
    /// matches an allowed value.
    #[test]
    fn test_allowlist_contains_default_metric() {
        assert!(ALLOWED_DISTANCE_METRICS.contains(&"ring_arc"));
        assert!(ALLOWED_DISTANCE_METRICS.contains(&default_distance_metric().as_str()));
    }

    /// Issue #254: ``v2_minimal`` is the 2-house x 4-agent PPO learnability
    /// diagnostic (option E from architect proposal #234). Reward shape
    /// mirrors ``minimal_specialization``; ignition rate is bumped to 0.05
    /// because a 2-house ring at 0.02 has too many fire-free episodes.
    #[test]
    fn test_v2_minimal_values() {
        let scenario = SCENARIOS.get("v2_minimal").unwrap();
        assert_eq!(scenario.num_houses, 2);
        assert_eq!(scenario.team_reward_house_survives, 10.0);
        assert_eq!(scenario.team_penalty_house_burns, 10.0);
        assert_eq!(scenario.prob_house_catches_fire, 0.05);
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.25);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.5);
        assert_eq!(scenario.cost_to_work_one_night, 0.5);
        assert_eq!(scenario.min_nights, 8);
        assert!(scenario
            .reward_own_house_survives
            .iter()
            .all(|&v| v == 50.0));
        assert!(scenario
            .reward_other_house_survives
            .iter()
            .all(|&v| v == 0.0));
        assert!(scenario.penalty_own_house_burns.iter().all(|&v| v == 100.0));
        assert!(scenario.penalty_other_house_burns.iter().all(|&v| v == 0.0));
    }

    /// Issue #254: every non-v2 scenario must keep `num_houses = 10` so the
    /// pre-#254 14 scenarios are bit-exactly preserved (the engine paths
    /// that previously hardcoded `10` now read `scenario.num_houses`, but
    /// the result is identical when `num_houses == 10`).
    #[test]
    fn test_non_v2_scenarios_have_ten_houses() {
        for (name, scenario) in SCENARIOS.iter() {
            if name.starts_with("v2_") {
                continue;
            }
            assert_eq!(
                scenario.num_houses, 10,
                "Pre-#254 scenario '{}' must keep num_houses = 10 (bit-exact)",
                name
            );
        }
    }

    /// Issue #254: `num_houses` is optional in JSON and defaults to 10.
    /// Verifies the serde default function is wired up.
    #[test]
    fn test_num_houses_defaults_to_ten_in_json() {
        let json = r#"{
            "prob_fire_spreads_to_neighbor": 0.25,
            "prob_solo_agent_extinguishes_fire": 0.5,
            "prob_house_catches_fire": 0.02,
            "team_reward_house_survives": 100.0,
            "team_penalty_house_burns": 100.0,
            "reward_own_house_survives": 1.0,
            "reward_other_house_survives": 0.0,
            "penalty_own_house_burns": 2.0,
            "penalty_other_house_burns": 0.0,
            "cost_to_work_one_night": 0.5,
            "min_nights": 12
        }"#;
        let scenario: Scenario = serde_json::from_str(json).unwrap();
        assert_eq!(scenario.num_houses, 10);
    }

    /// Issue #254: explicit `num_houses` in JSON is preserved (not silently
    /// overridden by the default).
    #[test]
    fn test_num_houses_explicit_value_preserved() {
        let json = r#"{
            "num_houses": 2,
            "prob_fire_spreads_to_neighbor": 0.25,
            "prob_solo_agent_extinguishes_fire": 0.5,
            "prob_house_catches_fire": 0.02,
            "team_reward_house_survives": 100.0,
            "team_penalty_house_burns": 100.0,
            "reward_own_house_survives": 1.0,
            "reward_other_house_survives": 0.0,
            "penalty_own_house_burns": 2.0,
            "penalty_other_house_burns": 0.0,
            "cost_to_work_one_night": 0.5,
            "min_nights": 12
        }"#;
        let scenario: Scenario = serde_json::from_str(json).unwrap();
        assert_eq!(scenario.num_houses, 2);
    }
}
