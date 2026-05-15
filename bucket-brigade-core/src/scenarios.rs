use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Maximum number of agents supported. Mirrors the 10-house ring constraint
/// in the engine: at most one owner per house, so num_agents <= 10. Used as
/// the default vector length for the four ownership reward fields.
const MAX_AGENTS: usize = 10;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
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
        assert!(scenario
            .reward_own_house_survives
            .iter()
            .all(|&v| v == 1.0));
        assert!(scenario
            .penalty_own_house_burns
            .iter()
            .all(|&v| v == 2.0));
    }

    #[test]
    fn test_scenario_count() {
        let count = SCENARIOS.keys().count();
        assert_eq!(count, 12, "Expected 12 predefined scenarios (3 difficulty + 9 research)");
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
        assert!(scenario.prob_solo_agent_extinguishes_fire >= 0.8, "Trivial cooperation should have high extinguish rate");
        assert!(scenario.prob_fire_spreads_to_neighbor <= 0.2, "Trivial cooperation should have low spread rate");
        assert!(scenario.cost_to_work_one_night <= 0.5, "Trivial cooperation should have low work cost");
    }

    #[test]
    fn test_greedy_neighbor_creates_social_dilemma() {
        let scenario = SCENARIOS.get("greedy_neighbor").unwrap();
        // Should have high work cost to create free-riding incentive
        assert!(scenario.cost_to_work_one_night >= 0.8, "Greedy neighbor should have high work cost to create social dilemma");
    }

    #[test]
    fn test_early_containment_is_aggressive() {
        let scenario = SCENARIOS.get("early_containment").unwrap();
        // Should have high spread rate requiring fast response
        assert!(scenario.prob_fire_spreads_to_neighbor >= 0.3, "Early containment should have high spread rate");
    }

    #[test]
    fn test_chain_reaction_has_highest_spread() {
        let scenario = SCENARIOS.get("chain_reaction").unwrap();
        // Chain reaction should have the highest spread rate
        assert!(scenario.prob_fire_spreads_to_neighbor >= 0.4, "Chain reaction should have very high spread rate");
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
        assert!(scenario.prob_solo_agent_extinguishes_fire >= 0.9, "Rest trap should have very high extinguish rate");
        assert!(scenario.prob_fire_spreads_to_neighbor <= 0.1, "Rest trap should have very low spread rate");
        assert!(scenario.cost_to_work_one_night <= 0.3, "Rest trap should have low work cost");
    }

    #[test]
    fn test_sparse_heroics_has_long_games() {
        let scenario = SCENARIOS.get("sparse_heroics").unwrap();
        // Sparse heroics should have longer minimum game length
        assert!(scenario.min_nights >= 15, "Sparse heroics should have longer minimum nights");
    }

    #[test]
    fn test_deceptive_calm_has_long_games() {
        let scenario = SCENARIOS.get("deceptive_calm").unwrap();
        // Deceptive calm should have longer games with occasional sparks
        assert!(scenario.min_nights >= 15, "Deceptive calm should have longer minimum nights");
        assert!(scenario.prob_house_catches_fire >= 0.03, "Deceptive calm should have occasional sparks");
    }

    #[test]
    fn test_overcrowding_has_low_efficiency() {
        let scenario = SCENARIOS.get("overcrowding").unwrap();
        // Overcrowding should have low extinguish efficiency
        assert!(scenario.prob_solo_agent_extinguishes_fire <= 0.4, "Overcrowding should have low extinguish efficiency");
        // May also have lower team reward
        assert!(scenario.team_reward_house_survives <= 100.0, "Overcrowding may have reduced rewards");
    }

    #[test]
    fn test_easy_vs_hard_difficulty_ordering() {
        let easy = SCENARIOS.get("easy").unwrap();
        let default_scenario = SCENARIOS.get("default").unwrap();
        let hard = SCENARIOS.get("hard").unwrap();

        // Easy should have easier parameters than default
        assert!(easy.prob_solo_agent_extinguishes_fire > default_scenario.prob_solo_agent_extinguishes_fire,
                "Easy should have higher extinguish rate than default");
        assert!(easy.prob_fire_spreads_to_neighbor < default_scenario.prob_fire_spreads_to_neighbor,
                "Easy should have lower spread rate than default");

        // Hard should have harder parameters than default
        assert!(hard.prob_solo_agent_extinguishes_fire < default_scenario.prob_solo_agent_extinguishes_fire,
                "Hard should have lower extinguish rate than default");
        assert!(hard.prob_fire_spreads_to_neighbor > default_scenario.prob_fire_spreads_to_neighbor,
                "Hard should have higher spread rate than default");
    }

    #[test]
    fn test_all_scenarios_use_standard_rewards() {
        // Most scenarios should use standard 100/100 rewards (except overcrowding)
        for (name, scenario) in SCENARIOS.iter() {
            if name != &"overcrowding" {
                assert_eq!(scenario.team_reward_house_survives, 100.0,
                          "Scenario '{}' should use standard reward (100)", name);
                assert_eq!(scenario.team_penalty_house_burns, 100.0,
                          "Scenario '{}' should use standard penalty (100)", name);
            }
        }
    }

    /// Suppress the "unused" warning if `per_agent` is only used by some tests.
    #[test]
    fn test_per_agent_helper_compiles() {
        let v = per_agent(3.5);
        assert_eq!(v.len(), MAX_AGENTS);
        assert!(v.iter().all(|&x| x == 3.5));
    }
}
