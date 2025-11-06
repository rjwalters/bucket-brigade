use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    // Fire dynamics
    pub prob_fire_spreads_to_neighbor: f32, // Probability fire spreads to adjacent house
    pub prob_solo_agent_extinguishes_fire: f32, // Probability one agent extinguishes fire
    pub prob_house_catches_fire: f32,       // Probability house catches fire each night

    // Team scoring (collective outcome)
    pub team_reward_house_survives: f32, // Team reward for each house that survives
    pub team_penalty_house_burns: f32,   // Team penalty for each house that burns

    // Individual rewards (ownership-based, for future use in issue #52)
    pub reward_own_house_survives: f32, // Individual reward when own house survives
    pub reward_other_house_survives: f32, // Individual reward when other house survives
    pub penalty_own_house_burns: f32,   // Individual penalty when own house burns
    pub penalty_other_house_burns: f32, // Individual penalty when other house burns

    // Costs and structure
    pub cost_to_work_one_night: f32, // Cost incurred when agent chooses to work
    pub min_nights: u32,             // Minimum nights before game can end
}

pub const SCENARIOS: phf::Map<&'static str, Scenario> = phf::phf_map! {
    // Standard difficulty scenarios
    "default" => Scenario {
        prob_fire_spreads_to_neighbor: 0.25,
        prob_solo_agent_extinguishes_fire: 0.5,
        prob_house_catches_fire: 0.02,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.5,
        min_nights: 12,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "easy" => Scenario {
        prob_fire_spreads_to_neighbor: 0.1,
        prob_solo_agent_extinguishes_fire: 0.8,
        prob_house_catches_fire: 0.01,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.5,
        min_nights: 10,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "hard" => Scenario {
        prob_fire_spreads_to_neighbor: 0.4,
        prob_solo_agent_extinguishes_fire: 0.3,
        prob_house_catches_fire: 0.05,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.5,
        min_nights: 15,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    // Research scenarios testing specific cooperation dynamics
    "trivial_cooperation" => Scenario {
        prob_fire_spreads_to_neighbor: 0.15,
        prob_solo_agent_extinguishes_fire: 0.9,
        prob_house_catches_fire: 0.0, // Note: Python uses rho_ignite=0.1 + no sparks
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.5,
        min_nights: 12,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "early_containment" => Scenario {
        prob_fire_spreads_to_neighbor: 0.35,
        prob_solo_agent_extinguishes_fire: 0.6,
        prob_house_catches_fire: 0.02,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.5,
        min_nights: 12,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "greedy_neighbor" => Scenario {
        prob_fire_spreads_to_neighbor: 0.15,
        prob_solo_agent_extinguishes_fire: 0.4,
        prob_house_catches_fire: 0.02,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 1.0, // High work cost creates social dilemma
        min_nights: 12,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "sparse_heroics" => Scenario {
        prob_fire_spreads_to_neighbor: 0.1,
        prob_solo_agent_extinguishes_fire: 0.5,
        prob_house_catches_fire: 0.02,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.8,
        min_nights: 20, // Longer games
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "rest_trap" => Scenario {
        prob_fire_spreads_to_neighbor: 0.05,
        prob_solo_agent_extinguishes_fire: 0.95,
        prob_house_catches_fire: 0.02,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.2,
        min_nights: 12,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "chain_reaction" => Scenario {
        prob_fire_spreads_to_neighbor: 0.45,
        prob_solo_agent_extinguishes_fire: 0.6,
        prob_house_catches_fire: 0.03,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.7,
        min_nights: 15,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "deceptive_calm" => Scenario {
        prob_fire_spreads_to_neighbor: 0.25,
        prob_solo_agent_extinguishes_fire: 0.6,
        prob_house_catches_fire: 0.05, // Occasional sparks
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.4,
        min_nights: 20, // Long games
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "overcrowding" => Scenario {
        prob_fire_spreads_to_neighbor: 0.2,
        prob_solo_agent_extinguishes_fire: 0.3, // Low efficiency
        prob_house_catches_fire: 0.02,
        team_reward_house_survives: 50.0, // Lower reward
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.6,
        min_nights: 12,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "mixed_motivation" => Scenario {
        prob_fire_spreads_to_neighbor: 0.3,
        prob_solo_agent_extinguishes_fire: 0.5,
        prob_house_catches_fire: 0.03,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.6,
        min_nights: 15,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },
};

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.9); // Updated to match new value
        assert_eq!(scenario.team_reward_house_survives, 100.0);
        assert_eq!(scenario.team_penalty_house_burns, 100.0);
        assert_eq!(scenario.cost_to_work_one_night, 0.5);
        assert_eq!(scenario.prob_house_catches_fire, 0.0); // Updated to match new value
        assert_eq!(scenario.min_nights, 12);
    }

    #[test]
    fn test_early_containment_values() {
        let scenario = SCENARIOS.get("early_containment").unwrap();
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.35);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.6); // Updated to match new value
        assert_eq!(scenario.prob_house_catches_fire, 0.02); // Updated to match new value
    }

    #[test]
    fn test_greedy_neighbor_values() {
        let scenario = SCENARIOS.get("greedy_neighbor").unwrap();
        assert_eq!(scenario.cost_to_work_one_night, 1.0);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.4); // Updated to match new value
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
        for (name, scenario) in SCENARIOS.entries() {
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

    #[test]
    fn test_scenario_count() {
        let count = SCENARIOS.keys().count();
        assert_eq!(count, 12, "Expected 12 predefined scenarios (3 difficulty + 9 research)");
    }

    #[test]
    fn test_scenarios_have_positive_rewards() {
        for (name, scenario) in SCENARIOS.entries() {
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
        for (name, other) in SCENARIOS.entries() {
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
        for (name, scenario) in SCENARIOS.entries() {
            if name != &"overcrowding" {
                assert_eq!(scenario.team_reward_house_survives, 100.0,
                          "Scenario '{}' should use standard reward (100)", name);
                assert_eq!(scenario.team_penalty_house_burns, 100.0,
                          "Scenario '{}' should use standard penalty (100)", name);
            }
        }
    }
}
