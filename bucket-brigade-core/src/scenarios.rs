use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    // Fire dynamics
    #[serde(alias = "beta")]
    pub fire_spread_prob: f32,        // Probability fire spreads to neighbor (β)
    #[serde(alias = "kappa")]
    pub extinguish_efficiency: f32,   // Efficiency of firefighting (κ)
    #[serde(alias = "rho_ignite")]
    pub initial_fire_fraction: f32,   // Fraction of houses burning at start (ρ)
    #[serde(alias = "p_spark")]
    pub spontaneous_ignition_prob: f32, // Probability of random fire
    #[serde(alias = "n_spark")]
    pub spontaneous_ignition_nights: u32, // How long sparks can occur

    // Rewards and costs
    #[serde(alias = "a")]
    pub team_reward_per_house: f32,   // Team reward for each saved house
    #[serde(alias = "l")]
    pub team_penalty_per_house: f32,  // Team penalty for each ruined house
    #[serde(alias = "c")]
    pub work_cost_per_night: f32,     // Cost to work one night
    #[serde(alias = "a_own")]
    pub owned_house_value: f32,       // Individual reward for own house saved
    #[serde(alias = "a_neighbor")]
    pub neighbor_house_value: f32,    // Individual reward for neighbor house saved

    // Game structure
    #[serde(alias = "n_min")]
    pub min_nights: u32,              // Minimum nights before game can end
    pub num_agents: usize,            // Number of agents in game
}

pub const SCENARIOS: phf::Map<&'static str, Scenario> = phf::phf_map! {
    "trivial_cooperation" => Scenario {
        fire_spread_prob: 0.15,
        extinguish_efficiency: 0.9,
        team_reward_per_house: 100.0,
        team_penalty_per_house: 100.0,
        work_cost_per_night: 0.5,
        initial_fire_fraction: 0.1,
        min_nights: 12,
        spontaneous_ignition_prob: 0.0,
        spontaneous_ignition_nights: 12,
        num_agents: 4,
        owned_house_value: 100.0,      // Owned house saved = +100
        neighbor_house_value: 50.0,    // Neighbor house saved = +50
    },

    "early_containment" => Scenario {
        fire_spread_prob: 0.35,
        extinguish_efficiency: 0.6,
        team_reward_per_house: 100.0,
        team_penalty_per_house: 100.0,
        work_cost_per_night: 0.5,
        initial_fire_fraction: 0.3,
        min_nights: 12,
        spontaneous_ignition_prob: 0.02,
        spontaneous_ignition_nights: 12,
        num_agents: 4,
        owned_house_value: 100.0,      // Owned house saved = +100
        neighbor_house_value: 50.0,    // Neighbor house saved = +50
    },

    "greedy_neighbor" => Scenario {
        fire_spread_prob: 0.15,
        extinguish_efficiency: 0.4,
        team_reward_per_house: 100.0,
        team_penalty_per_house: 100.0,
        work_cost_per_night: 1.0,
        initial_fire_fraction: 0.2,
        min_nights: 12,
        spontaneous_ignition_prob: 0.02,
        spontaneous_ignition_nights: 12,
        num_agents: 4,
        owned_house_value: 150.0,      // Higher ownership incentive
        neighbor_house_value: 25.0,    // Lower neighbor value - creates greed
    },

    "random" => Scenario {
        fire_spread_prob: 0.25,
        extinguish_efficiency: 0.5,
        team_reward_per_house: 100.0,
        team_penalty_per_house: 100.0,
        work_cost_per_night: 0.5,
        initial_fire_fraction: 0.2,
        min_nights: 12,
        spontaneous_ignition_prob: 0.02,
        spontaneous_ignition_nights: 12,
        num_agents: 4,
        owned_house_value: 100.0,      // Owned house saved = +100
        neighbor_house_value: 50.0,    // Neighbor house saved = +50
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenarios_exist() {
        // Test that all expected scenarios are available
        assert!(SCENARIOS.get("trivial_cooperation").is_some());
        assert!(SCENARIOS.get("early_containment").is_some());
        assert!(SCENARIOS.get("greedy_neighbor").is_some());
        assert!(SCENARIOS.get("random").is_some());
    }

    #[test]
    fn test_scenario_not_found() {
        // Test that non-existent scenarios return None
        assert!(SCENARIOS.get("nonexistent").is_none());
        assert!(SCENARIOS.get("").is_none());
    }

    #[test]
    fn test_trivial_cooperation_values() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap();
        assert_eq!(scenario.fire_spread_prob, 0.15);
        assert_eq!(scenario.extinguish_efficiency, 0.9);
        assert_eq!(scenario.team_reward_per_house, 100.0);
        assert_eq!(scenario.team_penalty_per_house, 100.0);
        assert_eq!(scenario.work_cost_per_night, 0.5);
        assert_eq!(scenario.initial_fire_fraction, 0.1);
        assert_eq!(scenario.min_nights, 12);
        assert_eq!(scenario.spontaneous_ignition_prob, 0.0); // No sparks in trivial cooperation
        assert_eq!(scenario.spontaneous_ignition_nights, 12);
        assert_eq!(scenario.num_agents, 4);
    }

    #[test]
    fn test_early_containment_values() {
        let scenario = SCENARIOS.get("early_containment").unwrap();
        assert_eq!(scenario.fire_spread_prob, 0.35); // Higher spread rate
        assert_eq!(scenario.extinguish_efficiency, 0.6); // Lower extinguish efficiency
        assert_eq!(scenario.initial_fire_fraction, 0.3); // More initial fires
        assert_eq!(scenario.spontaneous_ignition_prob, 0.02); // Has sparks
    }

    #[test]
    fn test_greedy_neighbor_values() {
        let scenario = SCENARIOS.get("greedy_neighbor").unwrap();
        assert_eq!(scenario.work_cost_per_night, 1.0); // Higher work cost - discourages cooperation
        assert_eq!(scenario.extinguish_efficiency, 0.4); // Lower extinguish efficiency
    }

    #[test]
    fn test_random_values() {
        let scenario = SCENARIOS.get("random").unwrap();
        assert_eq!(scenario.fire_spread_prob, 0.25);
        assert_eq!(scenario.extinguish_efficiency, 0.5);
        assert_eq!(scenario.initial_fire_fraction, 0.2);
    }

    #[test]
    fn test_all_scenarios_have_4_agents() {
        for (name, scenario) in SCENARIOS.entries() {
            assert_eq!(
                scenario.num_agents, 4,
                "Scenario '{}' should have 4 agents",
                name
            );
        }
    }

    #[test]
    fn test_all_scenarios_valid_probabilities() {
        for (name, scenario) in SCENARIOS.entries() {
            assert!(
                scenario.fire_spread_prob >= 0.0 && scenario.fire_spread_prob <= 1.0,
                "Scenario '{}' has invalid fire_spread_prob: {}",
                name,
                scenario.fire_spread_prob
            );
            assert!(
                scenario.extinguish_efficiency >= 0.0 && scenario.extinguish_efficiency <= 1.0,
                "Scenario '{}' has invalid extinguish_efficiency: {}",
                name,
                scenario.extinguish_efficiency
            );
            assert!(
                scenario.initial_fire_fraction >= 0.0 && scenario.initial_fire_fraction <= 1.0,
                "Scenario '{}' has invalid initial_fire_fraction: {}",
                name,
                scenario.initial_fire_fraction
            );
            assert!(
                scenario.spontaneous_ignition_prob >= 0.0 && scenario.spontaneous_ignition_prob <= 1.0,
                "Scenario '{}' has invalid spontaneous_ignition_prob: {}",
                name,
                scenario.spontaneous_ignition_prob
            );
        }
    }

    #[test]
    fn test_scenario_clone() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        assert_eq!(scenario.fire_spread_prob, 0.15);
        assert_eq!(scenario.num_agents, 4);
    }

    #[test]
    fn test_scenario_serialization() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap();
        let json = serde_json::to_string(scenario).unwrap();
        assert!(json.contains("\"fire_spread_prob\":0.15"));
        assert!(json.contains("\"num_agents\":4"));

        // Test deserialization round-trip
        let deserialized: Scenario = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.fire_spread_prob, scenario.fire_spread_prob);
        assert_eq!(deserialized.num_agents, scenario.num_agents);
    }

    #[test]
    fn test_scenario_count() {
        let count = SCENARIOS.keys().count();
        assert_eq!(count, 4, "Expected 4 predefined scenarios");
    }

    #[test]
    fn test_scenarios_have_positive_rewards() {
        for (name, scenario) in SCENARIOS.entries() {
            assert!(
                scenario.team_reward_per_house > 0.0,
                "Scenario '{}' should have positive reward for saved houses",
                name
            );
            assert!(
                scenario.team_penalty_per_house > 0.0,
                "Scenario '{}' should have positive penalty for ruined houses",
                name
            );
        }
    }
}
