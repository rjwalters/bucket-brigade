use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub beta: f32,       // Fire spread probability
    pub kappa: f32,      // Extinguish efficiency
    pub a: f32,          // Reward per saved house
    pub l: f32,          // Penalty per ruined house
    pub c: f32,          // Work cost per night
    pub rho_ignite: f32, // Initial burn fraction
    pub n_min: u32,      // Minimum nights
    pub p_spark: f32,    // Spark probability
    pub n_spark: u32,    // Spark duration
    pub num_agents: usize,
}

pub const SCENARIOS: phf::Map<&'static str, Scenario> = phf::phf_map! {
    "trivial_cooperation" => Scenario {
        beta: 0.15,
        kappa: 0.9,
        a: 100.0,
        l: 100.0,
        c: 0.5,
        rho_ignite: 0.1,
        n_min: 12,
        p_spark: 0.0,
        n_spark: 12,
        num_agents: 4,
    },

    "early_containment" => Scenario {
        beta: 0.35,
        kappa: 0.6,
        a: 100.0,
        l: 100.0,
        c: 0.5,
        rho_ignite: 0.3,
        n_min: 12,
        p_spark: 0.02,
        n_spark: 12,
        num_agents: 4,
    },

    "greedy_neighbor" => Scenario {
        beta: 0.15,
        kappa: 0.4,
        a: 100.0,
        l: 100.0,
        c: 1.0,
        rho_ignite: 0.2,
        n_min: 12,
        p_spark: 0.02,
        n_spark: 12,
        num_agents: 4,
    },

    "random" => Scenario {
        beta: 0.25,
        kappa: 0.5,
        a: 100.0,
        l: 100.0,
        c: 0.5,
        rho_ignite: 0.2,
        n_min: 12,
        p_spark: 0.02,
        n_spark: 12,
        num_agents: 4,
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
        assert_eq!(scenario.beta, 0.15);
        assert_eq!(scenario.kappa, 0.9);
        assert_eq!(scenario.a, 100.0);
        assert_eq!(scenario.l, 100.0);
        assert_eq!(scenario.c, 0.5);
        assert_eq!(scenario.rho_ignite, 0.1);
        assert_eq!(scenario.n_min, 12);
        assert_eq!(scenario.p_spark, 0.0); // No sparks in trivial cooperation
        assert_eq!(scenario.n_spark, 12);
        assert_eq!(scenario.num_agents, 4);
    }

    #[test]
    fn test_early_containment_values() {
        let scenario = SCENARIOS.get("early_containment").unwrap();
        assert_eq!(scenario.beta, 0.35); // Higher spread rate
        assert_eq!(scenario.kappa, 0.6); // Lower extinguish efficiency
        assert_eq!(scenario.rho_ignite, 0.3); // More initial fires
        assert_eq!(scenario.p_spark, 0.02); // Has sparks
    }

    #[test]
    fn test_greedy_neighbor_values() {
        let scenario = SCENARIOS.get("greedy_neighbor").unwrap();
        assert_eq!(scenario.c, 1.0); // Higher work cost - discourages cooperation
        assert_eq!(scenario.kappa, 0.4); // Lower extinguish efficiency
    }

    #[test]
    fn test_random_values() {
        let scenario = SCENARIOS.get("random").unwrap();
        assert_eq!(scenario.beta, 0.25);
        assert_eq!(scenario.kappa, 0.5);
        assert_eq!(scenario.rho_ignite, 0.2);
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
                scenario.beta >= 0.0 && scenario.beta <= 1.0,
                "Scenario '{}' has invalid beta: {}",
                name,
                scenario.beta
            );
            assert!(
                scenario.kappa >= 0.0 && scenario.kappa <= 1.0,
                "Scenario '{}' has invalid kappa: {}",
                name,
                scenario.kappa
            );
            assert!(
                scenario.rho_ignite >= 0.0 && scenario.rho_ignite <= 1.0,
                "Scenario '{}' has invalid rho_ignite: {}",
                name,
                scenario.rho_ignite
            );
            assert!(
                scenario.p_spark >= 0.0 && scenario.p_spark <= 1.0,
                "Scenario '{}' has invalid p_spark: {}",
                name,
                scenario.p_spark
            );
        }
    }

    #[test]
    fn test_scenario_clone() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
        assert_eq!(scenario.beta, 0.15);
        assert_eq!(scenario.num_agents, 4);
    }

    #[test]
    fn test_scenario_serialization() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap();
        let json = serde_json::to_string(scenario).unwrap();
        assert!(json.contains("\"beta\":0.15"));
        assert!(json.contains("\"num_agents\":4"));

        // Test deserialization round-trip
        let deserialized: Scenario = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.beta, scenario.beta);
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
                scenario.a > 0.0,
                "Scenario '{}' should have positive reward for saved houses",
                name
            );
            assert!(
                scenario.l > 0.0,
                "Scenario '{}' should have positive penalty for ruined houses",
                name
            );
        }
    }
}
