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
    pub num_agents: usize,           // Number of agents in game
}

pub const SCENARIOS: phf::Map<&'static str, Scenario> = phf::phf_map! {
    "trivial_cooperation" => Scenario {
        prob_fire_spreads_to_neighbor: 0.15,
        prob_solo_agent_extinguishes_fire: 0.7,
        prob_house_catches_fire: 0.01,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.5,
        min_nights: 12,
        num_agents: 4,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "early_containment" => Scenario {
        prob_fire_spreads_to_neighbor: 0.35,
        prob_solo_agent_extinguishes_fire: 0.45,
        prob_house_catches_fire: 0.03,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.5,
        min_nights: 12,
        num_agents: 4,
        reward_own_house_survives: 100.0,
        reward_other_house_survives: 50.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "greedy_neighbor" => Scenario {
        prob_fire_spreads_to_neighbor: 0.15,
        prob_solo_agent_extinguishes_fire: 0.33,
        prob_house_catches_fire: 0.02,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 1.0,
        min_nights: 12,
        num_agents: 4,
        reward_own_house_survives: 150.0,
        reward_other_house_survives: 25.0,
        penalty_own_house_burns: 0.0,
        penalty_other_house_burns: 0.0,
    },

    "random" => Scenario {
        prob_fire_spreads_to_neighbor: 0.25,
        prob_solo_agent_extinguishes_fire: 0.39,
        prob_house_catches_fire: 0.02,
        team_reward_house_survives: 100.0,
        team_penalty_house_burns: 100.0,
        cost_to_work_one_night: 0.5,
        min_nights: 12,
        num_agents: 4,
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
    fn test_scenarios_exist() {
        assert!(SCENARIOS.get("trivial_cooperation").is_some());
        assert!(SCENARIOS.get("early_containment").is_some());
        assert!(SCENARIOS.get("greedy_neighbor").is_some());
        assert!(SCENARIOS.get("random").is_some());
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
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.7);
        assert_eq!(scenario.team_reward_house_survives, 100.0);
        assert_eq!(scenario.team_penalty_house_burns, 100.0);
        assert_eq!(scenario.cost_to_work_one_night, 0.5);
        assert_eq!(scenario.prob_house_catches_fire, 0.01);
        assert_eq!(scenario.min_nights, 12);
        assert_eq!(scenario.num_agents, 4);
    }

    #[test]
    fn test_early_containment_values() {
        let scenario = SCENARIOS.get("early_containment").unwrap();
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.35);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.45);
        assert_eq!(scenario.prob_house_catches_fire, 0.03);
    }

    #[test]
    fn test_greedy_neighbor_values() {
        let scenario = SCENARIOS.get("greedy_neighbor").unwrap();
        assert_eq!(scenario.cost_to_work_one_night, 1.0);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.33);
    }

    #[test]
    fn test_random_values() {
        let scenario = SCENARIOS.get("random").unwrap();
        assert_eq!(scenario.prob_fire_spreads_to_neighbor, 0.25);
        assert_eq!(scenario.prob_solo_agent_extinguishes_fire, 0.39);
        assert_eq!(scenario.prob_house_catches_fire, 0.02);
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
        assert_eq!(scenario.num_agents, 4);
    }

    #[test]
    fn test_scenario_serialization() {
        let scenario = SCENARIOS.get("trivial_cooperation").unwrap();
        let json = serde_json::to_string(scenario).unwrap();
        assert!(json.contains("\"prob_fire_spreads_to_neighbor\":0.15"));
        assert!(json.contains("\"num_agents\":4"));

        let deserialized: Scenario = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.prob_fire_spreads_to_neighbor,
            scenario.prob_fire_spreads_to_neighbor
        );
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
}
