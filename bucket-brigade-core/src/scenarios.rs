use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub beta: f32,         // Fire spread probability
    pub kappa: f32,        // Extinguish efficiency
    pub a: f32,           // Reward per saved house
    pub l: f32,           // Penalty per ruined house
    pub c: f32,           // Work cost per night
    pub rho_ignite: f32,   // Initial burn fraction
    pub n_min: u32,        // Minimum nights
    pub p_spark: f32,      // Spark probability
    pub n_spark: u32,      // Spark duration
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
