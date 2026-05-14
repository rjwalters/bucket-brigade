// Team and scenario presets used by SimpleDashboard.
//
// Extracted from pages/SimpleDashboard.tsx so the data is reusable across
// dashboard components (HeroSection, TeamSelector, ScenarioSelector) and
// easier to maintain independently from UI code.

// Team presets with their archetype compositions
export interface TeamPreset {
  id: string;
  name: string;
  description: string;
  archetypes: string[];
}

// Scenario configurations for testing different cooperation dynamics
export interface ScenarioConfig {
  id: string;
  name: string;
  description: string;
  params: {
    beta: number;
    kappa: number;
    team_reward_house_survives: number;
    team_penalty_house_burns: number;
    reward_own_house_survives: number;
    reward_other_house_survives: number;
    penalty_own_house_burns: number;
    penalty_other_house_burns: number;
    c: number;
    N_min: number;
    p_spark: number;
    num_agents: number;
  };
}

// Note: These are parameterized heuristic agents for demonstration.
// Research uses evolved strategies (genetic algorithms) and Nash equilibria.
// Each archetype has 10 behavioral parameters (click agent name to see details).
export const TEAM_PRESETS: TeamPreset[] = [
  {
    id: 'solo_hero',
    name: 'Solo: Hero',
    description: 'One agent against the flames - ultimate test of individual skill (1 agent)',
    archetypes: ['hero']
  },
  {
    id: 'duo_firefighters',
    name: 'Duo: Firefighters',
    description: 'Two honest firefighters - minimal team, maximum cooperation (2 agents)',
    archetypes: ['firefighter', 'firefighter']
  },
  {
    id: 'trio_mixed',
    name: 'Trio: Mixed',
    description: 'Small diverse team - tests cooperation with limited resources (3 agents)',
    archetypes: ['firefighter', 'coordinator', 'hero']
  },
  {
    id: 'all_firefighters',
    name: 'All Firefighters',
    description: 'Honest, hard-working, cooperative agents - classic teamwork (4 agents)',
    archetypes: ['firefighter', 'firefighter', 'firefighter', 'firefighter']
  },
  {
    id: 'mixed_balanced',
    name: 'Mixed: Balanced',
    description: 'Diverse team with complementary strengths (4 agents)',
    archetypes: ['firefighter', 'coordinator', 'hero', 'free_rider']
  },
  {
    id: 'large_coordinators',
    name: 'Large: Coordinators',
    description: 'Big team with high coordination - tests scalability (5 agents)',
    archetypes: ['coordinator', 'coordinator', 'coordinator', 'coordinator', 'coordinator']
  },
  {
    id: 'large_mixed',
    name: 'Large: Mixed',
    description: 'Diverse large team - complex social dynamics (6 agents)',
    archetypes: ['firefighter', 'firefighter', 'coordinator', 'hero', 'free_rider', 'liar']
  },
  {
    id: 'full_town',
    name: 'Full Town',
    description: 'Maximum team size - every house has a dedicated owner (10 agents)',
    archetypes: ['firefighter', 'firefighter', 'coordinator', 'coordinator', 'hero', 'hero', 'free_rider', 'free_rider', 'liar', 'liar']
  },
  {
    id: 'random',
    name: 'Random Team',
    description: 'Randomly generated team (1-10 agents, varied archetypes)',
    archetypes: ['random', 'random', 'random', 'random']
  }
];

// Research scenarios aligned with bucket_brigade/envs/scenarios.py
export const TEST_SCENARIOS: ScenarioConfig[] = [
  {
    id: 'random',
    name: 'Random',
    description: 'Random parameters each game - tests generalization across diverse conditions',
    params: {
      beta: 0.25,
      kappa: 0.5,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 15,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'default',
    name: 'Default',
    description: 'Balanced scenario with moderate fire spread and good extinguish efficiency',
    params: {
      beta: 0.25,
      kappa: 0.5,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'easy',
    name: 'Easy',
    description: 'Low fire spread and high extinguish efficiency - cooperation should succeed',
    params: {
      beta: 0.1,
      kappa: 0.8,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 10,
      p_spark: 0.01,
      num_agents: 4
    }
  },
  {
    id: 'hard',
    name: 'Hard',
    description: 'High fire spread and low extinguish efficiency - requires strong coordination',
    params: {
      beta: 0.4,
      kappa: 0.3,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 15,
      p_spark: 0.05,
      num_agents: 4
    }
  },
  {
    id: 'trivial_cooperation',
    name: 'Trivial Cooperation',
    description: 'Fires are rare and extinguish easily - minimal cooperation needed',
    params: {
      beta: 0.15,
      kappa: 0.9,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 12,
      p_spark: 0.0,
      num_agents: 4
    }
  },
  {
    id: 'early_containment',
    name: 'Early Containment',
    description: 'Fires start aggressive but can be stopped early with coordination',
    params: {
      beta: 0.35,
      kappa: 0.6,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'greedy_neighbor',
    name: 'Greedy Neighbor',
    description: 'Social dilemma between self-interest and cooperation - high work cost',
    params: {
      beta: 0.15,
      kappa: 0.4,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 150,
      reward_other_house_survives: 25,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 1.0,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'sparse_heroics',
    name: 'Sparse Heroics',
    description: 'Few workers can make the difference - tests heroic action under cost',
    params: {
      beta: 0.1,
      kappa: 0.5,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.8,
      N_min: 20,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'rest_trap',
    name: 'Rest Trap',
    description: 'Fires usually extinguish themselves, but not always - tempts free-riding',
    params: {
      beta: 0.05,
      kappa: 0.95,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.2,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'chain_reaction',
    name: 'Chain Reaction',
    description: 'High spread requires distributed teams - tests spatial coordination',
    params: {
      beta: 0.45,
      kappa: 0.6,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.7,
      N_min: 15,
      p_spark: 0.03,
      num_agents: 4
    }
  },
  {
    id: 'deceptive_calm',
    name: 'Deceptive Calm',
    description: 'Occasional flare-ups reward honest signaling - tests communication',
    params: {
      beta: 0.25,
      kappa: 0.6,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.4,
      N_min: 20,
      p_spark: 0.05,
      num_agents: 4
    }
  },
  {
    id: 'overcrowding',
    name: 'Overcrowding',
    description: 'Too many workers reduce efficiency - tests resource allocation',
    params: {
      beta: 0.2,
      kappa: 0.3,
      team_reward_house_survives: 50,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.6,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'mixed_motivation',
    name: 'Mixed Motivation',
    description: 'Ownership creates self-interest conflicts - tests fairness',
    params: {
      beta: 0.3,
      kappa: 0.5,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.6,
      N_min: 15,
      p_spark: 0.03,
      num_agents: 4
    }
  }
];
