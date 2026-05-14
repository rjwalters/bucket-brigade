// Random generators used by the SimpleDashboard "Random" preset options.
//
// Extracted from pages/SimpleDashboard.tsx so the helpers are independently
// testable and don't clutter the page component.

import type { ScenarioConfig } from '../data/dashboardPresets';

/**
 * Generate a random scenario parameter set.
 *
 * Picks moderate values for beta/kappa, a 50/50 chance of sparks, an N_min
 * in [10, 20], and a team size in [1, 10].
 */
export const generateRandomScenario = (): ScenarioConfig['params'] => {
  const hasSparks = Math.random() < 0.5; // 50% chance of sparks
  const N_min = Math.floor(Math.random() * (20 - 10 + 1)) + 10; // 10-20
  const num_agents = Math.floor(Math.random() * 10) + 1; // 1-10 agents

  return {
    beta: Math.random() * (0.35 - 0.15) + 0.15, // 0.15-0.35
    kappa: Math.random() * (0.6 - 0.4) + 0.4, // 0.4-0.6
    team_reward_house_survives: 100,
    team_penalty_house_burns: 100,
    reward_own_house_survives: 100,
    reward_other_house_survives: 50,
    penalty_own_house_burns: 0,
    penalty_other_house_burns: 0,
    c: 0.5,
    N_min: N_min,
    p_spark: hasSparks ? (Math.random() * (0.05 - 0.01) + 0.01) : 0, // 0 or 0.01-0.05
    num_agents: num_agents
  };
};

/**
 * Generate a random team composition of 1-10 agents drawn from the canonical
 * archetypes.
 */
export const generateRandomAgents = (): string[] => {
  const agentTypes = ['firefighter', 'free_rider', 'hero', 'coordinator', 'liar'];
  const teamSize = Math.floor(Math.random() * 10) + 1; // 1-10 agents
  return Array.from({ length: teamSize }, () =>
    agentTypes[Math.floor(Math.random() * agentTypes.length)]
  );
};
