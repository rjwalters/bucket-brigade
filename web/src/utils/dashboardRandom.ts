// Random generators used by the SimpleDashboard "Random" preset options.
//
// Extracted from pages/SimpleDashboard.tsx so the helpers are independently
// testable and don't clutter the page component.

import type { ScenarioConfig } from '../data/dashboardPresets';

/**
 * Generate a random scenario parameter set.
 *
 * Picks moderate values for fire spread / extinguish probabilities, a 50/50
 * chance of sparks, a min_nights in [10, 20], and a team size in [1, 10].
 */
export const generateRandomScenario = (): ScenarioConfig['params'] => {
  const hasSparks = Math.random() < 0.5; // 50% chance of sparks
  const min_nights = Math.floor(Math.random() * (20 - 10 + 1)) + 10; // 10-20
  const num_agents = Math.floor(Math.random() * 10) + 1; // 1-10 agents

  return {
    prob_fire_spreads_to_neighbor: Math.random() * (0.35 - 0.15) + 0.15, // 0.15-0.35
    prob_solo_agent_extinguishes_fire: Math.random() * (0.6 - 0.4) + 0.4, // 0.4-0.6
    prob_house_catches_fire: hasSparks ? (Math.random() * (0.05 - 0.01) + 0.01) : 0, // 0 or 0.01-0.05
    team_reward_house_survives: 100,
    team_penalty_house_burns: 100,
    reward_own_house_survives: 100,
    reward_other_house_survives: 50,
    penalty_own_house_burns: 0,
    penalty_other_house_burns: 0,
    cost_to_work_one_night: 0.5,
    min_nights: min_nights,
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
