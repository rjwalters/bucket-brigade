/**
 * Scenario Generator
 *
 * Generates randomized game scenarios for tournaments
 */

import type { Scenario } from '../types';

/**
 * Scenario type categories
 */
export const SCENARIO_TYPES = {
  TRIVIAL_COOPERATION: 'trivial_cooperation',
  EARLY_CONTAINMENT: 'early_containment',
  GREEDY_NEIGHBOR: 'greedy_neighbor',
  SPARSE_HEROICS: 'sparse_heroics',
  REST_TRAP: 'rest_trap',
  CHAIN_REACTION: 'chain_reaction',
  DECEPTIVE_CALM: 'deceptive_calm',
  OVERCROWDING: 'overcrowding',
  HONEST_VS_LIAR: 'honest_vs_liar',
  MIXED_MOTIVATION: 'mixed_motivation',
} as const;

export type ScenarioType = (typeof SCENARIO_TYPES)[keyof typeof SCENARIO_TYPES];

/**
 * Scenario type definitions with parameter ranges
 */
const SCENARIO_TEMPLATES: Record<
  ScenarioType,
  {
    name: string;
    description: string;
    parameters: Partial<Scenario>;
  }
> = {
  [SCENARIO_TYPES.TRIVIAL_COOPERATION]: {
    name: 'Trivial Cooperation',
    description: 'Easy fires reward universal cooperation',
    parameters: {
      beta: 0.15,
      kappa: 0.9,
      A: 100,
      L: 100,
      c: 0.5,
      N_min: 12,
      p_spark: 0.0,
      N_spark: 12,
    },
  },
  [SCENARIO_TYPES.EARLY_CONTAINMENT]: {
    name: 'Early Containment',
    description: 'Time pressure requires coordinated early action',
    parameters: {
      beta: 0.35,
      kappa: 0.6,
      A: 100,
      L: 100,
      c: 0.5,
      N_min: 12,
      p_spark: 0.01,
      N_spark: 12,
    },
  },
  [SCENARIO_TYPES.GREEDY_NEIGHBOR]: {
    name: 'Greedy Neighbor',
    description: 'Social dilemma between self-interest and helping others',
    parameters: {
      beta: 0.15,
      kappa: 0.4,
      A: 100,
      L: 100,
      c: 1.0,
      N_min: 15,
      p_spark: 0.02,
      N_spark: 15,
    },
  },
  [SCENARIO_TYPES.SPARSE_HEROICS]: {
    name: 'Sparse Heroics',
    description: 'Minimal workers needed, overwork is wasteful',
    parameters: {
      beta: 0.1,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 0.8,
      N_min: 20,
      p_spark: 0.01,
      N_spark: 20,
    },
  },
  [SCENARIO_TYPES.REST_TRAP]: {
    name: 'Rest Trap',
    description: 'Usually safe to rest, but occasional disasters require response',
    parameters: {
      beta: 0.05,
      kappa: 0.95,
      A: 100,
      L: 100,
      c: 0.2,
      N_min: 25,
      p_spark: 0.02,
      N_spark: 25,
    },
  },
  [SCENARIO_TYPES.CHAIN_REACTION]: {
    name: 'Chain Reaction',
    description: 'High spread demands distributed firefighting teams',
    parameters: {
      beta: 0.45,
      kappa: 0.6,
      A: 100,
      L: 100,
      c: 0.7,
      N_min: 15,
      p_spark: 0.03,
      N_spark: 15,
    },
  },
  [SCENARIO_TYPES.DECEPTIVE_CALM]: {
    name: 'Deceptive Calm',
    description: 'Honest signaling rewarded during occasional flare-ups',
    parameters: {
      beta: 0.25,
      kappa: 0.6,
      A: 100,
      L: 100,
      c: 0.4,
      N_min: 20,
      p_spark: 0.05,
      N_spark: 20,
    },
  },
  [SCENARIO_TYPES.OVERCROWDING]: {
    name: 'Overcrowding',
    description: 'Too many workers reduce efficiency',
    parameters: {
      beta: 0.2,
      kappa: 0.3,
      A: 50,
      L: 100,
      c: 0.6,
      N_min: 15,
      p_spark: 0.02,
      N_spark: 15,
    },
  },
  [SCENARIO_TYPES.HONEST_VS_LIAR]: {
    name: 'Honest vs Liar',
    description: 'Mixed population of honest and dishonest agents',
    parameters: {
      beta: 0.25,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 0.6,
      N_min: 15,
      p_spark: 0.02,
      N_spark: 15,
    },
  },
  [SCENARIO_TYPES.MIXED_MOTIVATION]: {
    name: 'Mixed Motivation',
    description: 'House ownership creates conflicting incentives',
    parameters: {
      beta: 0.3,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 0.6,
      N_min: 15,
      p_spark: 0.02,
      N_spark: 15,
    },
  },
};

/**
 * Generate a random scenario of a specific type
 */
export function generateScenario(
  type: ScenarioType,
  numAgents: number,
  _seed?: number,
): Scenario {
  const template = SCENARIO_TEMPLATES[type];

  // Add small random variations to parameters
  const variance = 0.1; // 10% variation
  const randomize = (base: number) => {
    const factor = 1 + (Math.random() - 0.5) * variance;
    return Math.max(0, base * factor);
  };

  return {
    beta: randomize(template.parameters.beta ?? 0.25),
    kappa: randomize(template.parameters.kappa ?? 0.5),
    A: template.parameters.A ?? 100,
    L: template.parameters.L ?? 100,
    c: randomize(template.parameters.c ?? 0.5),
    N_min: template.parameters.N_min ?? 12,
    p_spark: randomize(template.parameters.p_spark ?? 0.02),
    N_spark: template.parameters.N_spark ?? 12,
    num_agents: numAgents,
  };
}

/**
 * Generate random scenario (any type)
 */
export function generateRandomScenario(numAgents: number, seed?: number): Scenario {
  const types = Object.values(SCENARIO_TYPES);
  const randomType = types[Math.floor(Math.random() * types.length)];
  return generateScenario(randomType, numAgents, seed);
}

/**
 * Generate balanced distribution of scenarios
 */
export function generateScenarioSet(
  numScenarios: number,
  numAgents: number,
  seed?: number,
): Array<{ scenario: Scenario; type: ScenarioType }> {
  const scenarios: Array<{ scenario: Scenario; type: ScenarioType }> = [];
  const types = Object.values(SCENARIO_TYPES);

  // Distribute evenly across scenario types
  for (let i = 0; i < numScenarios; i++) {
    const type = types[i % types.length];
    scenarios.push({
      scenario: generateScenario(type, numAgents, seed),
      type,
    });
  }

  // Shuffle for randomness
  for (let i = scenarios.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [scenarios[i], scenarios[j]] = [scenarios[j], scenarios[i]];
  }

  return scenarios;
}

/**
 * Get scenario template info
 */
export function getScenarioTemplate(type: ScenarioType) {
  return SCENARIO_TEMPLATES[type];
}

/**
 * Get all scenario types
 */
export function getAllScenarioTypes(): ScenarioType[] {
  return Object.values(SCENARIO_TYPES);
}
