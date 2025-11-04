/**
 * Archetype parameter definitions from bucket_brigade/agents/archetypes.py
 *
 * Parameter order: [honesty_bias, work_tendency, neighbor_help_bias, own_house_priority,
 *                   risk_aversion, coordination_weight, exploration_rate, fatigue_memory,
 *                   rest_reward_bias, altruism_factor]
 */

export interface ArchetypeParams {
  honesty_bias: number;
  work_tendency: number;
  neighbor_help_bias: number;
  own_house_priority: number;
  risk_aversion: number;
  coordination_weight: number;
  exploration_rate: number;
  fatigue_memory: number;
  rest_reward_bias: number;
  altruism_factor: number;
}

export interface Archetype {
  id: string;
  name: string;
  description: string;
  params: ArchetypeParams;
}

export const ARCHETYPES: Record<string, Archetype> = {
  firefighter: {
    id: 'firefighter',
    name: 'Firefighter',
    description: 'Honest, hard-working, cooperative agent that prioritizes teamwork and fire suppression. Works most nights and trusts others\' signals.',
    params: {
      honesty_bias: 1.0,
      work_tendency: 0.9,
      neighbor_help_bias: 0.5,
      own_house_priority: 0.8,
      risk_aversion: 0.5,
      coordination_weight: 0.7,
      exploration_rate: 0.1,
      fatigue_memory: 0.0,
      rest_reward_bias: 0.0,
      altruism_factor: 0.8,
    },
  },
  free_rider: {
    id: 'free_rider',
    name: 'Free Rider',
    description: 'Selfish agent that avoids work and only cares about its own house. Ignores coordination signals and strongly prefers resting.',
    params: {
      honesty_bias: 0.7,
      work_tendency: 0.2,
      neighbor_help_bias: 0.0,
      own_house_priority: 0.9,
      risk_aversion: 0.0,
      coordination_weight: 0.0,
      exploration_rate: 0.1,
      fatigue_memory: 0.0,
      rest_reward_bias: 0.9,
      altruism_factor: 0.0,
    },
  },
  hero: {
    id: 'hero',
    name: 'Hero',
    description: 'Maximum effort, maximum cooperation agent that always works and helps everyone. Brave, consistent, and maximally altruistic.',
    params: {
      honesty_bias: 1.0,
      work_tendency: 1.0,
      neighbor_help_bias: 1.0,
      own_house_priority: 0.5,
      risk_aversion: 0.1,
      coordination_weight: 0.5,
      exploration_rate: 0.0,
      fatigue_memory: 0.9,
      rest_reward_bias: 0.0,
      altruism_factor: 1.0,
    },
  },
  coordinator: {
    id: 'coordinator',
    name: 'Coordinator',
    description: 'Balanced, trust-based strategy with high trust in signals. Moderately cooperative, cautious, and relies heavily on team coordination.',
    params: {
      honesty_bias: 0.9,
      work_tendency: 0.6,
      neighbor_help_bias: 0.7,
      own_house_priority: 0.6,
      risk_aversion: 0.8,
      coordination_weight: 1.0,
      exploration_rate: 0.05,
      fatigue_memory: 0.0,
      rest_reward_bias: 0.2,
      altruism_factor: 0.6,
    },
  },
  liar: {
    id: 'liar',
    name: 'Liar',
    description: 'Deceptive agent with selfish motives. Mostly dishonest in signaling, works when beneficial, and has low altruism.',
    params: {
      honesty_bias: 0.1,
      work_tendency: 0.7,
      neighbor_help_bias: 0.0,
      own_house_priority: 0.9,
      risk_aversion: 0.2,
      coordination_weight: 0.8,
      exploration_rate: 0.3,
      fatigue_memory: 0.0,
      rest_reward_bias: 0.4,
      altruism_factor: 0.2,
    },
  },
};

export const PARAMETER_DESCRIPTIONS: Record<keyof ArchetypeParams, { label: string; description: string }> = {
  honesty_bias: {
    label: 'Honesty',
    description: 'Probability of truthful signaling (0 = always lies, 1 = always truthful)',
  },
  work_tendency: {
    label: 'Work Tendency',
    description: 'Base tendency to work vs. rest (0 = never works, 1 = always works)',
  },
  neighbor_help_bias: {
    label: 'Neighbor Help',
    description: 'Preference for helping neighbor houses (0 = ignores neighbors, 1 = prioritizes neighbors)',
  },
  own_house_priority: {
    label: 'Self-Preservation',
    description: 'Priority given to own houses (0 = ignores own houses, 1 = only cares about own houses)',
  },
  risk_aversion: {
    label: 'Risk Aversion',
    description: 'Sensitivity to burning houses and fire spread (0 = ignores risk, 1 = highly cautious)',
  },
  coordination_weight: {
    label: 'Trust in Signals',
    description: 'Trust in others\' signals for decision-making (0 = ignores signals, 1 = fully trusts)',
  },
  exploration_rate: {
    label: 'Randomness',
    description: 'Amount of random exploration in decisions (0 = deterministic, 1 = fully random)',
  },
  fatigue_memory: {
    label: 'Consistency',
    description: 'Inertia to repeat previous actions (0 = no memory, 1 = highly consistent)',
  },
  rest_reward_bias: {
    label: 'Rest Preference',
    description: 'Intrinsic preference for resting (0 = no preference, 1 = strongly prefers rest)',
  },
  altruism_factor: {
    label: 'Altruism',
    description: 'Willingness to help others at personal cost (0 = selfish, 1 = maximally altruistic)',
  },
};

export function getArchetype(id: string): Archetype | undefined {
  return ARCHETYPES[id.toLowerCase()];
}

export function listArchetypes(): Archetype[] {
  return Object.values(ARCHETYPES);
}
