/**
 * Agent Archetype Definitions
 *
 * Pre-defined behavioral profiles for the team builder.
 * Each archetype represents a distinct strategy and personality.
 */

import type { AgentArchetype } from '../types/teamBuilder';

/**
 * All available agent archetypes
 */
export const AGENT_ARCHETYPES: Record<string, AgentArchetype> = {
  firefighter: {
    id: 'firefighter',
    name: 'Firefighter',
    icon: '🧑‍🚒',
    color: '#dc2626', // red-600
    tagline: 'First to respond, last to rest',
    description:
      'Reliable team player who prioritizes putting out fires over personal rest. Always signals honestly and helps neighbors proactively.',
    parameters: {
      honesty_bias: 1.0,
      work_tendency: 0.9,
      neighbor_help_bias: 0.7,
      own_house_priority: 0.4,
      risk_aversion: 0.5,
      coordination_weight: 0.7,
      exploration_rate: 0.1,
      fatigue_memory: 0.5,
      rest_reward_bias: 0.1,
      altruism_factor: 0.8,
    },
    strategyNotes: [
      'Always signals honestly',
      'Prioritizes urgent fires over rest',
      'Helps neighbors proactively',
      'Coordinates well with teammates',
    ],
  },

  free_rider: {
    id: 'free_rider',
    name: 'Free Rider',
    icon: '💤',
    color: '#7c3aed', // violet-600
    tagline: 'Let others do the heavy lifting',
    description:
      'Prefers to rest and let teammates handle the fires. Only works when own house is directly threatened. Maximizes personal rest rewards.',
    parameters: {
      honesty_bias: 0.7,
      work_tendency: 0.2,
      neighbor_help_bias: 0.2,
      own_house_priority: 0.9,
      risk_aversion: 0.8,
      coordination_weight: 0.3,
      exploration_rate: 0.2,
      fatigue_memory: 0.8,
      rest_reward_bias: 0.9,
      altruism_factor: 0.1,
    },
    strategyNotes: [
      'Signals work but often rests',
      'Only works when own house threatened',
      'Relies on teammates for firefighting',
      'Maximizes personal rest rewards',
    ],
  },

  coordinator: {
    id: 'coordinator',
    name: 'Coordinator',
    icon: '📋',
    color: '#2563eb', // blue-600
    tagline: 'Teamwork makes the dream work',
    description:
      'Excellent at reading signals and organizing team response. Avoids redundant work and fills gaps in team coverage.',
    parameters: {
      honesty_bias: 0.9,
      work_tendency: 0.6,
      neighbor_help_bias: 0.6,
      own_house_priority: 0.5,
      risk_aversion: 0.5,
      coordination_weight: 1.0,
      exploration_rate: 0.05,
      fatigue_memory: 0.4,
      rest_reward_bias: 0.4,
      altruism_factor: 0.6,
    },
    strategyNotes: [
      'Highly responsive to team signals',
      'Avoids redundant work on same fire',
      'Balances work and rest strategically',
      'Fills gaps in team coverage',
    ],
  },

  liar: {
    id: 'liar',
    name: 'Liar',
    icon: '🤥',
    color: '#16a34a', // green-600
    tagline: 'Trust me, I\'m working hard',
    description:
      'Sends false signals to mislead teammates. Creates confusion in coordination while pursuing self-interest.',
    parameters: {
      honesty_bias: 0.1,
      work_tendency: 0.5,
      neighbor_help_bias: 0.3,
      own_house_priority: 0.7,
      risk_aversion: 0.4,
      coordination_weight: 0.6,
      exploration_rate: 0.3,
      fatigue_memory: 0.5,
      rest_reward_bias: 0.6,
      altruism_factor: 0.2,
    },
    strategyNotes: [
      'Signals opposite of actual intent',
      'Creates confusion in team coordination',
      'Prioritizes self-interest',
      'Exploits others\' trust',
    ],
  },

  hero: {
    id: 'hero',
    name: 'Hero',
    icon: '🦸',
    color: '#eab308', // yellow-500
    tagline: 'I\'ll save everyone or die trying',
    description:
      'Maximum effort, maximum altruism. Never rests while fires burn. Ignores personal costs to help everyone equally.',
    parameters: {
      honesty_bias: 1.0,
      work_tendency: 1.0,
      neighbor_help_bias: 0.9,
      own_house_priority: 0.2,
      risk_aversion: 0.1,
      coordination_weight: 0.5,
      exploration_rate: 0.1,
      fatigue_memory: 0.9,
      rest_reward_bias: 0.0,
      altruism_factor: 1.0,
    },
    strategyNotes: [
      'Never rests while fires burn',
      'Helps everyone equally',
      'Ignores personal costs',
      'Consistent and predictable',
    ],
  },

  strategist: {
    id: 'strategist',
    name: 'Strategist',
    icon: '🎯',
    color: '#1e3a8a', // blue-900
    tagline: 'Calculated action over emotion',
    description:
      'Analyzes situation carefully before acting. Minimizes exploration and maximizes efficiency. Responds to actual risk level.',
    parameters: {
      honesty_bias: 0.9,
      work_tendency: 0.6,
      neighbor_help_bias: 0.5,
      own_house_priority: 0.5,
      risk_aversion: 0.7,
      coordination_weight: 0.9,
      exploration_rate: 0.05,
      fatigue_memory: 0.3,
      rest_reward_bias: 0.5,
      altruism_factor: 0.6,
    },
    strategyNotes: [
      'Calculates optimal response',
      'Avoids wasteful overwork',
      'Responds to risk level',
      'Minimal exploration, maximum efficiency',
    ],
  },

  opportunist: {
    id: 'opportunist',
    name: 'Opportunist',
    icon: '💰',
    color: '#ea580c', // orange-600
    tagline: 'My house, my rules',
    description:
      'Laser-focused on protecting own property. Ignores team fires and doesn\'t coordinate with others. Maximizes personal reward.',
    parameters: {
      honesty_bias: 0.6,
      work_tendency: 0.6,
      neighbor_help_bias: 0.1,
      own_house_priority: 1.0,
      risk_aversion: 0.6,
      coordination_weight: 0.2,
      exploration_rate: 0.2,
      fatigue_memory: 0.6,
      rest_reward_bias: 0.7,
      altruism_factor: 0.0,
    },
    strategyNotes: [
      'Only defends own house',
      'Ignores team fires',
      'Maximizes personal reward',
      'Doesn\'t coordinate with others',
    ],
  },

  cautious: {
    id: 'cautious',
    name: 'Cautious',
    icon: '😰',
    color: '#facc15', // yellow-400
    tagline: 'Better safe than sorry',
    description:
      'Avoids risky situations and prefers conservative approach. Works less when many fires present. Prioritizes self-preservation.',
    parameters: {
      honesty_bias: 0.9,
      work_tendency: 0.4,
      neighbor_help_bias: 0.4,
      own_house_priority: 0.7,
      risk_aversion: 0.9,
      coordination_weight: 0.8,
      exploration_rate: 0.05,
      fatigue_memory: 0.7,
      rest_reward_bias: 0.6,
      altruism_factor: 0.4,
    },
    strategyNotes: [
      'Works less when many fires present',
      'Prioritizes self-preservation',
      'Honest but conservative',
      'Avoids overcommitment',
    ],
  },

  maverick: {
    id: 'maverick',
    name: 'Maverick',
    icon: '🎲',
    color: '#ec4899', // pink-500
    tagline: 'Unpredictable by design',
    description:
      'High variance strategy that keeps opponents guessing. Tries different approaches and explores the action space.',
    parameters: {
      honesty_bias: 0.5,
      work_tendency: 0.5,
      neighbor_help_bias: 0.5,
      own_house_priority: 0.5,
      risk_aversion: 0.5,
      coordination_weight: 0.5,
      exploration_rate: 1.0,
      fatigue_memory: 0.3,
      rest_reward_bias: 0.5,
      altruism_factor: 0.5,
    },
    strategyNotes: [
      'High exploration rate',
      'Tries different strategies',
      'Unpredictable behavior',
      'Good for learning optimal patterns',
    ],
  },

  random: {
    id: 'random',
    name: 'Random',
    icon: '❓',
    color: '#6b7280', // gray-500
    tagline: 'Chaos incarnate',
    description:
      'All parameters randomized each game. Completely unpredictable. Useful for testing adaptability and as a baseline.',
    parameters: {
      // These will be randomized at runtime
      honesty_bias: 0.5,
      work_tendency: 0.5,
      neighbor_help_bias: 0.5,
      own_house_priority: 0.5,
      risk_aversion: 0.5,
      coordination_weight: 0.5,
      exploration_rate: 0.5,
      fatigue_memory: 0.5,
      rest_reward_bias: 0.5,
      altruism_factor: 0.5,
    },
    isRandomized: true,
    strategyNotes: [
      'Parameters change every game',
      'Completely unpredictable',
      'Useful for testing adaptability',
      'Baseline for comparison',
    ],
  },
};

/**
 * Get archetype by ID with fallback
 */
export function getArchetype(id: string): AgentArchetype | null {
  return AGENT_ARCHETYPES[id] || null;
}

/**
 * Get all archetype IDs in display order
 */
export function getArchetypeIds(): string[] {
  return [
    'firefighter',
    'coordinator',
    'hero',
    'strategist',
    'free_rider',
    'opportunist',
    'cautious',
    'liar',
    'maverick',
    'random',
  ];
}

/**
 * Get all archetypes in display order
 */
export function getAllArchetypes(): AgentArchetype[] {
  return getArchetypeIds().map((id) => AGENT_ARCHETYPES[id]);
}

/**
 * Create a randomized version of the Random archetype
 */
export function createRandomizedArchetype(): AgentArchetype {
  const base = AGENT_ARCHETYPES.random;

  return {
    ...base,
    parameters: {
      honesty_bias: Math.random(),
      work_tendency: Math.random(),
      neighbor_help_bias: Math.random(),
      own_house_priority: Math.random(),
      risk_aversion: Math.random(),
      coordination_weight: Math.random(),
      exploration_rate: Math.random(),
      fatigue_memory: Math.random(),
      rest_reward_bias: Math.random(),
      altruism_factor: Math.random(),
    },
  };
}

/**
 * Get stat display data for parameters
 */
export interface StatDisplay {
  label: string;
  value: number;
  description: string;
}

/**
 * Convert parameters to display stats
 */
export function getStatDisplays(params: AgentArchetype['parameters']): StatDisplay[] {
  return [
    {
      label: 'Honesty',
      value: params.honesty_bias,
      description: 'Signals true intent',
    },
    {
      label: 'Work Ethic',
      value: params.work_tendency,
      description: 'Willingness to work',
    },
    {
      label: 'Altruism',
      value: params.altruism_factor,
      description: 'Helps others selflessly',
    },
    {
      label: 'Coordination',
      value: params.coordination_weight,
      description: 'Responds to team signals',
    },
    {
      label: 'Risk Taking',
      value: 1 - params.risk_aversion,
      description: 'Works in dangerous situations',
    },
    {
      label: 'House Priority',
      value: params.own_house_priority,
      description: 'Defends own property',
    },
    {
      label: 'Rest Bias',
      value: params.rest_reward_bias,
      description: 'Prefers to rest',
    },
  ];
}
