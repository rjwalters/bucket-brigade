/**
 * Team Templates
 *
 * Pre-defined team compositions for quick start and testing.
 */

import type { TeamTemplate } from '../types/teamBuilder';

/**
 * Pre-defined team templates
 */
export const TEAM_TEMPLATES: Record<string, TeamTemplate> = {
  perfect_cooperation: {
    id: 'perfect_cooperation',
    name: 'Perfect Cooperation',
    description: 'Maximum teamwork with honest signals and coordinated response.',
    archetypeIds: [
      'firefighter',
      'firefighter',
      'coordinator',
      'coordinator',
      'hero',
      'hero',
      'strategist',
      'strategist',
      'firefighter',
      'coordinator',
    ],
    recommendedFor: ['Trivial Cooperation', 'Early Containment'],
  },

  social_dilemma: {
    id: 'social_dilemma',
    name: 'Social Dilemma',
    description: 'Mix of cooperators and free-riders to test cooperation dynamics.',
    archetypeIds: [
      'firefighter',
      'free_rider',
      'coordinator',
      'free_rider',
      'hero',
      'opportunist',
      'firefighter',
      'free_rider',
      'coordinator',
      'opportunist',
    ],
    recommendedFor: ['Greedy Neighbor', 'Mixed Motivation'],
  },

  chaos_squad: {
    id: 'chaos_squad',
    name: 'Chaos Squad',
    description: 'Unpredictable and experimental team with high variance.',
    archetypeIds: [
      'maverick',
      'random',
      'liar',
      'opportunist',
      'free_rider',
      'maverick',
      'random',
      'liar',
    ],
    recommendedFor: ['Deceptive Calm', 'Rest Trap'],
  },

  all_stars: {
    id: 'all_stars',
    name: 'All-Stars',
    description: 'Best performers from each category for maximum effectiveness.',
    archetypeIds: [
      'firefighter',
      'firefighter',
      'coordinator',
      'coordinator',
      'hero',
      'hero',
      'strategist',
      'strategist',
      'cautious',
      'firefighter',
    ],
    recommendedFor: ['Chain Reaction', 'Sparse Heroics'],
  },

  selfish_strategy: {
    id: 'selfish_strategy',
    name: 'Selfish Strategy',
    description: 'Every agent for themselves. Tests individual optimization.',
    archetypeIds: [
      'opportunist',
      'opportunist',
      'opportunist',
      'free_rider',
      'free_rider',
      'opportunist',
      'free_rider',
      'opportunist',
      'free_rider',
      'opportunist',
    ],
    recommendedFor: ['Mixed Motivation', 'Greedy Neighbor'],
  },

  honest_workers: {
    id: 'honest_workers',
    name: 'Honest Workers',
    description: 'No deception, pure effort. Transparent coordination.',
    archetypeIds: [
      'firefighter',
      'firefighter',
      'firefighter',
      'coordinator',
      'coordinator',
      'hero',
      'hero',
      'firefighter',
      'coordinator',
      'firefighter',
    ],
    recommendedFor: ['Trivial Cooperation', 'Early Containment'],
  },

  minimal_team: {
    id: 'minimal_team',
    name: 'Minimal Team',
    description: 'Small but mighty team with just 4 agents.',
    archetypeIds: ['firefighter', 'firefighter', 'coordinator', 'hero'],
    recommendedFor: ['Sparse Heroics', 'Overcrowding'],
  },

  balanced_research: {
    id: 'balanced_research',
    name: 'Balanced Research',
    description: 'One of each archetype for scientific comparison.',
    archetypeIds: [
      'firefighter',
      'free_rider',
      'coordinator',
      'liar',
      'hero',
      'strategist',
      'opportunist',
      'cautious',
      'maverick',
      'random',
    ],
    recommendedFor: [],
  },

  deception_masters: {
    id: 'deception_masters',
    name: 'Deception Masters',
    description: 'Team built around misleading signals and opportunism.',
    archetypeIds: ['liar', 'liar', 'opportunist', 'free_rider', 'liar', 'opportunist'],
    recommendedFor: ['Deceptive Calm', 'Honest vs Liar'],
  },

  risk_takers: {
    id: 'risk_takers',
    name: 'Risk Takers',
    description: 'Aggressive firefighting with high-risk tolerance.',
    archetypeIds: [
      'hero',
      'hero',
      'firefighter',
      'firefighter',
      'maverick',
      'hero',
      'firefighter',
      'strategist',
    ],
    recommendedFor: ['Chain Reaction', 'Early Containment'],
  },
};

/**
 * Get template by ID
 */
export function getTemplate(id: string): TeamTemplate | null {
  return TEAM_TEMPLATES[id] || null;
}

/**
 * Get all template IDs in display order
 */
export function getTemplateIds(): string[] {
  return [
    'perfect_cooperation',
    'all_stars',
    'honest_workers',
    'social_dilemma',
    'balanced_research',
    'minimal_team',
    'selfish_strategy',
    'deception_masters',
    'risk_takers',
    'chaos_squad',
  ];
}

/**
 * Get all templates in display order
 */
export function getAllTemplates(): TeamTemplate[] {
  return getTemplateIds().map((id) => TEAM_TEMPLATES[id]);
}

/**
 * Get template recommendations for a scenario type
 */
export function getTemplatesForScenario(scenarioType: string): TeamTemplate[] {
  return getAllTemplates().filter((template) =>
    template.recommendedFor?.includes(scenarioType),
  );
}
