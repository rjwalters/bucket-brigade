/**
 * Agent Radar Chart Utilities
 *
 * Converts agent parameters into radar chart dimensions for visual comparison.
 * Based on the canonical agent parameter definitions.
 */

export interface RadarProfile {
  cooperation: number;
  reliability: number;
  workEthic: number;
  selfPreservation: number;
  riskManagement: number;
  initiative: number;
}

export interface RadarDimension {
  key: keyof RadarProfile;
  label: string;
  description: string;
  min: number;
  max: number;
}

/**
 * Radar chart dimensions with descriptions
 */
export const RADAR_DIMENSIONS: RadarDimension[] = [
  {
    key: 'cooperation',
    label: 'Cooperation',
    description: 'Willingness to help neighbors and coordinate with team',
    min: 0,
    max: 10,
  },
  {
    key: 'reliability',
    label: 'Reliability',
    description: 'Consistency and trustworthiness in actions and signals',
    min: 0,
    max: 10,
  },
  {
    key: 'workEthic',
    label: 'Work Ethic',
    description: 'Energy and commitment to actively fighting fires',
    min: 0,
    max: 10,
  },
  {
    key: 'selfPreservation',
    label: 'Self-Preservation',
    description: 'Priority on protecting own houses vs. helping others',
    min: 0,
    max: 10,
  },
  {
    key: 'riskManagement',
    label: 'Risk Management',
    description: 'Caution and strategic thinking about fire spread',
    min: 0,
    max: 10,
  },
  {
    key: 'initiative',
    label: 'Initiative',
    description: 'Independence and willingness to act without coordination',
    min: 0,
    max: 10,
  },
];

export interface AgentParameters {
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

/**
 * Calculate radar profile from agent parameters
 *
 * This converts the 10-dimensional parameter space into 6 intuitive dimensions
 * for visual comparison.
 */
export function calculateRadarProfile(params: AgentParameters): RadarProfile {
  // Cooperation: neighbor_help_bias (40%) + altruism_factor (40%) + coordination_weight (20%)
  const cooperation = Math.round(
    (params.neighbor_help_bias * 0.4 + params.altruism_factor * 0.4 + params.coordination_weight * 0.2) * 10
  );

  // Reliability: honesty_bias (50%) + (1 - exploration_rate) (30%) + (1 - fatigue_memory) (20%)
  const reliability = Math.round(
    (params.honesty_bias * 0.5 + (1 - params.exploration_rate) * 0.3 + (1 - params.fatigue_memory) * 0.2) * 10
  );

  // Work Ethic: work_tendency (60%) + (1 - rest_reward_bias) (40%)
  const workEthic = Math.round((params.work_tendency * 0.6 + (1 - params.rest_reward_bias) * 0.4) * 10);

  // Self-Preservation: own_house_priority (60%) + (1 - altruism_factor) (40%)
  const selfPreservation = Math.round((params.own_house_priority * 0.6 + (1 - params.altruism_factor) * 0.4) * 10);

  // Risk Management: risk_aversion (70%) + (1 - exploration_rate) (30%)
  const riskManagement = Math.round((params.risk_aversion * 0.7 + (1 - params.exploration_rate) * 0.3) * 10);

  // Initiative: (1 - coordination_weight) (40%) + work_tendency (30%) + exploration_rate (30%)
  const initiative = Math.round(
    ((1 - params.coordination_weight) * 0.4 + params.work_tendency * 0.3 + params.exploration_rate * 0.3) * 10
  );

  return {
    cooperation,
    reliability,
    workEthic,
    selfPreservation,
    riskManagement,
    initiative,
  };
}

/**
 * Calculate similarity score between two radar profiles (0-100)
 *
 * Uses Euclidean distance normalized to 0-100 scale.
 * 100 = identical, 0 = maximum difference
 */
export function calculateSimilarity(profile1: RadarProfile, profile2: RadarProfile): number {
  const dimensions: (keyof RadarProfile)[] = [
    'cooperation',
    'reliability',
    'workEthic',
    'selfPreservation',
    'riskManagement',
    'initiative',
  ];

  const squaredDiffs = dimensions.map((dim) => {
    const diff = profile1[dim] - profile2[dim];
    return diff * diff;
  });

  const euclideanDist = Math.sqrt(squaredDiffs.reduce((sum, val) => sum + val, 0));

  // Maximum possible distance is sqrt(6 * 10^2) = ~24.49
  const maxDistance = Math.sqrt(6 * 100);

  // Convert to 0-100 similarity score
  const similarity = 100 * (1 - euclideanDist / maxDistance);

  return Math.round(similarity);
}

/**
 * Get top N most similar archetypes to a given profile
 */
export function findSimilarProfiles(
  targetProfile: RadarProfile,
  allProfiles: Map<string, RadarProfile>,
  topN: number = 3
): Array<{ id: string; similarity: number }> {
  const similarities = Array.from(allProfiles.entries())
    .map(([id, profile]) => ({
      id,
      similarity: calculateSimilarity(targetProfile, profile),
    }))
    .sort((a, b) => b.similarity - a.similarity);

  return similarities.slice(0, topN);
}

/**
 * Calculate team balance score (0-100)
 *
 * A balanced team has diverse radar profiles covering different dimensions.
 * Higher score = more balanced team.
 */
export function calculateTeamBalance(profiles: RadarProfile[]): number {
  if (profiles.length === 0) return 0;
  if (profiles.length === 1) return 50;

  const dimensions: (keyof RadarProfile)[] = [
    'cooperation',
    'reliability',
    'workEthic',
    'selfPreservation',
    'riskManagement',
    'initiative',
  ];

  // Calculate standard deviation for each dimension
  const stdDevs = dimensions.map((dim) => {
    const values = profiles.map((p) => p[dim]);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  });

  // Average standard deviation across dimensions
  const avgStdDev = stdDevs.reduce((sum, val) => sum + val, 0) / stdDevs.length;

  // Ideal std dev is ~3 (diverse but not extreme)
  // Penalize both low diversity (std dev < 2) and extreme diversity (std dev > 4)
  const idealStdDev = 3;
  const deviation = Math.abs(avgStdDev - idealStdDev);
  const balanceScore = Math.max(0, 100 - deviation * 20);

  return Math.round(balanceScore);
}

/**
 * Get team coverage - which dimensions are well-covered by the team
 *
 * Returns percentage coverage (0-100) for each dimension.
 * 100 = at least one agent with 8+ in this dimension
 */
export function calculateTeamCoverage(profiles: RadarProfile[]): Record<keyof RadarProfile, number> {
  if (profiles.length === 0) {
    return {
      cooperation: 0,
      reliability: 0,
      workEthic: 0,
      selfPreservation: 0,
      riskManagement: 0,
      initiative: 0,
    };
  }

  const dimensions: (keyof RadarProfile)[] = [
    'cooperation',
    'reliability',
    'workEthic',
    'selfPreservation',
    'riskManagement',
    'initiative',
  ];

  const coverage = {} as Record<keyof RadarProfile, number>;

  dimensions.forEach((dim) => {
    const maxValue = Math.max(...profiles.map((p) => p[dim]));
    // Scale: 0-4 = 0%, 5 = 50%, 8+ = 100%
    coverage[dim] = Math.min(100, Math.round((maxValue / 8) * 100));
  });

  return coverage;
}

/**
 * Format radar profile for chart libraries (e.g., Recharts, Chart.js)
 */
export function formatForChart(profile: RadarProfile): Array<{ dimension: string; value: number; max: number }> {
  return RADAR_DIMENSIONS.map((dim) => ({
    dimension: dim.label,
    value: profile[dim.key],
    max: dim.max,
  }));
}

/**
 * Get color for dimension value (for visual feedback)
 */
export function getDimensionColor(value: number, max: number = 10): string {
  const percentage = (value / max) * 100;

  if (percentage >= 80) return '#22c55e'; // green-500 (excellent)
  if (percentage >= 60) return '#eab308'; // yellow-500 (good)
  if (percentage >= 40) return '#f97316'; // orange-500 (moderate)
  return '#ef4444'; // red-500 (low)
}
