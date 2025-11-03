/**
 * Agent Comparison Utilities
 *
 * Core comparison and analysis logic for agent discovery and team building.
 * Provides matchup matrices, complementarity detection, and synergy analysis.
 */

import type { AgentParameters } from './agentRadarChart';
import { calculateRadarProfile, calculateSimilarity, type RadarProfile } from './agentRadarChart';

/**
 * Agent comparison entry - an agent with a unique ID
 */
export interface AgentEntry {
  id: string;
  name: string;
  parameters: AgentParameters;
}

/**
 * Matchup matrix result - NxN similarity matrix for all agents
 */
export interface MatchupMatrix {
  agentIds: string[];
  matrix: number[][]; // matrix[i][j] = similarity(agent_i, agent_j)
  profiles: Map<string, RadarProfile>;
}

/**
 * Complementarity score between two agents
 */
export interface ComplementarityScore {
  agentId: string;
  score: number; // 0-100: how complementary they are (high where target is low)
  dimensions: Record<keyof RadarProfile, number>; // Complementarity by dimension
}

/**
 * Synergy analysis result for a team
 */
export interface SynergyAnalysis {
  balanceScore: number; // 0-100: diversity and balance
  coverageScore: number; // 0-100: overall dimension coverage
  redundancyScore: number; // 0-100: lower = more redundancy
  suggestions: string[]; // Actionable improvement suggestions
  dimensionCoverage: Record<keyof RadarProfile, number>; // Coverage by dimension
}

/**
 * Comparison explanation between two agents
 */
export interface ComparisonExplanation {
  similarity: number;
  differences: Array<{
    dimension: keyof RadarProfile;
    agent1Value: number;
    agent2Value: number;
    difference: number;
  }>;
  summary: string;
}

/**
 * Build NxN matchup matrix showing similarity between all agent pairs
 *
 * @param agents - List of agents to compare
 * @returns Matchup matrix with similarities and radar profiles
 */
export function buildMatchupMatrix(agents: AgentEntry[]): MatchupMatrix {
  // Pre-compute radar profiles for all agents
  const profiles = new Map<string, RadarProfile>();
  agents.forEach((agent) => {
    profiles.set(agent.id, calculateRadarProfile(agent.parameters));
  });

  // Build NxN similarity matrix
  const agentIds = agents.map((a) => a.id);
  const n = agentIds.length;
  const matrix: number[][] = Array(n)
    .fill(0)
    .map(() => Array(n).fill(0));

  // Calculate similarities for all pairs
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const profile1 = profiles.get(agentIds[i])!;
      const profile2 = profiles.get(agentIds[j])!;
      matrix[i][j] = calculateSimilarity(profile1, profile2);
    }
  }

  return {
    agentIds,
    matrix,
    profiles,
  };
}

/**
 * Find agents that complement the target (high where target is low)
 *
 * @param targetAgent - Agent to find complements for
 * @param candidateAgents - Pool of agents to consider
 * @param topN - Number of top complements to return
 * @returns Sorted list of complementary agents (highest first)
 */
export function findComplementaryAgents(
  targetAgent: AgentEntry,
  candidateAgents: AgentEntry[],
  topN: number = 5
): ComplementarityScore[] {
  const targetProfile = calculateRadarProfile(targetAgent.parameters);

  const dimensions: (keyof RadarProfile)[] = [
    'cooperation',
    'reliability',
    'workEthic',
    'selfPreservation',
    'riskManagement',
    'initiative',
  ];

  // Calculate complementarity scores for each candidate
  const scores = candidateAgents
    .filter((candidate) => candidate.id !== targetAgent.id) // Exclude self
    .map((candidate) => {
      const candidateProfile = calculateRadarProfile(candidate.parameters);

      // Complementarity: high score when candidate is strong where target is weak
      const dimensionScores: Record<keyof RadarProfile, number> = {} as Record<keyof RadarProfile, number>;
      let totalCompScore = 0;

      dimensions.forEach((dim) => {
        const targetValue = targetProfile[dim];
        const candidateValue = candidateProfile[dim];

        // Complementarity formula: score high when target is low and candidate is high
        // Use inverse target value * candidate value, normalized
        const targetWeakness = 10 - targetValue; // Higher when target is weak
        const candidateStrength = candidateValue; // Higher when candidate is strong
        const compScore = (targetWeakness * candidateStrength) / 100; // Normalize to 0-1 range

        dimensionScores[dim] = Math.round(compScore * 100);
        totalCompScore += compScore;
      });

      // Average complementarity across all dimensions
      const avgCompScore = (totalCompScore / dimensions.length) * 100;

      return {
        agentId: candidate.id,
        score: Math.round(avgCompScore),
        dimensions: dimensionScores,
      };
    })
    .sort((a, b) => b.score - a.score); // Sort by complementarity (highest first)

  return scores.slice(0, topN);
}

/**
 * Analyze team synergy - balance, coverage, and redundancy
 *
 * @param agents - Team members to analyze
 * @returns Synergy analysis with scores and suggestions
 */
export function analyzeSynergy(agents: AgentEntry[]): SynergyAnalysis {
  if (agents.length === 0) {
    return {
      balanceScore: 0,
      coverageScore: 0,
      redundancyScore: 0,
      suggestions: ['Add at least one agent to the team'],
      dimensionCoverage: {
        cooperation: 0,
        reliability: 0,
        workEthic: 0,
        selfPreservation: 0,
        riskManagement: 0,
        initiative: 0,
      },
    };
  }

  const profiles = agents.map((agent) => calculateRadarProfile(agent.parameters));
  const dimensions: (keyof RadarProfile)[] = [
    'cooperation',
    'reliability',
    'workEthic',
    'selfPreservation',
    'riskManagement',
    'initiative',
  ];

  // Calculate dimension coverage
  const dimensionCoverage = {} as Record<keyof RadarProfile, number>;
  dimensions.forEach((dim) => {
    const maxValue = Math.max(...profiles.map((p) => p[dim]));
    // Scale: 0-4 = poor, 5-7 = moderate, 8-10 = excellent
    dimensionCoverage[dim] = Math.min(100, Math.round((maxValue / 8) * 100));
  });

  // Overall coverage score (average across dimensions)
  const coverageScore = Math.round(
    dimensions.reduce((sum, dim) => sum + dimensionCoverage[dim], 0) / dimensions.length
  );

  // Balance score: diversity without extremes
  const balanceScore = calculateBalanceScore(profiles);

  // Redundancy score: penalize similar agents (lower = more redundancy)
  const redundancyScore = calculateRedundancyScore(profiles);

  // Generate actionable suggestions
  const suggestions = generateSuggestions(dimensionCoverage, balanceScore, redundancyScore, agents.length);

  return {
    balanceScore,
    coverageScore,
    redundancyScore,
    suggestions,
    dimensionCoverage,
  };
}

/**
 * Calculate team balance score based on diversity
 */
function calculateBalanceScore(profiles: RadarProfile[]): number {
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
 * Calculate redundancy score (lower = more redundancy)
 */
function calculateRedundancyScore(profiles: RadarProfile[]): number {
  if (profiles.length <= 1) return 100; // No redundancy possible with ≤1 agent

  // Calculate pairwise similarities
  let totalSimilarity = 0;
  let pairCount = 0;

  for (let i = 0; i < profiles.length; i++) {
    for (let j = i + 1; j < profiles.length; j++) {
      totalSimilarity += calculateSimilarity(profiles[i], profiles[j]);
      pairCount++;
    }
  }

  const avgSimilarity = totalSimilarity / pairCount;

  // Convert to redundancy score (lower similarity = less redundancy = higher score)
  // 100 similarity = 0 redundancy score, 0 similarity = 100 redundancy score
  const redundancyScore = 100 - avgSimilarity;

  return Math.round(redundancyScore);
}

/**
 * Generate actionable suggestions for team improvement
 */
function generateSuggestions(
  dimensionCoverage: Record<keyof RadarProfile, number>,
  balanceScore: number,
  redundancyScore: number,
  teamSize: number
): string[] {
  const suggestions: string[] = [];

  // Check for weak coverage in specific dimensions
  const dimensions: (keyof RadarProfile)[] = [
    'cooperation',
    'reliability',
    'workEthic',
    'selfPreservation',
    'riskManagement',
    'initiative',
  ];

  const weakDimensions = dimensions.filter((dim) => dimensionCoverage[dim] < 60);
  if (weakDimensions.length > 0) {
    const dimNames = weakDimensions.map((d) => d.charAt(0).toUpperCase() + d.slice(1)).join(', ');
    suggestions.push(`Add agents with higher ${dimNames} to improve coverage`);
  }

  // Check for poor balance
  if (balanceScore < 50) {
    suggestions.push('Team lacks diversity - consider adding agents with different behavioral profiles');
  } else if (balanceScore > 90) {
    suggestions.push('Team may be too diverse - consider agents with more aligned strategies');
  }

  // Check for redundancy
  if (redundancyScore > 70) {
    suggestions.push('Team has high redundancy - agents are very similar, consider more diverse profiles');
  }

  // Check team size
  if (teamSize < 3) {
    suggestions.push('Small team - consider adding more agents for better coverage and resilience');
  } else if (teamSize > 8) {
    suggestions.push('Large team - ensure coordination strategies are in place to avoid inefficiency');
  }

  // If everything is good
  if (suggestions.length === 0) {
    suggestions.push('Team composition looks well-balanced!');
  }

  return suggestions;
}

/**
 * Explain differences between two agents in human-readable format
 *
 * @param agent1 - First agent to compare
 * @param agent2 - Second agent to compare
 * @returns Detailed comparison explanation
 */
export function explainDifferences(agent1: AgentEntry, agent2: AgentEntry): ComparisonExplanation {
  const profile1 = calculateRadarProfile(agent1.parameters);
  const profile2 = calculateRadarProfile(agent2.parameters);

  const similarity = calculateSimilarity(profile1, profile2);

  const dimensions: (keyof RadarProfile)[] = [
    'cooperation',
    'reliability',
    'workEthic',
    'selfPreservation',
    'riskManagement',
    'initiative',
  ];

  // Calculate differences for each dimension
  const differences = dimensions
    .map((dim) => ({
      dimension: dim,
      agent1Value: profile1[dim],
      agent2Value: profile2[dim],
      difference: Math.abs(profile1[dim] - profile2[dim]),
    }))
    .sort((a, b) => b.difference - a.difference); // Sort by largest differences first

  // Generate summary
  let summary = '';
  if (similarity > 80) {
    summary = `${agent1.name} and ${agent2.name} are very similar (${similarity}% match).`;
  } else if (similarity > 60) {
    summary = `${agent1.name} and ${agent2.name} are moderately similar (${similarity}% match).`;
  } else if (similarity > 40) {
    summary = `${agent1.name} and ${agent2.name} are somewhat different (${similarity}% match).`;
  } else {
    summary = `${agent1.name} and ${agent2.name} are quite different (${similarity}% match).`;
  }

  // Add largest difference to summary
  if (differences.length > 0 && differences[0].difference > 3) {
    const largest = differences[0];
    const dimName = largest.dimension.charAt(0).toUpperCase() + largest.dimension.slice(1);
    const higherAgent = largest.agent1Value > largest.agent2Value ? agent1.name : agent2.name;
    summary += ` The biggest difference is in ${dimName}, where ${higherAgent} scores significantly higher.`;
  }

  return {
    similarity,
    differences,
    summary,
  };
}
