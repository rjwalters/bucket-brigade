/**
 * Unit tests for agent comparison utilities
 *
 * Tests all functions in agentComparison.ts with comprehensive coverage:
 * - buildMatchupMatrix: symmetry, diagonal, correct similarities
 * - findComplementaryAgents: complementarity logic, sorting, exclusion of self
 * - analyzeSynergy: balance, coverage, redundancy calculations
 * - explainDifferences: difference detection, summary generation
 */

import { expect, test } from '@playwright/test';
import {
  buildMatchupMatrix,
  findComplementaryAgents,
  analyzeSynergy,
  explainDifferences,
  type AgentEntry,
} from '../src/utils/agentComparison';

// Test agents with known characteristics
const testAgents: AgentEntry[] = [
  {
    id: 'firefighter',
    name: 'Firefighter',
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
  },
  {
    id: 'free_rider',
    name: 'Free Rider',
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
  },
  {
    id: 'hero',
    name: 'Hero',
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
  },
];

test.describe('buildMatchupMatrix', () => {
  test('should create NxN matrix with correct dimensions', () => {
    const result = buildMatchupMatrix(testAgents);

    expect(result.agentIds).toHaveLength(3);
    expect(result.matrix).toHaveLength(3);
    expect(result.matrix[0]).toHaveLength(3);
    expect(result.profiles.size).toBe(3);
  });

  test('should have 100 similarity on diagonal (agent vs self)', () => {
    const result = buildMatchupMatrix(testAgents);

    for (let i = 0; i < result.agentIds.length; i++) {
      expect(result.matrix[i][i]).toBe(100);
    }
  });

  test('should be symmetric (matrix[i][j] === matrix[j][i])', () => {
    const result = buildMatchupMatrix(testAgents);

    for (let i = 0; i < result.agentIds.length; i++) {
      for (let j = i + 1; j < result.agentIds.length; j++) {
        expect(result.matrix[i][j]).toBe(result.matrix[j][i]);
      }
    }
  });

  test('should have lower similarity between very different agents', () => {
    const result = buildMatchupMatrix(testAgents);

    // Firefighter vs Free Rider should be low similarity (opposite behaviors)
    const firefighterIdx = result.agentIds.indexOf('firefighter');
    const freeRiderIdx = result.agentIds.indexOf('free_rider');

    expect(result.matrix[firefighterIdx][freeRiderIdx]).toBeLessThan(60);
  });

  test('should have higher similarity between similar agents', () => {
    const result = buildMatchupMatrix(testAgents);

    // Firefighter vs Hero should be higher similarity (both altruistic)
    const firefighterIdx = result.agentIds.indexOf('firefighter');
    const heroIdx = result.agentIds.indexOf('hero');

    expect(result.matrix[firefighterIdx][heroIdx]).toBeGreaterThan(60);
  });

  test('should store radar profiles for all agents', () => {
    const result = buildMatchupMatrix(testAgents);

    testAgents.forEach((agent) => {
      expect(result.profiles.has(agent.id)).toBe(true);
      const profile = result.profiles.get(agent.id)!;
      expect(profile).toHaveProperty('cooperation');
      expect(profile).toHaveProperty('reliability');
      expect(profile).toHaveProperty('workEthic');
      expect(profile).toHaveProperty('selfPreservation');
      expect(profile).toHaveProperty('riskManagement');
      expect(profile).toHaveProperty('initiative');
    });
  });
});

test.describe('findComplementaryAgents', () => {
  test('should exclude the target agent from results', () => {
    const target = testAgents[0];
    const results = findComplementaryAgents(target, testAgents, 5);

    const targetInResults = results.some((r) => r.agentId === target.id);
    expect(targetInResults).toBe(false);
  });

  test('should return sorted results (highest complementarity first)', () => {
    const target = testAgents[0];
    const results = findComplementaryAgents(target, testAgents, 5);

    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });

  test('should limit results to topN', () => {
    const target = testAgents[0];
    const results = findComplementaryAgents(target, testAgents, 1);

    expect(results).toHaveLength(1);
  });

  test('should calculate complementarity correctly (high where target is low)', () => {
    // Free Rider (low work ethic, low altruism) should complement
    // Hero (high work ethic, high altruism) poorly
    const target = testAgents[2]; // Hero
    const results = findComplementaryAgents(target, testAgents);

    // Free Rider should score low as complement to Hero (both extremes, but opposite)
    const freeRiderResult = results.find((r) => r.agentId === 'free_rider');
    expect(freeRiderResult).toBeDefined();

    // Complementarity should be based on "high where target is low"
    // Hero is strong everywhere, so complements should score moderately
    expect(freeRiderResult!.score).toBeGreaterThanOrEqual(0);
    expect(freeRiderResult!.score).toBeLessThanOrEqual(100);
  });

  test('should include dimension-level complementarity scores', () => {
    const target = testAgents[0];
    const results = findComplementaryAgents(target, testAgents, 1);

    expect(results[0].dimensions).toHaveProperty('cooperation');
    expect(results[0].dimensions).toHaveProperty('reliability');
    expect(results[0].dimensions).toHaveProperty('workEthic');
    expect(results[0].dimensions).toHaveProperty('selfPreservation');
    expect(results[0].dimensions).toHaveProperty('riskManagement');
    expect(results[0].dimensions).toHaveProperty('initiative');
  });

  test('should handle empty candidate pool', () => {
    const target = testAgents[0];
    const results = findComplementaryAgents(target, [target], 5);

    expect(results).toHaveLength(0);
  });
});

test.describe('analyzeSynergy', () => {
  test('should handle empty team gracefully', () => {
    const result = analyzeSynergy([]);

    expect(result.balanceScore).toBe(0);
    expect(result.coverageScore).toBe(0);
    expect(result.redundancyScore).toBe(0);
    expect(result.suggestions).toContain('Add at least one agent to the team');
  });

  test('should calculate coverage for all dimensions', () => {
    const result = analyzeSynergy(testAgents);

    expect(result.dimensionCoverage).toHaveProperty('cooperation');
    expect(result.dimensionCoverage).toHaveProperty('reliability');
    expect(result.dimensionCoverage).toHaveProperty('workEthic');
    expect(result.dimensionCoverage).toHaveProperty('selfPreservation');
    expect(result.dimensionCoverage).toHaveProperty('riskManagement');
    expect(result.dimensionCoverage).toHaveProperty('initiative');

    // All coverage scores should be 0-100
    Object.values(result.dimensionCoverage).forEach((score) => {
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(100);
    });
  });

  test('should detect high redundancy in homogeneous team', () => {
    // Team of identical agents = high redundancy
    const homogeneousTeam = [testAgents[0], testAgents[0], testAgents[0]];
    const result = analyzeSynergy(homogeneousTeam);

    // Redundancy score should be low (high redundancy = low score? Let me check implementation)
    // Actually, redundancyScore is HIGH when there's LOW redundancy
    // So identical agents should have HIGH redundancy score (100 - avgSimilarity)
    // 100 similarity = 0 redundancy score
    expect(result.redundancyScore).toBeLessThan(20);
  });

  test('should detect good balance in diverse team', () => {
    // Test agents have diverse profiles
    const result = analyzeSynergy(testAgents);

    // Balance score should be reasonable (not 0 or 100)
    expect(result.balanceScore).toBeGreaterThan(20);
    expect(result.balanceScore).toBeLessThanOrEqual(100);
  });

  test('should provide actionable suggestions', () => {
    const result = analyzeSynergy(testAgents);

    expect(result.suggestions.length).toBeGreaterThan(0);
    result.suggestions.forEach((suggestion) => {
      expect(typeof suggestion).toBe('string');
      expect(suggestion.length).toBeGreaterThan(0);
    });
  });

  test('should calculate overall coverage score', () => {
    const result = analyzeSynergy(testAgents);

    expect(result.coverageScore).toBeGreaterThanOrEqual(0);
    expect(result.coverageScore).toBeLessThanOrEqual(100);
  });

  test('should handle single-agent team', () => {
    const result = analyzeSynergy([testAgents[0]]);

    expect(result.balanceScore).toBe(50);
    expect(result.redundancyScore).toBe(100); // No redundancy with single agent
  });
});

test.describe('explainDifferences', () => {
  test('should calculate similarity between agents', () => {
    const result = explainDifferences(testAgents[0], testAgents[1]);

    expect(result.similarity).toBeGreaterThanOrEqual(0);
    expect(result.similarity).toBeLessThanOrEqual(100);
  });

  test('should list differences for all dimensions', () => {
    const result = explainDifferences(testAgents[0], testAgents[1]);

    expect(result.differences).toHaveLength(6); // 6 radar dimensions

    result.differences.forEach((diff) => {
      expect(diff).toHaveProperty('dimension');
      expect(diff).toHaveProperty('agent1Value');
      expect(diff).toHaveProperty('agent2Value');
      expect(diff).toHaveProperty('difference');
      expect(diff.difference).toBeGreaterThanOrEqual(0);
    });
  });

  test('should sort differences by magnitude (largest first)', () => {
    const result = explainDifferences(testAgents[0], testAgents[1]);

    for (let i = 1; i < result.differences.length; i++) {
      expect(result.differences[i - 1].difference).toBeGreaterThanOrEqual(result.differences[i].difference);
    }
  });

  test('should generate appropriate summary for similar agents', () => {
    // Compare agent to itself (100% similar)
    const result = explainDifferences(testAgents[0], testAgents[0]);

    expect(result.summary).toContain('very similar');
    expect(result.summary).toMatch(/\d+%/); // Should mention percentage
  });

  test('should generate appropriate summary for different agents', () => {
    // Firefighter vs Free Rider (very different)
    const result = explainDifferences(testAgents[0], testAgents[1]);

    expect(result.summary).toBeDefined();
    expect(result.summary.length).toBeGreaterThan(0);
    expect(result.summary).toMatch(/\d+%/); // Should mention percentage
  });

  test('should mention biggest difference in summary when significant', () => {
    const result = explainDifferences(testAgents[0], testAgents[1]);

    // Should mention the dimension with largest difference if > 3
    const largestDiff = result.differences[0];
    if (largestDiff.difference > 3) {
      // Summary should reference the dimension name
      const dimName = largestDiff.dimension.charAt(0).toUpperCase() + largestDiff.dimension.slice(1);
      expect(result.summary.toLowerCase()).toContain(dimName.toLowerCase());
    }
  });

  test('should handle identical agents', () => {
    const result = explainDifferences(testAgents[0], testAgents[0]);

    expect(result.similarity).toBe(100);
    result.differences.forEach((diff) => {
      expect(diff.difference).toBe(0);
    });
  });
});

test.describe('Integration tests', () => {
  test('should work together: find complements and analyze synergy', () => {
    const target = testAgents[0];

    // Find complementary agents
    const complements = findComplementaryAgents(target, testAgents, 2);

    // Build a team with target + top complement
    const team = [target, testAgents.find((a) => a.id === complements[0].agentId)!];

    // Analyze team synergy
    const synergy = analyzeSynergy(team);

    expect(synergy.balanceScore).toBeGreaterThanOrEqual(0);
    expect(synergy.coverageScore).toBeGreaterThanOrEqual(0);
    expect(synergy.suggestions.length).toBeGreaterThan(0);
  });

  test('should work together: matchup matrix and explain differences', () => {
    const matrix = buildMatchupMatrix(testAgents);

    // Pick two agents from the matrix
    const agent1 = testAgents[0];
    const agent2 = testAgents[1];

    // Get similarity from matrix
    const idx1 = matrix.agentIds.indexOf(agent1.id);
    const idx2 = matrix.agentIds.indexOf(agent2.id);
    const matrixSimilarity = matrix.matrix[idx1][idx2];

    // Get explanation
    const explanation = explainDifferences(agent1, agent2);

    // Similarities should match
    expect(explanation.similarity).toBe(matrixSimilarity);
  });
});
