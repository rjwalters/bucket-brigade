/**
 * Tournament Engine
 *
 * Orchestrates running multiple games and aggregating results
 */

import type {
  TeamComposition,
  TournamentConfig,
  TournamentResult,
  TournamentProgress,
  ScenarioResult,
  TournamentStatistics,
  AgentContribution,
  ScenarioTypeStats,
} from '../types/teamBuilder';
import type { Scenario, GameReplay } from '../types';
import {
  type AgentObservation,
  type Agent,
  type GameResult,
} from './browserEngine';
import { generateScenarioSet } from './scenarioGenerator';
import type { ScenarioType } from './scenarioGenerator';
import { BrowserAgent } from './browserAgents';
import type { AgentArchetype } from '../types/teamBuilder';
import { createGameEngine, initWasm, isWasmInitialized } from './wasmEngine';
import { loadGameReplays, saveGameReplays } from './storage';

/**
 * Generate unique ID
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Convert GameResult to GameReplay format for replay viewer
 */
function convertToGameReplay(gameResult: GameResult): GameReplay {
  return {
    scenario: gameResult.scenario,
    nights: gameResult.nights,
  };
}

/**
 * Save interesting games from tournament to replay viewer
 * Saves best, worst, and median scoring games
 */
function saveTournamentGamesToReplay(results: ScenarioResult[]): void {
  if (results.length === 0) return;

  // Sort by score
  const sorted = [...results].sort((a, b) => b.teamScore - a.teamScore);

  // Select interesting games: best, worst, median, and a few random samples
  const gamesToSave: GameReplay[] = [];

  // Best game
  if (sorted[0]?.replayData) {
    gamesToSave.push(convertToGameReplay(sorted[0].replayData));
  }

  // Worst game
  if (sorted[sorted.length - 1]?.replayData) {
    gamesToSave.push(convertToGameReplay(sorted[sorted.length - 1].replayData));
  }

  // Median game
  const medianIdx = Math.floor(sorted.length / 2);
  if (sorted[medianIdx]?.replayData) {
    gamesToSave.push(convertToGameReplay(sorted[medianIdx].replayData));
  }

  // Add 3-5 random samples across the distribution
  const sampleIndices = [
    Math.floor(sorted.length * 0.25),
    Math.floor(sorted.length * 0.75),
    Math.floor(sorted.length * 0.33),
    Math.floor(sorted.length * 0.66),
  ];

  for (const idx of sampleIndices) {
    if (sorted[idx]?.replayData && gamesToSave.length < 8) {
      gamesToSave.push(convertToGameReplay(sorted[idx].replayData));
    }
  }

  // Load existing replays and append new ones
  const existingReplays = loadGameReplays();
  const allReplays = [...existingReplays, ...gamesToSave];

  // Keep only the most recent 25 games to avoid filling localStorage
  const recentReplays = allReplays
    .sort((a, b) => (b.timestamp ? Number(b.timestamp) : 0) - (a.timestamp ? Number(a.timestamp) : 0))
    .slice(0, 25);

  saveGameReplays(recentReplays);

  console.log(`💾 Saved ${gamesToSave.length} tournament games to replay viewer`);
}

/**
 * Heuristic Agent implementation using archetype parameters
 */
class HeuristicAgent extends BrowserAgent {
  private params: AgentArchetype['parameters'];

  constructor(id: number, name: string, params: AgentArchetype['parameters']) {
    super(id, name);
    this.params = params;
  }

  act(obs: AgentObservation): number[] {
    const { houses, signals } = obs;
    const agentId = this.id;

    // Count burning houses
    const burningCount = houses.filter((h: number) => h === 1).length;
    const burningFraction = burningCount / houses.length;

    // Decide whether to work based on parameters
    const workProbability =
      this.params.work_tendency *
      (1 - this.params.rest_reward_bias) *
      (1 - this.params.risk_aversion * burningFraction);

    const shouldWork = Math.random() < workProbability;
    const mode = shouldWork ? 1 : 0;

    // Find target house
    let targetHouse = agentId % houses.length; // Default to own house

    if (shouldWork) {
      // Find burning houses
      const burningHouses: number[] = [];
      for (let i = 0; i < houses.length; i++) {
        if (houses[i] === 1) {
          burningHouses.push(i);
        }
      }

      if (burningHouses.length > 0) {
        // Score each burning house
        const scores = burningHouses.map((house) => {
          let score = 0;

          // Own house priority
          if (house === agentId % houses.length) {
            score += this.params.own_house_priority * 10;
          }

          // Neighbor help bias
          const ownHouse = agentId % houses.length;
          const isNeighbor =
            house === (ownHouse - 1 + houses.length) % houses.length ||
            house === (ownHouse + 1) % houses.length;
          if (isNeighbor) {
            score += this.params.neighbor_help_bias * 5;
          }

          // Coordination - check if others signaling work there
          const othersSignalingWork = signals.filter((s: number) => s === 1).length;
          if (othersSignalingWork > 0) {
            score += this.params.coordination_weight * 3;
          }

          // Add randomness
          score += Math.random() * this.params.exploration_rate * 5;

          return score;
        });

        // Select house with highest score
        const maxScore = Math.max(...scores);
        const bestIndex = scores.indexOf(maxScore);
        targetHouse = burningHouses[bestIndex];
      }
    }

    return [targetHouse, mode];
  }
}

/**
 * Run a single game and return results
 */
async function runSingleGame(
  team: TeamComposition,
  scenario: Scenario,
  scenarioType: ScenarioType,
  forceJsEngine = false,
): Promise<ScenarioResult> {
  // Filter out null agents
  const activeAgents = team.positions.filter((a) => a != null);

  // Create engine with scenario (WASM if available, fallback to JS)
  const engine = await createGameEngine(scenario, !forceJsEngine);

  // Create agents
  const agents: Agent[] = activeAgents.map((archetype, i) =>
    new HeuristicAgent(i, archetype.name, archetype.parameters),
  );

  let nightsPlayed = 0;
  let done = false;

  // Run game until done
  while (!done && nightsPlayed < 100) {
    // Get observations for each agent
    const actions = agents.map((agent, i) => {
      const obs = engine.get_observation(i);
      return agent.act(obs);
    });

    // Execute step
    const result = engine.step(actions);
    done = result.done;

    nightsPlayed++;
  }

  // Get final results
  const gameResult = engine.get_result();
  const currentState = engine.get_current_state();

  // Count houses saved
  const housesSaved = currentState.houses.filter((h) => h === 0).length;

  return {
    scenarioId: generateId(),
    scenarioType,
    teamScore: gameResult.final_score,
    agentScores: gameResult.agent_scores,
    housesSaved,
    nightsPlayed,
    replayData: gameResult,
  };
}

/**
 * Calculate tournament statistics
 */
function calculateStatistics(results: ScenarioResult[]): TournamentStatistics {
  const scores = results.map((r) => r.teamScore);
  scores.sort((a, b) => a - b);

  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const median = scores[Math.floor(scores.length / 2)];
  const q25 = scores[Math.floor(scores.length * 0.25)];
  const q75 = scores[Math.floor(scores.length * 0.75)];

  const variance =
    scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
  const stdDev = Math.sqrt(variance);

  const successRate = results.filter((r) => r.housesSaved >= 5).length / results.length;
  const housesSavedAvg =
    results.reduce((sum, r) => sum + r.housesSaved, 0) / results.length;

  // Calculate work efficiency
  const avgNights = results.reduce((sum, r) => sum + r.nightsPlayed, 0) / results.length;
  const workEfficiency = housesSavedAvg / (avgNights * 0.1); // Normalized

  return {
    mean,
    median,
    stdDev,
    min: scores[0],
    max: scores[scores.length - 1],
    q25,
    q75,
    successRate,
    housesSavedAvg,
    workEfficiency,
  };
}

/**
 * Estimate agent contributions
 */
function estimateAgentContributions(
  results: ScenarioResult[],
  team: TeamComposition,
): AgentContribution[] {
  const activeAgents = team.positions.filter((a) => a != null);

  const contributions = activeAgents.map((agent, position) => {
    if (!agent) return null;

    // Average the agent's individual score across all scenarios
    const avgScore =
      results.reduce((sum, r) => sum + r.agentScores[position], 0) / results.length;

    // Calculate consistency (inverse of coefficient of variation)
    const scores = results.map((r) => r.agentScores[position]);
    const mean = avgScore;
    const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);
    const consistency = mean > 0 ? 1 / (1 + stdDev / Math.abs(mean)) : 0;

    // Count MVP appearances
    const mvpCount = results.filter((r) => {
      const maxScore = Math.max(...r.agentScores);
      return r.agentScores[position] === maxScore && maxScore > 0;
    }).length;

    return {
      position,
      archetype: agent.name,
      avgContribution: avgScore,
      consistency,
      mvpCount,
      rank: 0, // Will be set after sorting
    };
  }).filter((c) => c != null) as AgentContribution[];

  // Rank by contribution
  contributions.sort((a, b) => b.avgContribution - a.avgContribution);
  contributions.forEach((c, i) => (c.rank = i + 1));

  return contributions;
}

/**
 * Analyze performance by scenario type
 */
function analyzeScenarioTypes(
  results: ScenarioResult[],
): Record<string, ScenarioTypeStats> {
  const typeStats: Record<string, ScenarioTypeStats> = {};

  for (const result of results) {
    if (!typeStats[result.scenarioType]) {
      typeStats[result.scenarioType] = {
        count: 0,
        avgScore: 0,
        successRate: 0,
        avgHousesSaved: 0,
      };
    }

    const stats = typeStats[result.scenarioType];
    stats.count++;
    stats.avgScore += result.teamScore;
    stats.avgHousesSaved += result.housesSaved;
    if (result.housesSaved >= 5) {
      stats.successRate += 1;
    }
  }

  // Calculate averages
  for (const type in typeStats) {
    const stats = typeStats[type];
    stats.avgScore /= stats.count;
    stats.avgHousesSaved /= stats.count;
    stats.successRate /= stats.count;
  }

  return typeStats;
}

/**
 * Tournament Engine Class
 */
export class TournamentEngine {
  /**
   * Run tournament with progress callbacks
   */
  async runTournament(
    team: TeamComposition,
    config: TournamentConfig,
    forceJsEngine = false,
    onProgress?: (progress: TournamentProgress) => void,
  ): Promise<TournamentResult> {
    const startTime = Date.now();

    // Try to initialize WASM for better performance
    if (!isWasmInitialized()) {
      try {
        console.log('🚀 Initializing WASM engine for tournament...');
        await initWasm();
        console.log('✅ Using WASM engine (10-20x faster)');
      } catch (error) {
        console.warn('⚠️  WASM initialization failed, using JS engine:', error);
      }
    }

    const activeAgents = team.positions.filter((a) => a != null);
    const numAgents = activeAgents.length;

    // Generate scenarios
    const scenarioSet = generateScenarioSet(
      config.numScenarios,
      numAgents,
      config.seed,
    );

    const results: ScenarioResult[] = [];

    // Run each scenario
    for (let i = 0; i < scenarioSet.length; i++) {
      const { scenario, type } = scenarioSet[i];

      // Run game
      const result = await runSingleGame(team, scenario, type, forceJsEngine);
      results.push(result);

      // Report progress
      if (onProgress) {
        const currentStats = this.calculateProgressStats(results);
        const estimatedCompletion =
          startTime +
          ((Date.now() - startTime) / (i + 1)) * scenarioSet.length;

        onProgress({
          current: i + 1,
          total: scenarioSet.length,
          results,
          statistics: currentStats,
          startTime,
          estimatedCompletion,
        });
      }

      // Small delay to keep UI responsive
      if (i % 10 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    // Calculate final statistics
    const statistics = calculateStatistics(results);
    const agentContributions = estimateAgentContributions(results, team);
    const scenarioTypePerformance = analyzeScenarioTypes(results);

    // Save interesting games to replay viewer
    saveTournamentGamesToReplay(results);

    return {
      id: generateId(),
      teamId: team.id,
      teamName: team.name,
      timestamp: Date.now(),
      duration: Date.now() - startTime,
      config,
      scenarios: results,
      statistics,
      agentContributions,
      scenarioTypePerformance,
    };
  }

  /**
   * Calculate statistics for progress updates
   */
  private calculateProgressStats(results: ScenarioResult[]) {
    if (results.length === 0) {
      return {
        mean: 0,
        median: 0,
        stdDev: 0,
        min: 0,
        max: 0,
      };
    }

    const scores = results.map((r) => r.teamScore);
    scores.sort((a, b) => a - b);

    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const median = scores[Math.floor(scores.length / 2)];

    const variance =
      scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);

    return {
      mean,
      median,
      stdDev,
      min: scores[0],
      max: scores[scores.length - 1],
    };
  }
}

/**
 * Create tournament engine instance
 */
export function createTournamentEngine(): TournamentEngine {
  return new TournamentEngine();
}
