/**
 * Run a game simulation from team and scenario configuration
 */

import { BrowserBucketBrigade } from './browserEngine';
import { createAgentFromArchetype } from './archetypeAgents';
import type { GameReplay } from '../types';

interface TeamConfig {
  id: string;
  name: string;
  description: string;
  archetypes: string[];
}

interface ScenarioConfig {
  id: string;
  name: string;
  description: string;
  params: {
    beta: number;
    kappa: number;
    team_reward_house_survives: number;
    team_penalty_house_burns: number;
    reward_own_house_survives: number;
    reward_other_house_survives: number;
    penalty_own_house_burns: number;
    penalty_other_house_burns: number;
    c: number;
    N_min: number;
    p_spark: number;
    num_agents: number;
  };
}

/**
 * Convert scenario params to browser engine format
 */
function convertScenarioParams(params: ScenarioConfig['params']) {
  return {
    beta: params.beta,
    kappa: params.kappa,
    A: params.team_reward_house_survives,
    L: params.team_penalty_house_burns,
    c: params.c,
    N_min: params.N_min,
    p_spark: params.p_spark,
    N_spark: params.N_min, // Use N_min as spark duration
    num_agents: params.num_agents
  };
}

/**
 * Run a single game simulation
 */
export async function runGameSimulation(
  team: TeamConfig,
  scenario: ScenarioConfig,
  seed?: number
): Promise<GameReplay> {
  // Convert scenario parameters
  const engineScenario = convertScenarioParams(scenario.params);

  // Create game engine
  const engine = new BrowserBucketBrigade(engineScenario, seed);

  // Create agents from archetypes
  const agents = team.archetypes.map((archetype, id) =>
    createAgentFromArchetype(archetype, id)
  );

  // Run game simulation
  let step_count = 0;
  const max_steps = 100; // Safety limit

  while (!engine.get_current_state().done && step_count < max_steps) {
    // Get actions from all agents
    const actions = agents.map(agent => {
      const obs = engine.get_observation(agent.id);
      return agent.act(obs);
    });

    // Step the game
    engine.step(actions);
    step_count++;

    // Allow UI updates every 10 steps
    if (step_count % 10 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  // Get final result
  const result = engine.get_result();

  // Convert to GameReplay format (matching the schema)
  return {
    scenario: engineScenario,
    nights: result.nights,
    archetypes: team.archetypes
  };
}

/**
 * Run multiple game simulations and return statistics
 */
export async function runGameBatch(
  team: TeamConfig,
  scenario: ScenarioConfig,
  numGames: number = 100,
  onProgress?: (completed: number, total: number) => void
): Promise<{
  games: GameReplay[];
  statistics: {
    avgFinalScore: number;
    avgAgentScores: number[];
    avgNightsPlayed: number;
    avgHousesSaved: number;
    avgHousesRuined: number;
  };
}> {
  const games: GameReplay[] = [];
  const allAgentScores: number[][] = [];
  const allFinalScores: number[] = [];
  const allNights: number[] = [];
  const allHousesSaved: number[] = [];
  const allHousesRuined: number[] = [];

  for (let i = 0; i < numGames; i++) {
    const game = await runGameSimulation(team, scenario, i);
    games.push(game);

    // Calculate scores from rewards
    const agentScores = game.nights[0].rewards.map((_, agentIdx) => {
      return game.nights.reduce((sum, night) => sum + night.rewards[agentIdx], 0);
    });
    allAgentScores.push(agentScores);
    allFinalScores.push(agentScores.reduce((sum, score) => sum + score, 0));
    allNights.push(game.nights.length);

    // Count house outcomes
    const lastNight = game.nights[game.nights.length - 1];
    const saved = lastNight.houses.filter(h => h === 0).length;
    const ruined = lastNight.houses.filter(h => h === 2).length;
    allHousesSaved.push(saved);
    allHousesRuined.push(ruined);

    // Report progress
    if (onProgress) {
      onProgress(i + 1, numGames);
      // Allow UI update every 5 games
      if ((i + 1) % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
  }

  // Compute statistics
  const avgFinalScore = allFinalScores.reduce((a, b) => a + b, 0) / numGames;
  const numAgents = team.archetypes.length;
  const avgAgentScores = Array.from({ length: numAgents }, (_, agentIdx) => {
    const sum = allAgentScores.reduce((acc, scores) => acc + scores[agentIdx], 0);
    return sum / numGames;
  });
  const avgNightsPlayed = allNights.reduce((a, b) => a + b, 0) / numGames;
  const avgHousesSaved = allHousesSaved.reduce((a, b) => a + b, 0) / numGames;
  const avgHousesRuined = allHousesRuined.reduce((a, b) => a + b, 0) / numGames;

  // Compute standard errors
  const stdErrFinalScore = Math.sqrt(
    allFinalScores.reduce((sum, score) => sum + Math.pow(score - avgFinalScore, 2), 0) / numGames
  ) / Math.sqrt(numGames);

  const stdErrAgentScores = Array.from({ length: numAgents }, (_, agentIdx) => {
    const variance = allAgentScores.reduce((sum, scores) =>
      sum + Math.pow(scores[agentIdx] - avgAgentScores[agentIdx], 2), 0
    ) / numGames;
    return Math.sqrt(variance) / Math.sqrt(numGames);
  });

  return {
    games,
    statistics: {
      avgFinalScore,
      avgAgentScores,
      avgNightsPlayed,
      avgHousesSaved,
      avgHousesRuined,
      stdErrFinalScore,
      stdErrAgentScores,
      numGames
    }
  };
}
