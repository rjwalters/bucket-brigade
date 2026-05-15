import { z } from 'zod';

// House state enum (0 = SAFE, 1 = BURNING, 2 = RUINED)
export const HouseStateSchema = z.union([z.literal(0), z.literal(1), z.literal(2)]);

// Scenario schema
// Must match bucket_brigade/envs/scenarios_generated.py Scenario dataclass
// (source of truth: definitions/scenarios.json, mirrored in
// bucket-brigade-core/src/scenarios.rs Scenario struct).
export const ScenarioSchema = z.object({
  // Fire dynamics
  prob_fire_spreads_to_neighbor: z.number().min(0).max(1), // Probability fire spreads to adjacent house
  prob_solo_agent_extinguishes_fire: z.number().min(0).max(2), // Probability one agent extinguishes fire
  prob_house_catches_fire: z.number().min(0).max(1), // Probability house catches fire each night

  // Team scoring (collective outcome)
  team_reward_house_survives: z.number().nonnegative(), // Team reward for each house that survives
  team_penalty_house_burns: z.number().nonnegative(), // Team penalty for each house that burns

  // Individual rewards (ownership-based)
  reward_own_house_survives: z.number(), // Individual reward when own house survives
  reward_other_house_survives: z.number(), // Individual reward when other house survives
  penalty_own_house_burns: z.number(), // Individual penalty when own house burns
  penalty_other_house_burns: z.number(), // Individual penalty when other house burns

  // Costs and structure
  cost_to_work_one_night: z.number().nonnegative(), // Cost per worker per night
  min_nights: z.number().int().positive(), // Minimum nights before termination

  // Game setup
  num_agents: z.number().int().min(1).max(10), // Number of agents
});

// Game night schema
export const GameNightSchema = z.object({
  night: z.number().int().nonnegative(),
  houses: z.array(HouseStateSchema).length(10), // Always 10 houses in ring
  signals: z.array(z.number().int().min(0).max(1)), // Agent signals (0=REST, 1=WORK)
  locations: z.array(z.number().int().min(0).max(9)), // Agent locations (0-9)
  actions: z.array(z.array(z.number().int()).length(2)), // [house_index, mode_flag]
  rewards: z.array(z.number()), // Agent rewards
}).refine((data) => {
  // All arrays should have the same length (num_agents)
  const lengths = [data.signals.length, data.locations.length, data.actions.length, data.rewards.length];
  return lengths.every(len => len === lengths[0]);
}, {
  message: "All agent arrays must have the same length"
});

// Game replay schema
export const GameReplaySchema = z.object({
  scenario: ScenarioSchema,
  nights: z.array(GameNightSchema).min(1),
  archetypes: z.array(z.string()).optional(), // Agent archetype names
  statistics: z.object({
    avgAgentScores: z.array(z.number()),
    stdErrAgentScores: z.array(z.number()),
    avgFinalScore: z.number(),
    stdErrFinalScore: z.number(),
    numGames: z.number().int().positive(),
  }).optional(),
  timestamp: z.string().optional(), // ISO timestamp when game was run
  teamName: z.string().optional(), // Descriptive team name
  scenarioName: z.string().optional(), // Scenario identifier
}).refine((data) => {
  // Validate that all nights have the correct number of agents
  const numAgents = data.scenario.num_agents;
  const nightsValid = data.nights.every(night =>
    night.signals.length === numAgents &&
    night.locations.length === numAgents &&
    night.actions.length === numAgents &&
    night.rewards.length === numAgents
  );

  // If archetypes provided, should match num_agents
  const archetypesValid = !data.archetypes || data.archetypes.length === numAgents;

  // If statistics provided, arrays should match num_agents
  const statsValid = !data.statistics ||
    (data.statistics.avgAgentScores.length === numAgents &&
     data.statistics.stdErrAgentScores.length === numAgents);

  return nightsValid && archetypesValid && statsValid;
}, {
  message: "All nights must have data for the correct number of agents, and archetypes/statistics must match if provided"
});

// Batch result schema
export const BatchResultSchema = z.object({
  game_id: z.number().int().nonnegative(),
  scenario_id: z.number().int().nonnegative(),
  team: z.array(z.number().int().nonnegative()), // Agent IDs in team
  agent_params: z.array(z.array(z.number())), // Parameters for each agent
  team_reward: z.number(),
  agent_rewards: z.array(z.number()),
  nights_played: z.number().int().nonnegative(),
  saved_houses: z.number().int().min(0).max(10),
  ruined_houses: z.number().int().min(0).max(10),
  replay_path: z.string(),
}).refine((data) => {
  // Validate consistency
  return data.agent_rewards.length === data.team.length &&
         data.agent_params.length === data.team.length &&
         data.saved_houses + data.ruined_houses <= 10; // Can't have more than 10 total
}, {
  message: "Team composition and reward arrays must be consistent"
});

// Agent ranking schema
export const AgentRankingSchema = z.object({
  agent_id: z.number().int().nonnegative(),
  name: z.string().optional(),
  score: z.number(),
  uncertainty: z.number().optional(),
  games_played: z.number().int().nonnegative(),
  avg_reward: z.number(),
  win_rate: z.number().min(0).max(1).optional(),
});

// Export data schema (for backup/restore)
export const ExportDataSchema = z.object({
  games: z.array(GameReplaySchema),
  results: z.array(BatchResultSchema),
  exported_at: z.string().datetime(),
  version: z.string(),
});

// Type inference helpers
export type HouseState = z.infer<typeof HouseStateSchema>;
export type Scenario = z.infer<typeof ScenarioSchema>;
export type GameNight = z.infer<typeof GameNightSchema>;
export type GameReplay = z.infer<typeof GameReplaySchema>;
export type BatchResult = z.infer<typeof BatchResultSchema>;
export type AgentRanking = z.infer<typeof AgentRankingSchema>;
export type ExportData = z.infer<typeof ExportDataSchema>;

// Validation functions
export function validateGameReplay(data: unknown): GameReplay {
  return GameReplaySchema.parse(data);
}

export function validateBatchResult(data: unknown): BatchResult {
  return BatchResultSchema.parse(data);
}

// Safe validation functions (return null on error)
export function safeValidateGameReplay(data: unknown): GameReplay | null {
  try {
    return GameReplaySchema.parse(data);
  } catch {
    return null;
  }
}

export function safeValidateBatchResult(data: unknown): BatchResult | null {
  try {
    return BatchResultSchema.parse(data);
  } catch {
    return null;
  }
}

export function safeValidateExportData(data: unknown): ExportData | null {
  try {
    return ExportDataSchema.parse(data);
  } catch {
    return null;
  }
}
