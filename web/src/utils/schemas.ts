import { z } from 'zod';

// House state enum (0 = SAFE, 1 = BURNING, 2 = RUINED)
export const HouseStateSchema = z.union([z.literal(0), z.literal(1), z.literal(2)]);

// Scenario schema
export const ScenarioSchema = z.object({
  beta: z.number().min(0).max(1), // Fire spread probability
  kappa: z.number().min(0).max(2), // Extinguish efficiency
  A: z.number().positive(), // Reward per saved house
  L: z.number().positive(), // Penalty per ruined house
  c: z.number().nonnegative(), // Cost per worker per night
  rho_ignite: z.number().min(0).max(1), // Initial burning fraction
  N_min: z.number().int().positive(), // Minimum nights before termination
  p_spark: z.number().min(0).max(1), // Probability of spontaneous ignition
  N_spark: z.number().int().nonnegative(), // Number of nights with sparks active
  num_agents: z.number().int().min(2).max(10), // Number of agents
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
}).refine((data) => {
  // Validate that all nights have the correct number of agents
  const numAgents = data.scenario.num_agents;
  return data.nights.every(night =>
    night.signals.length === numAgents &&
    night.locations.length === numAgents &&
    night.actions.length === numAgents &&
    night.rewards.length === numAgents
  );
}, {
  message: "All nights must have data for the correct number of agents"
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

export function validateExportData(data: unknown): ExportData {
  return ExportDataSchema.parse(data);
}

export function validateAgentRanking(data: unknown): AgentRanking {
  return AgentRankingSchema.parse(data);
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
