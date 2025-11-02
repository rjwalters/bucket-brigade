// Game state types
export type HouseState = 0 | 1 | 2; // SAFE = 0, BURNING = 1, RUINED = 2

export interface Scenario {
  beta: number;
  kappa: number;
  A: number;
  L: number;
  c: number;
  rho_ignite: number;
  N_min: number;
  p_spark: number;
  N_spark: number;
  num_agents: number;
}

export interface GameNight {
  night: number;
  houses: HouseState[];
  signals: number[];
  locations: number[];
  actions: number[][];
  rewards: number[];
}

export interface GameReplay {
  scenario: Scenario;
  nights: GameNight[];
}

// Agent types
export interface Agent {
  id: number;
  name?: string;
  params?: number[];
}

// Batch result types
export interface BatchResult {
  game_id: number;
  scenario_id: number;
  team: number[];
  agent_params: number[][];
  team_reward: number;
  agent_rewards: number[];
  nights_played: number;
  saved_houses: number;
  ruined_houses: number;
  replay_path: string;
}

// Ranking types
export interface AgentRanking {
  agent_id: number;
  name?: string;
  score: number;
  uncertainty?: number;
  games_played: number;
  avg_reward: number;
  win_rate?: number;
}

// UI state types
export interface GameFilters {
  minReward?: number;
  maxReward?: number;
  minNights?: number;
  maxNights?: number;
  agentCount?: number;
}

export interface ReplayState {
  currentNight: number;
  isPlaying: boolean;
  speed: number; // milliseconds per frame
}

// Session storage keys
export const STORAGE_KEYS = {
  GAME_REPLAYS: 'bucket_brigade_replays',
  BATCH_RESULTS: 'bucket_brigade_results',
  AGENT_RANKINGS: 'bucket_brigade_rankings',
  UI_SETTINGS: 'bucket_brigade_settings'
} as const;
