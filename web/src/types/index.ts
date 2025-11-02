// Re-export types from Zod schemas for consistency
export type {
  HouseState,
  Scenario,
  GameNight,
  GameReplay,
  BatchResult,
  AgentRanking,
  ExportData
} from '../utils/schemas';

// Legacy agent interface (keeping for compatibility)
export interface Agent {
  id: number;
  name?: string;
  params?: number[];
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
