/**
 * Team Builder Tournament - Type Definitions
 *
 * Types for the team builder feature where users compose teams
 * of agents and test them against randomized scenarios.
 */

import type { RadarProfile } from '../utils/agentRadarChart';

/**
 * Agent behavioral parameters (0-1 scale)
 */
export interface AgentParameters {
  /** Probability of truthful signaling */
  honesty_bias: number;
  /** Base probability to choose mode=WORK */
  work_tendency: number;
  /** Preference for helping burning neighbor houses */
  neighbor_help_bias: number;
  /** Preference for defending own house */
  own_house_priority: number;
  /** Sensitivity to number of burning houses */
  risk_aversion: number;
  /** Trust in other agents' signals */
  coordination_weight: number;
  /** Randomness in action selection */
  exploration_rate: number;
  /** Inertia to repeat last action */
  fatigue_memory: number;
  /** Preference for rest if fires are low */
  rest_reward_bias: number;
  /** Willingness to work even if personal cost high */
  altruism_factor: number;
}

/**
 * Pre-defined agent archetype with personality and behavior
 */
export interface AgentArchetype {
  /** Unique identifier */
  id: string;
  /** Display name */
  name: string;
  /** Emoji or image URL */
  icon: string;
  /** Theme color (hex) */
  color: string;
  /** Short catchphrase */
  tagline: string;
  /** Longer description */
  description: string;
  /** Behavioral parameters */
  parameters: AgentParameters;
  /** Strategy explanation bullets */
  strategyNotes: string[];
  /** Whether parameters randomize each game */
  isRandomized?: boolean;
  /** Radar chart profile for visual comparison */
  radarProfile?: RadarProfile;
  /** Additional metadata for enhanced display */
  metadata?: {
    /** Top strengths for this agent */
    strengths?: string[];
    /** Known weaknesses */
    weaknesses?: string[];
    /** Filter tags (e.g., "cooperative", "honest", "deceptive") */
    tags?: string[];
  };
}

/**
 * Team composition with up to 10 agent positions
 */
export interface TeamComposition {
  /** Unique team ID */
  id: string;
  /** User-defined team name */
  name: string;
  /** Array of agent assignments (null = empty slot) */
  positions: (AgentArchetype | null)[];
  /** Creation timestamp */
  createdAt: number;
  /** Last modification timestamp */
  modifiedAt: number;
  /** References to tournament results */
  tournamentHistory: string[];
}

/**
 * Configuration for tournament execution
 */
export interface TournamentConfig {
  /** Team to test */
  teamId: string;
  /** Number of scenarios to run */
  numScenarios: number;
  /** Optional scenario type filter */
  scenarioTypes?: string[];
  /** Random seed for reproducibility */
  seed?: number;
}

/**
 * Real-time tournament progress tracking
 */
export interface TournamentProgress {
  /** Current scenario number */
  current: number;
  /** Total scenarios */
  total: number;
  /** Results collected so far */
  results: ScenarioResult[];
  /** Running statistics */
  statistics: {
    mean: number;
    median: number;
    stdDev: number;
    min: number;
    max: number;
  };
  /** Start timestamp */
  startTime: number;
  /** Estimated completion time */
  estimatedCompletion: number;
}

/**
 * Complete tournament results
 */
export interface TournamentResult {
  /** Unique result ID */
  id: string;
  /** Team that was tested */
  teamId: string;
  /** Team name at time of tournament */
  teamName: string;
  /** Completion timestamp */
  timestamp: number;
  /** Execution duration (ms) */
  duration: number;
  /** Tournament configuration */
  config: TournamentConfig;
  /** All scenario results */
  scenarios: ScenarioResult[];
  /** Aggregate statistics */
  statistics: TournamentStatistics;
  /** Individual agent performance */
  agentContributions: AgentContribution[];
  /** Performance by scenario type */
  scenarioTypePerformance: Record<string, ScenarioTypeStats>;
}

/**
 * Aggregate tournament statistics
 */
export interface TournamentStatistics {
  /** Average team score */
  mean: number;
  /** Median team score */
  median: number;
  /** Standard deviation */
  stdDev: number;
  /** Minimum score */
  min: number;
  /** Maximum score */
  max: number;
  /** 25th percentile */
  q25: number;
  /** 75th percentile */
  q75: number;
  /** Percentage of scenarios "won" */
  successRate: number;
  /** Average houses saved (0-10) */
  housesSavedAvg: number;
  /** Work efficiency metric */
  workEfficiency: number;
}

/**
 * Individual agent contribution analysis
 */
export interface AgentContribution {
  /** Position in team (0-9) */
  position: number;
  /** Archetype name */
  archetype: string;
  /** Estimated marginal contribution */
  avgContribution: number;
  /** Performance consistency (0-1) */
  consistency: number;
  /** Times agent was top performer */
  mvpCount: number;
  /** Overall rank in team */
  rank: number;
}

/**
 * Single scenario execution result
 */
export interface ScenarioResult {
  /** Scenario identifier */
  scenarioId: string;
  /** Scenario category */
  scenarioType: string;
  /** Team total score */
  teamScore: number;
  /** Per-agent scores */
  agentScores: number[];
  /** Number of houses saved (0-10) */
  housesSaved: number;
  /** Game duration */
  nightsPlayed: number;
  /** Full replay data */
  replayData: any;
}

/**
 * Performance statistics for a scenario type
 */
export interface ScenarioTypeStats {
  /** Number of scenarios of this type */
  count: number;
  /** Average score */
  avgScore: number;
  /** Success rate */
  successRate: number;
  /** Average houses saved */
  avgHousesSaved: number;
}

/**
 * Team comparison analysis
 */
export interface TeamComparison {
  /** First team's results */
  teamA: TournamentResult;
  /** Second team's results */
  teamB: TournamentResult;
  /** Winner determination */
  winner: 'A' | 'B' | 'tie';
  /** Score difference */
  scoreDifference: number;
  /** Statistical significance (p-value) */
  significance: number;
  /** Scenario wins breakdown */
  scenarioWins: {
    A: number;
    B: number;
    ties: number;
  };
  /** Performance by scenario type */
  scenarioTypeComparison: Record<
    string,
    {
      A: number;
      B: number;
    }
  >;
}

/**
 * Pre-defined team template
 */
export interface TeamTemplate {
  /** Template identifier */
  id: string;
  /** Template name */
  name: string;
  /** Description */
  description: string;
  /** Agent archetype IDs for each position */
  archetypeIds: string[];
  /** Recommended for specific scenarios */
  recommendedFor?: string[];
}

/**
 * Storage schema for session/local storage
 */
export interface TeamBuilderStorage {
  /** All saved teams */
  teams: Record<string, TeamComposition>;
  /** All tournament results */
  tournaments: Record<string, TournamentResult>;
  /** Currently active team ID */
  activeTeamId: string | null;
  /** Pre-defined templates */
  templates: Record<string, TeamTemplate>;
}
