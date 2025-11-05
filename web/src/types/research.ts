// TypeScript types for scenario research data

export interface ResearchInsight {
  question: string;
  finding: string;
  evidence: string[];
  implication: string;
}

export interface MethodInsights {
  nash?: ResearchInsight[];
  evolution?: ResearchInsight[];
  ppo?: ResearchInsight[];
  comparative?: ResearchInsight[];
}

export interface ScenarioConfig {
  scenario: string;
  description: string;
  story: string;
  parameters: {
    beta: number;
    kappa: number;
    c: number;
    num_agents: number;
  };
  research_questions?: string[];
  research_insights?: ResearchInsight[];  // Legacy format
  method_insights?: MethodInsights;  // New structured format
}

export interface AgentParameters {
  honesty: number;
  work_tendency: number;
  neighbor_help: number;
  own_priority: number;
  risk_aversion: number;
  coordination: number;
  exploration: number;
  fatigue_memory: number;
  rest_bias: number;
  altruism: number;
}

export interface HeuristicTeamResult {
  composition: string;
  mean_payoff: number;
  std_payoff: number;
  payoffs: number[];
}

export interface HeuristicResults {
  scenario: string;
  parameters: ScenarioConfig['parameters'];
  homogeneous_teams: HeuristicTeamResult[];
  mixed_teams: HeuristicTeamResult[];
  best_homogeneous: HeuristicTeamResult;
  best_mixed: HeuristicTeamResult;
  ranking: {
    homogeneous: Array<{ name: string; mean_payoff: number; std_payoff: number }>;
    mixed: Array<{ composition: string; mean_payoff: number; std_payoff: number }>;
  };
}

export interface BestAgent {
  scenario: string;
  fitness: number;
  generation: number;
  genome: number[];
  parameters: AgentParameters;
}

export interface GenerationSnapshot {
  generation: number;
  best_fitness: number;
  mean_fitness: number;
  std_fitness: number;
  diversity: number;
}

export interface EvolutionTrace {
  scenario: string;
  config: {
    population_size: number;
    num_generations: number;
    elite_size: number;
    selection_strategy: string;
    crossover_strategy: string;
    crossover_rate: number;
    mutation_strategy: string;
    mutation_rate: number;
    mutation_scale: number;
    fitness_type: string;
    games_per_individual: number;
    seed: number | null;
  };
  generations: GenerationSnapshot[];
  best_agent: {
    generation: number;
    fitness: number;
    genome: number[];
  };
  convergence: {
    converged: boolean;
    generation: number | null;
    elapsed_time: number;
  };
}

export interface ComparisonResults {
  scenario: string;
  strategies: {
    [key: string]: number[]; // strategy name -> genome
  };
  tournament: {
    [key: string]: {
      mean_payoff: number;
      std_payoff: number;
      payoffs: number[];
    };
  };
  distances: {
    [key: string]: number; // strategy1_vs_strategy2 -> distance
  };
  ranking: Array<{
    name: string;
    mean_payoff: number;
    std_payoff: number;
  }>;
  insights: Record<string, unknown>;
}

export interface NashStrategy {
  index: number;
  probability: number;
  classification: string;
  closest_archetype: string;
  archetype_distance: number;
  parameters: AgentParameters;
  genome: number[];
}

export interface NashResults {
  scenario: string;
  parameters: ScenarioConfig['parameters'];
  algorithm: {
    method: string;
    num_simulations: number;
    max_iterations: number;
    epsilon: number;
    seed: number;
  };
  equilibrium: {
    type: 'pure' | 'mixed';
    support_size: number;
    expected_payoff: number;
    distribution: Record<string, number>;
    strategy_pool: NashStrategy[];
  };
  convergence: {
    converged: boolean;
    iterations: number;
    elapsed_time: number;
  };
  interpretation: {
    equilibrium_type: string;
    cooperation_rate: number;
    free_riding_rate: number;
  };
}

export interface ScenarioResearchData {
  config: ScenarioConfig;
  heuristics?: HeuristicResults;
  evolved?: {
    trace: EvolutionTrace;
    best_agent: BestAgent;
  };
  nash?: NashResults;
  comparison?: ComparisonResults;
}

// Available scenario names
export const SCENARIOS = [
  'greedy_neighbor',
  'trivial_cooperation',
  'sparse_heroics',
  'early_containment',
  'rest_trap',
  'chain_reaction',
  'deceptive_calm',
  'overcrowding',
  'mixed_motivation',
] as const;

export type ScenarioName = typeof SCENARIOS[number];
