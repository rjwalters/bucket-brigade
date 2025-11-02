/**
 * Browser-compatible Bucket Brigade game engine.
 *
 * This implements the core game logic in TypeScript for browser execution,
 * allowing agents to run directly in the browser environment.
 */

// Type definitions
export type HouseState = 0 | 1 | 2; // SAFE, BURNING, RUINED

export interface Scenario {
  beta: number;      // Fire spread probability
  kappa: number;     // Extinguish efficiency
  A: number;         // Reward per saved house
  L: number;         // Penalty per ruined house
  c: number;         // Work cost per night
  rho_ignite: number; // Initial burn fraction
  N_min: number;     // Minimum nights
  p_spark: number;   // Spark probability
  N_spark: number;   // Spark duration
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

export interface GameResult {
  scenario: Scenario;
  nights: GameNight[];
  final_score: number;
  agent_scores: number[];
  winner?: string;
}

export interface Agent {
  id: number;
  name: string;
  act: (obs: AgentObservation) => number[];
  reset?: () => void;
}

export interface AgentObservation {
  signals: number[];
  locations: number[];
  houses: HouseState[];
  last_actions: number[][];
  scenario_info: number[];
  agent_id: number;
  night: number;
}

// Browser-compatible random number generator
class BrowserRNG {
  private seed: number;

  constructor(seed?: number) {
    this.seed = seed ?? Math.random() * 1000000;
  }

  random(): number {
    // Simple LCG random number generator
    this.seed = (this.seed * 9301 + 49297) % 233280;
    return this.seed / 233280;
  }

  randint(min: number, max: number): number {
    return Math.floor(this.random() * (max - min)) + min;
  }

  choice<T>(array: T[]): T {
    return array[this.randint(0, array.length)];
  }
}

// Core game engine
export class BrowserBucketBrigade {
  private houses!: HouseState[];
  private agent_positions!: number[];
  private agent_signals!: number[];
  private last_actions!: number[][];
  private night!: number;
  private done!: boolean;
  private rewards!: number[];
  private rng: BrowserRNG;
  private scenario: Scenario;
  private trajectory!: GameNight[];

  constructor(scenario: Scenario, seed?: number) {
    this.scenario = scenario;
    this.rng = new BrowserRNG(seed);
    this.reset();
  }

  reset(): void {
    this.houses = new Array(10).fill(0);
    this.agent_positions = new Array(this.scenario.num_agents).fill(0);
    this.agent_signals = new Array(this.scenario.num_agents).fill(0);
    this.last_actions = new Array(this.scenario.num_agents).fill(null).map(() => [0, 0]);
    this.night = 0;
    this.done = false;
    this.rewards = new Array(this.scenario.num_agents).fill(0);
    this.trajectory = [];

    // Initialize fires
    const num_burning = Math.round(this.scenario.rho_ignite * 10);
    const burn_indices = new Set<number>();
    while (burn_indices.size < num_burning) {
      burn_indices.add(this.rng.randint(0, 10));
    }
    burn_indices.forEach(idx => this.houses[idx] = 1);

    this.record_night();
  }

  step(actions: number[][]): { rewards: number[]; done: boolean; info: any } {
    if (this.done) {
      throw new Error("Game is already finished");
    }

    // Store previous house states for reward calculation
    const prev_houses = [...this.houses];

    // 1. Signal phase (signals are implicit in actions for now)
    this.agent_signals = actions.map(action => action[1]);

    // 2. Action phase: update agent positions
    this.last_actions = actions.map(action => [...action]);
    this.agent_positions = actions.map(action => action[0]);

    // 3. Extinguish phase
    this.extinguish_fires(actions);

    // 4. Spread phase
    this.spread_fires();

    // 5. Burn-out phase
    this.burn_out_houses();

    // 6. Spark phase (if active)
    if (this.night < this.scenario.N_spark) {
      this.spark_fires();
    }

    // 7. Compute rewards
    this.rewards = this.compute_rewards(actions, prev_houses);

    // 8. Check termination
    this.done = this.check_termination();

    // 9. Record this night
    this.record_night();

    // 10. Advance to next night
    this.night++;

    return {
      rewards: [...this.rewards],
      done: this.done,
      info: {}
    };
  }

  private extinguish_fires(actions: number[][]): void {
    for (let house_idx = 0; house_idx < 10; house_idx++) {
      if (this.houses[house_idx] !== 1) continue;

      // Count workers at this house
      const workers_here = actions.filter(action => action[0] === house_idx && action[1] === 1).length;

      // Probability of extinguishing
      const p_extinguish = 1 - Math.exp(-this.scenario.kappa * workers_here);

      if (this.rng.random() < p_extinguish) {
        this.houses[house_idx] = 0;
      }
    }
  }

  private spread_fires(): void {
    const new_houses = [...this.houses];

    for (let house_idx = 0; house_idx < 10; house_idx++) {
      if (this.houses[house_idx] !== 1) continue;

      // Check neighbors
      const neighbors = [
        (house_idx - 1 + 10) % 10,
        (house_idx + 1) % 10
      ];

      for (const neighbor of neighbors) {
        if (this.houses[neighbor] === 0 && this.rng.random() < this.scenario.beta) {
          new_houses[neighbor] = 1;
        }
      }
    }

    this.houses = new_houses;
  }

  private burn_out_houses(): void {
    for (let i = 0; i < 10; i++) {
      if (this.houses[i] === 1) {
        this.houses[i] = 2;
      }
    }
  }

  private spark_fires(): void {
    for (let house_idx = 0; house_idx < 10; house_idx++) {
      if (this.houses[house_idx] === 0 && this.rng.random() < this.scenario.p_spark) {
        this.houses[house_idx] = 1;
      }
    }
  }

  private compute_rewards(actions: number[][], prev_houses: HouseState[]): number[] {
    // Count outcomes
    const saved_houses = this.houses.filter(h => h === 0).length;
    const ruined_houses = this.houses.filter(h => h === 2).length;

    // Team reward
    const team_reward = this.scenario.A * (saved_houses / 10) - this.scenario.L * (ruined_houses / 10);

    const rewards = new Array(this.scenario.num_agents);

    for (let agent_idx = 0; agent_idx < this.scenario.num_agents; agent_idx++) {
      let reward = 0;

      // Work/rest cost
      if (actions[agent_idx][1] === 1) {
        reward -= this.scenario.c; // Work cost
      } else {
        reward += 0.5; // Rest reward
      }

      // Ownership bonus/penalty
      const owned_house = agent_idx % 10;
      if (prev_houses[owned_house] === 0 && this.houses[owned_house] === 0) {
        reward += 1.0; // Bonus for keeping house safe
      }
      if (this.houses[owned_house] === 2) {
        reward -= 2.0; // Penalty for ruined house
      }

      // Team reward share
      reward += 0.1 * team_reward;

      rewards[agent_idx] = reward;
    }

    return rewards;
  }

  private check_termination(): boolean {
    if (this.night < this.scenario.N_min) return false;

    const all_safe = this.houses.every(h => h === 0);
    const all_ruined = this.houses.every(h => h === 2);

    return all_safe || all_ruined;
  }

  private record_night(): void {
    this.trajectory.push({
      night: this.night,
      houses: [...this.houses],
      signals: [...this.agent_signals],
      locations: [...this.agent_positions],
      actions: this.last_actions.map(action => [...action]),
      rewards: [...this.rewards]
    });
  }

  get_observation(agent_id: number): AgentObservation {
    return {
      signals: [...this.agent_signals],
      locations: [...this.agent_positions],
      houses: [...this.houses],
      last_actions: this.last_actions.map(action => [...action]),
      scenario_info: [
        this.scenario.beta,
        this.scenario.kappa,
        this.scenario.A,
        this.scenario.L,
        this.scenario.c,
        this.scenario.rho_ignite,
        this.scenario.N_min,
        this.scenario.p_spark,
        this.scenario.N_spark,
        this.scenario.num_agents
      ],
      agent_id,
      night: this.night
    };
  }

  get_result(): GameResult {
    const agent_scores = this.trajectory.reduce((acc, night) => {
      night.rewards.forEach((reward, agentId) => {
        acc[agentId] = (acc[agentId] || 0) + reward;
      });
      return acc;
    }, {} as Record<number, number>);

    const final_score = Object.values(agent_scores).reduce((sum, score) => sum + score, 0);

    return {
      scenario: { ...this.scenario },
      nights: [...this.trajectory],
      final_score,
      agent_scores: Object.values(agent_scores)
    };
  }

  get_current_state() {
    return {
      houses: [...this.houses],
      night: this.night,
      done: this.done,
      agent_positions: [...this.agent_positions],
      agent_signals: [...this.agent_signals]
    };
  }
}

// Tournament runner
export class TournamentRunner {
  private games: BrowserBucketBrigade[] = [];
  private agents: Agent[] = [];
  private results: GameResult[] = [];
  private running = false;

  constructor(agents: Agent[], scenario: Scenario, num_games: number = 10) {
    this.agents = agents;

    // Create multiple game instances
    for (let i = 0; i < num_games; i++) {
      this.games.push(new BrowserBucketBrigade(scenario, i));
    }
  }

  async run_tournament(on_progress?: (completed: number, total: number) => void): Promise<GameResult[]> {
    this.running = true;
    this.results = [];

    const total_games = this.games.length;

    for (let game_idx = 0; game_idx < total_games; game_idx++) {
      if (!this.running) break;

      const game = this.games[game_idx];
      const game_result = await this.run_single_game(game);

      this.results.push(game_result);
      on_progress?.(game_idx + 1, total_games);

      // Allow UI updates
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    this.running = false;
    return this.results;
  }

  private async run_single_game(game: BrowserBucketBrigade): Promise<GameResult> {
    game.reset();

    while (!game.get_current_state().done) {
      // Get actions from all agents
      const actions = this.agents.map(agent => {
        const obs = game.get_observation(agent.id);
        return agent.act(obs);
      });

      // Step the game
      game.step(actions);

      // Allow UI updates
      await new Promise(resolve => setTimeout(resolve, 1));
    }

    return game.get_result();
  }

  stop(): void {
    this.running = false;
  }

  get_results(): GameResult[] {
    return [...this.results];
  }
}

// Predefined scenarios
export const SCENARIOS = {
  trivial_cooperation: {
    beta: 0.15,
    kappa: 0.9,
    A: 100,
    L: 100,
    c: 0.5,
    rho_ignite: 0.1,
    N_min: 12,
    p_spark: 0.0,
    N_spark: 12,
    num_agents: 4
  },

  early_containment: {
    beta: 0.35,
    kappa: 0.6,
    A: 100,
    L: 100,
    c: 0.5,
    rho_ignite: 0.3,
    N_min: 12,
    p_spark: 0.02,
    N_spark: 12,
    num_agents: 4
  },

  greedy_neighbor: {
    beta: 0.15,
    kappa: 0.4,
    A: 100,
    L: 100,
    c: 1.0,
    rho_ignite: 0.2,
    N_min: 12,
    p_spark: 0.02,
    N_spark: 12,
    num_agents: 4
  },

  random: {
    beta: 0.25,
    kappa: 0.5,
    A: 100,
    L: 100,
    c: 0.5,
    rho_ignite: 0.2,
    N_min: 12,
    p_spark: 0.02,
    N_spark: 12,
    num_agents: 4
  }
} as const;
