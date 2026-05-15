/**
 * Browser-compatible Bucket Brigade game engine.
 *
 * This implements the core game logic in TypeScript for browser execution,
 * allowing agents to run directly in the browser environment.
 */

// Type definitions
export type HouseState = 0 | 1 | 2; // SAFE, BURNING, RUINED

export interface Scenario {
  // Fire dynamics
  prob_fire_spreads_to_neighbor: number; // Probability fire spreads to adjacent house
  prob_solo_agent_extinguishes_fire: number; // Probability one agent extinguishes fire
  prob_house_catches_fire: number; // Probability house catches fire each night

  // Team scoring (collective outcome)
  team_reward_house_survives: number; // Team reward for each house that survives
  team_penalty_house_burns: number; // Team penalty for each house that burns

  // Individual rewards (ownership-based, per-agent vectors; scalar accepted)
  //
  // Per issue #198 these four fields are per-agent vectors of length
  // ``num_agents``. For backward compatibility (and parity with the Python
  // ``Scenario.__post_init__`` promotion), this engine accepts either a
  // scalar or an array; ``promoteOwnershipField`` below normalizes to an
  // array of length ``num_agents`` at observation/scoring time.
  reward_own_house_survives: number | number[]; // Per-agent reward when own house survives
  reward_other_house_survives: number | number[]; // Per-agent reward when other house survives
  penalty_own_house_burns: number | number[]; // Per-agent penalty when own house burns
  penalty_other_house_burns: number | number[]; // Per-agent penalty when other house burns

  // Costs and structure
  cost_to_work_one_night: number; // Cost per worker per night
  min_nights: number; // Minimum nights before termination

  // Game setup
  num_agents: number; // Number of agents
}

/**
 * Promote a scalar-or-array ownership reward field to a length-N array.
 *
 * Mirrors ``Scenario.__post_init__`` (Python) and
 * ``deserialize_scalar_or_vec`` (Rust) so that scalar JSON values continue
 * to work transparently after issue #198.
 */
export function promoteOwnershipField(value: number | number[], num_agents: number): number[] {
  if (typeof value === 'number') {
    return new Array(num_agents).fill(value);
  }
  return value;
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

    // Initialize fires - probabilistic per-house
    for (let house_idx = 0; house_idx < 10; house_idx++) {
      if (this.rng.random() < this.scenario.prob_house_catches_fire) {
        this.houses[house_idx] = 1;
      }
    }

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
    // Agents respond to fires visible at start of turn
    this.extinguish_fires(actions);

    // 4. Burn-out phase
    // Unextinguished fires become ruined houses
    this.burn_out_houses();

    // 5. Spread phase
    // Fires spread to neighbors (visible next turn)
    this.spread_fires();

    // 6. Spontaneous ignition phase
    // New fires can ignite on any night (visible next turn)
    this.spark_fires();

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
      const p_extinguish = 1 - Math.exp(-this.scenario.prob_solo_agent_extinguishes_fire * workers_here);

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
        if (this.houses[neighbor] === 0 && this.rng.random() < this.scenario.prob_fire_spreads_to_neighbor) {
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
      if (this.houses[house_idx] === 0 && this.rng.random() < this.scenario.prob_house_catches_fire) {
        this.houses[house_idx] = 1;
      }
    }
  }

  private compute_rewards(actions: number[][], prev_houses: HouseState[]): number[] {
    // Count outcomes
    const saved_houses = this.houses.filter(h => h === 0).length;
    const ruined_houses = this.houses.filter(h => h === 2).length;

    // Team reward
    const team_reward = this.scenario.team_reward_house_survives * (saved_houses / 10) - this.scenario.team_penalty_house_burns * (ruined_houses / 10);

    const rewards = new Array(this.scenario.num_agents);

    for (let agent_idx = 0; agent_idx < this.scenario.num_agents; agent_idx++) {
      let reward = 0;

      // Work/rest cost
      if (actions[agent_idx][1] === 1) {
        reward -= this.scenario.cost_to_work_one_night; // Work cost
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
    if (this.night < this.scenario.min_nights) return false;

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
    // The four ownership reward fields may be scalars or per-agent arrays
    // (issue #198). For the legacy 12-element scenario_info layout, use the
    // mean — for the uniform/scalar case this preserves the original scalar
    // value, keeping backward compatibility for any consumer.
    const meanOf = (value: number | number[]): number => {
      if (typeof value === 'number') return value;
      if (value.length === 0) return 0;
      return value.reduce((s, x) => s + x, 0) / value.length;
    };

    return {
      signals: [...this.agent_signals],
      locations: [...this.agent_positions],
      houses: [...this.houses],
      last_actions: this.last_actions.map(action => [...action]),
      scenario_info: [
        this.scenario.prob_fire_spreads_to_neighbor,
        this.scenario.prob_solo_agent_extinguishes_fire,
        this.scenario.prob_house_catches_fire,
        this.scenario.team_reward_house_survives,
        this.scenario.team_penalty_house_burns,
        meanOf(this.scenario.reward_own_house_survives),
        meanOf(this.scenario.reward_other_house_survives),
        meanOf(this.scenario.penalty_own_house_burns),
        meanOf(this.scenario.penalty_other_house_burns),
        this.scenario.cost_to_work_one_night,
        this.scenario.min_nights,
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
    prob_fire_spreads_to_neighbor: 0.15,
    prob_solo_agent_extinguishes_fire: 0.9,
    prob_house_catches_fire: 0.0,
    team_reward_house_survives: 100,
    team_penalty_house_burns: 100,
    reward_own_house_survives: 100,
    reward_other_house_survives: 50,
    penalty_own_house_burns: 0,
    penalty_other_house_burns: 0,
    cost_to_work_one_night: 0.5,
    min_nights: 12,
    num_agents: 4
  },

  early_containment: {
    prob_fire_spreads_to_neighbor: 0.35,
    prob_solo_agent_extinguishes_fire: 0.6,
    prob_house_catches_fire: 0.02,
    team_reward_house_survives: 100,
    team_penalty_house_burns: 100,
    reward_own_house_survives: 100,
    reward_other_house_survives: 50,
    penalty_own_house_burns: 0,
    penalty_other_house_burns: 0,
    cost_to_work_one_night: 0.5,
    min_nights: 12,
    num_agents: 4
  },

  greedy_neighbor: {
    prob_fire_spreads_to_neighbor: 0.15,
    prob_solo_agent_extinguishes_fire: 0.4,
    prob_house_catches_fire: 0.02,
    team_reward_house_survives: 100,
    team_penalty_house_burns: 100,
    reward_own_house_survives: 100,
    reward_other_house_survives: 50,
    penalty_own_house_burns: 0,
    penalty_other_house_burns: 0,
    cost_to_work_one_night: 1.0,
    min_nights: 12,
    num_agents: 4
  },

  random: {
    prob_fire_spreads_to_neighbor: 0.25,
    prob_solo_agent_extinguishes_fire: 0.5,
    prob_house_catches_fire: 0.02,
    team_reward_house_survives: 100,
    team_penalty_house_burns: 100,
    reward_own_house_survives: 100,
    reward_other_house_survives: 50,
    penalty_own_house_burns: 0,
    penalty_other_house_burns: 0,
    cost_to_work_one_night: 0.5,
    min_nights: 12,
    num_agents: 4
  }
} as const;
