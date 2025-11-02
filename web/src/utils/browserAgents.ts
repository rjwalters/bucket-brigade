/**
 * Browser-compatible agents for Bucket Brigade tournaments.
 *
 * These agents implement the browser agent interface and can run
 * directly in the browser environment.
 */

import { Agent, AgentObservation } from './browserEngine';

// Base agent class
export abstract class BrowserAgent implements Agent {
  id: number;
  name: string;

  constructor(id: number, name: string) {
    this.id = id;
    this.name = name;
  }

  abstract act(obs: AgentObservation): number[];

  reset?(): void {
    // Optional reset implementation
  }
}

// Random agent - baseline
export class RandomAgent extends BrowserAgent {
  constructor(id: number) {
    super(id, "Random");
  }

  act(_obs: AgentObservation): number[] {
    const house = Math.floor(Math.random() * 10);
    const mode = Math.floor(Math.random() * 2);
    return [house, mode];
  }
}

// Optimal agents for each scenario

export class TrivialCooperator extends BrowserAgent {
  constructor(id: number) {
    super(id, "TrivialCooperator");
  }

  act(obs: AgentObservation): number[] {
    // Find first burning house
    const burning_house = obs.houses.findIndex(house => house === 1);
    if (burning_house !== -1) {
      return [burning_house, 1]; // Work on burning house
    }
    return [this.id % 10, 1]; // Work on own house
  }
}

export class EarlyContainmentAgent extends BrowserAgent {
  constructor(id: number) {
    super(id, "EarlyContainment");
  }

  act(obs: AgentObservation): number[] {
    const burning_houses = obs.houses
      .map((house, idx) => ({ house, idx }))
      .filter(({ house }) => house === 1)
      .map(({ idx }) => idx);

    if (burning_houses.length > 0) {
      // Prioritize clusters
      const target_house = this.find_best_cluster_target(obs.houses, burning_houses);
      return [target_house, 1];
    }

    return [this.id % 10, 0]; // Rest
  }

  private find_best_cluster_target(houses: number[], burning_houses: number[]): number {
    if (burning_houses.length === 1) return burning_houses[0];

    // Score by cluster size
    let best_house = burning_houses[0];
    let best_score = 0;

    for (const house_idx of burning_houses) {
      const neighbors = [
        (house_idx - 1 + 10) % 10,
        (house_idx + 1) % 10
      ];
      const cluster_size = 1 + neighbors.filter(n => houses[n] === 1).length;

      if (cluster_size > best_score) {
        best_score = cluster_size;
        best_house = house_idx;
      }
    }

    return best_house;
  }
}

export class GreedyNeighborAgent extends BrowserAgent {
  constructor(id: number) {
    super(id, "GreedyNeighbor");
  }

  act(obs: AgentObservation): number[] {
    const own_house = this.id % 10;

    // If own house is burning, work on it
    if (obs.houses[own_house] === 1) {
      return [own_house, 1];
    }

    // Check if neighbors are burning (threat to own house)
    const neighbors = [
      (own_house - 1 + 10) % 10,
      (own_house + 1) % 10
    ];

    const threatening_neighbor = neighbors.find(n => obs.houses[n] === 1);
    if (threatening_neighbor !== undefined) {
      return [threatening_neighbor, 1];
    }

    // No immediate threats - rest
    return [own_house, 0];
  }
}

export class SparseHeroAgent extends BrowserAgent {
  constructor(id: number) {
    super(id, "SparseHero");
  }

  act(obs: AgentObservation): number[] {
    const burning_count = obs.houses.filter(h => h === 1).length;
    const own_house = this.id % 10;

    if (burning_count === 0) {
      return [own_house, 0]; // Rest if no fires
    }

    // Prioritize own house if burning
    if (obs.houses[own_house] === 1) {
      return [own_house, 1];
    }

    // Work on first burning house
    const burning_house = obs.houses.findIndex(h => h === 1);
    return [burning_house, 1];
  }
}

export class HonestSignaler extends BrowserAgent {
  constructor(id: number) {
    super(id, "HonestSignaler");
  }

  private last_houses: number[] = [];

  reset(): void {
    this.last_houses = [];
  }

  act(obs: AgentObservation): number[] {
    // Detect new fires
    const new_fires: number[] = [];
    if (this.last_houses.length > 0) {
      obs.houses.forEach((house, idx) => {
        if (house === 1 && this.last_houses[idx] !== 1) {
          new_fires.push(idx);
        }
      });
    }

    this.last_houses = [...obs.houses];

    // Work on new fires first, then any burning house
    const burning_houses = obs.houses
      .map((house, idx) => ({ house, idx }))
      .filter(({ house }) => house === 1)
      .map(({ idx }) => idx);

    if (burning_houses.length > 0) {
      const target_house = new_fires.length > 0 ? new_fires[0] : burning_houses[0];
      return [target_house, 1]; // Honest work signal
    }

    return [this.id % 10, 0]; // Rest honestly
  }
}

// Agent registry
export const BUILT_IN_AGENTS = {
  Random: RandomAgent,
  TrivialCooperator,
  EarlyContainmentAgent,
  GreedyNeighborAgent,
  SparseHeroAgent,
  HonestSignaler,
} as const;

export type AgentType = keyof typeof BUILT_IN_AGENTS;

// Create agent instances
export function create_agent(type: AgentType, id: number): Agent {
  const AgentClass = BUILT_IN_AGENTS[type];
  return new AgentClass(id);
}

// User agent template for custom agents
export class UserAgent extends BrowserAgent {
  private user_function: (obs: AgentObservation) => number[];

  constructor(id: number, name: string, act_function: (obs: AgentObservation) => number[]) {
    super(id, name);
    this.user_function = act_function;
  }

  act(obs: AgentObservation): number[] {
    try {
      const result = this.user_function(obs);

      // Validate result
      if (!Array.isArray(result) || result.length !== 2) {
        throw new Error("Agent must return [house_index, mode]");
      }

      const [house, mode] = result;
      if (typeof house !== 'number' || house < 0 || house > 9) {
        throw new Error("House index must be 0-9");
      }

      if (typeof mode !== 'number' || (mode !== 0 && mode !== 1)) {
        throw new Error("Mode must be 0 (REST) or 1 (WORK)");
      }

      return result;
    } catch (error) {
      console.error(`Agent ${this.name} error:`, error);
      // Fallback to random action
      return [Math.floor(Math.random() * 10), Math.floor(Math.random() * 2)];
    }
  }
}

// Create user agent from code string
export function create_user_agent_from_code(id: number, name: string, code: string): UserAgent | null {
  try {
    // Create a safe context for code execution
    const context = {
      Math,
      console: {
        log: (...args: any[]) => console.log(`[Agent ${name}]`, ...args),
        error: (...args: any[]) => console.error(`[Agent ${name}]`, ...args),
      }
    };

    // Create the act function
    const act_function = new Function(
      'obs',
      'Math',
      'console',
      `
        "use strict";
        try {
          ${code}
        } catch (error) {
          console.error("Agent execution error:", error);
          return [Math.floor(Math.random() * 10), Math.floor(Math.random() * 2)];
        }
      `
    ).bind(null, context.Math, context.console);

    return new UserAgent(id, name, act_function);
  } catch (error) {
    console.error(`Failed to create agent from code:`, error);
    return null;
  }
}

// Tournament agent combinations
export const TOURNAMENT_AGENTS = [
  { type: 'Random' as AgentType, count: 2 },
  { type: 'TrivialCooperator' as AgentType, count: 1 },
  { type: 'EarlyContainmentAgent' as AgentType, count: 1 },
  { type: 'GreedyNeighborAgent' as AgentType, count: 1 },
  { type: 'SparseHeroAgent' as AgentType, count: 1 },
  { type: 'HonestSignaler' as AgentType, count: 1 },
] as const;

export function create_tournament_agents(include_user_agent?: { name: string; act_function: (obs: AgentObservation) => number[] }): Agent[] {
  const agents: Agent[] = [];
  let agent_id = 0;

  // Add built-in agents
  for (const { type, count } of TOURNAMENT_AGENTS) {
    for (let i = 0; i < count; i++) {
      agents.push(create_agent(type, agent_id++));
    }
  }

  // Add user agent if provided
  if (include_user_agent) {
    const user_agent = new UserAgent(agent_id, include_user_agent.name, include_user_agent.act_function);
    agents.push(user_agent);
  }

  return agents;
}
