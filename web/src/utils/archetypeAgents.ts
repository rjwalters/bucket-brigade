/**
 * Agents based on archetype definitions
 * Maps archetype names to agent behaviors
 */

import { Agent, AgentObservation } from './browserEngine';
import { BrowserAgent } from './browserAgents';

/**
 * Firefighter - Reliable team player who prioritizes putting out fires
 */
export class FirefighterAgent extends BrowserAgent {
  constructor(id: number) {
    super(id, 'Firefighter');
  }

  act(obs: AgentObservation): number[] {
    // Find all burning houses
    const burning_houses = obs.houses
      .map((house, idx) => ({ house, idx }))
      .filter(({ house }) => house === 1)
      .map(({ idx }) => idx);

    if (burning_houses.length > 0) {
      // Prioritize fires closest to own house
      const own_house = this.id % 10;
      const sorted = burning_houses.sort((a, b) => {
        const dist_a = Math.min(Math.abs(a - own_house), 10 - Math.abs(a - own_house));
        const dist_b = Math.min(Math.abs(b - own_house), 10 - Math.abs(b - own_house));
        return dist_a - dist_b;
      });
      return [sorted[0], 1]; // Work on nearest fire
    }

    return [this.id % 10, 0]; // Rest if no fires
  }
}

/**
 * Free Rider - Prefers to rest and let teammates handle the fires
 */
export class FreeRiderAgent extends BrowserAgent {
  constructor(id: number) {
    super(id, 'FreeRider');
  }

  act(obs: AgentObservation): number[] {
    const own_house = this.id % 10;

    // Only work if own house is burning
    if (obs.houses[own_house] === 1) {
      return [own_house, 1];
    }

    // Check if immediate neighbors are burning (threat to own house)
    const neighbors = [
      (own_house - 1 + 10) % 10,
      (own_house + 1) % 10
    ];

    const threatening_neighbor = neighbors.find(n => obs.houses[n] === 1);
    if (threatening_neighbor !== undefined) {
      return [threatening_neighbor, 1];
    }

    // Otherwise, rest
    return [own_house, 0];
  }
}

/**
 * Hero - Takes risks and works hard
 */
export class HeroAgent extends BrowserAgent {
  constructor(id: number) {
    super(id, 'Hero');
  }

  act(obs: AgentObservation): number[] {
    // Always work if there are any fires
    const burning_house = obs.houses.findIndex(h => h === 1);
    if (burning_house !== -1) {
      return [burning_house, 1];
    }

    // Even if no fires, patrol own house
    return [this.id % 10, 1];
  }
}

/**
 * Coordinator - Excellent at organizing team response
 */
export class CoordinatorAgent extends BrowserAgent {
  constructor(id: number) {
    super(id, 'Coordinator');
  }

  act(obs: AgentObservation): number[] {
    // Find all burning houses
    const burning_houses = obs.houses
      .map((house, idx) => ({ house, idx }))
      .filter(({ house }) => house === 1)
      .map(({ idx }) => idx);

    if (burning_houses.length === 0) {
      return [this.id % 10, 0]; // Rest if no fires
    }

    // Avoid redundant work - find house with fewest workers
    const worker_counts = new Array(10).fill(0);
    obs.last_actions.forEach(action => {
      if (action[1] === 1) { // If working
        worker_counts[action[0]]++;
      }
    });

    // Find burning house with fewest workers
    let best_house = burning_houses[0];
    let min_workers = worker_counts[best_house];

    for (const house_idx of burning_houses) {
      if (worker_counts[house_idx] < min_workers) {
        min_workers = worker_counts[house_idx];
        best_house = house_idx;
      }
    }

    return [best_house, 1];
  }
}

/**
 * Liar - Signals dishonestly but may still help sometimes
 */
export class LiarAgent extends BrowserAgent {
  constructor(id: number) {
    super(id, 'Liar');
  }

  act(obs: AgentObservation): number[] {
    const own_house = this.id % 10;

    // Prioritize own house
    if (obs.houses[own_house] === 1) {
      return [own_house, 1];
    }

    // Sometimes work on other fires, sometimes rest
    const burning_houses = obs.houses
      .map((house, idx) => ({ house, idx }))
      .filter(({ house }) => house === 1)
      .map(({ idx }) => idx);

    if (burning_houses.length > 0 && Math.random() < 0.5) {
      return [burning_houses[0], 1];
    }

    return [own_house, 0]; // Rest most of the time
  }
}

/**
 * Create agent instance from archetype name
 */
export function createAgentFromArchetype(archetype: string, id: number): Agent {
  switch (archetype.toLowerCase()) {
    case 'firefighter':
      return new FirefighterAgent(id);
    case 'free_rider':
      return new FreeRiderAgent(id);
    case 'hero':
      return new HeroAgent(id);
    case 'coordinator':
      return new CoordinatorAgent(id);
    case 'liar':
      return new LiarAgent(id);
    default:
      // Default to firefighter for unknown archetypes
      console.warn(`Unknown archetype: ${archetype}, using Firefighter`);
      return new FirefighterAgent(id);
  }
}
