/**
 * WASM Game Engine Wrapper
 *
 * Thin wrapper around the Rust WASM core engine, providing a clean TypeScript interface.
 * The WASM engine handles only the game simulation (fire spread, rewards, state transitions).
 * Agent logic remains in JavaScript for flexibility and ease of development.
 */

import init, {
  WasmBucketBrigade,
  get_scenario,
  get_scenario_names,
} from '../../../bucket-brigade-core/pkg/bucket_brigade_core';
import type { Scenario, GameResult, AgentObservation } from './browserEngine';

// WASM initialization state
let wasmInitialized = false;
let wasmInitPromise: Promise<void> | null = null;

/**
 * Initialize the WASM module (call once at startup)
 */
export async function initWasm(): Promise<void> {
  if (wasmInitialized) return;

  if (wasmInitPromise) {
    return wasmInitPromise;
  }

  wasmInitPromise = (async () => {
    try {
      await init();
      wasmInitialized = true;
      console.log('✅ WASM engine initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize WASM:', error);
      throw error;
    }
  })();

  return wasmInitPromise;
}

/**
 * Check if WASM is initialized
 */
export function isWasmInitialized(): boolean {
  return wasmInitialized;
}

/**
 * Get available scenario names from WASM
 */
export function getWasmScenarioNames(): string[] {
  if (!wasmInitialized) {
    throw new Error('WASM not initialized. Call initWasm() first.');
  }
  return get_scenario_names();
}

/**
 * Get predefined scenario from WASM
 */
export function getWasmScenario(name: string): Scenario {
  if (!wasmInitialized) {
    throw new Error('WASM not initialized. Call initWasm() first.');
  }
  const scenarioJson = get_scenario(name);
  return JSON.parse(scenarioJson) as Scenario;
}

/**
 * WASM-backed game engine
 *
 * Drop-in replacement for BrowserBucketBrigade, using Rust WASM for 10-20x performance.
 * The API is identical to the TypeScript engine, so they can be swapped seamlessly.
 */
export class WasmGameEngine {
  private engine: WasmBucketBrigade;
  private scenario: Scenario;

  constructor(scenario: Scenario) {
    if (!wasmInitialized) {
      throw new Error('WASM not initialized. Call initWasm() first.');
    }

    this.scenario = scenario;
    const scenarioJson = JSON.stringify(scenario);
    this.engine = new WasmBucketBrigade(scenarioJson);
  }

  /**
   * Reset the game to initial state
   */
  reset(): void {
    this.engine.reset();
  }

  /**
   * Execute one game night
   */
  step(actions: number[][]): { rewards: number[]; done: boolean; info: any } {
    const actionsJson = JSON.stringify(actions);
    const resultJson = this.engine.step(actionsJson);
    return JSON.parse(resultJson);
  }

  /**
   * Get observation for a specific agent
   */
  get_observation(agentId: number): AgentObservation {
    const obsJson = this.engine.get_observation(agentId);
    return JSON.parse(obsJson) as AgentObservation;
  }

  /**
   * Get current game state
   */
  get_current_state(): {
    houses: number[];
    night: number;
    done: boolean;
    agent_positions: number[];
    agent_signals: number[];
  } {
    const stateJson = this.engine.get_current_state();
    return JSON.parse(stateJson);
  }

  /**
   * Get final game result
   */
  get_result(): GameResult {
    const resultJson = this.engine.get_result();
    return JSON.parse(resultJson) as GameResult;
  }

  /**
   * Check if game is done
   */
  is_done(): boolean {
    return this.engine.is_done();
  }

  /**
   * Get scenario configuration
   */
  get_scenario(): Scenario {
    return this.scenario;
  }
}

/**
 * Create game engine (WASM if available, fallback to JS)
 */
export async function createGameEngine(
  scenario: Scenario,
  preferWasm = true,
): Promise<WasmGameEngine | import('./browserEngine').BrowserBucketBrigade> {
  if (preferWasm) {
    try {
      await initWasm();
      return new WasmGameEngine(scenario);
    } catch (error) {
      console.warn('WASM engine unavailable, falling back to JS:', error);
    }
  }

  // Fallback to JS engine
  const { BrowserBucketBrigade } = await import('./browserEngine');
  return new BrowserBucketBrigade(scenario);
}
