/**
 * WASM Game Engine Wrapper
 *
 * Thin wrapper around the Rust WASM core engine, providing a clean TypeScript interface.
 * The WASM engine handles only the game simulation (fire spread, rewards, state transitions).
 * Agent logic remains in JavaScript for flexibility and ease of development.
 */

import type { Scenario, GameResult, AgentObservation } from './browserEngine';

// WASM module types (will be loaded dynamically)
type WasmModule = {
  default: () => Promise<void>;
  WasmBucketBrigade: new (scenarioJson: string) => {
    reset(): void;
    step(actionsJson: string): string;
    // Issue #252 / #331: two-phase non-binding signaling step. Required for
    // scenarios with `commitment_mode == "two_phase"`; calling `step()` on
    // such scenarios panics in the Rust engine.
    step_two_phase(round1SignalsJson: string, round2ActionsJson: string): string;
    get_observation(agentId: number): string;
    get_current_state(): string;
    get_result(): string;
    is_done(): boolean;
  };
  WasmScenario: new (scenarioJson: string) => any;
  get_scenario: (name: string) => string;
  get_scenario_names: () => string[];
};

// WASM initialization state
let wasmInitialized = false;
let wasmInitPromise: Promise<void> | null = null;
let wasmModule: WasmModule | null = null;

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
      // Try to dynamically import the WASM module
      wasmModule = await import('../../../bucket-brigade-core/pkg/bucket_brigade_core') as unknown as WasmModule;
      await wasmModule.default();
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
  if (!wasmInitialized || !wasmModule) {
    throw new Error('WASM not initialized. Call initWasm() first.');
  }
  return wasmModule.get_scenario_names();
}

/**
 * Get predefined scenario from WASM
 */
export function getWasmScenario(name: string): Scenario {
  if (!wasmInitialized || !wasmModule) {
    throw new Error('WASM not initialized. Call initWasm() first.');
  }
  const scenarioJson = wasmModule.get_scenario(name);
  return JSON.parse(scenarioJson) as Scenario;
}

/**
 * WASM-backed game engine
 *
 * Drop-in replacement for BrowserBucketBrigade, using Rust WASM for 10-20x performance.
 * The API is identical to the TypeScript engine, so they can be swapped seamlessly.
 */
export class WasmGameEngine {
  private engine: InstanceType<WasmModule['WasmBucketBrigade']>;
  private scenario: Scenario;

  constructor(scenario: Scenario) {
    if (!wasmInitialized || !wasmModule) {
      throw new Error('WASM not initialized. Call initWasm() first.');
    }

    this.scenario = scenario;
    const scenarioJson = JSON.stringify(scenario);
    this.engine = new wasmModule.WasmBucketBrigade(scenarioJson);
  }

  /**
   * Reset the game to initial state
   */
  reset(): void {
    this.engine.reset();
  }

  /**
   * Execute one game night.
   *
   * Issue #235: each action is now a 3-element array
   * ``[house_index, mode, signal]`` (was 2-element ``[house, mode]``).
   * The TypeScript signature ``number[][]`` is permissive and will NOT
   * catch a shape mismatch — only the Rust-side ``serde_json`` will. The
   * WASM step path also accepts legacy 2-element arrays for transitional
   * compatibility and promotes them to honest 3-element actions
   * (``signal := mode``); new code should emit 3-element arrays so it can
   * broadcast a signal independent of its actual mode.
   */
  step(actions: number[][]): { rewards: number[]; done: boolean; info: any } {
    const actionsJson = JSON.stringify(actions);
    const resultJson = this.engine.step(actionsJson);
    return JSON.parse(resultJson);
  }

  /**
   * Issue #252 / #331: two-phase non-binding signaling step.
   *
   * Required when the scenario was constructed with
   * ``commitment_mode == "two_phase"``. Mirrors the PyO3 surface in
   * ``bucket-brigade-core/src/python.rs::PyBucketBrigade::step_two_phase``
   * and the WASM binding in ``bucket-brigade-core/src/wasm.rs``.
   *
   * One call advances the night by one step, internally fusing the
   * signal-phase write and the action-phase step:
   *   1. Round 1: ``round1Signals`` (length num_agents, each 0/1) become
   *      visible in subsequent observations via ``round1_signals``.
   *   2. Round 2: ``round2Actions`` is applied through the regular
   *      ``step()`` pipeline (phases 0-9).
   *
   * The deception channel survives: round-2 mode (`action[1]`) is not
   * constrained by the round-1 signal. Policies can emit
   * ``round1Signal=1 (Work)`` and then ``round2Mode=0 (Rest)`` — this is
   * the "lie" mechanic.
   *
   * @param round1Signals - per-agent round-1 signals (length num_agents,
   *   each in {0, 1}).
   * @param round2Actions - per-agent round-2 actions
   *   ``[house, mode, signal]`` (length 3) or legacy ``[house, mode]``
   *   which is promoted to honest length-3 (signal := mode).
   */
  step_two_phase(
    round1Signals: number[],
    round2Actions: number[][],
  ): { rewards: number[]; done: boolean; info: any } {
    const round1Json = JSON.stringify(round1Signals);
    const actionsJson = JSON.stringify(round2Actions);
    const resultJson = this.engine.step_two_phase(round1Json, actionsJson);
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
 * Create game engine (WASM if available, fallback to JS).
 *
 * Issue #252 / #331: two-phase scenarios (``commitment_mode == "two_phase"``)
 * require the WASM engine — the pure-JS fallback does not implement the
 * round-1 signal channel. If WASM init fails and the scenario is two-phase,
 * this function throws instead of falling back (the JS engine would otherwise
 * throw at construction time with the same message). Single-phase scenarios
 * fall back to JS as before.
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
      if (scenario.commitment_mode === 'two_phase') {
        // Two-phase scenarios cannot fall back — surface the WASM init
        // failure directly so the caller knows the WASM engine is required.
        throw new Error(
          `WASM engine required for commitment_mode="two_phase" scenarios but ` +
            `failed to initialize: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
      console.warn('WASM engine unavailable, falling back to JS:', error);
    }
  }

  // Fallback to JS engine (single-phase only; the JS engine constructor
  // throws on two-phase scenarios so this is also safe.)
  const { BrowserBucketBrigade } = await import('./browserEngine');
  return new BrowserBucketBrigade(scenario);
}
