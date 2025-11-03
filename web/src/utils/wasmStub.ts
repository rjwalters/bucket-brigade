/**
 * WASM Module Stub
 *
 * This stub is used when the WASM module is not available (e.g., during CI builds).
 * It provides the same interface as the real WASM module but throws errors when used,
 * allowing the build to succeed while gracefully degrading to the JS engine at runtime.
 */

export default async function init(): Promise<void> {
  throw new Error('WASM module not available - this build does not include the Rust WASM engine');
}

export class WasmBucketBrigade {
  constructor(_scenarioJson: string) {
    throw new Error('WASM module not available');
  }

  reset(): void {
    throw new Error('WASM module not available');
  }

  step(_actionsJson: string): string {
    throw new Error('WASM module not available');
  }

  get_observation(_agentId: number): string {
    throw new Error('WASM module not available');
  }

  get_current_state(): string {
    throw new Error('WASM module not available');
  }

  get_result(): string {
    throw new Error('WASM module not available');
  }

  is_done(): boolean {
    throw new Error('WASM module not available');
  }
}

export function get_scenario(_name: string): string {
  throw new Error('WASM module not available');
}

export function get_scenario_names(): string[] {
  throw new Error('WASM module not available');
}
