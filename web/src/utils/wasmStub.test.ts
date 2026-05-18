import { describe, it, expect } from 'vitest'
import { WasmBucketBrigade } from './wasmStub'

/**
 * Issue #252 / #331: the WASM stub must expose the same surface as the
 * real WASM module so vite builds without the Rust artifact still typecheck.
 * For two-phase we add a stub method that throws — builds without WASM
 * cannot exercise two-phase scenarios (the JS fallback also rejects them).
 */

describe('wasmStub two-phase', () => {
  it('exposes step_two_phase that throws "WASM module not available"', () => {
    // Constructor itself throws; just check the method exists on prototype.
    expect(typeof WasmBucketBrigade.prototype.step_two_phase).toBe('function')
  })

  it('step_two_phase throws when called on a (somehow-constructed) instance', () => {
    // The constructor throws, so we can't instantiate normally. Probe via
    // prototype with a hand-rolled "this" to exercise the throw path.
    const stub = Object.create(WasmBucketBrigade.prototype) as WasmBucketBrigade
    expect(() => stub.step_two_phase('[0,1]', '[[0,1,1],[1,0,0]]')).toThrowError(
      /WASM module not available/,
    )
  })
})
