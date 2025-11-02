# WASM Integration Guide

This document describes the WASM integration for the Bucket Brigade web application.

## Overview

The Bucket Brigade web application uses a hybrid architecture:
- **Game Engine Core**: Rust WASM for high-performance simulation (10-20x faster)
- **Agent Logic**: TypeScript for flexibility and ease of development
- **Tournament Orchestration**: TypeScript for async handling and UI updates

## Architecture

```
┌─────────────────────────────────────────┐
│  JavaScript (Browser)                   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Tournament Runner (JS)         │   │
│  │  - Orchestrates games           │   │
│  │  - Manages agents               │   │
│  │  - Collects results             │   │
│  └────────────┬────────────────────┘   │
│               │                         │
│               ▼                         │
│  ┌─────────────────────────────────┐   │
│  │  Agent Logic (JS)               │   │
│  │  - Heuristic decisions          │   │
│  │  - Parameter-based behavior     │   │
│  └────────────┬────────────────────┘   │
│               │                         │
│               ▼                         │
│  ┌─────────────────────────────────┐   │
│  │  WASM Game Engine               │   │
│  │  (bucket-brigade-core)          │   │
│  │  - Fire spread simulation       │   │
│  │  - Reward calculation           │   │
│  │  - State transitions            │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Building WASM

### Prerequisites

Install `wasm-pack`:
```bash
cargo install wasm-pack
```

### Build Command

```bash
cd bucket-brigade-core
wasm-pack build --target web --features wasm
```

This generates the `pkg/` directory with:
- `bucket_brigade_core.js` - JS glue code
- `bucket_brigade_core_bg.wasm` - Binary WASM module
- `bucket_brigade_core.d.ts` - TypeScript types

### Integration into Web App

The web app automatically imports the WASM module from `../bucket-brigade-core/pkg/`:

```typescript
import init, { WasmBucketBrigade } from '../../../bucket-brigade-core/pkg/bucket_brigade_core';
```

Vite handles the WASM module bundling automatically.

## Usage

### Initializing WASM

```typescript
import { initWasm, isWasmInitialized } from './utils/wasmEngine';

// Initialize once at app startup
await initWasm();
```

### Creating Game Engine

The `createGameEngine` function automatically uses WASM if available, with fallback to JS:

```typescript
import { createGameEngine } from './utils/wasmEngine';

// Automatically uses WASM (fallback to JS if WASM fails)
const engine = await createGameEngine(scenario);
```

### Direct WASM Usage

```typescript
import { WasmGameEngine, initWasm } from './utils/wasmEngine';

// Initialize WASM
await initWasm();

// Create WASM engine
const engine = new WasmGameEngine(scenario);

// Use same API as JS engine
const actions = [[0, 1], [1, 1], [2, 0], [3, 1]]; // [house, mode]
const result = engine.step(actions);
const obs = engine.get_observation(0);
const isDone = engine.is_done();
const finalResult = engine.get_result();
```

## Fallback Strategy

The app gracefully falls back to the JavaScript engine if WASM fails to load:

```typescript
try {
  await initWasm();
  console.log('✅ Using WASM engine (10-20x faster)');
} catch (error) {
  console.warn('⚠️  WASM failed, using JS engine:', error);
}
```

Both engines implement the same interface, so they're drop-in replacements.

## Performance

- **WASM Engine**: ~10-20x faster than JavaScript
- **Best for**: Large tournaments (50+ games), evolutionary algorithms
- **JS Engine**: Slower but zero build complexity
- **Best for**: Development, debugging, single game replays

## Development

### Rebuilding WASM

After modifying Rust code:

```bash
cd bucket-brigade-core
wasm-pack build --target web --features wasm
```

Then rebuild the web app:

```bash
cd ../web
pnpm run build
```

### Hot Reloading

The Vite dev server automatically reloads when WASM changes, but you must manually rebuild the WASM package first.

## Troubleshooting

### WASM Module Not Found

Ensure the WASM package is built:
```bash
cd bucket-brigade-core && wasm-pack build --target web --features wasm
```

### WASM Initialization Fails

Check browser console for errors. Common issues:
- Browser doesn't support WASM (very old browsers)
- CORS issues (if loading from different origin)
- Build artifacts missing

The app will automatically fall back to the JS engine.

### Type Errors

Regenerate TypeScript types:
```bash
cd bucket-brigade-core
wasm-pack build --target web --features wasm
```

## Agent Development

**Important**: Agent logic remains in TypeScript, not Rust/WASM.

### Why TypeScript for Agents?

1. **Flexibility**: Easy to modify and debug
2. **No compilation**: Change parameters without rebuilding
3. **User-friendly**: Non-Rust developers can create agents
4. **DevTools**: Full browser debugging support

### Example Agent

```typescript
class MyAgent extends BrowserAgent {
  act(obs: AgentObservation): number[] {
    const { houses, signals } = obs;

    // Decision logic in TypeScript
    const burningHouse = houses.findIndex(h => h === 1);
    if (burningHouse !== -1) {
      return [burningHouse, 1]; // Work on burning house
    }

    return [this.id % 10, 0]; // Rest
  }
}
```

The agent calls into the WASM engine for each step, but the decision-making stays in JS.

## What WASM Does (and Doesn't Do)

### WASM Responsibilities

- ✅ Fire spread simulation
- ✅ Extinguish probability calculation
- ✅ Reward computation
- ✅ State transitions
- ✅ Victory condition checking

### Not WASM (Stays in TypeScript)

- ❌ Agent decision-making
- ❌ Tournament orchestration
- ❌ UI rendering
- ❌ Data persistence
- ❌ Progress callbacks

## API Reference

See [`wasmEngine.ts`](./src/utils/wasmEngine.ts) for the full API documentation.

Key exports:
- `initWasm()` - Initialize WASM module
- `isWasmInitialized()` - Check initialization status
- `WasmGameEngine` - WASM-backed game engine
- `createGameEngine()` - Auto-select WASM or JS engine
- `getWasmScenario()` - Load predefined scenario from WASM
- `getWasmScenarioNames()` - List available scenarios
