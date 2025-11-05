# Bucket Brigade Core

**Canonical source of truth** for the Bucket Brigade multi-agent cooperation game engine.

This Rust implementation serves as the authoritative game logic, with Python and WASM bindings for different use cases.

## Features

- **Canonical game implementation**: Definitive rules and scenarios for Bucket Brigade
- **10-20x faster** than pure Python implementations
- **Memory safe** and thread-safe
- **Python bindings** via PyO3 for research and RL training
- **WASM support** for browser deployment
- **Deterministic RNG** for reproducible experiments
- **12 predefined scenarios**: 3 difficulty levels + 9 research-focused cooperation scenarios

## Installation

**Python Version Requirement**: This module requires Python 3.9-3.13. Python 3.14+ is not yet supported by PyO3 0.22.6.

### Quick Build (Recommended)

```bash
./build.sh
```

This script:
- Checks your Python version
- Builds the Rust library with `cargo`
- Copies the `.so` file to the correct location
- Verifies the installation

### Manual Build

If you prefer manual installation:

```bash
# Using pip (may not trigger Rust build correctly)
pip install -e .

# Using cargo directly (more reliable)
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo build --release --features python
# Then manually copy target/release/libbucket_brigade_core.so to bucket_brigade_core/
```

## Usage

```python
from bucket_brigade_core import BucketBrigade, SCENARIOS

# Create environment
scenario = SCENARIOS["trivial_cooperation"]
env = BucketBrigade(scenario)

# Reset and run a game
env.reset()

# Get observation for agent 0
obs = env.get_observation(0)
actions = [[obs.houses.index(1), 1]]  # Work on first burning house

# Step the environment
rewards, done, info = env.step(actions)

# Get final results
result = env.get_result()
print(f"Final score: {result.final_score}")
```

## Performance

| Implementation | Single Game (50 nights) | Tournament (100 games) |
|----------------|------------------------|------------------------|
| Python (NumPy) | ~100ms               | ~10s                  |
| **Rust Core**  | **~5ms**              | **~0.5s**             |
| **Speedup**    | **20x faster**        | **20x faster**        |

## Architecture

This crate is the **source of truth** for Bucket Brigade game mechanics:

- **Core Engine**: Pure Rust implementation defining canonical game rules (`src/engine/`)
- **Scenarios**: All 12 official scenarios defined in Rust (`src/scenarios.rs`)
- **Python Bindings**: PyO3 exposes Rust engine to Python for RL training (`src/python.rs`)
- **WASM Support**: Compile to WebAssembly for browser deployment (`src/wasm.rs`)
- **Deterministic RNG**: Reproducible results across all platforms (`src/rng.rs`)

**Design Principle**: Rust defines the rules, Python and WASM consume them. Any game logic changes must be made here first.

## Development

```bash
# Build Rust library
cargo build

# Build Python extension
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --features python

# Run tests
cargo test
```

## License

MIT
