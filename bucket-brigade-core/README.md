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

This script uses the **setuptools-rust** build backend declared in
`pyproject.toml`. It installs `setuptools-rust`, removes any stale
CFFI-shadow artifacts, and runs `pip install -e . --no-build-isolation`.
It sets the env vars described below defensively, so it works on a fresh
machine with no extra configuration.

### Required environment variables

The build relies on two environment settings. `build.sh` exports both for
you; if you are running a build command manually, set them yourself:

- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` — required so PyO3 0.22.x builds
  against newer Python versions (e.g. 3.13).
- `RUSTC_WRAPPER=` (i.e. unset) — many other docs in this repo export
  `RUSTC_WRAPPER=sccache` for caching. **If `sccache` is not installed on
  your machine, the build will fail or produce a broken artifact.** Either
  install sccache (`cargo install sccache`) or unset `RUSTC_WRAPPER` before
  building.

### Manual Build

If you prefer manual installation:

```bash
# Recommended: uses the setuptools-rust backend declared in pyproject.toml.
# This matches what build.sh does.
uv pip install setuptools-rust
RUSTC_WRAPPER= PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
    uv run python -m pip install -e . --no-build-isolation
```

Note: the `--features python` flag is configured in `pyproject.toml`
(`[tool.setuptools-rust] features = ["python"]`) so you do not need to
pass it explicitly when using the setuptools-rust path.

### Troubleshooting

**`ImportError: cannot import name 'PyBucketBrigade' from 'bucket_brigade_core.bucket_brigade_core'`**

You have hit the CFFI-shadow trap. An older `maturin develop` build (or
one that ran without `--features python`) produced a nested
`bucket_brigade_core/bucket_brigade_core/` directory containing CFFI shim
files that shadow the real PyO3 module. Clean and rebuild:

```bash
cd bucket-brigade-core
rm -rf bucket_brigade_core/bucket_brigade_core
./build.sh
```

After a successful build, the artifact layout should be:

- `bucket_brigade_core/bucket_brigade_core.cpython-<TAG>-<PLATFORM>.so` (the real PyO3 module)
- `bucket_brigade_core/__init__.py`
- **No** `bucket_brigade_core/bucket_brigade_core/` directory
- **No** `ffi.py`, **no** `libbucket_brigade_core.dylib`/`.so` next to `__init__.py`

**Build hangs or fails complaining about `sccache`**

Your shell has `RUSTC_WRAPPER=sccache` exported but sccache is not
installed. Either `cargo install sccache` or `unset RUSTC_WRAPPER` before
running `build.sh`. `build.sh` unsets it for its own invocation but cannot
clean up your interactive shell environment.

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

# Build Python extension (the importable Python module).
# See "Installation" above for the canonical recipe.
./build.sh

# Build the underlying staticlib/cdylib via cargo only (no Python install).
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --features python

# Run tests
cargo test
```

## License

MIT
