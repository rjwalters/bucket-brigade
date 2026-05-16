# Bucket Brigade

A research platform for studying cooperation, deception, and collective intelligence in multi-agent systems.

**[Interactive demo](https://rjwalters.info/bucket-brigade/)** — run tournaments in your browser.

> **Terminology.** "Agent" means two things in this repo:
> - **Game policies/strategies** — the AI decision-makers playing the bucket brigade game (the research focus).
> - **Loom roles** — automation workers that run development tasks (see [`.loom/LOOM_AGENTS.md`](.loom/LOOM_AGENTS.md)).
>
> Game-context use is the default. See [docs/GLOSSARY.md](docs/GLOSSARY.md) for full terminology.

## The game

Ten houses stand in a circle. Fires erupt, spread, and burn houses down. A group of 4–10 agents — each owning one house — can either **work** (try to extinguish a fire at some location) or **rest** (save energy). Each night every agent first broadcasts a signal (`Work` or `Rest`, *which may be a lie*) and then chooses an action.

Working costs energy; resting is free. Saving houses gives a team reward; losing them costs everyone. Individual ownership creates conflict between self-interest and the collective. The result is a small, fully-specifiable game in which trust, deception, free-riding, and coordination all emerge.

The full rules live in [docs/game_mechanics.md](docs/game_mechanics.md). Design philosophy and research framing are in [docs/game_description.md](docs/game_description.md).

## Research goals

- **Tournaments** — large mixed-team play to extract individual policy value (see [docs/RANKING_METHODOLOGY.md](docs/RANKING_METHODOLOGY.md)).
- **Evolution** — discover effective heuristic policies via GAs (see [experiments/evolution/README.md](experiments/evolution/README.md)).
- **Reinforcement learning** — PPO and population-based training ([docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md), [docs/POPULATION_TRAINING.md](docs/POPULATION_TRAINING.md)).
- **Nash equilibrium analysis** — Double Oracle with LP solver over symmetric strategies (see [docs/NASH_BENCHMARKS.md](docs/NASH_BENCHMARKS.md)).

## Quickstart

```bash
# Install uv + pnpm
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://get.pnpm.io/install.sh | sh -

# Python deps
uv sync

# Rust core (required for performant work — gives ~100x speedup)
cd bucket-brigade-core && ./build.sh && cd ..

# Frontend deps + dev server
pnpm install
pnpm run dev   # http://localhost:5173
```

Run a game, a batch, or Nash analysis:

```bash
uv run python scripts/run_one_game.py
uv run python scripts/run_batch.py --num-games 50 --num-agents 6
uv run python scripts/analyze_nash_equilibrium.py --scenario greedy_neighbor
```

Train an RL policy:

```bash
uv run python scripts/train_simple.py --scenario hard --num-steps 100000
uv run python scripts/train_puffer_gpu.py --scenario hard --num-steps 5000000   # GPU
```

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for the full training workflow.

## Repo layout

```
bucket_brigade/        # Python: envs, agents, equilibrium, evolution, orchestration
bucket-brigade-core/   # Rust core (PyO3 + WASM bindings) — 100x faster
scripts/               # CLI tools and experiment runners
experiments/           # Research experiments (evolution, nash, marl, …)
tests/                 # pytest suite
web/                   # TypeScript front-end (visualizer + dashboard)
docs/                  # Documentation
```

Architecture details: [docs/SIMPLIFIED_ARCHITECTURE.md](docs/SIMPLIFIED_ARCHITECTURE.md).

## Scenarios

Thirteen named scenarios live in `bucket-brigade-core/src/scenarios.rs` (the canonical source): `default`, `easy`, `hard`, `trivial_cooperation`, `early_containment`, `greedy_neighbor`, `sparse_heroics`, `rest_trap`, `chain_reaction`, `deceptive_calm`, `overcrowding`, `mixed_motivation`, `minimal_specialization`. Each isolates a different dynamic — coordination pressure, social dilemma, sparse work, deceptive signaling, etc. See [docs/game_description.md](docs/game_description.md) for narrative descriptions.

## Custom agents

User policies subclass `AgentBase` and implement `act(obs) -> np.ndarray([house_index, mode_flag])`. Examples for each scenario live in `bucket_brigade/agents/scenario_optimal/`. The archetype library (`bucket_brigade/agents/archetypes.py`) gives parameterized starting points (Firefighter, Free Rider, Hero, Coordinator, Liar).

## Testing

```bash
uv run pytest                    # Python
pnpm run test                    # Web unit tests (Vitest)
pnpm run test:e2e                # Playwright
```

Code quality: `uv run ruff check . --fix`, `uv run mypy .`, `pnpm run lint:biome`, `pnpm run typecheck`. Pre-commit hooks: `pre-commit install`.

## Development

This repo uses **Loom** for AI-assisted development orchestration. See [CLAUDE.md](CLAUDE.md) for the full guide, [.loom/LOOM_AGENTS.md](.loom/LOOM_AGENTS.md) for role workflows, and [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

For documentation, start at [docs/README.md](docs/README.md).
