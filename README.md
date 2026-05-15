# 🔥 Bucket Brigade: The Ultimate Cooperation Challenge

*A groundbreaking platform for studying cooperation, deception, and collective intelligence in multi-agent systems.*

**[🎮 Try the Interactive Demo](https://rjwalters.info/bucket-brigade/)** - Run tournaments in your browser

---

> **📖 Terminology Note:** This project uses "agent" in two distinct contexts:
> - **Game Policies/Strategies**: AI decision-makers in the bucket brigade game (the focus of this research)
> - **Loom Roles**: Development automation workers (see [LOOM_AGENTS.md](LOOM_AGENTS.md) for development workflow)
>
> When we say "agent" in the game context below, we mean the AI policies that play the bucket brigade game.
>
> See [GLOSSARY.md](GLOSSARY.md) for complete terminology definitions.

## 🧭 The Story & Challenge

Imagine a frontier town where **10 houses stand in a circle**, connected by paths that carry not just people, but also the relentless spread of fire. When flames erupt, they leap from house to house with terrifying speed. The townsfolk have formed a **Bucket Brigade** — but not everyone wants to be a hero.

Some are exhausted from long workdays. Others prioritize their own home over the community's needs. A few might even spread false information or work against the group. In this microcosm of human nature, **cooperation isn't guaranteed — it's earned**.

**Bucket Brigade** transforms this dramatic scenario into a research platform where AI agents navigate the complex dance between self-interest and collective good. Every "night," agents make two crucial decisions:

1. **Signal** their intent (work or rest) — but they might be lying
2. **Choose an action**: where to go and whether to fight fires

The result? Endless fascinating dynamics of **trust, deception, coordination, and sacrifice**.

## 🎯 Core Game Mechanics

See [docs/game_mechanics.md](docs/game_mechanics.md) for the complete game rules and mechanics.

## 🎯 Research Goals

- **Run large-scale tournaments** with diverse teams and scenarios (1000+ games)
- **Extract individual policy performance** with statistical validation
- **Evolve optimal heuristic agents** through evolutionary algorithms - see [Evolution Research](experiments/evolution/README.md)
- **Train neural network policies** using reinforcement learning:
  - Population-based training (recommended) - see [POPULATION_TRAINING.md](POPULATION_TRAINING.md)
  - Single agent training - see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Analyze Nash equilibria** to find stable strategic configurations
- **Understand cooperation dynamics** through scenario analysis and agent ranking

---

## 🧩 Project Architecture

```
bucket-brigade/
├── README.md
├── pyproject.toml
├── uv.lock
│
├── bucket_brigade/           # Python implementation
│ ├── envs/                   # Simulation environments
│ │ ├── bucket_brigade_env.py
│ │ ├── puffer_env_rust.py    # Rust-backed RL environment (100x faster)
│ │ ├── scenarios.py
│ │ └── __init__.py
│ │
│ ├── agents/                 # Heuristic + learned agents
│ │ ├── agent_base.py
│ │ ├── heuristic_agent.py
│ │ ├── archetypes.py         # Predefined strategy profiles
│ │ └── __init__.py
│ │
│ ├── equilibrium/            # Nash equilibrium analysis
│ │ ├── payoff_evaluator_rust.py  # Rust-backed evaluator (100x faster)
│ │ ├── best_response.py      # Best response computation
│ │ ├── double_oracle.py      # Nash equilibrium finder
│ │ ├── nash_solver.py        # Linear programming solver
│ │ └── __init__.py
│ │
│ ├── evolution/              # Evolutionary algorithms
│ │ ├── fitness_rust.py       # Rust-backed fitness (100x faster)
│ │ ├── genetic_algorithm.py
│ │ └── __init__.py
│ │
│ ├── orchestration/          # Statistical analysis + ranking
│ │ ├── ranking_model.py
│ │ ├── summary.py            # Statistical summary generation
│ │ └── __init__.py
│ │
│ ├── utils/                  # Statistical utilities
│ │ ├── statistics.py         # Confidence intervals, Shapley values, etc.
│ │ └── __init__.py
│ │
│ └── visualizer_api/         # Replay export / web bridge
│ └── __init__.py
│
├── bucket-brigade-core/      # Rust implementation (100x faster)
│ ├── Cargo.toml
│ ├── pyproject.toml
│ ├── src/
│ │ ├── lib.rs
│ │ ├── engine.rs
│ │ ├── scenarios.rs
│ │ ├── rng.rs
│ │ ├── python.rs            # PyO3 bindings for Python integration
│ │ └── wasm.rs              # WebAssembly bindings for browser
│ └── bucket_brigade_core/
│ └── __init__.py
│
├── scripts/                   # CLI + experiment runners
│ ├── run_one_game.py
│ ├── run_batch.py            # With --generate-summary flag
│ ├── analyze_summaries.py    # Statistical analysis CLI
│ ├── evolve_agents.py        # Evolutionary optimization
│ ├── train_simple.py         # RL training with PufferLib
│ ├── analyze_nash_equilibrium.py  # Nash equilibrium analysis
│ ├── test_rust_payoff.py     # Verify Rust payoff evaluation
│ └── test_rust_fitness.py    # Verify Rust fitness evaluation
│
├── tests/                     # Unit tests (pytest)
│ ├── test_environment.py
│ ├── test_agents.py
│ ├── test_orchestration.py
│ └── test_rust_integration.py
│
└── web/                       # Front-end visualizer (TypeScript)
├── src/
│   ├── components/           # Town, AgentLayer, GameAnalysis
│   ├── pages/                # SimpleDashboard, GameReplay, Settings
│   ├── types/                # TypeScript definitions
│   ├── utils/                # Storage utilities
│   └── main.tsx              # App entry point
└── public/
```


---

## ⚙️ Environment Summary

| Feature | Description |
|----------|-------------|
| **World** | 10 houses in a ring, each Safe, Burning, or Ruined |
| **Agents** | 4–10 agents, each owning one house |
| **Signals** | Broadcast intent (`Work` or `Rest`) each night |
| **Actions** | `(house, mode)` → choose where and whether to work |
| **Fire spread** | Burning houses ignite neighbors with probability `prob_fire_spreads_to_neighbor` |
| **Extinguishing** | `P(extinguish) = 1 - (1 - prob_solo_agent_extinguishes_fire)^workers` |
| **Termination** | After ≥ min_nights and all fires are out or all houses ruined (max 100 nights) |
| **Rewards** | Team and individual components based on saved/ruined houses and effort cost |

**Detailed mechanics**: See [docs/game_mechanics.md](docs/game_mechanics.md) for the complete turn sequence and rules.

---

## 🎮 Nash Equilibrium Analysis

Find **stable strategic configurations** where no agent can improve by unilaterally changing strategy.

### What is Nash Equilibrium?

In the Bucket Brigade game, a Nash equilibrium represents a strategic configuration where:
- Every agent is playing a best response to others' strategies
- No single agent can improve their payoff by changing their strategy alone
- The system is in a stable state (no incentive to deviate)

### Key Algorithms

1. **Payoff Evaluation** (Monte Carlo)
   - Estimate expected rewards for strategy profiles
   - **Performance**: 100 simulations in ~0.4s (Rust-backed)
   - Uses parallel execution for faster computation

2. **Best Response Computation**
   - Find optimal strategy against given opponents
   - Uses scipy.optimize with bounds and constraints
   - Supports both local (L-BFGS-B) and global (differential evolution) optimization

3. **Double Oracle Algorithm**
   - Iteratively build strategy pool
   - Add best responses until convergence
   - Solves for mixed strategy equilibria

4. **Nash Solver** (Linear Programming)
   - Computes symmetric Nash equilibria
   - Uses scipy linear programming solver
   - Returns probability distribution over strategies

### Usage Example

```bash
# Analyze Nash equilibrium for a specific scenario
uv run python scripts/analyze_nash_equilibrium.py --scenario greedy_neighbor

# With custom parameters
uv run python scripts/analyze_nash_equilibrium.py \
    --scenario early_containment \
    --num-simulations 1000 \
    --max-iterations 10

# Quick test with minimal simulations
uv run python scripts/test_nash_minimal.py
```

### Agent Archetypes

Predefined strategy profiles for testing:
- **Firefighter**: High work tendency, honest signaling
- **Free Rider**: Low work tendency, rest bias
- **Hero**: Extreme work tendency, own house priority
- **Coordinator**: High coordination weight, honest signals
- **Liar**: Dishonest signaling, strategic deception

### Performance

| Operation | Time | Speedup |
|-----------|------|---------|
| 100 simulations | 0.4s | **2250x** faster than Python |
| Payoff matrix (2×2) | ~2s | Enables practical analysis |
| Full Double Oracle | Minutes | Was previously hours/days |

---

## 🧠 Ranking Orchestration

The **ranking system** runs batches of simulated games to estimate each agent's marginal value.

### Workflow
1. Randomly sample teams and scenarios.
2. Run games via `BucketBrigadeEnv`.
3. Record outcomes (team composition, rewards, replay path).
4. Fit a surrogate model:

   $$
   R_{\text{team}} = \alpha + \sum_{i\in\text{team}} \theta_i + \langle w, \phi_c\rangle + \varepsilon
   $$

5. Rank agents by estimated contribution $\theta_i$.
6. Optionally, adaptively select new team combinations to reduce uncertainty.

All results are logged to a local SQLite database and saved as JSON replays for analysis and visualization.

---

## 🧱 Implementation Roadmap

### ✅ Completed Foundation
| Stage | Goal | Deliverable |
|-------|------|-------------|
| ✅ 1 | Define environment dynamics | `bucket_brigade_env.py` |
| ✅ 2 | Design architecture & repo layout | *(this document)* |
| ✅ 3 | Implement heuristic scripted agents | `HeuristicAgent` with 10 parameters |
| ✅ 4 | Add replay logging + JSON exporter | One file per episode |
| ✅ 5 | Build ranking orchestration loop | Batch runner + basic analysis |
| ✅ 6 | Create TypeScript web visualizer | Game replay + ranking dashboard |
| ✅ 7 | **Rust core engine** | `bucket-brigade-core/` - 100x faster |
| ✅ 8 | **PufferLib integration** | Train learned policies with PPO - see [TRAINING_GUIDE.md](TRAINING_GUIDE.md) |
| ✅ 9 | **Nash equilibrium analysis** | Complete framework with Rust-backed performance |
| ✅ 10 | **100x performance boost** | All critical modules use Rust backend |

### 🚧 Phase 1: Statistical Validation & Analysis
| Feature | Goal | Status |
|-------|------|--------|
| Statistical summaries | Generate aggregate statistics from 1000+ game replays | ✅ Complete |
| Confidence intervals | Parametric and bootstrap CI for team performance | ✅ Complete |
| Agent contributions | Shapley value estimation for individual impact | ✅ Complete |
| Significance tests | Mann-Whitney and Welch's t-test for comparisons | ✅ Complete |
| Analysis CLI | Tools for comparing experiments and ranking teams | ✅ Complete |

### 🤖 Phase 2: RL Training Enhancements
| Issue | Goal | Status |
|-------|------|--------|
| [#10](../../issues/10) | Longer training runs for better policies | 📋 Planned |
| [#11](../../issues/11) | Hyperparameter tuning for PPO | 🚫 Blocked |
| [#12](../../issues/12) | Scenario-specific policy training | 📋 Planned |
| [#13](../../issues/13) | TensorBoard logging integration | 📋 Planned |
| [#14](../../issues/14) | Curriculum learning for progressive difficulty | 📋 Planned |

### 🔍 Phase 3: Agent Discovery & Analysis Tools
| Issue | Goal | Status |
|-------|------|--------|
| [#20](../../issues/20) | Core agent comparison utilities | 📋 Planned |
| [#21](../../issues/21) | Simple finder tools (Similar & Counter agents) | 🚫 Blocked |
| [#22](../../issues/22) | Matchup matrix visualization | 📋 Planned |
| [#23](../../issues/23) | Agent comparer (side-by-side) | 📋 Planned |
| [#24](../../issues/24) | Team synergy analyzer | 📋 Planned |
| [#31](../../issues/31) | Parameter explorer with live preview | 📋 Planned |
| [#32](../../issues/32) | Integration & polish (routing, navigation, testing) | 📋 Planned |
| [#15](../../issues/15) | React Radar Chart components | 📋 Planned |
| [#16](../../issues/16) | Expand team templates library | 📋 Planned |
| [#18](../../issues/18) | Parameter space visualizations | 📋 Planned |

### 📚 Phase 4: Documentation & Cleanup
| Issue | Goal | Status |
|-------|------|--------|
| [#19](../../issues/19) | Fix documentation inconsistencies | 📋 Planned |
| [#25](../../issues/25) | Remove duplicate venv/ directory | 🔨 In Progress |

---

## 🧰 Dependencies

**Python Version Requirement**: Python 3.9 - 3.13 (3.14+ not yet supported)

This project uses PyO3 for Rust-Python bindings in `bucket-brigade-core`. PyO3 0.22.6 currently supports Python up to 3.13. Python 3.14 support will be available in future PyO3 releases.

We recommend Python 3.12 (specified in `.python-version`).

```bash
pip install numpy pandas scikit-learn matplotlib
```

for development and testing:
```bash
pip install pytest ruff mypy typer
```

for Rust core (**required** for Nash equilibrium and fast training, provides 100x speedup):
```bash
cd bucket-brigade-core && ./build.sh
```

This automated build script checks your Python version, compiles the Rust library, and installs it correctly.

Alternatively, manual method (matches what `build.sh` does — uses the
setuptools-rust backend declared in `bucket-brigade-core/pyproject.toml`):
```bash
cd bucket-brigade-core
uv pip install setuptools-rust
RUSTC_WRAPPER= PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
    uv run python -m pip install -e . --no-build-isolation
```

> **Pitfall**: many of this repo's example commands set `RUSTC_WRAPPER=sccache`.
> If sccache is not installed, the build may fail or produce a broken
> artifact. Either install sccache (`cargo install sccache`) or unset
> `RUSTC_WRAPPER` before building. See `bucket-brigade-core/README.md` for
> troubleshooting (e.g. recovering from the CFFI-shadow trap).

🚀 Quickstart

⚠️ **Important**: For research and performance-critical work, use the **Rust-backed environment** (`bucket-brigade-core`) via PyO3 bindings. The pure Python environment (`bucket_brigade/envs/bucket_brigade_env.py`) is provided for reference and demos only. See `experiments/evolution/RUST_SINGLE_SOURCE_OF_TRUTH.md` for migration details.

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install pnpm (Node.js package manager)
curl -fsSL https://get.pnpm.io/install.sh | sh -

# Install all dependencies
pnpm run install:all

# Run a single game with random agents
uv run python scripts/run_one_game.py

# Test different agent types
uv run python scripts/test_agents.py

# Run a batch of games for ranking experiments
uv run python scripts/run_batch.py --num-games 50 --num-agents 6

# Test specific scenarios with known optimal strategies
uv run python scripts/test_scenarios.py trivial_cooperation
uv run python scripts/test_scenarios.py greedy_neighbor
uv run python scripts/test_scenarios.py sparse_heroics

# Analyze Nash equilibrium for game-theoretic insights
uv run python scripts/analyze_nash_equilibrium.py --scenario greedy_neighbor
uv run python scripts/test_nash_minimal.py  # Quick verification

# Test Rust performance (should be ~100x faster)
uv run python scripts/test_rust_payoff.py
uv run python scripts/test_rust_fitness.py

# Submit your own agent
uv run python scripts/submit_agent.py --create-template  # Create template
uv run python scripts/submit_agent.py my_agent.py        # Validate & submit

# Analyze batch results
uv run python scripts/analyze_rankings.py results/

# Launch the web visualizer (starts dev server on http://localhost:5173)
pnpm run dev
```

## 🧵 Development Orchestration

This repository is set up to use **Loom** for AI-powered development orchestration in future development stages. See `LOOM_AGENTS.md` and `CLAUDE.md` for details on the autonomous Loom agent workflow system (Builder, Judge, Curator, etc.) that will be used for managing complex development tasks like implementing the full Bayesian ranking system and PufferLib integration.

**Note**: The term "agent" in this project refers to two different concepts:
- **Loom Agents** (development): Builder, Judge, Curator - autonomous development workers (see `LOOM_AGENTS.md`)
- **Game Agents** (AI): Firefighter, Hero, Free Rider - AI players in the simulation (see `docs/AGENT_ROSTER.md`)

## 🧪 Testing & Quality

### Python Testing
```bash
# Run Python tests
uv run pytest

# Run with coverage
uv run pytest --cov=bucket_brigade

# Run specific test file
uv run pytest tests/test_environment.py
```

### Web Testing
```bash
# Run unit tests (Vitest + React Testing Library)
pnpm run test           # one-off run
pnpm run test:watch     # watch mode
pnpm run test:coverage  # with coverage report

# Run Playwright e2e tests (renamed from `pnpm test`)
pnpm run test:e2e

# Run e2e in headed mode (visible browser)
pnpm run test:e2e:headed

# Run e2e with UI mode
pnpm run test:e2e:ui
```

**Breaking change (issue #158):** `pnpm test` now runs Vitest unit tests instead
of Playwright. Use `pnpm test:e2e` for the Playwright suite.

### Code Quality
```bash
# Format Python code
uv run ruff format .

# Lint Python code
uv run ruff check . --fix

# Type check Python
uv run mypy .

# Format web code
pnpm run format

# Lint web code
pnpm run lint:biome

# Type check web code
pnpm run typecheck
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
uv pip install pre-commit
pre-commit install

# Run all checks manually
pre-commit run --all-files
```

### Monorepo Scripts
```bash
# Install all dependencies
pnpm run install:all

# Run all tests (web unit tests via Vitest; use `test:e2e` for Playwright)
pnpm run test

# Format all code
pnpm run format

# Lint all code
pnpm run lint:fix

# Type check everything
pnpm run typecheck
```

## 🤖 Reinforcement Learning Training

Train RL agents using **PufferLib** to learn optimal firefighting strategies:

```bash
# Train a policy on a specific scenario
uv run python scripts/train_policy.py --scenario trivial_cooperation --total-timesteps 500000

# Train with custom opponent mix
uv run python scripts/train_policy.py --scenario early_containment \
    --opponent-policies firefighter coordinator free_rider \
    --run-name my_training_run

# Train on harder scenarios for advanced strategies
uv run python scripts/train_policy.py --scenario chain_reaction \
    --num-opponents 5 --total-timesteps 2000000
```

### Evaluate Trained Policies

Test trained models against expert agents:

```bash
# Evaluate a trained model
uv run python scripts/evaluate_policy.py models/my_training_run/final_policy.pt \
    --scenario greedy_neighbor --num-games 100

# Compare against different opponent types
uv run python scripts/evaluate_policy.py models/my_run/best_policy_500000.pt \
    --scenario sparse_heroics --num-opponents 4
```

### Training Features

- **Multi-agent training**: Learn against diverse opponent strategies
- **Scenario curriculum**: Train from easy to hard scenarios
- **Parallel environments**: Efficient training with vectorization
- **Automatic evaluation**: Regular assessment against expert agents
- **Model checkpointing**: Save best models during training

## 🧪 Scenario Testing & Validation

The platform includes **10 carefully designed test scenarios** with known optimal strategies:

- **Trivial Cooperation**: Easy fires reward universal cooperation
- **Early Containment**: Time pressure requires coordinated early action
- **Greedy Neighbor**: Social dilemma between self-interest and helping others
- **Sparse Heroics**: Minimal workers needed, overwork is wasteful
- **Rest Trap**: Usually safe to rest, but occasional disasters require response
- **Chain Reaction**: High spread demands distributed firefighting teams
- **Deceptive Calm**: Honest signaling rewarded during occasional flare-ups
- **Overcrowding**: Too many workers reduce efficiency
- **Mixed Motivation**: House ownership creates conflicting incentives

Each scenario tests different aspects of agent intelligence:
- **Strategic timing** (when to work vs rest)
- **Cooperation incentives** (help others vs focus on self)
- **Resource allocation** (efficient use of limited workers)
- **Honesty vs deception** (signaling and trust dynamics)

Use these scenarios to validate agent learning and ranking accuracy:

```bash
# Test ranking system with known optimal strategies
uv run python scripts/test_scenarios.py trivial_cooperation --num-games 30

# Compare different agent types in challenging scenarios
uv run python scripts/test_scenarios.py chain_reaction --num-games 50
```

## 🤖 Agent Development

### Creating Custom Agents

The platform supports user-submitted agents for research and competition:

```bash
# 1. Create agent template
uv run python scripts/submit_agent.py --create-template

# 2. Implement your strategy in my_agent.py
# Edit the MyCustomAgent class with your logic

# 3. Validate and test your agent
uv run python scripts/submit_agent.py my_agent.py

# 4. Submit for evaluation in tournaments
```

### Agent Interface

Your agent must implement this interface:

```python
class MyAgent(AgentBase):
    def __init__(self, agent_id: int, name: str = "MyAgent"):
        super().__init__(agent_id, name)

    def reset(self):
        # Reset between games
        pass

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        # Return [house_index, mode_flag]
        # obs contains: signals, locations, houses, last_actions, scenario_info
        return np.array([0, 1])  # Example: work on house 0
```

### Security & Validation

- **Sandboxed execution** in isolated environments
- **Static analysis** prevents dangerous code
- **Behavioral testing** validates agent interface
- **Only standard libraries** allowed (numpy, typing, etc.)

### Example Agents

See `bucket_brigade/agents/scenario_optimal/` for examples of optimal strategies for each test scenario.

### Community Agent Registry

Discover and share high-performing agent configurations through the **distributed agent discovery system**:

- **Browse known-good agents** - Explore community-discovered configurations in the registry
- **Submit your discoveries** - Share agents that perform well in your tournaments
- **Crowdsource optimization** - Treat agent discovery as distributed parameter search
- **Build collective knowledge** - Contribute to understanding of effective strategies

```bash
# View the community registry
cat web/public/data/known-good-agents.json

# Submit an agent via GitHub Issue
# Go to: https://github.com/rjwalters/bucket-brigade/issues/new/choose
# Select: "Agent Submission" template
```

**Learn more**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

🧠 Future Work

Bayesian team-composition optimization

Cross-game generalization (ranking transferable to other environments)

Real-time leaderboard visualization

Integration with reinforcement learning pipelines via PufferLib
