# 🔥 Bucket Brigade: The Ultimate Cooperation Challenge

*A groundbreaking platform for studying cooperation, deception, and collective intelligence in multi-agent systems.*

**[🎮 Try the Interactive Demo](https://rjwalters.github.io/bucket-brigade/)** - Run tournaments in your browser

---

## 🧭 The Story & Challenge

Imagine a frontier town where **10 houses stand in a circle**, connected by paths that carry not just people, but also the relentless spread of fire. When flames erupt, they leap from house to house with terrifying speed. The townsfolk have formed a **Bucket Brigade** — but not everyone wants to be a hero.

Some are exhausted from long workdays. Others prioritize their own home over the community's needs. A few might even spread false information or work against the group. In this microcosm of human nature, **cooperation isn't guaranteed — it's earned**.

**Bucket Brigade** transforms this dramatic scenario into a research platform where AI agents navigate the complex dance between self-interest and collective good. Every "night," agents make two crucial decisions:

1. **Signal** their intent (work or rest) — but they might be lying
2. **Choose an action**: where to go and whether to fight fires or conserve energy

The result? Endless fascinating dynamics of **trust, deception, coordination, and sacrifice**.

## 🎯 Research Goals

- **Estimate each agent's marginal contribution** to team performance
- **Compare cooperation strategies** across diverse scenarios
- **Rank agents fairly** using advanced statistical methods
- **Understand emergent behaviors** in multi-agent systems

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
│ │ ├── scenarios.py
│ │ └── __init__.py
│ │
│ ├── agents/                 # Heuristic + learned agents
│ │ ├── agent_base.py
│ │ ├── heuristic_agent.py
│ │ ├── agent_loader.py
│ │ ├── agent_template.py
│ │ └── __init__.py
│ │
│ ├── orchestration/          # Ranking + batch orchestration
│ │ ├── ranking_model.py
│ │ └── __init__.py
│ │
│ ├── data/                   # Results + replays
│ │ └── replays/
│ │
│ ├── utils/                  # Shared utilities
│ │ └── __init__.py
│ │
│ └── visualizer_api/         # Replay export / web bridge
│ └── __init__.py
│
├── bucket-brigade-core/      # Rust implementation (10-20x faster)
│ ├── Cargo.toml
│ ├── pyproject.toml
│ ├── src/
│ │ ├── lib.rs
│ │ ├── engine.rs
│ │ ├── scenarios.rs
│ │ ├── rng.rs
│ │ ├── python.rs
│ │ └── wasm.rs
│ └── bucket_brigade_core/
│ └── __init__.py
│
├── scripts/                   # CLI + experiment runners
│ ├── run_one_game.py
│ ├── run_batch.py
│ └── analyze_rankings.py
│
├── tests/                     # Unit tests (pytest)
│ ├── test_environment.py
│ ├── test_agents.py
│ ├── test_orchestration.py
│ └── test_rust_integration.py
│
└── web/                       # Front-end visualizer (TypeScript)
├── src/
│   ├── components/           # GameBoard, ReplayControls, GameInfo
│   ├── pages/                # Dashboard, GameReplay, Rankings, Settings
│   ├── types/                # TypeScript definitions
│   ├── utils/                # Storage utilities
│   └── main.tsx              # App entry point
└── public/
```


---

## ⚙️ Environment Summary

| Feature | Description |
|----------|-------------|
| **World** | 10 houses in a ring, each `Safe`, `Burning`, or `Ruined` |
| **Agents** | 4–10 agents, each owning one or more houses |
| **Signals** | Broadcast intent (`Work` or `Rest`) each night |
| **Actions** | `(house, mode)` → choose where and whether to work |
| **Fire spread** | Burning houses ignite neighbors with probability β |
| **Extinguishing** | `P(extinguish) = 1 - exp(-κ * workers)` |
| **Termination** | After ≥ N_min nights and all fires are out or all houses ruined (max 100 nights) |
| **Rewards** | Team and individual components based on saved/ruined houses and effort cost |

### 🎮 Turn Order & Game Mechanics

Each "night" in Bucket Brigade follows this sequence:

1. **Observation Phase**: Agents observe current fire state (fires from previous night's spread/sparks)
2. **Signal Phase**: Agents broadcast their intended mode (`Work` or `Rest`)
3. **Action Phase**: Agents simultaneously choose destination and mode `(house_index, work/rest)`
4. **Extinguish Phase**: Workers attempt to put out fires at their chosen locations
   - Probability: `P(extinguish) = 1 - exp(-κ * num_workers)`
5. **Burn-out Phase**: Unextinguished fires become RUINED houses
6. **Spread Phase**: Fires spread to neighboring SAFE houses (probability β)
   - **Important**: New fires are visible NEXT turn, giving agents time to respond
7. **Spark Phase**: Random spontaneous fires ignite on SAFE houses (if night < N_spark)
   - **Important**: New sparks are visible NEXT turn
8. **Reward Calculation**: Based on houses saved/ruined and work costs
9. **Termination Check**: Game ends if `night ≥ N_min AND (all_safe OR all_ruined OR night ≥ 100)`

**Key Design Decision**: Fires spread and spark at the END of each turn, making them visible for the NEXT turn. This allows agents to observe fire locations and coordinate strategic responses, rewarding teamwork over luck.

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
| ✅ 7 | **Rust core engine** | `bucket-brigade-core/` - 10-20x faster |
| ✅ 8 | **PufferLib integration** | Train learned policies with PPO - see [TRAINING_GUIDE.md](TRAINING_GUIDE.md) |

### 🚧 Phase 1: Backend Infrastructure (Tournament System)
| Issue | Goal | Status |
|-------|------|--------|
| [#4](../../issues/4) | PostgreSQL schema & Agent Registry | 📋 Planned |
| [#5](../../issues/5) | Job Queue with priority support | 📋 Planned |
| [#6](../../issues/6) | Tournament Coordinator with adaptive sampling | 📋 Planned |
| [#7](../../issues/7) | Worker Pool for parallel execution | 📋 Planned |
| [#8](../../issues/8) | Ranking Service with periodic re-computation | 🚫 Blocked |
| [#9](../../issues/9) | Web API for live rankings & history | 🚫 Blocked |

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

```bash
pip install numpy pandas scikit-learn matplotlib
```

for development and testing:
```bash
pip install pytest ruff mypy typer
```

for Rust core (optional, provides 10-20x speedup):
```bash
cd bucket-brigade-core && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install -e .
```

🚀 Quickstart
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
# Run Playwright tests
pnpm run test

# Run in headed mode (visible browser)
pnpm run test:headed

# Run with UI mode
pnpm run test:ui
```

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

# Run all tests
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

**Learn more**: See [docs/AGENT_SUBMISSION_GUIDE.md](docs/AGENT_SUBMISSION_GUIDE.md) for complete submission guidelines.

🧠 Future Work

Bayesian team-composition optimization

Cross-game generalization (ranking transferable to other environments)

Real-time leaderboard visualization

Integration with reinforcement learning pipelines via PufferLib
