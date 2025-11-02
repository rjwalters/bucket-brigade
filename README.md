# 🔥 Bucket Brigade

*A multi-agent cooperation game and ranking engine for team-based environments.*

---

## 🧭 Overview

**Bucket Brigade** is a research platform for studying cooperation, deception, and skill ranking in multi-agent systems.

Agents play repeated “nights” in a small town arranged as a **ring of 10 houses**.  
Each night, they can **signal** whether they will work or rest, and then choose an **action**:

- **Work** on a specific house to fight fires.  
- **Rest** to save energy (and perhaps mislead others).  

Fires spread probabilistically, and the team’s total reward depends on how many houses are saved versus ruined.  
Agents can lie, coordinate, or free-ride — the dynamics create natural tension and emergent strategies.

The long-term goal is to estimate each agent’s **marginal contribution** to team performance using a scalable **ranking orchestration system** inspired by Elo, Bradley-Terry, and Bayesian optimization methods.

---

## 🧩 Project Architecture

```
bucket-brigade/
├── README.md
├── pyproject.toml / setup.py
├── requirements.txt
│
├── bucket_brigade/
│ ├── envs/ # Simulation environments
│ │ ├── bucket_brigade_env.py
│ │ └── scenarios.py
│ │
│ ├── agents/ # Heuristic + learned agents
│ │ ├── heuristic_agent.py
│ │ ├── random_agent.py
│ │ └── puffer_adapter.py
│ │
│ ├── orchestration/ # Ranking + batch orchestration
│ │ ├── orchestrator.py
│ │ ├── ranking_model.py
│ │ └── database.py
│ │
│ ├── data/ # Results + replays
│ │ ├── results.db
│ │ └── replays/
│ │
│ ├── utils/ # Shared utilities
│ │ ├── logging.py
│ │ └── serialization.py
│ │
│ └── visualizer_api/ # Replay export / web bridge
│ └── exporter.py
│
├── scripts/ # CLI + experiment runners
│ ├── run_one_game.py
│ ├── run_batch.py
│ └── analyze_rankings.py
│
├── tests/ # Unit tests (pytest)
│
└── web/ # Front-end visualizer (TypeScript)
├── src/
│   ├── components/     # GameBoard, ReplayControls, GameInfo
│   ├── pages/          # Dashboard, GameReplay, Rankings, Settings
│   ├── types/          # TypeScript definitions
│   ├── utils/          # Storage utilities
│   └── main.tsx        # App entry point
└── public/
```


---

## ⚙️ Environment Summary

| Feature | Description |
|----------|-------------|
| **World** | 10 houses in a ring, each `Safe`, `Burning`, or `Ruined` |
| **Agents** | 4–10 agents, each owning a house |
| **Signals** | Broadcast intent (`Work` or `Rest`) each night |
| **Actions** | `(house, mode)` → choose where and whether to work |
| **Fire spread** | Burning houses ignite neighbors with probability β |
| **Extinguishing** | `P(extinguish) = 1 - exp(-κ * workers)` |
| **Termination** | After ≥ N_min nights and all fires are out or all houses ruined |
| **Rewards** | Team and individual components based on saved/ruined houses and effort cost |

---

## 🧠 Ranking Orchestration

The **ranking system** runs batches of simulated games to estimate each agent’s marginal value.

### Workflow
1. Randomly sample teams and scenarios.  
2. Run games via `BucketBrigadeEnv`.  
3. Record outcomes (team composition, rewards, replay path).  
4. Fit a surrogate model:

   \[
   R_{\text{team}} = \alpha + \sum_{i\in\text{team}} \theta_i + \langle w, \phi_c\rangle + \varepsilon
   \]

5. Rank agents by estimated contribution \( \theta_i \).  
6. Optionally, adaptively select new team combinations to reduce uncertainty.

All results are logged to a local SQLite database and saved as JSON replays for analysis and visualization.

---

## 🧱 Implementation Roadmap

| Stage | Goal | Deliverable |
|-------|------|-------------|
| ✅ 1 | Define environment dynamics | `bucket_brigade_env.py` |
| ✅ 2 | Design architecture & repo layout | *(this document)* |
| ✅ 3 | Implement heuristic scripted agents | `HeuristicAgent` with 10 parameters |
| ✅ 4 | Add replay logging + JSON exporter | One file per episode |
| ✅ 5 | Build ranking orchestration loop | Batch runner + basic analysis |
| ✅ 6 | Create TypeScript web visualizer | Game replay + ranking dashboard |
| 🔜 7 | (Future) Integrate PufferLib | Train learned policies |

---

## 🧰 Dependencies

```bash
pip install pufferlib numpy pandas scikit-learn matplotlib
```

for development and testing:
```bash
pip install pytest black ruff mypy typer
```

🚀 Quickstart
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install pnpm (Node.js package manager)
npm install -g pnpm

# Install all dependencies
npm run install:all

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

# Launch the web visualizer
npm run dev
```

## 🧵 Development Orchestration

This repository is set up to use **Loom** for AI-powered development orchestration in future development stages. See `AGENTS.md` and `CLAUDE.md` for details on the autonomous agent workflow system that will be used for managing complex development tasks like implementing the full Bayesian ranking system and PufferLib integration.

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
uv run black .

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
npm run install:all

# Run all tests
npm run test

# Format all code
npm run format

# Lint all code
npm run lint:fix

# Type check everything
npm run typecheck
```

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

🧠 Future Work

Bayesian team-composition optimization

Cross-game generalization (ranking transferable to other environments)

Real-time leaderboard visualization

Integration with reinforcement learning pipelines via PufferLib

