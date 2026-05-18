# 🏗️ Simplified Architecture - Bucket Brigade MVP

**Version**: 2.0 (Simplified)
**Last Updated**: 2025-11-03

---

## 🎯 Design Principles

1. **Simplicity First** - Remove unnecessary complexity
2. **Fast Iteration** - Quick experiments over premature scaling
3. **Research Focus** - Optimize for understanding, not production
4. **No Backend** - Static site + local computation
5. **Data on Demand** - Generate games when needed, store summaries only

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                     (Browser - Static Site)                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │   Dashboard    │   │  Game Replay   │   │   Settings     │  │
│  │                │   │                │   │                │  │
│  │ • Team Select  │   │ • Visualization│   │ • Speed        │  │
│  │ • Scenario     │   │ • Controls     │   │ • Theme        │  │
│  │ • Run Game     │   │ • Analysis     │   │ • Presets      │  │
│  └────────────────┘   └────────────────┘   └────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ React Router
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        GAME ENGINE                               │
│                   (Browser / WASM / Python)                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │          Rust Core Engine (SOURCE OF TRUTH)                 │ │
│  │  • bucket-brigade-core - Canonical game implementation      │ │
│  │  • Defines all scenarios and game mechanics                 │ │
│  │  • Exposed via PyO3 (Python) and wasm-bindgen (WASM)       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│              ┌───────────────┼───────────────┐                 │
│              ▼               ▼               ▼                  │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ TypeScript   │  │ WASM        │  │ Python      │           │
│  │ (fallback)   │  │ (browser)   │  │ (research)* │           │
│  │              │  │             │  │             │           │
│  │ browserEng.  │  │ wasmEng.    │  │ PyO3        │           │
│  │ ~50ms/game   │  │ ~5ms/game   │  │ ~2ms/game   │           │
│  └──────────────┘  └─────────────┘  └─────────────┘           │
│                                                                  │
│  * Python: Use Rust-backed PyO3 bindings (bucket-brigade-core) │
│    for research. Pure Python env (bucket_brigade.envs) is      │
│    DEPRECATED - see experiments/evolution/                      │
│    RUST_SINGLE_SOURCE_OF_TRUTH.md                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ Game Data
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │          Browser Storage (Ephemeral)                        │ │
│  │  • sessionStorage - Recent replays (last 10)                │ │
│  │  • localStorage - User preferences, favorites               │ │
│  │  • No backend - all client-side                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │          Filesystem (Research Artifacts)                    │ │
│  │  • results/summaries/*.json - Statistical results           │ │
│  │  • models/*.pt - Trained RL policies                        │ │
│  │  • Optional: SQLite for querying summaries                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Single Game Flow (Web Demo)

```
User Interaction
       │
       ▼
┌─────────────────┐
│   Dashboard     │  1. User selects team + scenario
│                 │  2. Clicks "Run Game"
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Browser Engine  │  3. Initialize game state
│                 │  4. Run simulation (50ms)
│                 │  5. Generate replay data
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Session Storage │  6. Store replay in sessionStorage
│                 │     (overwrites old replays after 10)
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Game Replay    │  7. Visualize with controls
│                 │  8. Show analysis
│                 │  9. User can download JSON
└─────────────────┘
```

### Batch Analysis Flow (Research CLI)

```
Researcher CLI Command
       │
       ▼
┌──────────────────────┐
│  scripts/run_batch   │  1. Define: team × scenario × N runs
│                      │  2. Example: 1000 games per config
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│   Python/Rust Engine │  3. Run games (5ms × 1000 = 5 seconds)
│                      │  4. Aggregate statistics
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Statistical Summary │  5. Compute:
│                      │     • Mean/std team reward
│                      │     • Individual contributions
│                      │     • Win rate, houses saved
│                      │     • Confidence intervals
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│   JSON Export        │  6. Save to:
│                      │     results/summaries/
│                      │       team_X_scenario_Y_1000runs.json
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Analysis Script     │  7. scripts/analyze_rankings
│                      │  8. Generate plots, rankings, insights
└──────────────────────┘
```

---

## 📦 Component Breakdown

### Frontend (React + TypeScript)

```
web/
├── src/
│   ├── components/
│   │   ├── Town.tsx                    # Main game board (circle of houses)
│   │   ├── AgentLayer.tsx              # Agent positions & animations
│   │   ├── ReplayControls.tsx          # Play/pause/speed controls
│   │   ├── GameSidebar.tsx             # Game state info (night, houses)
│   │   └── GameAnalysis.tsx            # Post-game analysis (NEW)
│   │
│   ├── pages/
│   │   ├── Dashboard.tsx               # SIMPLIFIED - Team + Scenario picker
│   │   ├── GameReplay.tsx              # Main game visualization page
│   │   └── Settings.tsx                # SIMPLIFIED - Basic settings
│   │
│   ├── utils/
│   │   ├── browserEngine.ts            # Pure TS game engine
│   │   ├── wasmEngine.ts               # WASM engine loader (optional)
│   │   ├── browserAgents.ts            # Agent logic in TS
│   │   ├── storage.ts                  # localStorage/sessionStorage helpers
│   │   └── schemas.ts                  # Type definitions
│   │
│   └── App.tsx                         # Main router (3 pages only)
│
└── public/
    └── wasm/                           # Compiled Rust WASM (optional)
```

### Backend (Python - CLI Only)

```
bucket_brigade/
├── envs/
│   ├── bucket_brigade_env.py           # Core game logic (Rust-backed, drives JointPPOTrainer)
│   ├── scenarios.py                    # 10 test scenarios
│   └── macro_action_env.py             # Sutton-options wrapper
│
├── agents/
│   ├── agent_base.py                   # Base agent class
│   ├── heuristic_agent.py              # Parameterized agents
│   └── scenario_optimal/               # Expert agents (6-7 files)
│
├── evolution/
│   ├── genetic_algorithm.py            # GA main loop
│   ├── population.py                   # Population management
│   ├── fitness.py                      # Fitness evaluation
│   └── operators.py                    # Crossover, mutation
│
└── orchestration/
    └── ranking_model.py                # Statistical ranking algorithms

scripts/
├── run_one_game.py                     # Single game runner
├── run_batch.py                        # Batch experiments
├── test_scenarios.py                   # Scenario validation
├── test_team.py                        # Team testing
├── compare_teams.py                    # Statistical comparison
├── analyze_rankings.py                 # Analysis & visualization
└── evolve_agents.py                    # GA optimization

# RL training lives under experiments/p3_specialization/train*.py
# and uses bucket_brigade.training.joint_trainer.JointPPOTrainer.

bucket-brigade-core/                    # Rust engine (optional, 10x faster)
├── src/
│   ├── engine.rs                       # Core game loop
│   ├── scenarios.rs                    # Scenario configs
│   ├── python.rs                       # PyO3 bindings
│   └── wasm.rs                         # WASM bindings
```

---

## 💾 Data Structures

### Statistical Summary (Primary Storage)

```typescript
interface StatisticalSummary {
  // Experiment metadata
  team: string[];                       // ["firefighter", "coordinator", "hero"]
  scenario: string;                     // "early_containment"
  num_runs: number;                     // 1000
  timestamp: string;                    // ISO 8601

  // Aggregate statistics
  statistics: {
    mean_team_reward: number;           // 241.6
    std_team_reward: number;            // 45.2
    confidence_interval_95: [number, number];  // [238.8, 244.4]

    mean_individual_rewards: number[];  // [32.5, 28.0, 45.2]
    std_individual_rewards: number[];   // [12.3, 10.5, 15.8]

    win_rate: number;                   // 0.73 (houses_saved >= 7)
    avg_nights: number;                 // 18.4
    std_nights: number;                 // 5.2

    houses_saved_avg: number;           // 7.2
    houses_saved_std: number;           // 1.8
  };

  // Individual contributions (Shapley values or marginal)
  agent_contributions: {
    [agent_name: string]: number;       // Estimated marginal value
  };

  // Optional: Performance breakdown
  performance_by_phase?: {
    early_game: { mean_reward: number; std_reward: number };
    mid_game: { mean_reward: number; std_reward: number };
    late_game: { mean_reward: number; std_reward: number };
  };
}
```

### Game Replay (Generated On-Demand)

```typescript
interface GameReplay {
  scenario: Scenario;
  nights: GameNight[];
}

interface GameNight {
  night: number;
  houses: HouseState[];          // [0-2] for each house
  signals: number[];             // [0-1] for each agent
  locations: number[];           // [0-9] for each agent
  actions: number[][];           // [[house, mode], ...] for each agent
  rewards: number[];             // Reward for each agent
}
```

---

## 🚀 Deployment Architecture

### Production (Static Site)

```
┌─────────────────────────────────────────────┐
│          GitHub Pages / Netlify / Vercel    │
│                                             │
│  https://yourusername.github.io/            │
│         bucket-brigade/                     │
│                                             │
│  ┌────────────────────────────────────────┐│
│  │  Static Files (web/dist/)              ││
│  │  • index.html                          ││
│  │  • main.js (bundled React app)        ││
│  │  • main.css                            ││
│  │  • wasm/bucket_brigade_core.wasm      ││
│  └────────────────────────────────────────┘│
│                                             │
│  No backend required ✅                     │
│  No database required ✅                    │
│  CDN cached globally ✅                     │
└─────────────────────────────────────────────┘

User Browser
  │
  ├─→ Load HTML/JS/CSS from CDN
  ├─→ Run games locally (client-side)
  └─→ Store data in browser storage
```

### Research Environment (Local)

```
Researcher's Machine
  │
  ├─→ Python Environment (uv)
  │    └─→ bucket_brigade package installed
  │
  ├─→ Rust Engine (optional, for speed)
  │    └─→ bucket-brigade-core compiled
  │
  ├─→ Scripts for batch processing
  │    ├─→ run_batch.py (1000s of games)
  │    ├─→ evolve_agents.py (GA optimization)
  │    └─→ experiments/p3_specialization/train*.py (RL training via JointPPOTrainer)
  │
  └─→ Results stored locally
       ├─→ results/summaries/*.json
       ├─→ models/*.pt (RL checkpoints)
       └─→ plots/*.png (analysis)
```

---

## 🔌 API Contracts (Internal Only)

### Browser Engine API

```typescript
// Initialize game
const engine = new BucketBrigadeEngine(scenario, agents);

// Run one step
const night: GameNight = engine.step();

// Check if done
const isDone: boolean = engine.isDone();

// Get full replay
const replay: GameReplay = engine.getReplay();

// Get statistics
const stats = {
  teamReward: engine.getTeamReward(),
  individualRewards: engine.getIndividualRewards(),
  nightsPlayed: engine.getNightsPlayed(),
  housesSaved: engine.getHousesSaved(),
};
```

### Python Engine API

```python
from bucket_brigade.envs import BucketBrigadeEnv, default_scenario
from bucket_brigade.agents import HeuristicAgent

# Create environment
scenario = default_scenario(num_agents=4)
env = BucketBrigadeEnv(scenario)

# Create agents
agents = [HeuristicAgent(i, f"Agent{i}", params) for i in range(4)]

# Run game
obs = env.reset()
while not env.done:
    actions = [agent.act(obs) for agent in agents]
    obs, rewards, dones, info = env.step(actions)

# Get results
team_reward = env.get_team_reward()
individual_rewards = env.get_individual_rewards()
```

---

## 🧪 Testing Strategy

### Unit Tests

```
tests/
├── test_environment.py          # Game mechanics
├── test_agents.py              # Agent behaviors
├── test_scenarios.py           # Scenario configs
├── test_evolution.py           # GA operators
└── test_orchestration.py       # Ranking algorithms
```

### Integration Tests

```
tests/integration/
├── test_full_game.py           # End-to-end game
├── test_batch_runner.py        # Batch processing
└── test_rl_training.py         # Training pipeline
```

### End-to-End Tests (Web)

```
web/tests/
├── game-visualization.spec.ts  # Playwright test for game display
├── controls.spec.ts            # Test replay controls
└── team-selection.spec.ts      # Test team picker
```

---

## 📊 Performance Characteristics

### Game Execution Speed

| Implementation | Speed per Game | Throughput (1000 games) | Use Case |
|----------------|----------------|-------------------------|----------|
| TypeScript (Browser) | ~50ms | ~50 seconds | Single game demos |
| WASM (Browser) | ~5ms | ~5 seconds | Batch in browser |
| Python (Pure) | ~20ms | ~20 seconds | Development |
| Rust (PyO3) | ~2ms | ~2 seconds | Large tournaments |

### Storage Requirements

| Data Type | Size | Count | Total |
|-----------|------|-------|-------|
| Single game replay | ~10 KB | 10 (recent) | 100 KB |
| Statistical summary | ~5 KB | 100 configs | 500 KB |
| User preferences | ~1 KB | 1 | 1 KB |
| **Total (Browser)** | | | **~600 KB** |

---

## 🔐 Security Considerations

### Removed Attack Surfaces

✅ **No user code execution** - Agents are curated, not submitted
✅ **No backend API** - No server to compromise
✅ **No database** - No SQL injection, no data breach
✅ **No file uploads** - No malicious file attacks

### Remaining Considerations

- **Browser storage limits** - Max 10MB per domain (plenty for our use)
- **Client-side validation** - Trust browser, but validate inputs
- **WASM safety** - Compiled Rust is memory-safe

---

## 🔄 Migration Path (If Backend Needed Later)

If we decide to add a backend later:

1. **Add backend service** (FastAPI + PostgreSQL)
2. **Keep static site** - Backend is optional enhancement
3. **Gradual adoption** - Users can opt-in to cloud features
4. **Data sync** - Upload local summaries to backend
5. **Community features** - Leaderboards, shared results

But for MVP: **No backend needed!**

---

## 📖 Related Documentation

- [SIMPLIFICATION_PLAN.md](./SIMPLIFICATION_PLAN.md) - Removal plan
- [API.md](archive/API.md) - Data structures (archived, aspirational)
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - RL training
- [DEPLOYMENT.md](archive/DEPLOYMENT.md) - Static site deployment (archived, aspirational)

---

**Status**: ✅ Architecture designed, ready for implementation
