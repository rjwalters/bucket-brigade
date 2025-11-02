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
| ⏳ 3 | Implement heuristic scripted agents | Simple `act(obs)` interface |
| ⏳ 4 | Add replay logging + JSON exporter | One file per episode |
| ⏳ 5 | Build ranking orchestration loop | Batch runner + ridge regression ranking |
| 🔜 6 | Create TypeScript web visualizer | Load & replay saved games |
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

🚀 Quickstart (after initial commit)
```bash
# Clone and install
git clone https://github.com/<your-org>/bucket-brigade.git
cd bucket-brigade
pip install -e .

# Run a simple test game
python scripts/run_one_game.py

# Run a batch for ranking
python scripts/run_batch.py --num-games 50
```

🧠 Future Work

Bayesian team-composition optimization

Cross-game generalization (ranking transferable to other environments)

Real-time leaderboard visualization

Integration with reinforcement learning pipelines via PufferLib

