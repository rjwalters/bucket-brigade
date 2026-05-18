# 🎯 Bucket Brigade Simplification Plan

**Status**: Draft for Review
**Created**: 2025-11-03
**Goal**: Refocus on core research MVP - tournaments, performance analysis, and heuristic evolution

---

## 📊 Executive Summary

This plan removes agent submission infrastructure and backend API complexity to focus on:

1. **Tournament Engine** - Run diverse teams across diverse scenarios
2. **Performance Analysis** - Extract individual agent contributions
3. **Heuristic Evolution** - Optimize scripted agent parameters
4. **Statistical Validation** - Run 1000s of games for robust scenario analysis
5. **Simple Web Demo** - Single-game visualization with analysis
6. **RL Training Path** - Keep PufferLib integration, remove submission pipeline

**Key Insight**: Games are so fast to simulate (~5ms each) that we should store **statistical summaries** (team × scenario × N runs), not individual game replays.

---

## ❌ Features to Remove

### 1. Agent Submission Infrastructure

**Why Remove**: Adds complexity without immediate research value. Focus on curated agents first.

**Files to Delete**:
```
bucket_brigade/agents/agent_loader.py          # Security validation, sandboxing
bucket_brigade/agents/agent_template.py        # User submission template
bucket_brigade/services/agent_registry.py      # Agent submission service
bucket_brigade/services/api.py                 # REST API for submissions
bucket_brigade/services/job_queue.py           # Job queue (unused)
scripts/submit_agent.py                        # CLI submission tool
web/public/data/known-good-agents.json         # Community registry
docs/AGENT_SUBMISSION_GUIDE.md                 # Submission docs
docs/AGENT_REGISTRY_API.md                     # API docs
```

**Functions to Remove**:
- `load_agent_from_file()` with validation
- `load_agent_from_string()`
- `validate_agent_code()` - security checks
- `validate_agent_behavior()` - behavioral tests
- `AgentValidationError`, `AgentSecurityError` exceptions

**Keep**:
- `agent_base.py` - Base class still needed
- `heuristic_agent.py` - Our curated agents
- `scenario_optimal/` - Expert agents for validation

---

### 2. Database & Backend API

**Why Remove**: PostgreSQL adds deployment complexity. Local computation is fast enough. If storage is needed later, use SQLite for summaries or JSON for session data.

**Files to Delete**:
```
bucket_brigade/db/models.py                    # SQLAlchemy models
bucket_brigade/db/connection.py                # Database connection
bucket_brigade/db/migrations/                  # Alembic migrations
bucket_brigade/db/__init__.py
bucket_brigade/services/                       # Entire services directory
docs/DATABASE_SETUP.md                         # Database docs
```

**Dependencies to Remove** (from `pyproject.toml`):
```python
"sqlalchemy>=2.0.0",
"psycopg2-binary>=2.9.0",
"alembic>=1.12.0",
"fastapi>=0.104.0",
"uvicorn>=0.24.0",
"python-multipart>=0.0.6",
```

---

### 3. Web UI - Complex Features

**Why Simplify**: Focus on single-game demo. Remove multi-tournament dashboard complexity.

**Pages to Remove/Simplify**:
```
web/src/pages/TeamBuilder.tsx                  # Complex team builder (simplify)
web/src/pages/Tournament.tsx                   # Multi-tournament dashboard
web/src/pages/Rankings.tsx                     # Global rankings page
web/src/components/team-builder/TournamentRunner.tsx
web/src/components/team-builder/TournamentResults.tsx
```

**Keep & Simplify**:
- `GameReplay.tsx` - Core visualization ✅
- `Dashboard.tsx` - Simplified to single-game launcher
- `Settings.tsx` - Basic app settings

---

### 4. Unused/Planned Features

**Files to Remove** (not implemented or unused):
```
docs/features/TEAM_BUILDER_TOURNAMENT.md       # Planned feature
docs/implementation/rust-wasm-plan.md          # Implementation notes (archive)
```

---

## ✅ Core Features to Keep & Enhance

### 1. Game Engine & Environment

**Keep All**:
```
bucket_brigade/envs/bucket_brigade_env.py      # Core game logic ✅
bucket_brigade/envs/scenarios.py               # 10 test scenarios ✅
bucket_brigade/envs/puffer_env.py              # RL training wrapper ✅
bucket-brigade-core/                           # Rust engine (10-20x faster) ✅
```

---

### 2. Agent System (Curated)

**Keep**:
```
bucket_brigade/agents/agent_base.py            # Base class
bucket_brigade/agents/heuristic_agent.py       # 10-parameter agents
bucket_brigade/agents/scenario_optimal/        # Expert agents
bucket_brigade/agents/__init__.py              # Exports
```

**Simplify**: Remove submission/validation code from `__init__.py`, keep only:
- `AgentBase` class
- `HeuristicAgent` class
- Helper to instantiate agents by name

---

### 3. Evolution & Optimization

**Keep All**:
```
bucket_brigade/evolution/genetic_algorithm.py  # GA for heuristic tuning ✅
bucket_brigade/evolution/population.py         # Population management ✅
bucket_brigade/evolution/fitness.py            # Fitness functions ✅
bucket_brigade/evolution/operators.py          # Crossover, mutation ✅
scripts/evolve_agents.py                       # CLI tool ✅
```

---

### 4. RL Training (JointPPOTrainer)

> **Status (2026-05, issue #335)**: The PufferLib path and its
> `scripts/train_simple.py` / `scripts/evaluate_simple.py` /
> `scripts/train_policy.py` / `scripts/train_curriculum.py` /
> `scripts/evaluate_policy.py` orchestration scripts were removed. RL training
> now lives in `bucket_brigade/training/joint_trainer.py` (`JointPPOTrainer`)
> and is driven by experiment-specific entry points such as
> `experiments/p3_specialization/train.py`. This section is retained for
> historical context only.

**Keep**:
```
bucket_brigade/training/joint_trainer.py       # JointPPOTrainer ✅
experiments/p3_specialization/train.py         # P3 training entry point ✅
TRAINING_GUIDE.md                              # Documentation ✅
```

---

### 5. Analysis & Testing

**Keep All**:
```
scripts/run_one_game.py                        # Single game runner ✅
scripts/run_batch.py                           # Batch experiments ✅
scripts/test_scenarios.py                      # Scenario validation ✅
scripts/test_agents.py                         # Agent testing ✅
scripts/test_team.py                           # Team testing ✅
scripts/compare_teams.py                       # Team comparison ✅
scripts/analyze_rankings.py                    # Statistical analysis ✅
bucket_brigade/orchestration/ranking_model.py  # Ranking algorithms ✅
```

---

### 6. Web Visualization (Simplified)

**Keep & Refactor**:
```
web/src/components/Town.tsx                    # Game board visualization ✅
web/src/components/AgentLayer.tsx              # Agent rendering ✅
web/src/components/ReplayControls.tsx          # Playback controls ✅
web/src/components/GameSidebar.tsx             # Game info display ✅
web/src/pages/GameReplay.tsx                   # Main game page ✅
web/src/utils/browserEngine.ts                 # Client-side engine ✅
web/src/utils/wasmEngine.ts                    # WASM integration ✅
```

**Simplify**:
- `Dashboard.tsx` - Single game launcher (team + scenario selector)
- `Settings.tsx` - Basic app preferences

---

## 🏗️ New Simplified Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  Dashboard  │  │  Game Replay │  │    Settings    │ │
│  │             │  │              │  │                │ │
│  │ - Pick Team │  │ - Visualize  │  │ - Speed        │ │
│  │ - Pick Scen │  │ - Controls   │  │ - Themes       │ │
│  │ - Run Game  │  │ - Analysis   │  │ - Presets      │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Browser Engine                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  TypeScript Game Engine (browserEngine.ts)       │  │
│  │  - Run single games in-browser                   │  │
│  │  - Fast enough for demos (50ms per game)         │  │
│  │  - Optional WASM for speed (5ms per game)        │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Session Storage                         │
│  - Recent game replays (last 10)                        │
│  - User preferences                                      │
│  - Team/scenario favorites                               │
│  (All ephemeral - no backend needed)                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│             Python CLI (Research Use)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Tournament  │  │  Evolution   │  │  RL Training   │ │
│  │ Engine      │  │  (GA)        │  │  (PufferLib)   │ │
│  │             │  │              │  │                │ │
│  │ - Run 1000s │  │ - Optimize   │  │ - Train NNs    │ │
│  │   of games  │  │   heuristics │  │ - Eval vs exp  │ │
│  │ - Stats     │  │ - Tournament │  │ - Save models  │ │
│  │   analysis  │  │   validation │  │                │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Filesystem Storage (Optional)               │
│  - JSON summaries of statistical results                │
│  - CSV exports for analysis                              │
│  - Saved RL policy checkpoints                           │
│  (No database - just files for research artifacts)      │
└─────────────────────────────────────────────────────────┘
```

---

## 💾 Data Storage Strategy

### What to Store

**DON'T Store**: Individual game replays (generate on-demand)

**DO Store**: Statistical summaries

```python
# Example: Statistical summary for team × scenario
{
  "team": ["firefighter", "coordinator", "hero"],
  "scenario": "early_containment",
  "num_runs": 1000,
  "timestamp": "2025-11-03T10:00:00Z",
  "statistics": {
    "mean_team_reward": 241.6,
    "std_team_reward": 45.2,
    "mean_individual_rewards": [32.5, 28.0, 45.2],
    "win_rate": 0.73,
    "avg_nights": 18.4,
    "houses_saved_avg": 7.2
  },
  "agent_contributions": {
    "firefighter": 0.35,
    "coordinator": 0.28,
    "hero": 0.37
  }
}
```

### Storage Location

**Option 1 (Simplest)**: JSON files in `results/summaries/`
```
results/
  summaries/
    team_abc_scenario_early_containment_1000runs_20251103.json
    team_def_scenario_chain_reaction_1000runs_20251103.json
```

**Option 2 (Future)**: SQLite for querying
```python
# Single local SQLite file
results/statistics.db
  tables: team_scenario_stats, agent_rankings, evolution_history
```

**Option 3 (Never)**: PostgreSQL - Too heavy for this use case

---

## 🎨 Simplified Web UI Design

### New Page Structure

```
┌────────────────────────────────────────────────────┐
│  Bucket Brigade - Game Demo                        │
├────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────┐  ┌──────────────────────────┐│
│  │   Team Select   │  │   Scenario Select        ││
│  │                 │  │                          ││
│  │ ○ Firefighters  │  │ ○ Trivial Cooperation    ││
│  │ ○ Coordinators  │  │ ○ Early Containment      ││
│  │ ○ Mixed Team    │  │ ○ Greedy Neighbor        ││
│  │ ○ Custom...     │  │ ○ Chain Reaction         ││
│  │   🎲 Random     │  │   🎲 Random              ││
│  └─────────────────┘  └──────────────────────────┘│
│                                                     │
│        [▶ Run Game]    [⚙️ Settings]               │
│                                                     │
├────────────────────────────────────────────────────┤
│                                                     │
│              🎮 Live Game Visualization            │
│                                                     │
│  ┌────────────────────────────────────────────────┐│
│  │                                                ││
│  │        [Game Board - Town with 10 Houses]     ││
│  │                                                ││
│  │  🏠 🏠 🏠 🏠 🏠 🏠 🏠 🏠 🏠 🏠              ││
│  │                                                ││
│  └────────────────────────────────────────────────┘│
│                                                     │
│  Night: 12/30    Houses Safe: 7/10                │
│  ◀◀  ◀  ▶  ▶▶   Speed: ━━━━━○────                │
│                                                     │
├────────────────────────────────────────────────────┤
│                                                     │
│  📊 Game Analysis                                  │
│  ┌────────────────────────────────────────────────┐│
│  │ Final Score: 241.6                            ││
│  │ Team Performance: Excellent (top 20%)         ││
│  │                                                ││
│  │ Agent Contributions:                           ││
│  │   • Firefighter: 32.5 (cooperated well)       ││
│  │   • Coordinator: 28.0 (efficient signaling)   ││
│  │   • Hero: 45.2 (saved critical houses)        ││
│  │                                                ││
│  │ Strategy: Early containment prevented spread  ││
│  │ Key Moments: Night 8 - coordinated response   ││
│  └────────────────────────────────────────────────┘│
│                                                     │
│  [📥 Download Replay]  [🔄 Run Again]             │
└────────────────────────────────────────────────────┘
```

### Features

1. **Team Selection** - Dropdown with presets + randomize
2. **Scenario Selection** - Dropdown with all 10 scenarios + randomize
3. **Single Game Visualization** - Live replay with controls
4. **Post-Game Analysis** - Score breakdown, strategy summary
5. **Minimal Settings** - Speed, theme, display options

---

## 📋 Implementation Roadmap

### Phase 1: Remove Infrastructure (1-2 days)

**Tasks**:
1. ✅ Delete agent submission files
2. ✅ Delete database/backend files
3. ✅ Remove dependencies from pyproject.toml
4. ✅ Update imports throughout codebase
5. ✅ Remove submission docs
6. ✅ Update README to reflect new scope

**PR**: "refactor: Remove agent submission and backend infrastructure"

---

### Phase 2: Simplify Web UI (2-3 days)

**Tasks**:
1. ✅ Create simplified Dashboard (team + scenario picker)
2. ✅ Remove Tournament and Rankings pages
3. ✅ Simplify Settings page
4. ✅ Update routing
5. ✅ Add post-game analysis component
6. ✅ Test all visualizations still work

**PR**: "refactor: Simplify web UI to single-game demo"

---

### Phase 3: Statistical Analysis Tools (2-3 days)

**Tasks**:
1. ✅ Create batch runner for statistical validation (1000+ runs)
2. ✅ Implement summary statistics generator
3. ✅ Add JSON export for team × scenario results
4. ✅ Create CLI tool for analyzing summaries
5. ✅ Add visualization of confidence intervals

**PR**: "feat: Add statistical validation tools for scenario analysis"

---

### Phase 4: Documentation Update (1 day)

**Tasks**:
1. ✅ Update README with new focus
2. ✅ Update API.md (remove backend endpoints)
3. ✅ Update DEPLOYMENT.md (static site only)
4. ✅ Update CONTRIBUTING.md (remove submission guide)
5. ✅ Archive removed feature docs
6. ✅ Update ROADMAP with new priorities

**PR**: "docs: Update documentation for simplified scope"

---

## 🎯 Success Criteria

After simplification, the project should:

1. ✅ **Focus on Core Research**
   - Run tournaments easily
   - Extract individual contributions
   - Evolve optimal heuristics
   - Validate scenarios statistically

2. ✅ **Simple Demo Experience**
   - Pick team + scenario → watch game → see analysis
   - No login, no backend, no complexity
   - Fast loading, works offline

3. ✅ **Easy to Understand**
   - Clear code organization
   - Focused documentation
   - Obvious entry points for researchers

4. ✅ **Easy to Deploy**
   - Static site (GitHub Pages)
   - No database setup
   - No server management

5. ✅ **Research Ready**
   - Run 1000s of games locally
   - Statistical validation built-in
   - Export results for analysis
   - RL training path clear

---

## 📊 Metrics

**Current Complexity** (Before):
- Lines of Code: ~15,000
- Dependencies: 28 Python packages
- Backend Components: PostgreSQL, FastAPI, Alembic
- Deployment: Multi-service (frontend + backend + DB)
- Maintainability: Medium-High complexity

**Target Complexity** (After):
- Lines of Code: ~10,000 (-33%)
- Dependencies: 20 Python packages (-8 backend deps)
- Backend Components: None (static site only)
- Deployment: Single static site (GitHub Pages)
- Maintainability: Low-Medium complexity

---

## 🚀 Future Additions (Post-MVP)

After establishing the clean MVP, consider adding:

1. **Batch Upload Results** - Share statistical summaries
2. **Community Leaderboard** - Based on shared summaries (not agents)
3. **Multi-Game Comparison** - Compare team performance across scenarios
4. **Advanced Evolution** - Genetic algorithms with more sophisticated fitness
5. **Neural Net Showcase** - Demo trained RL policies vs heuristics

But only after the core is solid and well-understood.

---

## ❓ Questions for Discussion

1. **Storage Format**: JSON files or SQLite for statistical summaries?
2. **Web Demo Scope**: Should we allow users to customize agent parameters in browser?
3. **RL Integration**: Keep all PufferLib training scripts or simplify further?
4. **Evolution Tools**: Should GA tools be CLI-only or add to web UI?
5. **Documentation**: Archive removed docs or delete entirely?

---

## 📝 Notes

- This plan is **reversible** - all removed code is in git history
- Focus on **understanding game dynamics** before scaling
- **Simpler = easier to reason about** = better science
- Can always add back features later if needed

---

**Next Steps**: Review this plan, discuss questions, then start Phase 1 implementation.
