# 📋 Bucket Brigade Simplification - Executive Summary

**Date**: 2025-11-03
**Status**: ✅ Planning Complete - Ready for Implementation
**Goal**: Focus on core research MVP

---

## 🎯 What We're Doing

**Removing complexity** to focus on what matters:
1. Tournament engine for diverse teams × scenarios
2. Statistical analysis (1000+ games per config)
3. Heuristic agent evolution
4. Simple web demo (single-game visualization)
5. RL training path (PufferLib)

---

## ❌ What's Being Removed

### 1. Agent Submission Infrastructure (~2,500 LOC)
- Security validation & sandboxing
- Agent registry service
- Backend API for submissions
- Community agent database
- Submission CLI tool

**Rationale**: Adds complexity without immediate research value. Focus on curated agents first.

### 2. Database & Backend API (~1,500 LOC)
- PostgreSQL schema
- SQLAlchemy models
- FastAPI REST API
- Alembic migrations
- 8 backend dependencies

**Rationale**: Games are fast (~5ms). Store statistical summaries, not individual replays. No backend needed.

###  3. Complex Web Features (~1,000 LOC)
- Multi-tournament dashboard
- Global rankings page
- Complex team builder UI
- Tournament batch runner

**Rationale**: Single-game demo is clearer and more educational. Multi-game analysis belongs in research CLI.

**Total Reduction**: ~5,000 LOC removed (~33% of codebase)

---

## ✅ What We're Keeping & Enhancing

### Core Engine ✅
- `bucket_brigade_env.py` - Pure Python game logic
- `bucket-brigade-core/` - Rust engine (10-20x faster)
- `scenarios.py` - 10 test scenarios
- `puffer_env.py` - RL training wrapper

### Agent System ✅
- `agent_base.py` - Base class
- `heuristic_agent.py` - 10-parameter agents
- `scenario_optimal/` - Expert agents (6-7 types)

### Evolution & Optimization ✅
- Genetic algorithm for heuristic tuning
- Population management
- Fitness functions
- CLI evolution tools

### RL Training ✅
- PPO training scripts
- Policy evaluation
- Curriculum learning
- PufferLib integration

### Analysis Tools ✅
- Batch game runners
- Scenario validation
- Team comparison
- Statistical ranking

### Web Demo ✅ (Simplified)
- Single-game visualization
- Team + scenario picker
- Replay controls
- Post-game analysis

---

## 🏗️ New Architecture

### Data Flow

```
Browser (Static Site)
  ├─→ Dashboard: Pick team + scenario
  ├─→ Run game (50ms in browser)
  ├─→ Visualize with controls
  └─→ Show analysis

Python CLI (Research)
  ├─→ Run 1000 games (~5 seconds with Rust)
  ├─→ Generate statistical summaries
  ├─→ Export to JSON files
  └─→ Analyze with scripts
```

### Storage Strategy

**Don't Store**: Individual game replays (generate on-demand)

**Do Store**: Statistical summaries
```json
{
  "team": ["firefighter", "coordinator", "hero"],
  "scenario": "early_containment",
  "num_runs": 1000,
  "statistics": {
    "mean_team_reward": 241.6,
    "std_team_reward": 45.2,
    "win_rate": 0.73,
    "houses_saved_avg": 7.2
  },
  "agent_contributions": {
    "firefighter": 0.35,
    "coordinator": 0.28,
    "hero": 0.37
  }
}
```

Stored in: `results/summaries/*.json`

---

## 📋 Implementation Plan

### Phase 1: Remove Infrastructure (1-2 days)

**Files to Delete**:
```
bucket_brigade/agents/agent_loader.py
bucket_brigade/agents/agent_template.py
bucket_brigade/services/
bucket_brigade/db/
scripts/submit_agent.py
web/public/data/known-good-agents.json
docs/AGENT_SUBMISSION_GUIDE.md
docs/AGENT_REGISTRY_API.md
docs/DATABASE_SETUP.md
```

**Dependencies to Remove** (pyproject.toml):
```python
sqlalchemy, psycopg2-binary, alembic,
fastapi, uvicorn, python-multipart
```

**PR**: `refactor: Remove agent submission and backend infrastructure`

---

### Phase 2: Simplify Web UI (2-3 days)

**Create**:
- New simplified Dashboard (team + scenario picker)
- GameAnalysis component (post-game insights)
- Custom team modal (simple agent picker)

**Remove**:
- Tournament.tsx
- Rankings.tsx
- Complex TeamBuilder.tsx
- TournamentRunner.tsx
- TournamentResults.tsx

**Simplify**:
- Settings.tsx (basic preferences only)
- App.tsx routing (3 pages total)

**PR**: `refactor: Simplify web UI to single-game demo`

---

### Phase 3: Statistical Tools (2-3 days)

**Create**:
- Batch runner for 1000+ game experiments
- Statistical summary generator
- JSON export utilities
- Analysis visualization scripts
- Confidence interval calculations

**Enhance**:
- `scripts/run_batch.py` - Add summary generation
- `scripts/analyze_rankings.py` - Read from summaries
- New: `scripts/generate_summary.py`

**PR**: `feat: Add statistical validation tools`

---

### Phase 4: Documentation Update (1 day)

**Update**:
- README.md - New focus and scope
- API.md - Remove backend endpoints
- DEPLOYMENT.md - Static site only
- CONTRIBUTING.md - Remove submission guide
- TRAINING_GUIDE.md - Keep, update links

**Create**:
- SIMPLIFICATION_PLAN.md ✅ (done)
- SIMPLIFIED_ARCHITECTURE.md ✅ (done)
- WEB_UI_MOCKUP.md ✅ (done)
- SIMPLIFICATION_SUMMARY.md ✅ (this file)

**Archive**:
- Move removed feature docs to `docs/archive/`

**PR**: `docs: Update documentation for simplified scope`

---

## 📊 Expected Outcomes

### Before Simplification
- **Complexity**: High
- **LOC**: ~15,000
- **Dependencies**: 28 packages
- **Backend**: PostgreSQL + FastAPI
- **Deployment**: Multi-service
- **Focus**: Agent submissions & rankings

### After Simplification
- **Complexity**: Medium-Low
- **LOC**: ~10,000 (-33%)
- **Dependencies**: 20 packages (-8)
- **Backend**: None (static site)
- **Deployment**: GitHub Pages
- **Focus**: Research & understanding

### Research Capabilities
✅ Run thousands of games easily
✅ Extract individual contributions
✅ Evolve optimal heuristics
✅ Validate scenarios statistically
✅ Train RL policies with PufferLib
✅ Simple, clear web demo

---

## 🎯 Success Metrics

### Technical
- [ ] No backend dependencies in package.json
- [ ] Static site deploys to GitHub Pages
- [ ] All tests pass after refactor
- [ ] Web demo loads in <1 second
- [ ] Games run in <100ms (browser)

### Research
- [ ] Can run 1000-game experiments easily
- [ ] Statistical summaries export to JSON
- [ ] Ranking algorithms produce clear results
- [ ] Scenario validation is reproducible
- [ ] RL training path is clear

### User Experience
- [ ] Dashboard is immediately understandable
- [ ] Game visualization is smooth
- [ ] Post-game analysis provides insights
- [ ] No confusing features or complexity
- [ ] Documentation is focused and clear

---

## ❓ Open Questions (for Discussion)

1. **Storage Format**
   - Option A: JSON files in `results/summaries/`
   - Option B: Single SQLite file for querying
   - **Recommendation**: Start with JSON, add SQLite if needed

2. **Web Demo Scope**
   - Should users be able to customize agent parameters in browser?
   - **Recommendation**: No, keeps it simple. Use presets only.

3. **RL Integration**
   - Keep all PufferLib training scripts?
   - **Recommendation**: Yes, core research capability

4. **Evolution Tools**
   - CLI-only or add to web UI?
   - **Recommendation**: CLI-only for now

5. **Documentation**
   - Archive or delete removed docs?
   - **Recommendation**: Archive to `docs/archive/`

---

## 📚 Related Documents

- **[SIMPLIFICATION_PLAN.md](./SIMPLIFICATION_PLAN.md)** - Detailed removal plan
- **[SIMPLIFIED_ARCHITECTURE.md](./SIMPLIFIED_ARCHITECTURE.md)** - New architecture design
- **[WEB_UI_MOCKUP.md](./WEB_UI_MOCKUP.md)** - Web interface design

---

## 🚀 Next Steps

1. **Review Documents** - Read all 4 planning docs
2. **Discuss Questions** - Resolve open questions
3. **Start Phase 1** - Begin removing infrastructure
4. **Test After Each Phase** - Ensure nothing breaks
5. **Update as We Go** - Refine plan based on learnings

---

## 💡 Key Insights

### Why This Simplification?

1. **Fast Simulation** - Games are so fast (5ms) that backend storage is unnecessary
2. **Research Focus** - Need to understand dynamics before scaling
3. **Deployment Simplicity** - Static site is trivial to deploy and maintain
4. **Code Clarity** - Less code = easier to reason about = better science
5. **Iteration Speed** - Simpler codebase = faster experiments

### What This Enables

- **Quick Experiments** - Run 1000 games in 5 seconds
- **Clear Understanding** - See exactly what's happening
- **Easy Sharing** - GitHub Pages demo, no setup needed
- **Solid Foundation** - Build on clean, simple base
- **Future Growth** - Can add back features if truly needed

---

**Status**: ✅ Planning complete, ready to execute
**Estimated Timeline**: 6-9 days for full implementation
**Risk Level**: Low (all changes are reversible via git)
