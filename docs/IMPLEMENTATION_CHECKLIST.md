# Bucket Brigade Simplification - Implementation Checklist

**See also**:
- [SIMPLIFICATION_SUMMARY.md](./SIMPLIFICATION_SUMMARY.md) - Executive summary
- [SIMPLIFICATION_PLAN.md](./SIMPLIFICATION_PLAN.md) - Detailed plan
- [SIMPLIFIED_ARCHITECTURE.md](./SIMPLIFIED_ARCHITECTURE.md) - Architecture design
- [WEB_UI_MOCKUP.md](./WEB_UI_MOCKUP.md) - UI design

---

## Phase 1: Remove Infrastructure

### Delete Files
- [ ] `bucket_brigade/agents/agent_loader.py`
- [ ] `bucket_brigade/agents/agent_template.py`
- [ ] `bucket_brigade/services/agent_registry.py`
- [ ] `bucket_brigade/services/api.py`
- [ ] `bucket_brigade/services/job_queue.py`
- [ ] `bucket_brigade/services/__init__.py`
- [ ] `bucket_brigade/db/models.py`
- [ ] `bucket_brigade/db/connection.py`
- [ ] `bucket_brigade/db/migrations/init_db.py`
- [ ] `bucket_brigade/db/migrations/migrate_from_sqlite.py`
- [ ] `bucket_brigade/db/__init__.py`
- [ ] `scripts/submit_agent.py`
- [ ] `web/public/data/known-good-agents.json`
- [ ] `docs/AGENT_SUBMISSION_GUIDE.md`
- [ ] `docs/AGENT_REGISTRY_API.md`
- [ ] `docs/DATABASE_SETUP.md`

### Update Dependencies
- [ ] Remove from `pyproject.toml`: sqlalchemy, psycopg2-binary, alembic
- [ ] Remove from `pyproject.toml`: fastapi, uvicorn, python-multipart
- [ ] Run `uv lock` to update lock file
- [ ] Test that package still installs: `uv pip install -e .`

### Fix Imports
- [ ] Search codebase for imports from removed modules
- [ ] Update `bucket_brigade/agents/__init__.py` (remove loader exports)
- [ ] Check all test files for broken imports
- [ ] Run test suite: `uv run pytest`

### Git Cleanup
- [ ] Stage all deletions
- [ ] Create commit: "refactor: Remove agent submission and backend infrastructure"
- [ ] Create PR with description from plan

---

## Phase 2: Simplify Web UI

### Delete Web Files
- [ ] `web/src/pages/Tournament.tsx`
- [ ] `web/src/pages/Rankings.tsx`
- [ ] `web/src/components/team-builder/TournamentRunner.tsx`
- [ ] `web/src/components/team-builder/TournamentResults.tsx`
- [ ] `web/src/utils/tournamentEngine.ts` (if not used elsewhere)

### Create New Components
- [ ] Create `web/src/pages/SimpleDashboard.tsx`
  - Team selector with presets
  - Scenario selector with descriptions
  - "Run Game" button
- [ ] Create `web/src/components/GameAnalysis.tsx`
  - Final scores
  - Individual contributions
  - Strategy summaries
  - Key insights
- [ ] Create `web/src/components/CustomTeamModal.tsx`
  - Simple agent picker for 4 slots
  - Agent descriptions

### Update Existing Components
- [ ] Simplify `web/src/pages/Settings.tsx`
  - Remove complex options
  - Keep theme, speed, defaults
- [ ] Update `web/src/App.tsx`
  - Remove Tournament/Rankings routes
  - Update to 3 pages only: Dashboard, GameReplay, Settings
- [ ] Update `web/src/pages/GameReplay.tsx`
  - Add conditional GameAnalysis at bottom
  - Add "Back to Dashboard" button

### Update Storage
- [ ] Update `web/src/utils/storage.ts`
  - Remove tournament/ranking storage
  - Keep recent game replays (max 10)
  - Keep user preferences

### Test Web UI
- [ ] Dashboard loads and shows team/scenario pickers
- [ ] Can select preset teams
- [ ] Can open custom team modal
- [ ] "Run Game" navigates to replay
- [ ] Game visualization still works
- [ ] Post-game analysis appears when done
- [ ] Settings page works
- [ ] All routing works

### Git Cleanup
- [ ] Stage all changes
- [ ] Create commit: "refactor: Simplify web UI to single-game demo"
- [ ] Create PR with screenshots

---

## Phase 3: Statistical Analysis Tools

### Create Summary Generator
- [ ] Create `bucket_brigade/orchestration/summary.py`
  - `generate_statistical_summary()` function
  - Takes: team, scenario, replays
  - Returns: StatisticalSummary dict
  - Computes: mean, std, CI, contributions

### Update Batch Runner
- [ ] Update `scripts/run_batch.py`
  - Add `--generate-summary` flag
  - Add `--output-dir` for results/summaries/
  - Run N games and aggregate statistics
  - Export JSON summary file

### Create Analysis Script
- [ ] Create `scripts/analyze_summaries.py`
  - Load all summaries from directory
  - Compare teams across scenarios
  - Generate plots (matplotlib)
  - Export CSV for further analysis

### Add Statistical Utilities
- [ ] Add `bucket_brigade/utils/statistics.py`
  - Confidence interval calculation
  - Shapley value estimation
  - Performance ranking
  - Significance testing

### Test Statistical Tools
- [ ] Run batch with 100 games, check summary format
- [ ] Run batch with 1000 games, check performance (<10s)
- [ ] Verify JSON summaries are valid
- [ ] Verify analysis script produces correct results
- [ ] Test with different teams/scenarios

### Documentation
- [ ] Add docstrings to all new functions
- [ ] Create usage examples in docstrings
- [ ] Update TRAINING_GUIDE with statistical validation section

### Git Cleanup
- [ ] Stage all changes
- [ ] Create commit: "feat: Add statistical validation tools"
- [ ] Create PR with example outputs

---

## Phase 4: Documentation Update

### Update Main Docs
- [ ] Update `README.md`
  - New focus: tournaments, stats, evolution
  - Remove agent submission sections
  - Update quickstart commands
  - Update roadmap
- [ ] Update `API.md`
  - Remove backend API endpoints
  - Keep data structures
  - Add StatisticalSummary format
  - Update to reflect client-only architecture
- [ ] Update `DEPLOYMENT.md`
  - Remove backend deployment sections
  - Keep static site deployment only
  - Update to GitHub Pages focus
- [ ] Update `CONTRIBUTING.md`
  - Remove agent submission guidelines
  - Update development workflow
  - Keep RL training contribution path

### Archive Removed Docs
- [ ] Create `docs/archive/` directory
- [ ] Move `AGENT_SUBMISSION_GUIDE.md` to archive
- [ ] Move `AGENT_REGISTRY_API.md` to archive
- [ ] Move `DATABASE_SETUP.md` to archive
- [ ] Add `docs/archive/README.md` explaining archived docs

### Create New Docs Index
- [ ] Update `docs/README.md`
  - List all active documentation
  - Remove references to removed docs
  - Add links to new planning docs

### Add Planning Docs to Repo
- [x] `docs/SIMPLIFICATION_PLAN.md` ✅
- [x] `docs/SIMPLIFIED_ARCHITECTURE.md` ✅
- [x] `docs/WEB_UI_MOCKUP.md` ✅
- [x] `docs/SIMPLIFICATION_SUMMARY.md` ✅
- [x] `docs/IMPLEMENTATION_CHECKLIST.md` ✅ (this file)

### Git Cleanup
- [ ] Stage all changes
- [ ] Create commit: "docs: Update documentation for simplified scope"
- [ ] Create PR

---

## Phase 5: Final Testing & Polish

### Comprehensive Testing
- [ ] Run full Python test suite: `uv run pytest`
- [ ] Run web tests: `pnpm run test`
- [ ] Test RL training still works: `uv run python scripts/train_simple.py --num-steps 1000`
- [ ] Test evolution still works: `uv run python scripts/evolve_agents.py`
- [ ] Test all scripts in `scripts/` directory
- [ ] Manual web UI testing (all flows)

### Performance Validation
- [ ] Single game runs in <100ms (browser)
- [ ] 1000 games run in <10s (Python + Rust)
- [ ] Web demo loads in <1s
- [ ] No console errors in browser
- [ ] No memory leaks in browser

### Documentation Review
- [ ] All links in docs are valid
- [ ] README quickstart actually works
- [ ] API.md matches code
- [ ] TRAINING_GUIDE examples work
- [ ] No references to removed features

### Deploy Test
- [ ] Build web: `pnpm run build`
- [ ] Preview: `pnpm run preview`
- [ ] Deploy to GitHub Pages (or test branch)
- [ ] Verify demo works in production

---

## Success Criteria

### Technical
- [ ] All phases completed
- [ ] All tests pass
- [ ] No broken imports
- [ ] No console errors
- [ ] Documentation is accurate
- [ ] Static site deploys successfully

### Research Capabilities
- [ ] Can run 1000-game experiments
- [ ] Statistical summaries export correctly
- [ ] Ranking algorithms work
- [ ] Scenario validation is reproducible
- [ ] RL training path is clear

### User Experience
- [ ] Dashboard is clear and simple
- [ ] Game visualization is smooth
- [ ] Post-game analysis provides value
- [ ] No confusing complexity
- [ ] Documentation is focused

---

## Estimated Timeline

- **Phase 1**: 1-2 days (infrastructure removal)
- **Phase 2**: 2-3 days (web UI simplification)
- **Phase 3**: 2-3 days (statistical tools)
- **Phase 4**: 1 day (documentation)
- **Phase 5**: 1 day (testing & polish)

**Total**: 7-10 days

---

## Risk Mitigation

1. **All changes in git** - Can revert any time
2. **One PR per phase** - Easier to review and test
3. **Tests after each phase** - Catch breakage early
4. **Branch protection** - Require reviews before merge
5. **Incremental deployment** - Test on staging first

---

## GitHub Issues Template

Create these 5 issues (one per phase):

### Issue #1: Remove Infrastructure
```
Title: refactor: Remove agent submission and backend infrastructure
Labels: refactor, simplification, phase-1
Milestone: Simplification MVP

See Phase 1 in IMPLEMENTATION_CHECKLIST.md for detailed tasks.

- Delete agent submission pipeline
- Remove database & backend API
- Remove related dependencies
- Fix broken imports
```

### Issue #2: Simplify Web UI
```
Title: refactor: Simplify web UI to single-game demo
Labels: refactor, simplification, phase-2, frontend
Milestone: Simplification MVP

See Phase 2 in IMPLEMENTATION_CHECKLIST.md for detailed tasks.

- Remove Tournament/Rankings pages
- Create simplified Dashboard
- Add GameAnalysis component
- Simplify Settings
```

### Issue #3: Statistical Tools
```
Title: feat: Add statistical validation tools
Labels: feature, simplification, phase-3
Milestone: Simplification MVP

See Phase 3 in IMPLEMENTATION_CHECKLIST.md for detailed tasks.

- Create summary generator
- Update batch runner
- Add analysis scripts
- Statistical utilities
```

### Issue #4: Update Documentation
```
Title: docs: Update documentation for simplified scope
Labels: documentation, simplification, phase-4
Milestone: Simplification MVP

See Phase 4 in IMPLEMENTATION_CHECKLIST.md for detailed tasks.

- Update main docs (README, API, DEPLOYMENT, CONTRIBUTING)
- Archive removed feature docs
- Add planning docs
```

### Issue #5: Testing & Polish
```
Title: test: Final testing and polish for simplified MVP
Labels: testing, simplification, phase-5
Milestone: Simplification MVP

See Phase 5 in IMPLEMENTATION_CHECKLIST.md for detailed tasks.

- Comprehensive testing
- Performance validation
- Documentation review
- Deploy test
```
