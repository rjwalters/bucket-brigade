# Large Files Refactoring Analysis

## 📊 Summary

Found **7 files >500 lines** across the codebase:

| File | Lines | Type | Priority |
|------|-------|------|----------|
| ScenarioResearch.tsx | 1,045 | Web UI | 🔴 High |
| scenarios.py | 809 | Core | 🟡 Medium |
| GameReplay.tsx | 661 | Web UI | 🟡 Medium |
| SimpleDashboard.tsx | 640 | Web UI | 🟡 Medium |
| test_evolution.py | 581 | Tests | 🟢 Low |
| generate_insights.py | 578 | Scripts | 🟢 Low |
| analyze_evolved_experts.py | 504 | Scripts | 🟢 Low |

---

## 🔴 Priority 1: ScenarioResearch.tsx (1,045 lines)

**Location:** `web/src/pages/ScenarioResearch.tsx`

**Current Issues:**
- Single massive component handling multiple concerns
- 8 useState/useEffect hooks in one component
- 4+ nested return statements
- Mixed data loading, transformation, and presentation logic

**Refactoring Strategy:**

### Split into Multiple Components:

1. **ScenarioSelector.tsx** (~50 lines)
   - Dropdown for selecting scenarios
   - Simple controlled component

2. **hooks/useScenarioData.ts** (~100 lines)
   - Custom hook: `useScenarioData(scenarioName)`
   - Handles async loading, error states
   - Data transformation logic

3. **components/research/HeuristicsSection.tsx** (~150 lines)
   - Display heuristic benchmark results
   - Table/chart components

4. **components/research/EvolutionSection.tsx** (~200 lines)
   - Evolution trace visualization
   - Best agent display
   - Info modal logic

5. **components/research/NashSection.tsx** (~150 lines)
   - Nash equilibrium results
   - Payoff matrices

6. **components/research/ComparisonSection.tsx** (~150 lines)
   - Cross-scenario comparison charts
   - Performance metrics

7. **pages/ScenarioResearch.tsx** (~200 lines)
   - Main layout component
   - Compose all sections
   - State orchestration

**Benefits:**
- Each component <200 lines
- Single Responsibility Principle
- Easier testing
- Better code reuse
- Improved performance (component-level memoization)

---

## 🟡 Priority 2: scenarios.py (809 lines)

**Location:** `bucket_brigade/envs/scenarios.py`

**Current Issues:**
- 40 functions/classes in one file
- Mixed concerns: Scenario dataclass + factory functions + utilities
- Hard to navigate and maintain
- Adding new scenarios requires modifying a large file

**Refactoring Strategy:**

### Split into Module Package:

```
bucket_brigade/envs/scenarios/
├── __init__.py           # Public API exports
├── core.py               # Scenario dataclass (~100 lines)
├── factory.py            # Factory functions (~150 lines)
├── definitions/
│   ├── __init__.py
│   ├── baseline.py       # default, easy, hard, etc.
│   ├── research.py       # Phase 2A, 2B, 2C scenarios
│   ├── mechanism.py      # Phase 2D scenarios
│   └── boundary.py       # Phase 2A.1 boundary scenarios
└── utils.py              # Helper functions (~50 lines)
```

**Migration Path:**
1. Create new package structure
2. Move Scenario dataclass to `core.py`
3. Group scenario factory functions into `definitions/` by category
4. Update `__init__.py` to maintain backward compatibility
5. Update imports across codebase
6. Add deprecation warnings for old import paths
7. Remove old file after migration

**Benefits:**
- Scenario definitions grouped by research phase/purpose
- Easy to add new scenario categories
- Clear separation of data vs. logic
- Each file <200 lines
- Better organization for future expansion

---

## 🟡 Priority 3: GameReplay.tsx (661 lines)

**Location:** `web/src/pages/GameReplay.tsx`

**Current Issues:**
- Game replay visualization with timeline controls
- Similar issues to ScenarioResearch but smaller
- Mixed UI and game state logic

**Refactoring Strategy:**

1. **components/replay/TimelineControls.tsx** (~100 lines)
   - Play/pause, speed control, scrubbing
   - Progress bar

2. **components/replay/GameStateVisualizer.tsx** (~150 lines)
   - Houses display
   - Agents positioning
   - Fire visualization

3. **hooks/useReplayData.ts** (~100 lines)
   - Load and parse replay JSON
   - State management for playback
   - Custom hook: `useReplayData(replayId)`

4. **pages/GameReplay.tsx** (~200 lines)
   - Main orchestration
   - Layout and routing
   - Playback state

**Benefits:**
- Reusable timeline controls
- Testable game visualization
- Clear data loading separation

---

## 🟡 Priority 4: SimpleDashboard.tsx (640 lines)

**Location:** `web/src/pages/SimpleDashboard.tsx`

**Current Issues:**
- Single component handling simulation controls, execution, and results
- Complex state management for running simulations
- Mixed concerns

**Refactoring Strategy:**

1. **components/simulation/SimulationControls.tsx** (~100 lines)
   - Scenario selector
   - Agent configuration
   - Parameter inputs

2. **hooks/useSimulation.ts** (~150 lines)
   - Run button, game loop logic
   - Custom hook: `useSimulation(config)`
   - WASM engine integration

3. **components/simulation/ResultsDisplay.tsx** (~150 lines)
   - Charts and metrics
   - Statistics tables
   - Export functionality

4. **pages/SimpleDashboard.tsx** (~200 lines)
   - Layout orchestration
   - Top-level state coordination

**Benefits:**
- Reusable simulation controls
- Testable simulation logic
- Cleaner results presentation

---

## 🟢 Low Priority Files

### test_evolution.py (581 lines)
**Location:** `tests/test_evolution.py`
**Status:** Test files can be longer - acceptable
**If needed:** Group related tests into separate test classes/files by feature

### generate_insights.py (578 lines)
**Location:** `experiments/scripts/generate_insights.py`
**Status:** Script file - lower priority
**If needed:** Extract helper functions into `insights_utils.py`

### analyze_evolved_experts.py (504 lines)
**Location:** `experiments/scripts/analyze_evolved_experts.py`
**Status:** Script file - lower priority
**If needed:** Extract analysis functions into reusable module

---

## 🎯 Recommended Action Plan

### Phase 1: High-Impact Web Components (Week 1-2)
1. **Refactor ScenarioResearch.tsx** (biggest win)
   - Most complex, highest lines
   - Extract 6-7 focused components
   - Target: ~200 lines each
   - **Estimated effort:** 8-12 hours
   - **Impact:** High - improves maintainability and performance

### Phase 2: Core Architecture (Week 3)
2. **Refactor scenarios.py into module**
   - Improves maintainability
   - Easier to add new scenarios
   - Better organization for research phases
   - **Estimated effort:** 4-6 hours
   - **Impact:** Medium - improves developer experience

### Phase 3: Additional UI Components (Week 4-5)
3. **Refactor GameReplay.tsx and SimpleDashboard.tsx**
   - Apply patterns from ScenarioResearch refactor
   - Consistent component structure
   - **Estimated effort:** 6-8 hours each
   - **Impact:** Medium - consistency across codebase

---

## 📐 General Refactoring Principles

### For React Components:
- **Target:** <200 lines per component
- **Extract:** Data hooks, sections, modals, controls
- **Pattern:** Container/Presentational split
- **Hooks:** Custom hooks for data fetching and complex state
- **Testing:** Each extracted component should be unit testable
- **Performance:** Use React.memo for expensive components

### For Python Modules:
- **Target:** <300 lines per file
- **Extract:** Related functions into focused modules
- **Pattern:** Clear separation of data/logic/utils
- **Structure:** Use packages with `__init__.py` for public API
- **Imports:** Maintain backward compatibility during migration
- **Documentation:** Update module docstrings

### General Best Practices:
- **One concern per file:** Single Responsibility Principle
- **Vertical organization:** Related code stays together
- **Shallow nesting:** Avoid deeply nested structures
- **Clear naming:** File names should indicate purpose
- **Documentation:** Update README/docs when restructuring

---

## 📝 Notes

- **Backward Compatibility:** All refactors should maintain existing API contracts
- **Testing:** Add/update tests during refactoring
- **Incremental:** Each phase can be done independently
- **PR Size:** Keep PRs focused (1-2 files per PR maximum)
- **Documentation:** Update component/module docs as you refactor

---

*Last updated: 2025-11-06*
*Analysis generated using file line counts and structural review*
