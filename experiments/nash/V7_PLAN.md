# Nash Equilibrium V7 Plan: Correct-Mechanics Analysis

**Created**: 2025-11-07
**Status**: 🚀 Ready to Execute
**Priority**: 🔴 CRITICAL - Blocks Phase 2.5 analysis

---

## Executive Summary

Nash V5 computed equilibria using **V5 evolved agents trained with incorrect game mechanics**. V7 evolution has completed with **correct mechanics** for all 12 scenarios. Nash V7 will recompute equilibria using V7 agents to answer the fundamental question:

**Does the Nash vs Evolution paradox persist when both use correct game mechanics?**

### The Paradox (from Phase 2.5 analysis)

**Nash V5 Result**: Hero strategy dominates (8/9 scenarios)
- Pure prosocial strategy: `{honesty: 1.0, work: 1.0, neighbor_help: 1.0}`
- Homogeneous population assumption

**V7 Evolution Result**: Diverse strategies with low work tendency
- Mean work_tendency: 0.06 (range: 0.00-0.21)
- Heterogeneous tournament fitness
- Context-dependent cooperation

**Core Question**: Is this difference due to:
1. **Evaluation context** (homogeneous vs heterogeneous)?
2. **Fitness function** (self-play vs tournament)?
3. **Strategy space coverage** (archetypes vs continuous parameters)?

---

## Motivation: Why Nash V7 is Critical

### 1. Nash V5 is Invalidated

Nash V5 used **V5 agents evolved with wrong mechanics**:
- ❌ Wrong fire extinguishing formula (exponential vs independent probabilities)
- ❌ Wrong spontaneous ignition (stopped after N_spark vs continuous)
- ❌ Inconsistent Python/Rust implementations

**All Nash V5 results must be recomputed with correct-mechanics agents.**

### 2. Phase 2.5 Analysis is Blocked

From `2025-11-08_phase_2_5_analysis_plan.md`:

> **Priority 1A: Zero-Compute Analyses**
> - Extract Hero parameters from Nash V5 results
> - Compare to v7 evolved agents side-by-side
> - **BLOCKED**: Nash V5 used wrong-mechanics agents, comparison invalid

We cannot proceed with Phase 2.5 analysis until we have Nash equilibria computed with V7 agents.

### 3. Critical Research Question

**Research Question**: Why does game theory predict Hero while evolution discovers diverse strategies?

**Two competing hypotheses**:

**Hypothesis A: Evaluation Context Matters**
- Nash: Homogeneous teams (all agents play same strategy)
- Evolution: Heterogeneous tournaments (mixed agent types)
- **Prediction**: Nash V7 with heterogeneous fitness will match evolution

**Hypothesis B: Fitness Function Shapes Discovery**
- Nash: Maximize payoff against same strategy (self-play)
- Evolution: Maximize payoff against diverse opponents (tournament)
- **Prediction**: Nash V7 with heterogeneous support will find diverse equilibria

---

## Nash V7 Experimental Design

### Core Innovation: Dual Analysis Approach

Run **both** Nash computation methods to understand the discrepancy:

#### Method 1: Traditional Nash (Homogeneous)
**Setup**: Compute Nash equilibrium for homogeneous teams
- All 4 agents play the same strategy
- Standard symmetric game assumption
- Best response: What strategy should I play if my 3 teammates play the same strategy?

**Strategy Pool**:
- 4 archetypes (firefighter, hero, free_rider, coordinator)
- 12 V7 evolved agents (one per scenario)
- **Total**: 16 strategies

**Algorithm**: Double Oracle with Rust evaluator

#### Method 2: Heterogeneous Nash (NEW)
**Setup**: Compute Nash equilibrium for heterogeneous teams
- Each agent can play a different strategy
- Asymmetric game with 4 player positions
- Best response: What strategy should agent_i play given the other 3 agents' strategies?

**Strategy Pool**: Same 16 strategies per position

**Algorithm**: Extended Double Oracle for asymmetric games

### Why Both Methods?

**If homogeneous Nash ≈ V7 evolved**:
→ Evolution found the Nash equilibrium for self-play
→ Validates evolutionary approach

**If heterogeneous Nash ≈ V7 evolved**:
→ Evolution optimizes for mixed teams, not self-play
→ Confirms hypothesis that fitness function shapes discovery

**If neither matches V7**:
→ Evolution discovers non-equilibrium strategies
→ Suggests evolutionary dynamics differ fundamentally from Nash

---

## Implementation Plan

### Phase 1: Homogeneous Nash V7 (4-6 hours compute)

**Script**: `experiments/scripts/compute_nash_v2.py` (already exists, just need V7 agents)

**For each of 9 core scenarios**:

```bash
uv run python experiments/scripts/compute_nash_v2.py \
  {scenario} \
  --evolved-versions v7 \
  --simulations 2000 \
  --max-iterations 20 \
  --output-dir experiments/nash/v2_results_v7/{scenario} \
  --seed 42
```

**Strategy pool per scenario**:
- 4 archetypes (firefighter, hero, free_rider, coordinator)
- 1 V7 evolved agent (from that scenario)
- **Total**: 5 strategies

**Output**: `experiments/nash/v2_results_v7/{scenario}/equilibrium_v2.json`

**Expected Runtime**: ~30-40 min per scenario × 9 scenarios = 4.5-6 hours

### Phase 2: Heterogeneous Nash V7 (8-12 hours compute)

**Script**: NEW - `experiments/scripts/compute_nash_heterogeneous.py`

**Key differences from homogeneous**:
- 4 separate strategy pools (one per agent position)
- Best-response computed per position
- Support can include different strategies per agent

**For each of 9 core scenarios**:

```bash
uv run python experiments/scripts/compute_nash_heterogeneous.py \
  {scenario} \
  --evolved-versions v7 \
  --simulations 2000 \
  --max-iterations 30 \
  --output-dir experiments/nash/heterogeneous_v7/{scenario} \
  --seed 42
```

**Strategy pool** (same for all 4 positions):
- 4 archetypes + 1 V7 evolved = 5 strategies per position
- Total game space: 5^4 = 625 possible team compositions

**Output**: `experiments/nash/heterogeneous_v7/{scenario}/equilibrium.json`

**Expected Runtime**: ~50-80 min per scenario × 9 scenarios = 8-12 hours

---

## Overnight Execution Plan

### Option A: Homogeneous Only (Conservative)

**What**: Run Nash V7 with homogeneous teams (Method 1 only)
**Where**: `rwalters-sandbox-1` (CPU server)
**Runtime**: 4-6 hours
**Deliverable**: Comparable results to Nash V5, but with correct mechanics

```bash
# SSH to remote
ssh rwalters-sandbox-1

# Create launch script
cat > ~/bucket-brigade/scripts/launch_nash_v7_homogeneous.sh <<'SCRIPT'
#!/bin/bash
cd ~/bucket-brigade

scenarios=(
  "chain_reaction" "deceptive_calm" "early_containment"
  "greedy_neighbor" "mixed_motivation" "overcrowding"
  "rest_trap" "sparse_heroics" "trivial_cooperation"
)

for scenario in "${scenarios[@]}"; do
  echo "=== Nash V7: $scenario ==="

  tmux new-session -d -s "nash_v7_$scenario" \
    "cd ~/bucket-brigade && \
     uv run python experiments/scripts/compute_nash_v2.py $scenario \
       --evolved-versions v7 \
       --simulations 2000 \
       --max-iterations 20 \
       --output-dir experiments/nash/v2_results_v7/$scenario \
       --seed 42 \
       2>&1 | tee logs/nash_v7_$scenario.log; \
     echo 'Press any key to close'; read"

  echo "Launched tmux session: nash_v7_$scenario"
  sleep 2
done

echo ""
echo "All Nash V7 sessions launched!"
echo "Monitor with: tmux ls"
echo "Attach to session: tmux attach -t nash_v7_chain_reaction"
SCRIPT

chmod +x scripts/launch_nash_v7_homogeneous.sh

# Launch
./scripts/launch_nash_v7_homogeneous.sh
```

### Option B: Both Methods (Comprehensive)

**What**: Run both homogeneous AND heterogeneous Nash
**Where**: `rwalters-sandbox-1` (CPU server, has 32 cores)
**Runtime**: 12-18 hours (can run in parallel)
**Deliverable**: Complete analysis of both evaluation contexts

```bash
# Launch homogeneous (9 sessions)
./scripts/launch_nash_v7_homogeneous.sh

# THEN launch heterogeneous (9 more sessions)
# Wait 30 min to avoid resource contention, or run on separate cores
./scripts/launch_nash_v7_heterogeneous.sh
```

**Resource Management**:
- 32 cores available
- Each Nash run uses ~2-4 cores
- Can run 9 homogeneous + 9 heterogeneous concurrently if careful
- Or run sequentially: homogeneous first (4-6 hours), then heterogeneous (8-12 hours)

### Option C: Staged Approach (Recommended)

**Tonight**: Homogeneous Nash V7 only (Option A)
- **Why**: Faster, lower risk, directly comparable to Nash V5
- **Runtime**: 4-6 hours
- **Outcome**: Can start Phase 2.5 analysis tomorrow

**Tomorrow Night**: Heterogeneous Nash V7 (after reviewing homogeneous results)
- **Why**: More complex, needs validation of approach
- **Runtime**: 8-12 hours
- **Outcome**: Complete understanding of Nash vs Evolution discrepancy

---

## Expected Outcomes & Interpretation

### Scenario 1: Homogeneous Nash V7 = Hero (like V5)

**Result**: Nash still predicts Hero equilibrium for 8/9 scenarios

**Interpretation**:
- Mechanics fix didn't change equilibrium structure
- **Confirmed**: Homogeneous self-play favors prosocial strategies
- **Next**: Run heterogeneous Nash to test if mixed teams change equilibria

### Scenario 2: Homogeneous Nash V7 = V7 Evolved

**Result**: Nash finds same diverse strategies as evolution

**Interpretation**:
- Nash V5 was wrong due to incorrect mechanics
- **Confirmed**: Evolution found true Nash equilibria
- **Validates**: Evolutionary approach for this game
- **Phase 1 complete**: Evolution = Nash for symmetric games

### Scenario 3: Homogeneous Nash V7 ≠ V7 Evolved, but Heterogeneous Nash = V7 Evolved

**Result**: Homogeneous favors Hero, heterogeneous matches evolution

**Interpretation**:
- **Critical finding**: Fitness function (homogeneous vs heterogeneous) determines optimal strategy
- Evolution optimizes for heterogeneous tournament, not self-play
- **Explains paradox**: Nash and Evolution solving different optimization problems
- **Insight**: Real-world teams are heterogeneous, evolution is more realistic

### Scenario 4: Neither Nash method matches V7 Evolved

**Result**: Both homogeneous and heterogeneous Nash differ from evolution

**Interpretation**:
- Evolution discovers non-equilibrium strategies
- Possible reasons:
  - Finite population effects
  - Selection pressure shapes trajectories
  - Local optima in evolutionary search
- **Next**: Analyze epsilon-equilibrium distance
- **Question**: Are V7 agents exploitable?

---

## Analysis Pipeline (After Completion)

### Step 1: Compare Nash V7 to Nash V5

**Script**: `experiments/scripts/compare_nash_versions.py`

```bash
uv run python experiments/scripts/compare_nash_versions.py \
  --v5-dir experiments/nash/v2_results_v5/ \
  --v7-dir experiments/nash/v2_results_v7/ \
  --output experiments/nash/V5_VS_V7_COMPARISON.md
```

**Questions**:
1. Did equilibrium types change? (pure vs mixed)
2. Did equilibrium strategies change? (Hero vs others)
3. Did payoffs change significantly?
4. Which scenarios most affected by mechanics fix?

### Step 2: Compare Nash V7 to V7 Evolution

**Script**: `experiments/scripts/compare_nash_vs_evolution.py`

```bash
uv run python experiments/scripts/compare_nash_vs_evolution.py \
  --nash-dir experiments/nash/v2_results_v7/ \
  --evolved-dir experiments/scenarios/*/evolved_v7/ \
  --output experiments/nash/NASH_VS_EVOLUTION_V7.md
```

**Metrics**:
- **Parameter distance**: L2 norm between Nash and evolved genomes
- **Behavioral similarity**: Action distribution overlap
- **Performance gap**: Nash payoff vs evolved fitness
- **Robustness**: Test Nash strategy in heterogeneous tournaments

### Step 3: Heterogeneous vs Homogeneous Nash

**Script**: `experiments/scripts/compare_nash_methods.py`

```bash
uv run python experiments/scripts/compare_nash_methods.py \
  --homogeneous experiments/nash/v2_results_v7/ \
  --heterogeneous experiments/nash/heterogeneous_v7/ \
  --output experiments/nash/HOMOGENEOUS_VS_HETEROGENEOUS.md
```

**Questions**:
1. Do heterogeneous equilibria include role specialization?
2. Are heterogeneous equilibria closer to V7 evolution?
3. What's the payoff difference between the two methods?
4. Can we predict which scenarios benefit from heterogeneity?

---

## Success Criteria

### Must Have (Critical Path)

1. ✅ **Nash V7 homogeneous complete** for all 9 scenarios
   - Using correct-mechanics V7 agents
   - Comparable to Nash V5 structure
   - Documented differences from V5

2. ✅ **Phase 2.5 unblocked**
   - Can extract Nash V7 parameters
   - Can compare to V7 evolution
   - Can proceed with Priority 1 analysis

3. ✅ **Paradox explained**
   - Clear understanding of Nash vs Evolution discrepancy
   - Evidence for Hypothesis A or B (or alternative)
   - Documented in research notebook

### Should Have (High Value)

4. ✅ **Heterogeneous Nash V7 complete** for key scenarios (3-5)
   - Tests hypothesis about evaluation context
   - Provides alternative equilibrium concept
   - Informs future experimental design

5. ✅ **Cross-scenario patterns identified**
   - Which scenarios favor cooperation vs selfishness?
   - Which scenarios have pure vs mixed equilibria?
   - Can we predict equilibrium type from scenario parameters?

### Nice to Have (Future Work)

6. ⚠️ **Epsilon-equilibrium analysis**
   - How close are V7 agents to Nash?
   - Are they exploitable?
   - Robustness metrics

7. ⚠️ **Mechanism design insights**
   - How to shift equilibria toward cooperation?
   - Parameter sensitivity analysis
   - Inform Phase 3 work

---

## Risk Mitigation

### Risk 1: Heterogeneous Nash doesn't converge

**Likelihood**: Medium (new algorithm, complex game space)
**Impact**: High (blocks hypothesis testing)

**Mitigation**:
- Start with homogeneous Nash (proven algorithm)
- Test heterogeneous on single scenario first
- Use warm-start from homogeneous equilibrium
- Increase simulation budget if needed (2000 → 5000)
- Fallback: Analyze homogeneous Nash only, defer heterogeneous to future

### Risk 2: Double Oracle too slow with V7 agents

**Likelihood**: Low (Rust evaluator is fast)
**Impact**: Medium (delays results)

**Mitigation**:
- Parallelize across scenarios (9 concurrent sessions)
- Use remote server with 32 cores
- Reduce strategy pool if needed (drop some archetypes)
- Cache payoff evaluations aggressively

### Risk 3: Results still don't explain paradox

**Likelihood**: Medium (complex system, multiple factors)
**Impact**: Low (negative result is still informative)

**Mitigation**:
- This is valuable research regardless of outcome
- Document what doesn't explain it (narrows hypothesis space)
- Consider additional factors:
  - Population dynamics
  - Selection pressure
  - Finite-population effects
  - Temporal evolution trajectories
- Design follow-up experiments

---

## Timeline

### Overnight (Option C - Recommended)

**Tonight** (4-6 hours):
- Launch Nash V7 homogeneous on all 9 scenarios
- Monitor first scenario completion (~30 min)
- Verify no errors, let run overnight

**Tomorrow Morning** (1 hour):
- Download results from remote
- Quick validation (check equilibrium files exist)
- Start comparison analysis (Nash V5 vs V7)

**Tomorrow Evening** (8-12 hours):
- If homogeneous results look good, launch heterogeneous
- OR proceed with Phase 2.5 analysis if homogeneous sufficient

### Full Week (Conservative)

**Day 1-2**: Nash V7 homogeneous
- Launch overnight run
- Analyze results
- Compare to Nash V5 and V7 evolution
- Write up findings

**Day 3**: Decision point
- If paradox explained → Proceed to Phase 2.5
- If paradox remains → Prepare heterogeneous Nash

**Day 4-5**: Nash V7 heterogeneous (if needed)
- Launch overnight run
- Analyze results
- Compare to homogeneous Nash
- Final synthesis

**Day 6-7**: Documentation
- Update research notebook
- Create visualization
- Publish findings to website
- Plan next experiments

---

## Deliverables

### Immediate (After Overnight Run)

1. **Raw results**: `experiments/nash/v2_results_v7/{scenario}/equilibrium_v2.json`
2. **Logs**: `logs/nash_v7_{scenario}.log`
3. **Quick validation**: Confirm all 9 scenarios completed successfully

### Short-term (1-2 days)

4. **Comparison report**: `experiments/nash/V5_VS_V7_COMPARISON.md`
   - What changed with mechanics fix?
   - Equilibrium structure differences
   - Payoff changes

5. **Nash vs Evolution**: `experiments/nash/NASH_VS_EVOLUTION_V7.md`
   - Parameter comparison tables
   - Hypothesis testing (A vs B)
   - Paradox explanation

6. **Research notebook entry**: `research_notebook/2025-11-08_nash_v7_results.md`
   - Key findings
   - Implications for Phase 2.5
   - Next steps

### Medium-term (3-7 days)

7. **Heterogeneous results** (if applicable): `experiments/nash/heterogeneous_v7/`
8. **Method comparison**: `experiments/nash/HOMOGENEOUS_VS_HETEROGENEOUS.md`
9. **Phase 2.5 readiness**: All data needed for Priority 1-3 analyses

---

## Open Questions (To Answer)

### Before Running

1. Should we include all 12 scenarios or just 9 core scenarios?
   - **Recommendation**: Start with 9 core (matches V5), add easy/hard/default later

2. Should we run heterogeneous Nash tonight or wait?
   - **Recommendation**: Homogeneous first (Option C), heterogeneous tomorrow

3. Do we need epsilon-equilibrium analysis now or later?
   - **Recommendation**: Later (after basic Nash V7 complete)

### After Homogeneous Nash V7

4. Does Nash V7 match Nash V5 (Hero equilibrium)?
5. Does Nash V7 match V7 evolution (diverse strategies)?
6. What explains the remaining gap (if any)?
7. Should we proceed with heterogeneous Nash?

### After Heterogeneous Nash V7

8. Do heterogeneous equilibria include role specialization?
9. Are heterogeneous equilibria closer to evolution?
10. What's the optimal team composition (homogeneous vs heterogeneous)?

---

## Integration with Phase 2.5 Analysis Plan

From `2025-11-08_phase_2_5_analysis_plan.md`, Nash V7 enables:

**Priority 1: Zero-Compute Analyses**
- ✅ Extract Nash V7 equilibrium parameters
- ✅ Compare Nash vs Evolution side-by-side
- ✅ Generate testable hypotheses

**Priority 3: Nash V7 Deep Dive**
- ✅ Are Nash equilibria Hero in heterogeneous tournaments?
- ✅ Would V7 agents beat Nash predictions in mixed games?
- ✅ Are V7 strategies epsilon-Nash equilibria?

**Decision Tree: Phase 2.6 Design**
- **If V7 agents are NOT Nash equilibria** → Compute Nash V7 resolves this
- **If cooperation scenarios cluster strongly** → Nash helps identify clusters
- **If parameter sensitivity reveals critical dimensions** → Nash provides baseline

---

## Key Takeaways

1. **Nash V5 is obsolete** - wrong mechanics invalidate all results
2. **Nash V7 is critical** - blocks Phase 2.5 analysis without it
3. **Two methods** - homogeneous (traditional) vs heterogeneous (novel)
4. **Clear hypothesis test** - does evaluation context explain the paradox?
5. **Overnight-ready** - can launch homogeneous Nash tonight (4-6 hours)
6. **Low risk** - proven algorithm, just need to run it with V7 agents

---

**Status**: 🚀 Ready to Execute
**Priority**: 🔴 CRITICAL
**Next Action**: Launch Nash V7 homogeneous overnight (Option C)
**Owner**: Nash research track
**Dependencies**: V7 evolution complete ✅, Rust evaluator working ✅, Remote server available ✅

---

## Immediate Next Steps

1. **Verify V7 agents accessible on remote server** (5 min)
   ```bash
   ssh rwalters-sandbox-1 "ls -lh ~/bucket-brigade/experiments/scenarios/*/evolved_v7/"
   ```

2. **Test Nash V2 script with V7 agents** (10 min)
   ```bash
   # Test on single scenario first
   ssh rwalters-sandbox-1 "cd ~/bucket-brigade && \
     uv run python experiments/scripts/compute_nash_v2.py trivial_cooperation \
       --evolved-versions v7 --simulations 100 --max-iterations 5"
   ```

3. **Launch overnight run** (5 min)
   ```bash
   ssh rwalters-sandbox-1 "./bucket-brigade/scripts/launch_nash_v7_homogeneous.sh"
   ```

4. **Monitor first completion** (30 min)
   ```bash
   ssh rwalters-sandbox-1 "tmux attach -t nash_v7_trivial_cooperation"
   # Verify output looks reasonable, then detach (Ctrl+B, D)
   ```

5. **Check in tomorrow morning** (15 min)
   ```bash
   # Check all sessions complete
   ssh rwalters-sandbox-1 "ls -lh ~/bucket-brigade/experiments/nash/v2_results_v7/"

   # Download results
   rsync -avz rwalters-sandbox-1:~/bucket-brigade/experiments/nash/v2_results_v7/ \
     experiments/nash/v2_results_v7/
   ```
