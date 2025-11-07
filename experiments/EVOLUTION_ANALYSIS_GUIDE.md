# Evolution Analysis Guide

**Last Updated**: 2025-11-07
**Phase**: Phase 1 - Learning to Train Agents That Perform Well

## Overview

This document describes our methodology for analyzing evolution experiment results in the Bucket Brigade research project. We are currently in **Phase 1** of our research program, focused on learning how to train agents that perform well through evolutionary optimization.

## Research Context

### Current Phase: Learning Effective Training

**Goal**: Discover and validate methods for training agents that achieve strong performance in multi-agent cooperation scenarios.

**Key Questions**:
1. Can evolution discover effective strategies for multi-agent cooperation?
2. How do evolved strategies compare to hand-designed heuristics?
3. What fitness functions and evolutionary parameters lead to robust agents?
4. Do strategies generalize across scenarios or require specialization?

**Success Metrics**:
- Evolved agents match or exceed heuristic baseline performance
- Training fitness correlates with tournament performance (no train/test gap)
- Strategies are stable and reproducible across random seeds
- Clear understanding of what makes agents perform well

### Evolution Versions

We track evolution experiments by version number (V3, V4, V5, V6, etc.):

| Version | Key Innovation | Population | Generations | Games/Eval | Status |
|---------|---------------|------------|-------------|------------|--------|
| V3 | Rust fitness evaluation | 100 | 2,500 | 50 | ✅ Complete |
| V4 | Extended training | 200 | 15,000 | 50 | ✅ Complete |
| V5 | Nash-based fitness | 100 | 12,000 | 100 | ✅ Complete |
| V6 | Tournament fitness | 200 | 200 | 100 | ✅ Complete |

Each version is stored in `experiments/scenarios/{scenario}/evolved_v{N}/` with:
- `best_agent.json` - Best evolved agent parameters and fitness
- `evolution_results.json` - Full evolution trace (all generations)
- `checkpoint_gen*.json` - Population snapshots at regular intervals
- `evolution_log.txt` - Text log of evolution progress

## Analysis Workflow

### Step 1: Data Collection

After evolution completes on remote server, retrieve results:

```bash
# Check completion status
ssh rwalters-sandbox-1 "cd bucket-brigade && ls -la experiments/scenarios/*/evolved_v6/best_agent.json"

# Sync results to local machine
rsync -avz --progress rwalters-sandbox-1:bucket-brigade/experiments/scenarios/*/evolved_v6/ \
  ./experiments/scenarios/ \
  --include='*/' --include='*/evolved_v6/**' --exclude='*'
```

**What to verify**:
- ✅ All scenarios have `best_agent.json`
- ✅ File sizes are reasonable (~300B for best agent, ~100KB for full trace)
- ✅ Timestamps indicate recent completion
- ✅ No error logs in evolution output

### Step 2: Quick Fitness Summary

First, get a quick overview of evolution performance:

```bash
# Summarize fitness across all scenarios
for scenario in chain_reaction deceptive_calm default early_containment easy \
                greedy_neighbor hard mixed_motivation overcrowding rest_trap \
                sparse_heroics trivial_cooperation; do
  echo -n "$scenario: "
  python3 -c "import json; data = json.load(open('experiments/scenarios/$scenario/evolved_v6/best_agent.json')); print(f'fitness={data[\"fitness\"]:.2f}, gen={data[\"generation\"]}')"
done
```

**Expected output**:
```
chain_reaction: fitness=130.00, gen=200
deceptive_calm: fitness=158.50, gen=200
...
```

**What to look for**:
- ✅ **Positive fitness**: Indicates agents are earning rewards
- ⚠️ **Negative fitness**: May indicate poor strategies or harsh scenarios
- ✅ **Variation across scenarios**: Different scenarios have different difficulty
- ⚠️ **Early convergence** (gen << max): May indicate premature convergence

### Step 3: Convergence Analysis

Examine how fitness evolved over generations:

```bash
# Plot evolution trajectory for a scenario
uv run python experiments/scripts/plot_evolution_trace.py chain_reaction \
  --version v6 \
  --output experiments/scenarios/chain_reaction/evolved_v6/convergence.png
```

**Key patterns to identify**:

1. **Healthy Convergence** ✅
   - Steady improvement over first 50-100 generations
   - Plateau at high fitness
   - Smooth curve with minor fluctuations
   - Example: V4 on trivial_cooperation

2. **Premature Convergence** ⚠️
   - Rapid improvement then long plateau
   - Convergence before 50% of generations
   - Low final fitness relative to scenario difficulty
   - Example: V5 on early_containment

3. **Unstable Evolution** ❌
   - Wild fitness swings throughout evolution
   - No clear upward trend
   - High variance in best fitness
   - Example: Early V3 experiments

4. **Local Minimum** ⚠️
   - Initial improvement then stuck
   - Fitness plateau well below theoretical optimum
   - Population diversity collapsed
   - Example: deceptive_calm in RESEARCH_PROGRESS.md

### Step 4: Strategy Characterization

Examine what strategies evolution discovered:

```bash
# Analyze evolved agent parameters
uv run python experiments/scripts/analyze_evolved_agent.py chain_reaction --version v6
```

**Parameter interpretation** (0-1 normalized):

| Parameter | Low (0.0-0.3) | Medium (0.3-0.7) | High (0.7-1.0) |
|-----------|---------------|------------------|----------------|
| **honesty_bias** | Lies frequently | Selective honesty | Always honest |
| **work_tendency** | Free-rider | Balanced work | Workaholic |
| **neighbor_help_bias** | Selfish | Helps occasionally | Altruistic |
| **own_house_priority** | Neglects own house | Balanced | Prioritizes own |
| **risk_aversion** | Risk-seeking | Neutral | Risk-averse |
| **coordination_weight** | Independent | Some coordination | Highly coordinated |
| **exploration_rate** | Exploitative | Balanced | Exploratory |
| **fatigue_memory** | Forgets quickly | Medium memory | Long memory |
| **rest_reward_bias** | Avoids rest | Neutral | Prefers rest |
| **altruism_factor** | Purely selfish | Mixed | Purely altruistic |

**Common evolved archetypes**:

1. **Lazy Free-Rider** (V3/V4 universal equilibrium)
   - Very low work_tendency (0.06)
   - Maximum rest_bias (1.0)
   - High own_priority (0.91)
   - Strategy: Let others do the work

2. **Balanced Cooperator**
   - Medium work_tendency (0.4-0.6)
   - Balanced priorities (0.4-0.6)
   - High coordination (0.7+)
   - Strategy: Fair contribution to team

3. **Altruistic Helper**
   - High neighbor_help_bias (0.8+)
   - High altruism (0.8+)
   - Low own_priority (0.2-)
   - Strategy: Help others at own expense

### Step 5: Tournament Validation

Critical step: Test evolved agents in tournaments against baselines:

```bash
# Run heterogeneous tournament (evolved vs archetypes)
uv run python experiments/scripts/run_heterogeneous_tournament.py chain_reaction \
  --agent-types evolved_v6 firefighter hero free_rider coordinator \
  --num-games 100 \
  --output experiments/scenarios/chain_reaction/tournament_v6/
```

**Why tournaments matter**:
- Evolution fitness may not match real performance (train/test gap)
- Agents may overfit to their training population
- Heterogeneous teams reveal strategy robustness
- Validates that fitness improvements are genuine

**Tournament metrics to analyze**:

1. **Mean Payoff**: Average reward across all games
   - Compare evolved agent to baselines
   - Should match or exceed best archetype
   - Consistency with training fitness indicates good fit

2. **Payoff Variance**: Standard deviation of rewards
   - Lower is better (more reliable)
   - High variance may indicate brittle strategies

3. **Win Rate**: Percentage of games where agent has highest payoff
   - Useful for competitive scenarios
   - May not be meaningful in cooperative scenarios

4. **Team Composition Effects**: Performance across different team makeups
   - Test with various mixtures of agent types
   - Robust agents perform well across compositions

### Step 6: Cross-Version Comparison

Compare current version against previous experiments:

```bash
# Compare V6 vs V5 vs V4 across all scenarios
uv run python experiments/scripts/compare_evolution_versions.py \
  --versions v4 v5 v6 \
  --scenarios all \
  --output experiments/evolution/version_comparison/
```

**Key comparisons**:

1. **Fitness Improvement**
   ```
   V4 → V5: +5.2% mean fitness (nash-based optimization)
   V5 → V6: +12.8% mean fitness (tournament-based training)
   ```

2. **Training Stability**
   - Convergence speed (generations to 95% of best)
   - Reproducibility across seeds
   - Population diversity maintenance

3. **Generalization**
   - Do V6 agents generalize better than V5?
   - Cross-scenario transfer performance
   - Robustness to parameter variations

4. **Computational Efficiency**
   ```
   V4: 15,000 gen × 200 pop = 3M evaluations
   V5: 12,000 gen × 100 pop = 1.2M evaluations
   V6: 200 gen × 200 pop = 40K evaluations ✅ Much faster!
   ```

### Step 7: Failure Mode Analysis

When evolution doesn't work, diagnose the issue:

**Common failure modes**:

1. **Train/Test Mismatch** (V4 critical bug)
   - **Symptom**: High training fitness, poor tournament performance
   - **Cause**: Python evaluation gave incorrect rewards during training
   - **Solution**: Use Rust as single source of truth (V5+)
   - **Example**: V4 claimed 58.50 fitness but actually -25.00 in Rust tournaments

2. **Premature Convergence**
   - **Symptom**: Fitness plateaus early, low final performance
   - **Cause**: Population diversity collapsed
   - **Solution**: Increase mutation rate, larger population, diversity maintenance
   - **Example**: deceptive_calm converged at gen 10 in RESEARCH_PROGRESS.md

3. **Free-Rider Equilibrium**
   - **Symptom**: Evolved agents have very low work_tendency
   - **Cause**: Fitness rewards team performance, allowing free-riding
   - **Solution**: Use individual fitness or tournament-based fitness
   - **Example**: V3/V4 universal "lazy free-rider" (work=0.06, rest=1.0)

4. **Overfitting to Training Population**
   - **Symptom**: Good fitness, poor tournament performance vs novel agents
   - **Cause**: Evolved against limited opponent diversity
   - **Solution**: Train with heterogeneous opposition (V6)
   - **Example**: V5 optimized for Nash equilibrium, failed in tournaments

5. **Unstable Fitness Landscape**
   - **Symptom**: High variance in best fitness, no convergence trend
   - **Cause**: Stochastic fitness estimates, insufficient evaluation games
   - **Solution**: Increase games per evaluation, use fitness averaging
   - **Example**: Early experiments with games=5 (too noisy)

### Step 8: Insight Generation

Extract research insights from results for documentation:

**Format for scenario configs** (`experiments/scenarios/{scenario}/config.json`):

```json
{
  "method_insights": {
    "evolution": [
      {
        "question": "What strategy did evolution discover?",
        "finding": "Evolution converged to a balanced cooperator strategy with medium work tendency (0.48) and high coordination (0.82).",
        "evidence": [
          "Final fitness: 4794.21 (rank #2 of 12 scenarios)",
          "Convergence: Generation 156 of 200",
          "Strategy: work=0.48, rest=0.35, coordination=0.82"
        ],
        "implication": "In easy scenarios, balanced cooperation outperforms both free-riding and excessive altruism."
      }
    ],
    "comparative": [
      {
        "question": "How does V6 compare to previous versions?",
        "finding": "V6 outperformed V5 by 12.8% and V4 by 18.3% in tournament play.",
        "evidence": [
          "V6 mean payoff: 4794.21",
          "V5 mean payoff: 4250.15 (-11.3%)",
          "V4 mean payoff: 4052.33 (-15.5%)"
        ],
        "implication": "Tournament-based fitness (V6) produces more robust strategies than Nash-based (V5) or population-based (V4) training."
      }
    ]
  }
}
```

**Document in research summaries**:
- Update `experiments/evolution/V{N}_RESULTS.md` with findings
- Add key insights to `experiments/RESEARCH_SUMMARY.md`
- Create visualizations for `web/public/research/scenarios/{scenario}/`

## Analysis Tools Reference

### Core Analysis Scripts

Located in `experiments/scripts/`:

1. **`analyze_evolved_agent.py`**
   - Examines single agent parameters
   - Compares to archetypes
   - Outputs strategy characterization

2. **`run_heterogeneous_tournament.py`**
   - Runs tournaments with mixed agent types
   - Generates ranking and payoff distributions
   - Tests strategy robustness

3. **`compare_evolution_versions.py`**
   - Compares multiple evolution versions
   - Tracks improvements over time
   - Identifies regressions

4. **`plot_evolution_trace.py`**
   - Visualizes convergence trajectories
   - Shows population diversity over time
   - Identifies convergence patterns

5. **`compute_nash_v2.py`**
   - Computes theoretical Nash equilibrium
   - Compares evolved strategies to equilibrium
   - Validates optimality claims

### Web Dashboard Integration

Results are displayed in the research dashboard at `web/src/pages/ScenarioResearch.tsx`:

**Data pipeline**:
```
Evolution Results
  ↓
experiments/scenarios/{scenario}/evolved_v6/
  ↓
web/public/research/scenarios/{scenario}/evolved/
  ↓
ScenarioResearch.tsx (loads and visualizes)
```

**What's displayed**:
- Evolution convergence chart (best/mean fitness over generations)
- Best agent parameter radar chart
- Comparison rankings (evolved vs archetypes)
- Research insights with evidence and implications

**To update dashboard**:
```bash
# Copy evolution results to web public directory
cp experiments/scenarios/easy/evolved_v6/evolution_results.json \
   web/public/research/scenarios/easy/evolved/evolution_trace.json

cp experiments/scenarios/easy/evolved_v6/best_agent.json \
   web/public/research/scenarios/easy/evolved/best_agent.json

# Rebuild web app
cd web && npm run build
```

## Best Practices

### Running Evolution Experiments

1. **Use Remote Compute** (rwalters-sandbox-1) for heavy experiments
   - Evolution with 200 population × 200 generations takes hours
   - Run in tmux for persistence
   - Log all output for debugging

2. **Version Everything**
   - Increment version number (V7, V8, etc.) for each major change
   - Track hyperparameters in `V{N}_PLAN.md`
   - Document rationale for changes

3. **Test Locally First**
   - Run 1 scenario with small population (pop=20, gen=10)
   - Verify fitness function works as expected
   - Check for obvious bugs before full run

4. **Monitor Progress**
   - Check logs periodically for errors
   - Plot intermediate checkpoints to verify convergence
   - Kill and restart if clearly stuck in local minimum

### Analyzing Results

1. **Always Run Tournaments**
   - Training fitness alone is insufficient
   - Test against diverse opponents
   - Validate with fresh random seeds

2. **Compare to Baselines**
   - Heuristic archetypes (firefighter, hero, etc.)
   - Previous evolution versions (V5, V4, etc.)
   - Theoretical Nash equilibrium (when available)

3. **Look for Patterns**
   - Are strategies consistent across seeds?
   - Do they generalize across scenarios?
   - What parameters matter most for performance?

4. **Document Failures**
   - Failed experiments teach us what doesn't work
   - Track failure modes in analysis documents
   - Use insights to improve next version

### Reporting Results

1. **Fitness Tables**
   - Sort by fitness (descending)
   - Include standard deviation
   - Mark best/worst scenarios

2. **Convergence Plots**
   - Show best and mean fitness
   - Highlight convergence point
   - Include population diversity if available

3. **Strategy Profiles**
   - Radar charts for parameter visualization
   - Compare evolved vs archetypes vs Nash
   - Highlight distinctive characteristics

4. **Research Insights**
   - Question → Finding → Evidence → Implication
   - Connect to broader research goals
   - Identify future work directions

## Common Pitfalls

### ❌ Don't: Trust Training Fitness Alone

**Why**: Train/test mismatch can be large (V4 bug: +58.50 train → -25.00 test)

**Do instead**: Always validate with independent tournaments using Rust evaluation

### ❌ Don't: Run Long Experiments Without Monitoring

**Why**: May converge early or get stuck, wasting compute

**Do instead**: Check intermediate checkpoints, plot progress, adjust if needed

### ❌ Don't: Compare Across Different Fitness Functions

**Why**: Nash-based fitness (V5) and tournament fitness (V6) optimize different objectives

**Do instead**: Compare final tournament performance, not training metrics

### ❌ Don't: Ignore Negative Results

**Why**: Failures teach us what doesn't work and guide better designs

**Do instead**: Document failure modes, analyze root causes, update methodology

### ❌ Don't: Assume More Compute = Better Results

**Why**: Wrong fitness function or evolutionary parameters can make things worse (see RESEARCH_PROGRESS.md)

**Do instead**: Test hypotheses with small experiments first, scale up only when promising

## Phase 1 Success Criteria

We'll consider Phase 1 complete when:

1. ✅ **Reproducible Training**: Evolved agents consistently achieve high fitness across seeds
2. ✅ **Tournament Validation**: Training fitness matches tournament performance (±5%)
3. ✅ **Baseline Performance**: Evolved agents match or beat best heuristic archetypes
4. ⚠️ **Generalization**: Strategies generalize reasonably across scenario variations
5. ⚠️ **Understanding**: Clear principles for what makes agents perform well

**Current Status** (as of V6):
- ✅ V6 achieves consistent positive fitness across all 12 scenarios
- ✅ Tournament-based training reduces train/test gap significantly
- ⚠️ Still learning optimal evolutionary parameters (population, generations, mutation)
- ⚠️ Generalization across scenarios varies (trivial_cooperation: +4826, hard: -2739)

## Next Steps

**Immediate** (for current V6 results):
1. Run V6 tournaments against V5/V4/archetypes
2. Analyze strategy profiles and identify patterns
3. Compare V6 to theoretical Nash equilibria
4. Update web dashboard with V6 results

**Short-term** (next experiments):
1. Test if V6 strategies generalize across parameter variations
2. Analyze which scenarios still need improvement (hard, greedy_neighbor)
3. Experiment with curriculum learning (easy → hard scenarios)
4. Compare evolution to PPO baseline (when PPO training works)

**Long-term** (Phase 2 preparation):
1. Solidify best practices from Phase 1 learnings
2. Establish benchmark suite for agent evaluation
3. Prepare for deep RL experiments (PPO, MAPPO)
4. Design experiments for mechanism design and heterogeneous teams

## References

**Key Documents**:
- `experiments/RESEARCH_SUMMARY.md` - Overall research progress
- `experiments/evolution/V6_PLAN.md` - V6 experiment design
- `experiments/evolution/V5_NEXT_STEPS.md` - V5 analysis workflow
- `experiments/RESEARCH_PROGRESS.md` - Lessons learned from failed experiments
- `CLAUDE.md` - Remote execution and compute guidelines

**Related Research Phases**:
- Phase 1: Learning to Train Agents (current)
- Phase 1.5: Cross-Scenario Generalization (completed in earlier work)
- Phase 2A: Boundary Testing (completed in earlier work)
- Phase 2: Advanced Multi-Agent Topics (future)

---

*This guide reflects our evolving understanding of what works in multi-agent evolution. Update as we learn more!*
