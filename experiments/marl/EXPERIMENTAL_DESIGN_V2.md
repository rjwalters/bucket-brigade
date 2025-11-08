# MARL Research: Experimental Design V2 (Updated)

**Phase**: 2 (Understanding Agent Effectiveness)
**Track**: Multi-Agent Reinforcement Learning (Neural Network Learning)
**Status**: 🔄 In Progress - Hard baseline complete
**Updated**: 2025-11-07

---

## Overview

This document provides an updated experimental protocol for Phase 2 MARL research, reflecting **actual progress** and aligning with Phase 2.5 priorities.

### What Changed Since V1

**Original V1 Plan**: Start with `chain_reaction` as baseline
**Actual Progress**: Started with `hard` scenario instead

**V1 Status**: Infrastructure built, one baseline complete
**V2 Focus**: Expand coverage, compare to evolution/Nash, prioritize analysis

### Core Questions (Unchanged)

1. **Can neural networks match evolution?** (Performance comparison)
2. **What do neural policies learn?** (Behavioral analysis)
3. **How does population-based training compare?** (Multi-agent learning dynamics)

---

## Current Progress Summary

### ✅ Completed Work

#### Infrastructure (Phase 1)
- ✅ GPU-accelerated PPO training (`train_gpu.py`)
- ✅ Vectorized environments (Rust-backed)
- ✅ TensorBoard logging and monitoring
- ✅ Checkpoint system (every 500K steps)
- ✅ Remote execution workflow (GPU server setup)

#### Baseline Training (Experiment 1 - Partial)
- ✅ **Hard scenario**: 5M timesteps, 116.9 min
  - Final reward: **9.45**
  - Model: `ppo_hard_5M_with_checkpoints.pt`
  - Checkpoints: 500K, 1M, 1.5M, 2M, 2.5M, 3M, 3.5M, 4M, 4.5M, 5M
  - **Using correct game mechanics** ✅

### ⏳ Pending Work

- ⏳ Baseline training for other scenarios
- ⏳ Learning curve analysis
- ⏳ Behavioral comparison to evolution
- ⏳ Population-based training experiments
- ⏳ Cross-scenario evaluation

---

## Updated Experimental Plan

### Experiment 1: Baseline PPO Training (Multi-Scenario)

**Goal**: Train PPO on priority scenarios and compare to evolution baselines

**Status**: 1/12 complete (hard)

#### Priority Scenarios (Ranked by Value)

**Tier 1: Critical Comparisons** (Run overnight)
1. ✅ **hard**: DONE (9.45 reward, V7 evolved = 43.21 fitness)
2. 🔴 **chain_reaction**: High cooperation, V7 evolved = 43.68
3. 🔴 **easy**: Baseline difficulty, V7 evolved = 17.11
4. 🔴 **greedy_neighbor**: Strategic selfishness, V7 evolved = 38.12

**Rationale**:
- Diverse scenario types (cooperation, difficulty, strategy)
- All have V7 evolved baselines for comparison
- Cover range of difficulty (easy 17.11 → chain_reaction 43.68)

**Tier 2: Extended Coverage** (Run after Tier 1 analysis)
5. **deceptive_calm**: Best V7 performance (48.80)
6. **trivial_cooperation**: Lowest V7 (6.50) - interesting edge case
7. **mixed_motivation**: Mid-range performance (44.68)
8. **rest_trap**: Low V7 (21.82) - difficulty test

**Tier 3: Complete Coverage** (Future work)
9-12. Remaining scenarios (default, early_containment, overcrowding, sparse_heroics)

#### Training Configuration (Per Scenario)

```bash
# Standard config (based on successful hard run)
uv run python experiments/marl/train_gpu.py \
  --scenario {scenario} \
  --steps 5000000 \
  --batch-size 2048 \
  --lr 3e-4 \
  --hidden-size 512 \
  --checkpoint-interval 500000 \
  --run-name baseline_{scenario}_v1 \
  --device cuda \
  --seed 42
```

**Expected runtime**: ~2 hours per scenario on L4 GPU

**Total for Tier 1**: 3 scenarios × 2 hours = 6 hours (perfect overnight run)

---

### Experiment 2: Learning Curve Analysis (Updated)

**Goal**: Compare PPO convergence to evolution across scenarios

**Status**: Ready to start (once Tier 1 complete)

#### 2A. Extract Learning Curves

For each completed baseline:

```bash
uv run python experiments/scripts/extract_ppo_learning_curves.py \
  --checkpoint-dir experiments/marl/checkpoints/baseline_{scenario}_v1/ \
  --output experiments/marl/learning_curves/baseline_{scenario}_v1.csv
```

**Output format**:
```csv
timestep,episode_reward,episode_length,loss,entropy
500000,5.2,48,0.034,1.23
1000000,6.8,52,0.028,1.18
...
5000000,9.45,55,0.021,1.05
```

#### 2B. Compare to Evolution

**PPO metrics**:
- Timesteps to convergence
- Sample efficiency (reward per timestep)
- Final performance

**Evolution metrics** (from V7):
- Generations to convergence
- Sample efficiency (fitness per generation)
- Final performance

**Key comparison**:
```python
# Sample efficiency comparison
ppo_efficiency = final_reward / total_timesteps
evolution_efficiency = final_fitness / (generations * population * games)

# Example: hard scenario
ppo_efficiency = 9.45 / 5M = 1.89e-6 reward/step
evolution_efficiency = 43.21 / (200 * 200 * 30) = 3.60e-6 fitness/game
```

**Note**: Direct comparison difficult (different metrics), but can compare convergence patterns

#### 2C. Visualization

```bash
uv run python experiments/scripts/plot_ppo_vs_evolution.py \
  --ppo-curves experiments/marl/learning_curves/baseline_*_v1.csv \
  --evolution-data experiments/scenarios/*/evolved_v7/evolution_results.json \
  --output plots/ppo_vs_evolution_convergence.png
```

**Visualizations**:
- Learning curves overlay (PPO timesteps vs Evolution generations)
- Convergence speed comparison
- Final performance comparison table

---

### Experiment 3: Behavioral Analysis (Critical for Phase 2.5)

**Goal**: Understand what PPO learns and compare to evolved heuristics

**Status**: Ready to start (needs analysis scripts)

**Priority**: 🔴 HIGH - Aligns with Phase 2.5 Priority 3

#### 3A. Extract PPO Behavior

```bash
uv run python experiments/scripts/analyze_ppo_behavior.py \
  --checkpoint experiments/marl/checkpoints/baseline_hard_v1/final.pt \
  --scenario hard \
  --num-episodes 100 \
  --output experiments/marl/behavior/ppo_hard_v1.json
```

**Metrics to extract**:
- **Cooperation rate**: Fraction of timesteps working (vs resting)
- **House visit distribution**: Which houses does agent prioritize?
- **Response to fires**: Action probability changes when fire detected
- **Spatial patterns**: Heatmap of house visits
- **Temporal patterns**: Work/rest cycles

#### 3B. Compare to V7 Evolved

**For hard scenario**:
- V7 evolved genome: `{honesty: 0.64, work: 0.05, neighbor_help: 0.0, risk_aversion: 0.88}`
- PPO behavior: Extract from rollouts

**Comparison questions**:
1. Do both have similar cooperation rates? (V7 work=0.05 is very low)
2. Do both prioritize own house? (V7 neighbor_help=0.0 suggests yes)
3. Do both avoid risk? (V7 risk_aversion=0.88 is high)
4. Are strategies converging or diverging?

**Expected finding**:
- If similar → Convergence to common optimal strategy
- If different → Multiple local optima or different objectives

---

### Experiment 4: PPO Evaluation Against V7 Evolved

**Goal**: Test PPO in heterogeneous tournaments (like V7 training)

**Status**: Ready to start

**Priority**: 🟡 MEDIUM - Validates PPO performance

#### 4A. Tournament Evaluation

```bash
uv run python experiments/scripts/evaluate_ppo_tournament.py \
  --ppo-checkpoint experiments/marl/checkpoints/baseline_hard_v1/final.pt \
  --opponents evolved_v7 firefighter hero free_rider coordinator \
  --scenario hard \
  --num-games 200 \
  --output experiments/marl/evaluation/ppo_hard_tournament.csv
```

**Output**:
```csv
game_id,agent_0,agent_1,agent_2,agent_3,payoff_0,payoff_1,payoff_2,payoff_3
1,ppo,evolved_v7,firefighter,hero,8.5,42.1,35.2,38.9
2,ppo,free_rider,free_rider,free_rider,2.1,5.2,4.8,6.1
...
```

**Analysis**:
- PPO mean payoff vs V7 evolved mean payoff
- PPO performance with good teammates (hero, firefighter)
- PPO performance with bad teammates (free_rider)
- Ranking: Where does PPO rank among 6 agent types?

**Success criteria**:
- PPO beats free_rider in heterogeneous teams
- PPO competitive with V7 evolved (within 20%)
- PPO doesn't catastrophically fail in worst-case teams

---

### Experiment 5: Population-Based Training (Deferred)

**Goal**: Test if training multiple policies improves diversity

**Status**: Not started

**Priority**: 🟣 LOW - Defer until baseline analysis complete

**Rationale**:
- Single-agent PPO working, no urgent need for population
- GPU utilization acceptable with single agent
- Focus on analysis before adding complexity

**Future work**: If single-agent PPO underperforms, try population training

---

## Overnight Execution Plan (Tier 1 Scenarios)

### Option A: Sequential Training (Recommended)

**What**: Train 3 scenarios sequentially on GPU server
**Where**: `rwalters-sandbox-2` (L4 GPU)
**Runtime**: ~6 hours total (2 hours × 3 scenarios)

```bash
# SSH to GPU server
ssh rwalters-sandbox-2

# Create launch script
cat > ~/bucket-brigade/scripts/launch_ppo_tier1.sh <<'SCRIPT'
#!/bin/bash
cd ~/bucket-brigade

scenarios=("chain_reaction" "easy" "greedy_neighbor")

for scenario in "${scenarios[@]}"; do
  echo "========================================="
  echo "Training PPO on $scenario"
  echo "========================================="

  tmux new-session -d -s "ppo_$scenario" \
    "cd ~/bucket-brigade && \
     uv run python experiments/marl/train_gpu.py \
       --scenario $scenario \
       --steps 5000000 \
       --batch-size 2048 \
       --lr 3e-4 \
       --hidden-size 512 \
       --checkpoint-interval 500000 \
       --run-name baseline_${scenario}_v1 \
       --device cuda \
       --seed 42 \
       2>&1 | tee logs/ppo_${scenario}_5M.log; \
     echo 'Training complete. Press any key to close'; read"

  echo "Launched: ppo_$scenario"

  # Wait for previous to finish before starting next
  # (sequential to avoid GPU contention)
  while tmux has-session -t "ppo_$scenario" 2>/dev/null; do
    sleep 60
  done
done

echo "All Tier 1 PPO training complete!"
SCRIPT

chmod +x scripts/launch_ppo_tier1.sh

# Launch
./scripts/launch_ppo_tier1.sh
```

**Monitoring**:
```bash
# Check status
ssh rwalters-sandbox-2 "tmux ls"

# Attach to active session
ssh rwalters-sandbox-2 -t "tmux attach -t ppo_chain_reaction"

# Check logs
ssh rwalters-sandbox-2 "tail -f ~/bucket-brigade/logs/ppo_chain_reaction_5M.log"
```

### Option B: Parallel Training (Aggressive)

**What**: Train all 3 scenarios in parallel
**Risk**: GPU memory contention (may OOM)
**Benefit**: Completes in ~2 hours instead of 6

**Recommendation**: Try Option A first. If GPU has spare capacity (check with `nvidia-smi`), can launch parallel next time.

---

## Analysis Pipeline (After Tier 1 Complete)

### Day 1: Data Collection (Morning after overnight run)

1. **Download models** (5 min)
   ```bash
   rsync -avz rwalters-sandbox-2:~/bucket-brigade/models/ppo_*_5M.pt models/
   rsync -avz rwalters-sandbox-2:~/bucket-brigade/experiments/marl/checkpoints/ \
     experiments/marl/checkpoints/
   ```

2. **Extract learning curves** (15 min)
   ```bash
   for scenario in hard chain_reaction easy greedy_neighbor; do
     uv run python experiments/scripts/extract_ppo_learning_curves.py \
       --checkpoint-dir experiments/marl/checkpoints/baseline_${scenario}_v1/ \
       --output experiments/marl/learning_curves/${scenario}_v1.csv
   done
   ```

3. **Quick validation** (30 min)
   ```bash
   # Sanity check: Run each model for 10 episodes
   for scenario in chain_reaction easy greedy_neighbor; do
     uv run python experiments/scripts/evaluate_ppo_quick.py \
       --checkpoint models/ppo_${scenario}_5M.pt \
       --scenario $scenario \
       --num-episodes 10
   done
   ```

### Day 2: Comparative Analysis (2-3 hours)

1. **PPO vs Evolution comparison**
   - Extract V7 evolved parameters for same 4 scenarios
   - Compare final performance (PPO reward vs V7 fitness)
   - Plot learning curves side-by-side
   - Document findings: `experiments/marl/PPO_VS_EVOLUTION_TIER1.md`

2. **Behavioral analysis** (for 1-2 scenarios)
   - Extract PPO action distributions
   - Compare to V7 genome interpretations
   - Identify similarities and differences

3. **Tournament evaluation** (1-2 scenarios)
   - Test PPO in heterogeneous teams
   - Compare performance to V7 in same setting

### Day 3: Documentation & Decision (1-2 hours)

1. **Synthesize findings**
   - What did PPO learn?
   - How does it compare to evolution?
   - Which method performs better for which scenarios?

2. **Decision: Tier 2 or Analysis Deepening**
   - **If PPO competitive**: Proceed to Tier 2 scenarios
   - **If PPO underperforms**: Debug, tune hyperparameters, try population training
   - **If PPO exceeds**: Document why, investigate novel strategies

---

## Success Criteria (Updated for V2)

### Must Have (Tier 1 Complete)

1. ✅ **Hard baseline**: DONE (9.45 reward)
2. 🔲 **Chain_reaction baseline**: PPO trained, reward documented
3. 🔲 **Easy baseline**: PPO trained, reward documented
4. 🔲 **Greedy_neighbor baseline**: PPO trained, reward documented
5. 🔲 **Learning curves extracted**: For all 4 Tier 1 scenarios
6. 🔲 **PPO vs Evolution comparison**: Initial comparison table

### Should Have (Analysis Phase)

7. 🔲 **Behavioral analysis**: For 2+ scenarios
8. 🔲 **Tournament evaluation**: PPO in heterogeneous teams (2+ scenarios)
9. 🔲 **Convergence analysis**: Sample efficiency comparison
10. 🔲 **Documentation**: Comprehensive Tier 1 results write-up

### Nice to Have (Future Work)

11. ⚠️ **Tier 2 training**: 4 additional scenarios
12. ⚠️ **Population training**: If single-agent insufficient
13. ⚠️ **Complete coverage**: All 12 scenarios trained

---

## Integration with Phase 2.5

From `research_notebook/2025-11-08_phase_2_5_analysis_plan.md`:

### Phase 2.5 Priority 3: Nash V7 Deep Dive 🎯

**MARL Contribution**:
- ✅ Test PPO agents in heterogeneous tournaments (same as Nash/Evolution)
- ✅ Compare PPO vs V7 evolved vs Nash equilibrium
- ✅ Three-way comparison: Gradient-based RL vs Evolutionary vs Game-theoretic

### Phase 2.5 Priority 4: Scenario Clustering 🗂️

**MARL Contribution**:
- ✅ PPO performance adds another dimension to scenario similarity
- ✅ Do PPO and Evolution struggle/excel on same scenarios?
- ✅ Identify scenarios where neural networks have advantage

### Phase 2.5 Priority 5: Cross-Scenario Transfer Matrix 🔀

**MARL Contribution**:
- ⚠️ Future: Test PPO policies trained on scenario A in scenario B
- ⚠️ Compare to evolution transfer results
- ⚠️ Identify if neural networks generalize better

---

## Risk Mitigation

### Risk 1: PPO significantly underperforms evolution

**Likelihood**: Medium (RL can struggle with sparse rewards)
**Impact**: High (questions neural network viability)

**Mitigation**:
- Hyperparameter tuning (learning rate, batch size, network architecture)
- Reward shaping (add intermediate rewards)
- Population-based training (diverse exploration)
- Curriculum learning (start easy, increase difficulty)
- **Fallback**: Accept evolution is better for this domain, focus on Nash

### Risk 2: GPU server unavailable or crashes

**Likelihood**: Low (server stable)
**Impact**: High (blocks all training)

**Mitigation**:
- Use tmux sessions (survive disconnects)
- Checkpoint frequently (every 500K steps)
- Monitor remotely (can restart if needed)
- **Fallback**: Use local CPU training (much slower, ~10x)

### Risk 3: Analysis scripts don't exist

**Likelihood**: High (many scripts need to be written)
**Impact**: Medium (manual analysis possible but tedious)

**Mitigation**:
- Prioritize critical scripts (learning curve extraction, comparison)
- Use simple tools for visualization (spreadsheets, matplotlib)
- Document manual analysis process
- Create scripts incrementally as needed

### Risk 4: PPO doesn't converge in 5M steps

**Likelihood**: Medium (some scenarios may be harder)
**Impact**: Medium (need longer training)

**Mitigation**:
- Check learning curves at checkpoints (500K, 1M, 2M, 3M, 4M, 5M)
- If still improving at 5M → Extend to 10M for those scenarios
- If plateaued early → May be local optimum, try different hyperparameters
- Compare convergence to evolution (did evolution converge?)

---

## Key Questions to Answer (Priority Order)

### Immediate (After Tier 1)

1. **Performance**: How do PPO rewards compare to V7 fitness values?
2. **Convergence**: Does PPO converge faster or slower than evolution?
3. **Robustness**: Does PPO perform well in heterogeneous tournaments?
4. **Behavior**: Do PPO and V7 learn similar strategies?

### Short-term (After Analysis)

5. **Generalization**: Can PPO policies transfer across scenarios?
6. **Sample efficiency**: Which method is more sample-efficient?
7. **Interpretability**: Can we understand what PPO learned?
8. **Population training**: Does it improve diversity/performance?

### Long-term (Phase 3+)

9. **Mechanism design**: Can PPO adapt to game rule changes?
10. **Curriculum learning**: Does staged training improve results?
11. **Multi-agent learning**: Does population training discover coordination?
12. **Hybrid approaches**: Combine evolution + RL fine-tuning?

---

## Timeline

### Week 1: Tier 1 Baseline Training

**Day 1** (Tonight): Launch overnight run (6 hours)
- chain_reaction, easy, greedy_neighbor

**Day 2** (Morning): Download and validate
- Extract learning curves
- Quick performance checks
- Document initial findings

**Day 3-4**: Comparative analysis
- PPO vs Evolution comparison
- Behavioral analysis (1-2 scenarios)
- Tournament evaluation

**Day 5**: Documentation and decision
- Synthesize Tier 1 findings
- Decide: Tier 2 or deeper analysis?
- Update roadmap

### Week 2: Extended Analysis or Tier 2

**If analysis focused**:
- Deep behavioral analysis (all 4 Tier 1 scenarios)
- Comprehensive tournament evaluation
- Learning curve analysis
- Write up: `MARL_PHASE2_TIER1_RESULTS.md`

**If Tier 2 training**:
- Launch 4 more scenarios (deceptive_calm, trivial_cooperation, mixed_motivation, rest_trap)
- Repeat analysis pipeline
- Expand comparison to 8 scenarios

---

## Deliverables

### Immediate (After Tier 1 Training)

1. **Trained models**: `models/ppo_{scenario}_5M.pt` (4 scenarios)
2. **Checkpoints**: Every 500K steps (10 checkpoints × 4 scenarios = 40 files)
3. **Training logs**: `logs/ppo_{scenario}_5M.log` (4 files)
4. **Learning curves**: `experiments/marl/learning_curves/{scenario}_v1.csv` (4 files)

### Short-term (1 Week)

5. **Performance comparison**: `experiments/marl/PPO_VS_EVOLUTION_TIER1.md`
   - Performance table (PPO vs V7 for 4 scenarios)
   - Learning curve plots
   - Convergence speed comparison

6. **Behavioral analysis**: `experiments/marl/BEHAVIORAL_ANALYSIS_TIER1.md`
   - Action distribution comparisons (2+ scenarios)
   - Strategy characterization
   - PPO vs V7 behavior comparison

7. **Tournament results**: `experiments/marl/TOURNAMENT_EVALUATION_TIER1.md`
   - PPO vs archetypes rankings
   - Heterogeneous team performance
   - Robustness analysis

### Medium-term (2-3 Weeks)

8. **Comprehensive report**: `experiments/marl/MARL_PHASE2_COMPREHENSIVE.md`
   - All findings synthesized
   - Three-way comparison: PPO vs Evolution vs Nash
   - Recommendations for Phase 3
   - Identification of best method per scenario type

9. **Research notebook entry**: `research_notebook/2025-11-XX_marl_tier1_complete.md`
   - Narrative summary of MARL findings
   - Integration with Evolution and Nash tracks
   - Updated research agenda

---

## Scripts to Create (Priority Order)

### Critical (Needed for basic analysis)

1. **`extract_ppo_learning_curves.py`**: Extract reward/loss over time from checkpoints
2. **`evaluate_ppo_quick.py`**: Quick validation (10-20 episodes)
3. **`compare_ppo_vs_evolution.py`**: Side-by-side performance comparison

### High Priority (Needed for deep analysis)

4. **`analyze_ppo_behavior.py`**: Extract action distributions, visit patterns
5. **`evaluate_ppo_tournament.py`**: Test PPO in heterogeneous tournaments
6. **`plot_ppo_vs_evolution.py`**: Visualization of learning curves

### Medium Priority (Nice to have)

7. **`compare_behavior_ppo_vs_v7.py`**: Behavioral similarity analysis
8. **`parameter_sensitivity_ppo.py`**: Hyperparameter sensitivity
9. **`cross_scenario_evaluation_ppo.py`**: Transfer learning tests

---

## Next Steps (Immediate)

1. **Launch Tier 1 training tonight** (5 min setup)
   ```bash
   ssh rwalters-sandbox-2
   cd ~/bucket-brigade
   ./scripts/launch_ppo_tier1.sh
   ```

2. **Verify launch** (5 min)
   ```bash
   # Check tmux sessions exist
   ssh rwalters-sandbox-2 "tmux ls"

   # Check first scenario started
   ssh rwalters-sandbox-2 "tail -20 ~/bucket-brigade/logs/ppo_chain_reaction_5M.log"
   ```

3. **Morning check-in** (15 min)
   ```bash
   # Check completion status
   ssh rwalters-sandbox-2 "ls -lh ~/bucket-brigade/models/ppo_*_5M.pt"

   # Download results
   rsync -avz rwalters-sandbox-2:~/bucket-brigade/models/ models/
   ```

4. **Begin analysis** (Day 2)
   - Extract learning curves
   - Quick performance validation
   - Start PPO vs Evolution comparison

---

**Status**: 🚀 Ready for Tier 1 Execution
**Priority**: 🟡 HIGH - Core MARL baseline expansion
**Next Action**: Launch overnight Tier 1 training (chain_reaction, easy, greedy_neighbor)
**Decision Point**: Day 5 (Tier 2 vs deeper analysis)
**Integration**: Aligns with Phase 2.5 three-way comparison (PPO vs Evolution vs Nash)
