# MARL Research: Experimental Design & Roadmap

**Phase**: 2 (Understanding Agent Effectiveness)
**Track**: Multi-Agent Reinforcement Learning (Neural Network Learning)
**Goal**: Establish baselines and understand learning dynamics

---

## Overview

This document provides a step-by-step experimental protocol for Phase 2 MARL research. Phase 1 established GPU-accelerated PPO infrastructure with vectorized environments. Phase 2 focuses on **understanding what neural networks learn** and **comparing to evolution/Nash baselines**.

### Core Questions

1. **Can neural networks match evolution?** (Performance comparison)
2. **What do neural policies learn?** (Behavioral analysis)
3. **How does population-based training compare?** (Multi-agent learning dynamics)

---

## Experiment 1: Baseline PPO Training (chain_reaction)

**Goal**: Train PPO on chain_reaction and compare to evolved baseline (58.50).

**Priority**: 🔴 Critical (Q1)

### Protocol

**Step 1: Configure training**

```bash
# Create training config
cat > experiments/marl/configs/baseline_chain_reaction.yaml <<EOF
scenario: chain_reaction
total_timesteps: 10000000  # 10M steps
num_envs: 256              # Vectorized environments
rollout_length: 256        # Steps per rollout
minibatch_size: 2048       # PPO batch size
learning_rate: 3e-4
num_epochs: 4              # PPO epochs per batch
hidden_size: 512           # Network capacity
device: cuda
seed: 42
EOF
```

**Step 2: Launch training**

```bash
# Using vectorized training script (GPU-optimized)
uv run python experiments/marl/train_vectorized_population.py \
  --scenario chain_reaction \
  --population-size 1 \
  --num-envs 256 \
  --total-timesteps 10000000 \
  --rollout-length 256 \
  --batch-size 2048 \
  --hidden-size 512 \
  --learning-rate 3e-4 \
  --num-epochs 4 \
  --device cuda \
  --seed 42 \
  --run-name baseline_chain_reaction_v1 \
  --checkpoint-interval 100000 \
  --log-interval 10
```

**Expected runtime**: 2-4 hours on L4 GPU

**Step 3: Monitor training**

```bash
# Track GPU utilization
ssh remote-gpu "nvidia-smi dmon -s u -d 10" &

# Watch training progress
tail -f experiments/marl/runs/baseline_chain_reaction_v1/events.out.tfevents.*

# Check checkpoint at 1M steps
ls -lh experiments/marl/checkpoints/baseline_chain_reaction_v1/
```

**Key metrics to monitor**:
- Episode reward (target: ≥58.50)
- GPU utilization (target: 60-95%)
- Steps per second (target: 500-2000)
- Convergence (reward plateaus)

**Step 4: Evaluate trained policy**

```bash
# Load checkpoint and run evaluation
uv run python experiments/scripts/evaluate_ppo_policy.py \
  --checkpoint experiments/marl/checkpoints/baseline_chain_reaction_v1/final_checkpoint.pt \
  --scenario chain_reaction \
  --num-games 200 \
  --output experiments/marl/eval_baseline_chain_reaction_v1.json
```

**Step 5: Compare to evolved baseline**

```bash
# Run tournament: PPO vs. Evolved
uv run python experiments/scripts/compare_ppo_vs_evolved.py \
  --ppo-checkpoint experiments/marl/checkpoints/baseline_chain_reaction_v1/final_checkpoint.pt \
  --evolved-agent experiments/scenarios/chain_reaction/evolved_v4/best_agent.json \
  --scenario chain_reaction \
  --num-games 200 \
  --output experiments/marl/BASELINE_RESULTS.md
```

### Expected Outcomes

**Scenario A: PPO matches evolution**
- PPO reward ≥ 55.00 (within 5% of 58.50)
- **Conclusion**: Neural networks can learn effective strategies

**Scenario B: PPO underperforms**
- PPO reward 40-50 (significantly below 58.50)
- **Conclusion**: Need hyperparameter tuning or architectural changes

**Scenario C: PPO exceeds evolution**
- PPO reward > 60.00 (better than 58.50)
- **Conclusion**: Neural networks discover novel strategies (investigate!)

### Deliverables

1. **Trained policy**: `experiments/marl/checkpoints/baseline_chain_reaction_v1/final_checkpoint.pt`
2. **Training logs**: `experiments/marl/runs/baseline_chain_reaction_v1/`
3. **Evaluation results**: `experiments/marl/eval_baseline_chain_reaction_v1.json`
4. **Comparison report**: `experiments/marl/BASELINE_RESULTS.md`
   - PPO vs. Evolved performance
   - Learning curve analysis
   - Convergence speed (timesteps to reach X% of final)

---

## Experiment 2: Learning Curve Analysis

**Goal**: Understand PPO convergence dynamics and sample efficiency.

**Priority**: 🟡 High (Q1)

### Protocol

**Step 1: Extract learning curves**

```bash
uv run python experiments/scripts/extract_learning_curves.py \
  experiments/marl/runs/baseline_chain_reaction_v1/ \
  --output experiments/marl/learning_curve_chain_reaction.csv
```

**Output format**:
```csv
timestep,episode_reward,episode_length,avg_reward_100ep
100000,25.3,48,22.1
200000,32.1,52,28.4
...
10000000,58.2,55,57.9
```

**Step 2: Analyze convergence**

```bash
uv run python experiments/scripts/analyze_convergence.py \
  experiments/marl/learning_curve_chain_reaction.csv \
  --baseline-payoff 58.50 \
  --output experiments/marl/LEARNING_CURVES.md
```

**Key metrics**:
- **Time to 50% of evolved**: Timesteps to reach 29.25 reward
- **Time to 90% of evolved**: Timesteps to reach 52.65 reward
- **Sample efficiency**: (timesteps / final_reward) compared to evolution (generations / fitness)
- **Stability**: Variance in last 100K timesteps

**Step 3: Compare to evolution**

Evolution V4 metrics:
- Final fitness: 58.50 (after 15,000 generations)
- Evaluation budget: 15,000 gen × 200 pop × 50 games = 150M game steps

PPO metrics:
- Final reward: X (after 10M timesteps)
- Evaluation budget: 10M timesteps

**Step 4: Visualize**

```bash
uv run python experiments/scripts/plot_learning_curves.py \
  experiments/marl/learning_curve_chain_reaction.csv \
  --evolved-baseline 58.50 \
  --output plots/ppo_learning_curve_chain_reaction.png
```

### Expected Outcomes

**Fast convergence** (< 2M timesteps to 90%):
- PPO is sample-efficient for this scenario
- Quick adaptation to cooperation dynamics

**Slow convergence** (> 5M timesteps to 90%):
- Scenario is difficult for gradient-based learning
- May need curriculum learning or better exploration

**Non-monotonic** (oscillations, plateaus):
- Non-stationarity from multi-agent interactions
- Population training may help stabilize

### Deliverables

1. **Learning curve data**: `experiments/marl/learning_curve_{scenario}.csv`
2. **Convergence analysis**: `experiments/marl/LEARNING_CURVES.md`
   - Convergence speed comparison (PPO vs. Evolution)
   - Sample efficiency analysis
   - Stability metrics
3. **Visualizations**:
   - Learning curve plot: `plots/ppo_learning_curve_chain_reaction.png`
   - Comparison overlay: `plots/ppo_vs_evolution_convergence.png`

---

## Experiment 3: Behavioral Analysis

**Goal**: Understand what strategies neural policies learn.

**Priority**: 🟡 High (Q1-Q2)

### Protocol

**Step 1: Extract action distributions**

```bash
uv run python experiments/scripts/analyze_ppo_behavior.py \
  --checkpoint experiments/marl/checkpoints/baseline_chain_reaction_v1/final_checkpoint.pt \
  --scenario chain_reaction \
  --num-episodes 100 \
  --output experiments/marl/behavior_ppo_chain_reaction.json
```

**Metrics to extract**:
- **Cooperation rate**: Fraction of timesteps spent working
- **Own-house bias**: Visits to own house / total visits
- **Spatial distribution**: Heatmap of house visits
- **Temporal patterns**: Work/rest cycles
- **Response to fires**: Action changes when fire detected

**Step 2: Compare to evolved agents**

```bash
uv run python experiments/scripts/compare_behavior.py \
  --ppo experiments/marl/behavior_ppo_chain_reaction.json \
  --evolved experiments/scenarios/chain_reaction/evolved_v4/best_agent.json \
  --scenario chain_reaction \
  --output experiments/marl/behavior_comparison_chain_reaction.md
```

**Comparison questions**:
- Do both strategies prioritize same houses?
- Do both have similar cooperation rates?
- Are temporal patterns similar (e.g., rest cycles)?
- Where do they differ behaviorally?

**Step 3: Interpretability analysis**

For evolved agents, we have explicit parameters:
```python
genome = {
  'work_tendency': 0.75,
  'signal_honesty': 0.82,
  'own_house_priority': 0.65,
  ...
}
```

For neural policies, we need to infer:
```bash
uv run python experiments/scripts/interpret_neural_policy.py \
  --checkpoint experiments/marl/checkpoints/baseline_chain_reaction_v1/final_checkpoint.pt \
  --scenario chain_reaction \
  --output experiments/marl/policy_interpretation_chain_reaction.md
```

**Methods**:
- **Activation analysis**: Which neurons fire for different states?
- **Gradient attribution**: Which observations influence actions?
- **Behavioral cloning**: Can we fit a heuristic to the policy?

**Step 4: Visualize strategy**

```bash
uv run python experiments/scripts/visualize_policy_strategy.py \
  experiments/marl/behavior_ppo_chain_reaction.json \
  experiments/scenarios/chain_reaction/evolved_v4/best_agent.json \
  --output plots/strategy_comparison_chain_reaction.png
```

**Visualizations**:
- House visit heatmap (PPO vs. Evolved)
- Action distribution pie charts
- Temporal pattern timeseries

### Expected Outcomes

**Similar strategies**:
- PPO learns similar cooperation rates to evolved
- Similar house prioritization patterns
- **Conclusion**: Convergence to shared optimal strategy

**Different strategies**:
- PPO finds qualitatively different approach
- Example: More exploration, less own-house bias
- **Conclusion**: Multiple optima in strategy space

**Incomplete learning**:
- PPO has lower cooperation rate
- Doesn't match evolved spatial patterns
- **Conclusion**: Needs more training or better architecture

### Deliverables

1. **Behavioral data**: `experiments/marl/behavior_ppo_{scenario}.json`
2. **Comparison report**: `experiments/marl/behavior_comparison_{scenario}.md`
3. **Interpretation**: `experiments/marl/policy_interpretation_{scenario}.md`
4. **Visualizations**:
   - Heatmaps: `plots/house_visit_heatmap_{scenario}.png`
   - Action distributions: `plots/action_dist_{scenario}.png`
   - Strategy comparison: `plots/strategy_comparison_{scenario}.png`

---

## Experiment 4: Population-Based Training Pilot

**Goal**: Test if training multiple agents simultaneously improves diversity and performance.

**Priority**: 🟢 Medium (Q1-Q2)

### Background

**Single-agent PPO**: One policy learns from own experiences
**Population-based training**: Multiple policies learn simultaneously, play against each other

**Hypothesis**: Population training provides:
- **Diversity**: Different strategies emerge
- **Robustness**: Less overfitting to specific opponents
- **Better GPU utilization**: More parallel computation

### Protocol

**Step 1: Configure population training**

```bash
# 4 agents (small pilot)
uv run python experiments/marl/train_vectorized_population.py \
  --scenario chain_reaction \
  --population-size 4 \
  --num-envs 256 \
  --total-timesteps 10000000 \
  --batch-size 512 \
  --hidden-size 512 \
  --device cuda \
  --seed 42 \
  --run-name pop4_chain_reaction_v1
```

**Population-specific parameters**:
- `--population-size`: Number of independent policies (4, 8, 16, 32)
- `--matchmaking`: How to pair agents (round-robin, random)
- `--update-interval`: How often policies sync with simulator

**Step 2: Monitor GPU utilization**

```bash
# Check if population training improves GPU usage
watch -n 1 nvidia-smi

# Expected: Higher utilization (60-95%) vs. single-agent (10-20%)
```

**Step 3: Analyze population diversity**

After training:
```bash
uv run python experiments/scripts/analyze_population_diversity.py \
  experiments/marl/checkpoints/pop4_chain_reaction_v1/ \
  --scenario chain_reaction \
  --output experiments/marl/diversity_pop4_chain_reaction.json
```

**Diversity metrics**:
- **Behavioral diversity**: Action distribution differences
- **Performance spread**: Variance in episode rewards
- **Strategy divergence**: Pairwise policy distance (e.g., KL divergence)

**Step 4: Compare to single-agent**

```bash
uv run python experiments/scripts/compare_single_vs_population.py \
  --single experiments/marl/checkpoints/baseline_chain_reaction_v1/ \
  --population experiments/marl/checkpoints/pop4_chain_reaction_v1/ \
  --scenario chain_reaction \
  --output experiments/marl/POPULATION_TRAINING_RESULTS.md
```

**Comparison axes**:
- Best agent performance: max(pop) vs. single
- Average agent performance: mean(pop) vs. single
- Diversity: pop variance vs. single (none)
- GPU utilization: pop % vs. single %

### Expected Outcomes

**Population improves performance**:
- Best pop agent > single agent
- **Conclusion**: Competition/collaboration helps learning

**Population improves diversity**:
- High behavioral variance within population
- **Conclusion**: Useful for opponent modeling

**Population improves GPU utilization**:
- 60-95% GPU vs. 10-20% single
- **Conclusion**: Better hardware efficiency

**Population doesn't help**:
- Best pop agent ≈ single agent
- No significant diversity
- **Conclusion**: Scenario doesn't benefit from population

### Deliverables

1. **Population checkpoints**: `experiments/marl/checkpoints/pop4_chain_reaction_v1/`
2. **Diversity analysis**: `experiments/marl/diversity_pop4_chain_reaction.json`
3. **Comparison report**: `experiments/marl/POPULATION_TRAINING_RESULTS.md`
4. **GPU optimization**: `experiments/marl/GPU_OPTIMIZATION.md`
   - Utilization comparison (single vs. population)
   - Throughput metrics (steps/sec)
   - Recommendations for scaling

---

## Experiment 5: Multi-Scenario Baseline Suite

**Goal**: Train PPO on all 12 scenarios to establish comprehensive baselines.

**Priority**: 🟢 Medium (Q2-Q3)

### Protocol

**Step 1: Prioritize scenarios**

**Tier 1** (already tested):
- chain_reaction

**Tier 2** (compare to evolution):
- greedy_neighbor
- mixed_motivation
- easy

**Tier 3** (complete coverage):
- All remaining scenarios

**Step 2: Batch training**

```bash
cat > experiments/marl/train_all_scenarios.sh <<'EOF'
#!/bin/bash
scenarios="chain_reaction greedy_neighbor mixed_motivation easy \
           trivial_cooperation overcrowding rest_trap sparse_heroics \
           deceptive_calm early_containment default hard"

for scenario in $scenarios; do
  echo "=== Training PPO on $scenario ==="
  uv run python experiments/marl/train_vectorized_population.py \
    --scenario $scenario \
    --population-size 1 \
    --num-envs 256 \
    --total-timesteps 10000000 \
    --batch-size 2048 \
    --hidden-size 512 \
    --device cuda \
    --seed 42 \
    --run-name baseline_${scenario}_v1 \
    2>&1 | tee logs/training_${scenario}_v1.log

  echo "=== Completed $scenario ==="
done
EOF

chmod +x experiments/marl/train_all_scenarios.sh
# Run sequentially (or parallelize if multiple GPUs)
./experiments/marl/train_all_scenarios.sh
```

**Expected total runtime**: 24-48 GPU-hours (sequential), 4-8 hours (parallel on 6 GPUs)

**Step 3: Evaluate all baselines**

```bash
uv run python experiments/scripts/evaluate_all_ppo_baselines.py \
  experiments/marl/checkpoints/ \
  --num-games 200 \
  --output experiments/marl/all_baselines_summary.csv
```

**Output format**:
```csv
scenario,ppo_reward,evolved_reward,nash_reward,ppo_vs_evolved_%
chain_reaction,57.2,58.5,2.94,97.8%
greedy_neighbor,60.1,62.3,58.8,96.5%
...
```

**Step 4: Cross-scenario analysis**

```bash
uv run python experiments/scripts/analyze_cross_scenario_ppo.py \
  experiments/marl/all_baselines_summary.csv \
  --output experiments/marl/CROSS_SCENARIO_ANALYSIS.md
```

**Analysis questions**:
- Which scenarios does PPO excel on?
- Which scenarios does PPO struggle with?
- Correlation: scenario difficulty (evolution) vs. PPO performance?
- Transfer: Can policies trained on A work on B?

### Deliverables

1. **All baseline checkpoints**: `experiments/marl/checkpoints/baseline_{scenario}_v1/`
2. **Summary table**: `experiments/marl/all_baselines_summary.csv`
3. **Cross-scenario analysis**: `experiments/marl/CROSS_SCENARIO_ANALYSIS.md`
4. **Visualizations**:
   - Performance comparison: `plots/ppo_vs_evolved_all_scenarios.png`
   - Scenario difficulty ranking: `plots/scenario_difficulty_ppo.png`

---

## Implementation Checklist

### Core Scripts

- [x] `experiments/marl/train_vectorized_population.py` - Vectorized PPO training
- [ ] `experiments/scripts/evaluate_ppo_policy.py` - Policy evaluation
- [ ] `experiments/scripts/compare_ppo_vs_evolved.py` - PPO vs. Evolution comparison
- [ ] `experiments/scripts/extract_learning_curves.py` - Learning curve extraction
- [ ] `experiments/scripts/analyze_convergence.py` - Convergence analysis
- [ ] `experiments/scripts/analyze_ppo_behavior.py` - Behavioral analysis
- [ ] `experiments/scripts/compare_behavior.py` - Behavior comparison
- [ ] `experiments/scripts/interpret_neural_policy.py` - Policy interpretation
- [ ] `experiments/scripts/analyze_population_diversity.py` - Population diversity
- [ ] `experiments/scripts/compare_single_vs_population.py` - Single vs. population
- [ ] `experiments/scripts/evaluate_all_ppo_baselines.py` - Batch evaluation
- [ ] `experiments/scripts/analyze_cross_scenario_ppo.py` - Cross-scenario analysis

### Visualization Scripts

- [ ] `experiments/scripts/plot_learning_curves.py` - Learning curve plots
- [ ] `experiments/scripts/visualize_policy_strategy.py` - Strategy visualization

### Batch Execution

- [ ] `experiments/marl/train_all_scenarios.sh` - Batch training script

---

## Timeline

### Q1 (Months 1-3)

**Week 1-2**: Experiment 1 (chain_reaction baseline)
- Train PPO
- Evaluate vs. evolved
- Document results

**Week 3-4**: Experiment 2 (learning curves)
- Extract convergence data
- Analyze sample efficiency
- Compare to evolution

**Week 5-8**: Experiment 3 (behavioral analysis)
- Extract action distributions
- Compare to evolved heuristics
- Interpret neural policy

**Week 9-12**: Experiment 4 (population pilot)
- Train 4-agent population
- Measure diversity
- GPU optimization study

### Q2 (Months 4-6)

**Week 1-8**: Experiment 5 (multi-scenario)
- Train Tier 2 scenarios (3 scenarios)
- Train Tier 3 scenarios (8 scenarios)
- Batch evaluation

### Q3 (Months 7-9)

**Week 1-4**: Cross-scenario analysis
- Performance patterns
- Transfer experiments
- Scenario taxonomy (PPO lens)

**Week 5-8**: Integration
- Compare to Nash/Evolution tracks
- Synthesize findings
- Technical report

---

## Success Criteria

### Quantitative

✅ **Performance**: PPO matches evolution (≥90% of evolved payoff) on chain_reaction
✅ **Coverage**: Baselines for all 12 scenarios
✅ **Efficiency**: GPU utilization >60% with population training
✅ **Diversity**: Population training produces varied strategies (high behavioral variance)

### Qualitative

✅ **Understanding**: Clear characterization of what neural policies learn
✅ **Comparison**: Behavioral analysis comparing PPO to evolved heuristics
✅ **Prediction**: Model relating scenario features to PPO performance
✅ **Integration**: Cross-validation with Nash and Evolution tracks

---

## Risk Mitigation

### Risk: PPO doesn't match evolution performance

**Likelihood**: Medium
**Impact**: High (challenges neural network viability)

**Mitigation**:
1. Hyperparameter tuning (learning rate, batch size, network size)
2. Architectural changes (attention, recurrence)
3. Curriculum learning (start simple, increase difficulty)
4. Population-based training (diverse exploration)

### Risk: GPU utilization remains low

**Likelihood**: Low (VectorEnv fixed)
**Impact**: Medium (slow training)

**Mitigation**:
1. Increase num_envs (256 → 512)
2. Increase population_size (1 → 8+)
3. Profile bottlenecks (CPU vs. GPU bound)
4. Optimize data transfer (reduce CPU→GPU copies)

### Risk: Behavioral analysis inconclusive

**Likelihood**: Medium
**Impact**: Low (can still compare performance)

**Mitigation**:
1. Use multiple analysis methods (activation, gradient, cloning)
2. Visualize intermediate representations
3. Ablation studies on network components
4. Compare to interpretable baselines (linear policies)

---

## Resources

### Computational

- **GPU**: High (2-10 GPU-hours per scenario)
- **Total**: 24-120 GPU-hours for full suite
- **Memory**: 16-24GB GPU RAM (L4, A10, or better)
- **Storage**: ~5GB per scenario (checkpoints + logs)

### Optimization Strategies

**Sequential training** (1 GPU):
- Train scenarios one at a time
- 24-48 hours wall-clock

**Parallel training** (6 GPUs):
- Train 6 scenarios simultaneously
- 4-8 hours wall-clock

**Population training**:
- Better GPU utilization (60-95%)
- Faster convergence per scenario
- More insights (diversity analysis)

---

## References

### Internal

- **Phase 2 Agenda**: `docs/PHASE_2_RESEARCH_AGENDA.md`
- **MARL README**: `experiments/marl/README.md`
- **Population Training Guide**: `POPULATION_TRAINING.md`
- **VectorEnv Fix**: `docs/VECTORIZED_TRAINING_FIX.md`

### External

- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Lowe et al. (2017): "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
- Paine et al. (2021): "Making Efficient Use of Demonstrations to Solve Hard Exploration Problems"
- Huang et al. (2022): "Sample Factory: Scaling RL to Many-core Machines"

---

**Status**: 🚀 Ready to Execute
**Owner**: MARL Track Lead
**Next Steps**: Begin Experiment 1 (chain_reaction baseline)
