# Research Progress Tracker

## Extended Evolution Experiments

### Objective
Improve evolutionary optimization to match or beat hand-designed heuristics by running longer evolution experiments with larger populations.

### Current Runs (In Progress)

| Scenario | Status | Population | Generations | Games | Started |
|----------|--------|------------|-------------|-------|---------|
| deceptive_calm | ✅ Complete | 50 | 50 | 20 | 2025-11-03 |
| sparse_heroics | 🔄 Running | 100 | 100 | 30 | 2025-11-03 |
| rest_trap | 🔄 Running | 100 | 100 | 30 | 2025-11-03 |
| trivial_cooperation | 🔄 Running | 100 | 100 | 30 | 2025-11-03 |

### Results

#### deceptive_calm (Complete)

**Quick Run** (baseline):
- Population: 20, Generations: 10, Games: 5
- Best evolved fitness: -0.5
- Tournament: best_heuristic 81.48 vs evolved 15.31
- Gap: -81%

**Extended Run** (50/50/20):
- Population: 50, Generations: 50, Games: 20
- Best evolved fitness: -2.60 (worse!)
- Tournament: best_heuristic 50.39 vs evolved 12.62
- Gap: -75%
- Converged: Generation 10 (early stopping)

**Evolved Strategy**:
```
neighbor_help: 1.0   ████████████████████ (maxed out)
altruism: 0.43       ████████░░░░░░░░░░░░
rest_bias: 0.53      ██████████░░░░░░░░░░
honesty: 0.28        █████░░░░░░░░░░░░░░░
risk_aversion: 0.24  ████░░░░░░░░░░░░░░░░
work_tendency: 0.11  ██░░░░░░░░░░░░░░░░░░ (very low)
own_priority: 0.0    ░░░░░░░░░░░░░░░░░░░░ (zero!)
```

**Analysis**:
- Strategy is "altruistic non-worker": helps neighbors but doesn't work
- This is suboptimal for deceptive_calm (high kappa=0.6, needs active firefighting)
- Extended run made things worse (early convergence to local minimum)
- Issue: Fitness function rewards team performance, but this strategy free-rides

**Lessons Learned**:
1. More generations != better results if fitness function is misaligned
2. Early convergence (gen 10) suggests population diversity collapsed
3. Need competitive co-evolution against best_heuristic specifically
4. Current fitness (mean_reward) allows free-riding

### Research Questions

**Q1: Why did extended evolution perform worse?**
- Hypothesis: Converged to local minimum (altruistic free-rider)
- Need: Diversity maintenance, better fitness function

**Q2: What fitness function improvements would help?**
Options:
- Individual reward (not team average)
- Head-to-head vs best_heuristic
- Multi-objective: performance + diversity
- Penalize free-riding (low work_tendency)

**Q3: Should we use warm-start from heuristics?**
- Pro: Start near known good solutions
- Con: May limit exploration
- Decision: Test on next scenario

### Next Steps

1. **Wait for current runs to complete** (sparse_heroics, rest_trap, trivial_cooperation)
2. **Analyze results** - do they show similar issues?
3. **If pattern continues**:
   - Implement competitive co-evolution
   - Add individual fitness option
   - Try warm-start initialization
4. **If some improve**:
   - Identify what makes those scenarios different
   - Adjust approach per scenario type

### Key Insights

**Evolution is Hard**:
- Our heuristics are surprisingly good
- 10-dimensional parameter space is large
- Team-based fitness allows free-riding
- Need smarter search, not just more compute

**Scenario Characteristics Matter**:
- deceptive_calm (high kappa, medium beta, low cost):
  - Needs active firefighting
  - Punishes free-riding less (fires extinguish easily)
  - Evolved agent learned to help neighbors but not work directly

**Next Experiments Should**:
- Test individual vs team fitness
- Add explicit anti-free-riding penalties
- Use competitive benchmarking during evolution
- Consider scenario-specific initialization

---

*Last updated: 2025-11-03*
*In progress: 3 scenarios running*
