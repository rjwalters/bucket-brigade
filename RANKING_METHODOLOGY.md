# Bucket Brigade: Policy Ranking Methodology

## Executive Summary

We rank individual policies by their **marginal contribution** to team performance across **diverse team compositions** and **scenarios**. This approach solves a fundamental challenge in cooperative multi-agent systems: agent performance is highly context-dependent, varying based on both teammates and environmental conditions.

**Key Innovation**: Instead of evaluating policies in isolation or fixed teams, we run large-scale tournaments with **mixed teams** across **diverse scenarios**, then use statistical modeling to estimate each policy's individual value.

---

## The Problem: Why Traditional Evaluation Fails

Traditional agent evaluation methods are insufficient for cooperative games like Bucket Brigade:

| Method | Limitation | Example Failure Mode |
|--------|------------|---------------------|
| **Self-play** | Doesn't reveal cooperation with diverse partners | A "Hero" agent works well with other Heroes, but fails with Free Riders |
| **Fixed teams** | Can't isolate individual contributions | Did Team A win because of Agent 1's skill or lucky teammates? |
| **Single scenarios** | Policies overfit to specific dynamics | "Greedy Neighbor specialist" fails in "Chain Reaction" scenarios |
| **Round-robin tournaments** | Only tests pairwise interactions | Misses emergent team dynamics (3+ agents) |

**The Core Challenge**: We need to answer "How much value does Agent X add to a random team?" while accounting for:
- Team composition effects (who are the teammates?)
- Scenario diversity (which game parameters?)
- Statistical uncertainty (how confident are we?)

---

## Our Solution: Mixed Team Tournaments

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RANKING PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Tournament Design                                        │
│     ├─ Generate diverse team compositions (4-10 agents)     │
│     ├─ Sample scenarios (9 named scenarios)                 │
│     └─ Run 100+ games per policy                            │
│                                                              │
│  2. Data Collection                                          │
│     ├─ Record team rewards (collective outcome)             │
│     ├─ Record individual rewards (per-agent outcome)        │
│     └─ Track scenario parameters (beta, kappa, c, etc.)     │
│                                                              │
│  3. Statistical Modeling                                     │
│     ├─ Fit surrogate model (additive or interaction)        │
│     ├─ Estimate individual skill parameters (θ_i)           │
│     └─ Compute marginal contributions (v_i)                 │
│                                                              │
│  4. Ranking & Uncertainty                                    │
│     ├─ Rank policies by marginal value                      │
│     ├─ Compute confidence intervals (Bayesian posterior)    │
│     └─ Test for significant differences (hypothesis tests)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Tournament Structure

### Mixed Team Composition

**Why Mixed Teams?**
- **Cooperation testing**: Reveals how policies work with diverse partners (not just clones)
- **Marginal value estimation**: Enables isolating individual contributions from team effects
- **Robustness**: Policies must succeed across team compositions, not just favorable matchups

**Team Design Principles**:
```python
# Example team compositions (4 agents per game)
Game 1:  [Firefighter, Firefighter, Free Rider, Coordinator]
Game 2:  [Free Rider, Free Rider, Hero, Opportunist]
Game 3:  [Firefighter, Coordinator, Hero, Strategist]
Game 4:  [Free Rider, Free Rider, Free Rider, Free Rider]  # Homogeneous test
...
```

**Key Features**:
- Team size varies (4-10 agents) to match deployment scenarios
- Random sampling ensures balanced coverage of all policies
- Homogeneous teams (all same policy) included as baseline
- Avoid near-duplicate rosters (diversity constraint)

### Diverse Scenario Testing

**Why Multiple Scenarios?**

The Bucket Brigade game has **9 named scenarios**, each with hand-tuned parameters that create distinct cooperation challenges:

| Scenario | Key Challenge | Parameter Profile |
|----------|---------------|-------------------|
| **trivial_cooperation** | Easy cooperation | High κ (easy extinguish), low β (slow spread) |
| **greedy_neighbor** | Social dilemma | High work cost (c=1.0), free-riding incentive |
| **sparse_heroics** | Efficiency matters | Few workers needed, coordination critical |
| **chain_reaction** | Distributed response | High spread (β), requires coverage |
| **deceptive_calm** | Trust & signaling | Rare flare-ups, honest signals rewarded |
| **overcrowding** | Too many cooks | Diminishing returns from extra workers |
| **mixed_motivation** | Self-interest vs team | Conflicting individual/collective optima |
| **rest_trap** | False security | Fires often self-extinguish (but not always!) |
| **early_containment** | Aggressive start | Fast action required, early coordination |

**Testing Strategy**:
- Run each policy across **all 9 scenarios** (or stratified sample)
- Compute **scenario-specific rankings** (identify specialists)
- Compute **aggregate rankings** (overall performance)
- Weight scenarios by deployment distribution (if applicable)

**Example**: A policy might rank:
- 🥇 #1 in "greedy_neighbor" (handles free-riding well)
- 🥉 #7 in "chain_reaction" (poor spatial coordination)
- 🥈 #3 overall (aggregate across scenarios)

---

## 2. Data Collection Format

Each completed game produces one row:

```python
{
    "episode_id": "ep_001234",
    "scenario_id": "greedy_neighbor",
    "team": [0, 0, 1, 2],  # Policy IDs: [FF, FF, FR, CO]
    "team_reward": 245.3,  # Collective outcome
    "agent_rewards": [62.1, 61.8, 58.4, 63.0],  # Individual outcomes
    "nights_played": 12,
    "houses_saved": 8,
    "replay_path": "replays/ep_001234.json"
}
```

**Key Fields**:
- `team`: List of policy IDs (which agents played)
- `team_reward`: Sum of all agent rewards (team performance)
- `agent_rewards`: Individual rewards for each agent (basis for ranking)
- `scenario_id`: Which game parameters were used

**Storage**:
- CSV/SQLite for tabular data (fast queries)
- JSON replays for visualization (optional, large)

---

## 3. Statistical Modeling: Estimating Individual Value

### The Challenge

Given mixed team data, how do we estimate each policy's **marginal contribution**?

**Example Problem**:
- Game 1: Team [FF, FF, FR, CO] → Rewards [62.1, 61.8, 58.4, 63.0]
- Game 2: Team [FR, FR, HE, OP] → Rewards [52.3, 51.9, 48.2, 46.3]

**Question**: Is Firefighter (FF) better than Free Rider (FR)?
- FF scored 62.1 in Game 1, but had good teammates (FF, CO)
- FR scored 52.3 in Game 2, but had poor teammates (FR, OP)
- **We can't compare these directly!** Need to account for team composition.

### Method A: Simple Average (Current Implementation)

**Approach**: Average individual rewards across all games

```python
θ_Firefighter = mean([62.1, 61.8, ...])  # Average across all FF appearances
θ_FreeRider = mean([58.4, 52.3, 51.9, ...])  # Average across all FR appearances
```

**Ranking**: Sort policies by θ_i

**Pros**:
- ✅ Simple, interpretable
- ✅ Fast to compute
- ✅ Works well if games are balanced (each policy plays similar team compositions)

**Cons**:
- ❌ Biased if team compositions are imbalanced
- ❌ No uncertainty quantification (confidence intervals)
- ❌ Ignores team synergies (interaction effects)

**When to Use**: Initial exploration, quick comparisons, balanced tournaments

### Method B: Bayesian Additive Model (Advanced)

**Approach**: Fit a statistical model that decomposes team rewards into individual contributions

**Model**:
```
y_g = α + μ_scenario(g) + Σ_{i ∈ team(g)} θ_i + ε_g

where:
  y_g = observed outcome for game g (e.g., team reward or log(team reward))
  α = intercept (baseline reward)
  μ_scenario(g) = scenario difficulty offset
  θ_i = skill parameter for policy i (what we want to estimate!)
  ε_g ~ Normal(0, σ²) = noise
```

**Interpretation**:
- `θ_i` represents policy i's **additive contribution** to team performance
- Higher θ_i → higher marginal value
- Model assumes **additive effects** (no synergies between policies)

**Bayesian Inference**:
```python
# Prior (regularization)
θ_i ~ Normal(0, τ²)  # Ridge prior

# Posterior (closed-form for linear model)
θ_post ~ Normal(θ_hat, Σ)

where:
  θ_hat = (X^T X + λI)^(-1) X^T y  # Ridge regression
  Σ = σ² (X^T X + λI)^(-1)  # Posterior covariance
```

**Design Matrix X**:
```
Game 1: [1, 0, 1, 0, ...]  # One-hot encoding: FF appears twice
Game 2: [0, 1, 0, 1, ...]  # FR appears twice, HE once, OP once
...
```

**Output**:
- **Point estimates**: θ_i (mean skill)
- **Uncertainty**: SE(θ_i) or 95% credible intervals
- **Rankings**: Sort by θ_i, with confidence bands

**Marginal Value**:
```python
# For additive model, marginal value = θ_i directly
v_i = θ_i

# For deployment-specific value:
v_i = E_{(S, scenario) ~ deployment} [f(S ∪ {i}) - f(S)]
```

**Pros**:
- ✅ Accounts for team composition bias
- ✅ Provides uncertainty quantification (confidence intervals)
- ✅ Statistically rigorous (Bayesian inference)
- ✅ Handles imbalanced data (regularization prevents overfitting)

**Cons**:
- ❌ Assumes additive effects (no synergies)
- ❌ More complex to implement
- ❌ Requires sufficient data (100+ games per policy)

**When to Use**: Final rankings, publication-quality results, imbalanced tournaments

### Method C: Additive + Interactions (Future Extension)

**Enhancement**: Add low-rank interaction terms to capture synergies

```
y_g = α + μ_scenario + Σ_i θ_i + <Σ_i u_i, Σ_i v_i> + ε_g

where:
  u_i, v_i ∈ R^k (k=4-16) = latent factors for policy i
  Interaction term captures synergies (e.g., FF + CO work well together)
```

**Use Case**: When additive model has large residuals (team effects matter)

---

## 4. Ranking Output & Interpretation

### Example Ranking (Method B)

```
┌─────────────────────────────────────────────────────────────┐
│                   POLICY RANKINGS                            │
│                 (Scenario: greedy_neighbor)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Rank | Policy        | θ_i (Skill) | 95% CI      | Games  │
│  ─────────────────────────────────────────────────────────  │
│   1   | Firefighter   |  62.1       | [59.8, 64.4]|  147   │
│   2   | Coordinator   |  61.5       | [59.0, 64.0]|  139   │
│   3   | Hero          |  58.3       | [55.9, 60.7]|  152   │
│   4   | Strategist    |  56.8       | [54.2, 59.4]|  128   │
│   5   | Free Rider    |  53.2       | [50.8, 55.6]|  144   │
│   6   | Opportunist   |  47.8       | [45.1, 50.5]|  131   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Statistical Significance Testing

**Question**: Is Firefighter significantly better than Coordinator?

```python
# Hypothesis test
H0: θ_FF = θ_CO  (no difference)
H1: θ_FF > θ_CO  (Firefighter is better)

# Test statistic (from posterior)
diff = θ_FF - θ_CO = 62.1 - 61.5 = 0.6
SE(diff) = sqrt(Var(θ_FF) + Var(θ_CO) - 2*Cov(θ_FF, θ_CO))

# p-value
p = P(Z > diff / SE(diff))  where Z ~ Normal(0,1)
```

**Interpretation**:
- If p < 0.05: Difference is statistically significant
- If p > 0.05: Not enough evidence (ranks may overlap)
- **Confidence intervals overlapping** → ranks not significantly different

### Cross-Scenario Aggregation

**Scenario-Specific Rankings**:
```
Policy         | trivial | greedy | sparse | chain | ... | Average
─────────────────────────────────────────────────────────────────
Firefighter    |   1     |   1    |   3    |   2   | ... |  1.8
Coordinator    |   2     |   2    |   2    |   1   | ... |  1.9
Hero           |   3     |   3    |   1    |   4   | ... |  2.6
...
```

**Aggregate Ranking Methods**:
1. **Average rank**: Mean rank across scenarios
2. **Weighted average**: Weight by scenario deployment frequency
3. **Average θ_i**: Refit model with all scenarios (pooled data)
4. **Borda count**: Tournament-style voting

**Use Cases**:
- **Scenario-specific**: "Best policy for greedy_neighbor"
- **Aggregate**: "Best all-around policy across all scenarios"

---

## 5. Adaptive Batch Design (Optional Advanced Feature)

### The Opportunity

After collecting initial data, we can **intelligently design the next batch** of games to reduce ranking uncertainty efficiently.

**Goal**: Minimize uncertainty (narrow confidence intervals) with fewest games

**Strategies**:

#### A. A-Optimal Design (Minimize Average Variance)
```python
# Choose next games to minimize trace(Σ_new)
# Prioritize games that reduce uncertainty for all policies equally
```

#### B. D-Optimal Design (Maximize Information)
```python
# Choose next games to maximize log det(X^T X)
# Prioritize games with novel team compositions
```

#### C. Thompson Sampling (Robust Default)
```python
# Sample θ_i from posterior, find games with highest disagreement
# Focuses on uncertain policy comparisons
```

**Implementation Status**:
- ✅ Mathematical framework documented (RANKING_SYSTEM.md)
- ⏳ Adaptive design planned for future (current: fixed tournaments)

**When to Use**:
- Large policy pools (50+ policies)
- Limited compute budget
- Real-time learning scenarios

---

## Why This Methodology Works

### 1. Mixed Teams Reveal Cooperation

**Problem**: Self-play only tests homogeneous teams
**Solution**: Mixed teams reveal how policies work with diverse partners

**Example**:
- **Policy A (Team Player)**: θ_A = 60 (works well with anyone)
- **Policy B (Lone Wolf)**: θ_B = 65 (only works well with clones)
- **Winner**: Policy A (more robust, higher marginal value in mixed teams)

### 2. Marginal Contribution is Fair

**Problem**: Fixed teams conflate individual skill with teammate quality
**Solution**: Statistical model isolates individual contributions

**Example**:
- Agent on strong team: High reward, but low θ_i (carried by teammates)
- Agent on weak team: Low reward, but high θ_i (valuable despite poor support)

### 3. Diverse Scenarios Prevent Overfitting

**Problem**: Policies can specialize for specific game dynamics
**Solution**: Test across 9 scenarios with different cooperation challenges

**Example**:
- **Policy C (Specialist)**: Rank #1 in "greedy_neighbor", #8 overall
- **Policy D (Generalist)**: Rank #3 in all scenarios, #2 overall
- **Winner**: Policy D (more robust across scenarios)

### 4. Statistical Rigor Enables Confident Decisions

**Problem**: Noisy data, small sample sizes
**Solution**: Bayesian inference + confidence intervals

**Example**:
- Policy E: θ_E = 62.0 ± 4.0 (wide CI, uncertain)
- Policy F: θ_F = 61.0 ± 1.5 (narrow CI, confident)
- **Interpretation**: Can't confidently say E > F (CIs overlap)

---

## Implementation Guide

### Quick Start (Method A: Simple Average)

```python
# 1. Run tournament
results = run_tournament(
    policies=["firefighter", "free_rider", "hero", "coordinator"],
    num_games=100,
    team_size=4,
    scenarios=["trivial_cooperation", "greedy_neighbor"]
)

# 2. Compute rankings
rankings = {}
for policy_id, policy_name in enumerate(policies):
    # Get all games where this policy played
    policy_games = [g for g in results if policy_id in g["team"]]

    # Extract individual rewards
    rewards = []
    for game in policy_games:
        positions = [i for i, p in enumerate(game["team"]) if p == policy_id]
        rewards.extend([game["agent_rewards"][i] for i in positions])

    # Average
    rankings[policy_name] = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "num_games": len(policy_games)
    }

# 3. Sort by mean reward
sorted_rankings = sorted(rankings.items(), key=lambda x: x[1]["mean_reward"], reverse=True)

# 4. Print
for rank, (policy, stats) in enumerate(sorted_rankings, 1):
    print(f"{rank}. {policy}: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
```

### Production (Method B: Bayesian Model)

See `bucket_brigade/orchestration/ranking_model.py` (planned implementation)

**Steps**:
1. Collect tournament data (CSV format)
2. Build design matrix X (agents + scenarios)
3. Fit ridge regression: `θ_hat = (X^T X + λI)^(-1) X^T y`
4. Compute posterior covariance: `Σ = σ² (X^T X + λI)^(-1)`
5. Extract θ_i and SE(θ_i) for each policy
6. Rank by θ_i with confidence intervals

---

## Example Use Cases

### Use Case 1: Policy Development

**Goal**: Evaluate newly designed heuristic policies

**Workflow**:
1. Define 5 candidate policies (different parameter sets)
2. Run 100-game tournament (mixed teams, 2 scenarios)
3. Rank by simple average (Method A)
4. Select top 2 policies for further testing
5. Run 500-game tournament with top 2 + baselines
6. Rank by Bayesian model (Method B)
7. Deploy best policy

### Use Case 2: Evolutionary Algorithm Validation

**Goal**: Compare evolved policies to hand-designed heuristics

**Workflow**:
1. Run evolution to discover best policy
2. Run comparison tournament:
   - Evolved champion
   - 5 hand-designed heuristics (firefighter, coordinator, etc.)
   - 3 scenarios (trivial, greedy, chain)
3. Rank using Bayesian model
4. Test hypothesis: "Evolved policy significantly better than best heuristic?"
5. Analyze failure modes (which scenarios does evolved policy struggle in?)

### Use Case 3: Nash Equilibrium Analysis

**Goal**: Identify stable strategic configurations

**Workflow**:
1. Compute Nash equilibrium (mixed strategy)
2. Sample policies from Nash distribution
3. Run tournament with Nash samples + heuristics
4. Rank using Bayesian model
5. Compare: "Do Nash strategies outperform heuristics?"
6. Analyze: "Does evolution converge to Nash equilibrium?"

---

## Technical Details & References

### Design Matrix Construction

For G games and N policies:

```python
# Example: 3 policies, 2 games
#   Game 1: [Policy 0, Policy 0, Policy 1]
#   Game 2: [Policy 1, Policy 2, Policy 2]

X = [
    [2, 1, 0],  # Game 1: Policy 0 appears twice, Policy 1 once
    [0, 1, 2],  # Game 2: Policy 1 once, Policy 2 twice
]

y = [
    180.5,  # Game 1 team reward
    165.3,  # Game 2 team reward
]
```

**Extensions**:
- Add scenario columns (one-hot or linear features)
- Center columns to improve conditioning
- Use sparse CSR format for large tournaments

### Regularization & Hyperparameters

**Ridge penalty**: `λ = N / SNR` where SNR ≈ signal-to-noise ratio
- Low λ: Less regularization (trust data)
- High λ: More regularization (shrink toward zero)
- **Heuristic**: Start with λ = 1.0, tune via cross-validation

**Noise variance**: `σ² = empirical residual variance` or fit via maximum likelihood

### Computational Complexity

**Tournament execution**:
- Single game: ~5ms (WASM engine)
- 100 games: ~500ms (sequential)
- With 4 workers: ~125ms (parallel)

**Ranking computation**:
- Simple average (Method A): O(G) time
- Bayesian model (Method B): O(N³ + GN) time (dominated by matrix inverse)
- For N=50 policies, G=1000 games: <1 second

### Storage Requirements

**Per game**:
- Tabular data: ~200 bytes (CSV row)
- Replay JSON: ~10 KB (full game state)

**For 1000 games**:
- CSV: ~200 KB
- Replays: ~10 MB

---

## Current Implementation Status

### ✅ Implemented
- Mixed team tournament orchestration (`scripts/run_batch.py`)
- CSV data collection format
- Simple average ranking (Method A)
- Scenario definitions (9 named scenarios)
- Web-based tournament runner (Team Builder feature)

### ⏳ Planned
- Bayesian ranking model (Method B) with ridge regression
- Confidence interval computation
- Adaptive batch design (A-optimal, Thompson sampling)
- Cross-scenario aggregation utilities
- Statistical significance testing

### 📚 Documentation
- Mathematical framework: `docs/game-design/RANKING_SYSTEM.md`
- Scenario research: `SCENARIO_RESEARCH.md`
- Tournament UI: `docs/features/TEAM_BUILDER_TOURNAMENT.md`
- This methodology overview: `RANKING_METHODOLOGY.md`

---

## Comparison to Alternatives

| Method | Mixed Teams? | Scenarios? | Uncertainty? | Complexity |
|--------|-------------|------------|--------------|------------|
| **Self-play** | ❌ No | Single | ❌ No | Low |
| **Round-robin** | ✅ Pairwise | Single | ❌ No | Medium |
| **Elo rating** | ✅ Yes | Single | ✅ Yes | Medium |
| **Our method (A)** | ✅ Yes | ✅ Multiple | ❌ No | Medium |
| **Our method (B)** | ✅ Yes | ✅ Multiple | ✅ Yes | High |

**Why Our Method?**
- Only approach that handles **mixed teams** + **diverse scenarios** + **uncertainty**
- Elo rating assumes pairwise comparisons (doesn't scale to 4-10 agent teams)
- Round-robin misses emergent team dynamics (3+ agents)
- Self-play fails to test cooperation with diverse partners

---

## Frequently Asked Questions

### Q1: Why not just use self-play?

**A**: Self-play only tests homogeneous teams (all agents are clones). This fails to reveal:
- How well a policy cooperates with diverse partners
- Whether a policy is carried by teammates or adds value
- Robustness to different team compositions

**Example**: A policy that only works well with clones scores high in self-play but low in mixed teams.

### Q2: How many games do I need per policy?

**A**: Depends on desired confidence level:
- **Quick test**: 20-50 games (rough comparison)
- **Standard**: 100-200 games (confident rankings)
- **Publication**: 500+ games (tight confidence intervals)

**Rule of thumb**: SE(θ_i) ≈ σ / √N where N = number of games policy appears in

### Q3: Should I use Method A or Method B?

**A**:
- **Method A (Simple Average)** if:
  - Quick exploration / prototyping
  - Balanced tournaments (each policy plays similar team compositions)
  - Don't need confidence intervals

- **Method B (Bayesian Model)** if:
  - Final rankings for publication
  - Imbalanced data (some policies play more games)
  - Need uncertainty quantification (confidence intervals)
  - Testing statistical significance of rank differences

### Q4: How do I interpret overlapping confidence intervals?

**A**: If 95% CIs overlap, you **cannot confidently say one policy is better** than another. Options:
- Collect more data (narrow CIs)
- Use paired comparison test (directly test difference)
- Accept that ranks are too close to distinguish

### Q5: Can I weight scenarios by importance?

**A**: Yes! Two approaches:
1. **Sampling**: Run more games in important scenarios
2. **Weighted aggregation**: `θ_i = Σ_s w_s * θ_{i,s}` where w_s = scenario weight

### Q6: What if I have computational constraints?

**A**: Use **adaptive batch design** (future feature):
- Start with small tournament (50 games)
- Identify uncertain comparisons (wide CIs)
- Design next batch to reduce uncertainty for those comparisons
- Repeat until convergence

**Benefit**: ~2-3x fewer games for same confidence level

---

## Citation

If you use this ranking methodology in your research, please cite:

```bibtex
@software{bucket_brigade_ranking,
  title = {Bucket Brigade: Mixed Team Tournament Ranking for Multi-Agent Evaluation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/rjwalters/bucket-brigade}
}
```

---

## Contact & Contributions

- **GitHub**: [rjwalters/bucket-brigade](https://github.com/rjwalters/bucket-brigade)
- **Documentation**: See `docs/` directory
- **Issues**: Report bugs or suggest improvements via GitHub Issues
- **Contributions**: Pull requests welcome!

---

*Last updated: 2025-11-03*
*Status: Active Development*
