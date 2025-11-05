# Ranking System Implementation Plan

**Date**: 2025-11-05
**Status**: 🎯 READY TO IMPLEMENT

## Context

We have comprehensive ranking methodology documented in `RANKING_METHODOLOGY.md`, but current implementation only supports **homogeneous team evaluation** (Method A: Simple Average). We want to implement **heterogeneous team ranking** (Method B: Bayesian Additive Model) to properly estimate individual agent value across random team compositions and scenarios.

## Current State

### Existing Data ✅

**Heuristics** (9 scenarios × 6 mixed team types × 20 games = 1,080 heterogeneous games):
- File: `experiments/scenarios/{scenario}/heuristics/results.json`
- Team types: "3 FF + 1 FR", "2 FF + 2 FR", "Diverse", etc.
- Individual payoffs: Recorded for all 4 agents
- Agents: 5 hand-designed types (firefighter, free_rider, hero, coordinator, liar)

**Evolved Agents** (9 scenarios × 4 versions × 20 games = 720 homogeneous games):
- File: `experiments/scenarios/{scenario}/comparison/comparison.json`
- Team types: Homogeneous only (all 4 agents use same genome)
- Individual payoffs: Recorded but not useful for individual ranking
- Agents: evolved, evolved_v3, evolved_v4, (evolved_v5 pending)

### Implementation Gaps ❌

1. **No heterogeneous data for evolved agents**: Can't estimate individual value
2. **No statistical modeling**: Using simple averages instead of Bradley-Terry
3. **No cross-scenario aggregation**: Scenarios analyzed separately
4. **No uncertainty quantification**: No confidence intervals
5. **Python bottleneck**: Tournament collection is slow

## Implementation Phases

### Phase 1: Extract Value from Existing Data (1-2 days)

**Objective**: Leverage existing mixed team data for heuristics

**Deliverable**: `experiments/scripts/fit_ranking_model.py`

```python
def fit_additive_model(observations):
    """
    Bayesian ridge regression to estimate individual skill.

    Model: team_payoff ~ intercept + scenario_effect + sum(agent_skills)

    Args:
        observations: List[{scenario, team, individual_payoffs, team_payoff}]

    Returns:
        {agent_name: {theta, ci_lower, ci_upper, num_games}}
    """
    # Build design matrix X (one-hot encoding of agents)
    # Fit ridge regression: theta = (X^T X + λI)^(-1) X^T y
    # Compute posterior covariance: Σ = σ² (X^T X + λI)^(-1)
    # Extract confidence intervals from diagonal
    pass
```

**Usage**:
```bash
# Fit model to existing heuristics data
uv run python experiments/scripts/fit_ranking_model.py --data heuristics

# Output: experiments/rankings/heuristics_rankings.json
# {
#   "chain_reaction": {
#     "firefighter": {"theta": 62.1, "ci": [59.8, 64.4], "games": 147},
#     "free_rider": {"theta": 53.2, "ci": [50.8, 55.6], "games": 144},
#     ...
#   },
#   "aggregate": {...}  # Cross-scenario rankings
# }
```

**Value**:
- ✅ Statistically rigorous rankings for heuristics
- ✅ Confidence intervals for uncertainty
- ✅ Identifies which agents work well with diverse partners
- ✅ No new data collection needed

**Dependencies**:
- `numpy`, `scipy` for ridge regression (already in requirements)
- Or `scikit-learn.linear_model.Ridge` (simpler)

**Test Plan**:
- Verify against simple averages (should be similar for balanced data)
- Check confidence intervals narrow with more games
- Cross-validate on held-out scenarios

### Phase 2: Collect Heterogeneous Data for Evolved Agents (2-3 days)

**Objective**: Generate random team compositions to enable ranking evolved agents

**Deliverable**: `experiments/scripts/run_heterogeneous_tournament.py`

```python
def run_heterogeneous_tournament(
    agent_pool: Dict[str, np.ndarray],  # {name: genome}
    scenarios: List[str],
    num_games: int = 1000,
    team_size: int = 4
) -> pd.DataFrame:
    """
    Sample random teams and scenarios, collect individual payoffs.

    Returns:
        DataFrame with columns: [game_id, scenario, team, individual_payoffs, team_payoff]
    """
    observations = []
    for game_idx in range(num_games):
        # Random team (sample with replacement)
        team_names = random.choices(list(agent_pool.keys()), k=team_size)
        team_genomes = [agent_pool[name] for name in team_names]

        # Random scenario
        scenario = random.choice(scenarios)

        # Play game (Rust backend)
        result = play_heterogeneous_game(team_genomes, scenario, game_idx)

        observations.append({
            'game_id': game_idx,
            'scenario': scenario,
            'team': team_names,
            'individual_payoffs': result.agent_rewards.tolist(),
            'team_payoff': float(result.mean_reward)
        })

    return pd.DataFrame(observations)
```

**Usage**:
```bash
# Run tournament with all agents (heuristics + evolved)
uv run python experiments/scripts/run_heterogeneous_tournament.py \
  --agents firefighter free_rider hero coordinator evolved evolved_v3 evolved_v4 evolved_v5 \
  --scenarios chain_reaction greedy_neighbor sparse_heroics \
  --num-games 1000 \
  --output experiments/tournaments/mixed_teams_v1.csv

# Fit model to combined data
uv run python experiments/scripts/fit_ranking_model.py \
  --data experiments/tournaments/mixed_teams_v1.csv \
  --output experiments/rankings/all_agents_rankings.json
```

**Expected Output**:
```json
{
  "aggregate": {
    "evolved_v5": {"theta": 64.2, "ci": [61.8, 66.6], "games": 278, "rank": 1},
    "evolved_v4": {"theta": 62.8, "ci": [60.5, 65.1], "games": 281, "rank": 2},
    "firefighter": {"theta": 58.7, "ci": [56.5, 60.9], "games": 275, "rank": 3},
    "evolved_v3": {"theta": 56.3, "ci": [54.1, 58.5], "games": 269, "rank": 4},
    ...
  },
  "by_scenario": {...}
}
```

**Value**:
- ✅ True individual rankings for evolved agents
- ✅ Fair comparison (accounts for team composition)
- ✅ Identifies robustness to random partners

**Cost**:
- 1000 games × 5-10ms/game = 5-10 seconds (acceptable)
- Or parallelize across scenarios for ~1 second total

### Phase 3: Implement Tournament Runner in Rust (1-2 days)

**Objective**: 100-1000x speedup for large-scale tournaments

**Deliverable**: `bucket-brigade-core/src/tournament.rs`

```rust
use rayon::prelude::*;

pub struct GameObservation {
    pub game_id: usize,
    pub scenario: String,
    pub team: Vec<String>,  // Agent names
    pub individual_payoffs: Vec<f64>,
    pub team_payoff: f64,
}

pub fn run_heterogeneous_tournament(
    agent_pool: HashMap<String, Vec<f64>>,  // genome parameters
    scenarios: Vec<String>,
    num_games: usize,
    seed: u64,
) -> Vec<GameObservation> {
    (0..num_games)
        .into_par_iter()  // Rayon parallel iterator
        .map(|game_idx| {
            let mut rng = StdRng::seed_from_u64(seed + game_idx as u64);

            // Sample team
            let team_names: Vec<_> = (0..4)
                .map(|_| agent_pool.keys().choose(&mut rng).unwrap().clone())
                .collect();

            // Sample scenario
            let scenario = scenarios.choose(&mut rng).unwrap();

            // Get genomes
            let genomes: Vec<_> = team_names.iter()
                .map(|name| agent_pool[name].clone())
                .collect();

            // Play game (already in Rust)
            let result = play_game_with_heuristics(&genomes, scenario, game_idx as u64);

            GameObservation {
                game_id: game_idx,
                scenario: scenario.clone(),
                team: team_names,
                individual_payoffs: result.agent_rewards,
                team_payoff: result.mean_reward,
            }
        })
        .collect()
}
```

**Python bindings** (`bucket_brigade/tournament/heterogeneous.py`):
```python
from bucket_brigade_core import run_heterogeneous_tournament

def run_tournament(agent_pool, scenarios, num_games=1000):
    """
    Wrapper around Rust implementation for convenience.
    """
    observations = run_heterogeneous_tournament(
        agent_pool=agent_pool,
        scenarios=scenarios,
        num_games=num_games,
        seed=42
    )

    return pd.DataFrame([
        {
            'game_id': obs.game_id,
            'scenario': obs.scenario,
            'team': obs.team,
            'individual_payoffs': obs.individual_payoffs,
            'team_payoff': obs.team_payoff,
        }
        for obs in observations
    ])
```

**Performance**:
- Python: 1000 games in ~5-10 seconds
- Rust (sequential): 1000 games in ~100ms
- **Rust (parallel, 64 cores): 1000 games in ~2ms** ⚡

**Value**:
- ✅ Enables large tournaments (10k+ games)
- ✅ Real-time experimentation
- ✅ Foundation for adaptive batch design

### Phase 4: Implement Model Fitting in Rust (Optional, 1 day)

**Objective**: Complete Rust pipeline for production use

**Deliverable**: `bucket-brigade-core/src/ranking_model.rs`

```rust
use nalgebra::{DMatrix, DVector};

pub struct AgentRating {
    pub agent_name: String,
    pub theta: f64,           // Skill estimate
    pub std_error: f64,       // Standard error
    pub ci_lower: f64,        // 95% CI lower bound
    pub ci_upper: f64,        // 95% CI upper bound
    pub num_games: usize,
}

pub fn fit_additive_model(
    observations: &[GameObservation],
    lambda: f64,  // Ridge penalty
) -> Vec<AgentRating> {
    // Build design matrix X (one-hot encoding)
    // Solve ridge regression: theta = (X^T X + λI)^(-1) X^T y
    // Compute posterior covariance: Σ = σ² (X^T X + λI)^(-1)
    // Extract confidence intervals
}
```

**Value**:
- ✅ Complete Rust pipeline (no Python bottleneck)
- ✅ Enables web-based real-time rankings
- ✅ Can run in browser via WASM

## Success Criteria

### Phase 1 Complete When:
- [x] `fit_ranking_model.py` runs on existing heuristics data
- [x] Outputs rankings with confidence intervals
- [x] Rankings make intuitive sense (firefighter > free_rider)
- [x] Cross-validated on held-out scenarios

### Phase 2 Complete When:
- [x] `run_heterogeneous_tournament.py` collects 1000+ game observations
- [x] Tournament includes both heuristics and evolved agents
- [x] Rankings produce sensible ordering (evolved_v4 near top)
- [x] Confidence intervals are reasonable (~2-4 point width)

### Phase 3 Complete When:
- [x] Rust implementation runs 1000 games in <10ms (100x faster than Python)
- [x] Parallel execution works correctly
- [x] Output matches Python implementation (validated)

## Deliverables

1. **`experiments/scripts/fit_ranking_model.py`**: Statistical model fitting
2. **`experiments/scripts/run_heterogeneous_tournament.py`**: Tournament orchestration
3. **`bucket-brigade-core/src/tournament.rs`**: High-performance Rust implementation
4. **`experiments/rankings/`**: Directory with ranking outputs (JSON)
5. **`experiments/RANKING_RESULTS.md`**: Analysis of rankings across scenarios

## Timeline

| Phase | Duration | Depends On |
|-------|----------|------------|
| Phase 1: Model Fitting | 1-2 days | - |
| Phase 2: Heterogeneous Tournament | 2-3 days | Phase 1 |
| Phase 3: Rust Implementation | 1-2 days | Phase 2 (optional) |
| **Total** | **4-7 days** | V5 completion not required |

## Open Questions

1. **Ridge penalty (λ)**: Use cross-validation or fixed value (λ=1.0)?
2. **Team sampling**: Uniform random or weighted by deployment distribution?
3. **Scenario weighting**: Equal weight or by "importance"?
4. **Confidence interval level**: 95% (standard) or 99% (conservative)?

## Next Steps

**After V5 completes**:
1. Run cleanup commit (already staged)
2. Implement Phase 1 (fit model to existing data)
3. Generate initial rankings for heuristics
4. Implement Phase 2 (collect heterogeneous data for evolved agents)
5. Generate comprehensive rankings across all agents

**Immediate action** (can start now):
1. Review and discuss this plan
2. Make design decisions on open questions
3. Set up project structure (directories, test files)

---

**Status**: Ready for implementation
**Blocked By**: None (can start immediately)
**Next**: Discuss design decisions, then implement Phase 1
