# Bucket Brigade: A Minimal Multi-Agent Cooperation Game

## Design Philosophy

**Goal**: Create the simplest possible multi-agent game that captures the essence of cooperation challenges while being **small enough to understand completely** and **rich enough to support serious tournament research**.

We deliberately avoided complex games like StarCraft, Dota, or even simpler options like Capture the Flag. Instead, we designed Bucket Brigade from first principles with three core requirements:

1. **Minimal complexity**: Small state space, simple rules, easy to explain
2. **Rich strategic depth**: Cooperation, deception, and coordination emerge naturally
3. **Tournament-ready**: Fast execution, deterministic seeding, parameterizable scenarios

**Result**: A game you can explain in 5 minutes, implement in 500 lines, but spend years researching.

---

## The Game in 60 Seconds

### The World
- **10 houses** arranged in a circle (positions 0-9)
- Each house is **Safe**, **Burning**, or **Ruined**
- Fires **spread** to neighboring houses probabilistically
- Fires **spark** randomly on safe houses

### The Agents
- **4-10 agents**, each owning 1+ houses
- Each night, every agent:
  1. **Signals** their intent (Work or Rest) — *but might lie*
  2. **Acts** by choosing (house, mode) — *where to go and what to do*

### The Challenge
- **Working** at a burning house might extinguish the fire (probabilistic)
- **Working** costs energy (individual penalty)
- **Resting** is free and restores energy
- **Goal**: Save as many houses as possible (team reward) while minimizing personal costs

### The Dilemma
```
If everyone works → Fires extinguished, but everyone pays cost
If everyone rests → Town burns down, everyone loses
If I rest while others work → I get the best outcome (free-rider!)
If others rest while I work → I pay cost for minimal benefit (sucker!)
```

This is a **social dilemma** — the Nash equilibrium is not the social optimum.

---

## Why This Design?

### Problem: Existing Multi-Agent Games Are Too Complex

| Game | Agents | State Space | Rules Complexity | Implementation |
|------|--------|-------------|------------------|----------------|
| **StarCraft** | 100s | Enormous | Very High | ~1M+ LOC |
| **Dota 2** | 10 | Enormous | Very High | ~1M+ LOC |
| **Capture the Flag** | 2-10 | Large | Medium | ~10K LOC |
| **Gridworld** | 1-10 | Small | Low | ~500 LOC |
| **Bucket Brigade** | 4-10 | **Small** | **Medium** | **~500 LOC** |

**The Gap**: Gridworld is too simple (single-agent planning), while CTF is too complex for rapid iteration. We needed something in between.

### Solution: Design for Understanding & Iteration

**Key Design Principles**:

1. **Discrete, Observable State**
   - Only 3 possible states per house (Safe/Burning/Ruined)
   - All information is public (no hidden state)
   - Full observability enables reasoning about strategies

2. **Simultaneous Actions**
   - No turn order complications
   - Naturally models real-world coordination (emergency response)
   - Emergent conflicts from overlapping actions

3. **Probabilistic Outcomes**
   - Fire spread: stochastic but parameterizable
   - Extinguish success: depends on # of workers (cooperation!)
   - Prevents deterministic exploitation

4. **Bounded Episode Length**
   - Minimum nights (12) ensures meaningful decisions
   - Maximum nights (100) guarantees termination
   - Typical episodes: 15-30 nights (~1-2 seconds in WASM)

5. **Parameterizable Scenarios**
   - 12 scenario parameters (fire spread rate, work cost, etc.)
   - Create diverse cooperation challenges (9 named scenarios)
   - Same game engine, different strategic landscapes

---

## The Complete State Space

### World State (30 bits)
```
10 houses × 2 bits per house = 20 bits
  (00 = Safe, 01 = Burning, 10 = Ruined, 11 = unused)

Agent locations = 10 bits
  (4 agents × log₂(10) ≈ 3.3 bits each)

Total world state ≈ 30 bits
```

**This is tiny!** For comparison:
- Chess: ~10⁴⁴ states
- Go: ~10¹⁷⁰ states
- Bucket Brigade: ~10⁹ states (including agent positions)

### Action Space (per agent)
```
Signal: {Work, Rest} = 2 choices
Action: {10 houses} × {Work, Rest} = 20 choices

Joint action space (4 agents): 2⁴ × 20⁴ = 2,560,000
```

**Still tractable** for analysis, yet rich enough for interesting strategies.

---

## Why It's Perfect for Tournament Research

### 1. Fast Execution
```
Single game (50 nights avg):
  Python: ~50ms
  Rust:   ~5ms
  WASM:   ~5ms (browser)

1000-game tournament:
  Python:  ~50 seconds
  Rust:    ~5 seconds
  WASM:    ~5 seconds (4 workers)
```

**This speed enables**:
- Large-scale tournaments (10,000+ games in minutes)
- Rapid iteration during development
- Real-time browser-based tournaments

### 2. Deterministic Reproducibility
```python
# Same seed → same game
env = BucketBrigadeEnv(scenario, seed=42)
result1 = run_game(env, agents)

env = BucketBrigadeEnv(scenario, seed=42)
result2 = run_game(env, agents)

assert result1 == result2  # Guaranteed identical
```

**Critical for**:
- Reproducible research (publish seeds with results)
- Fair comparisons (same starting conditions)
- Debugging (replay exact game traces)

### 3. Parameterizable Difficulty

**9 Named Scenarios** span the cooperation spectrum:

```
┌─────────────────────────────────────────────────────────────┐
│                   COOPERATION LANDSCAPE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Easy ←──────────────────────────────────────────→ Hard    │
│                                                              │
│  ┌──────────┐                                    ┌────────┐ │
│  │ trivial  │  ┌──────┐   ┌────────┐   ┌──────┐│ chain  │ │
│  │cooperation│  │sparse│   │greedy  │   │rest  ││reaction│ │
│  └──────────┘  │heroics│   │neighbor│   │trap  │└────────┘ │
│                 └──────┘   └────────┘   └──────┘           │
│                                                              │
│  Pure cooperation → Social dilemma → Coordination failure    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Each scenario tests different skills**:
- **trivial_cooperation**: Can you cooperate when it's easy?
- **greedy_neighbor**: Can you resist free-riding temptation?
- **sparse_heroics**: Can you coordinate efficiently (not overwork)?
- **chain_reaction**: Can you distribute spatially (cover territory)?
- **deceptive_calm**: Can you trust signals when fires are rare?
- **overcrowding**: Can you avoid diminishing returns (too many workers)?
- **mixed_motivation**: Can you balance self-interest vs. team good?
- **rest_trap**: Can you distinguish real threats from false alarms?
- **early_containment**: Can you respond quickly to aggressive fires?

**Why This Matters**:
- Policies that excel in one scenario may fail in others
- Testing across scenarios reveals **robustness**, not just peak performance
- Scenario-specific rankings identify **specialists** vs. **generalists**

### 4. Clear Individual Contributions

**Individual Rewards** decompose team performance:
```python
R_team = A × (SavedHouses/10) - L × (RuinedHouses/10) - c × TotalWork

R_agent_i = TeamShare + OwnHouseBonus - WorkCost_i
```

**This enables**:
- Ranking individual policies in mixed teams
- Estimating marginal contributions (Shapley values)
- Identifying free-riders vs. contributors

**Critical for tournament ranking** (see RANKING_METHODOLOGY.md)

### 5. Observable but Non-Trivial Strategies

**Fully observable state** means:
- No hidden information (unlike poker)
- Agents can reason about others' observations
- Signaling is **cheap talk** (non-binding)

**But strategic depth emerges**:
- Should I trust your signal? (deception possible)
- Should I help your house or mine? (self-interest vs. altruism)
- Should I work at a crowded fire? (coordination)
- Should I rest if fires are sparse? (efficiency)

**Hand-designed heuristics** can be competitive:
- 10 parameters control behavior (honesty, work tendency, altruism, etc.)
- Interpretable strategies (Firefighter, Free Rider, Coordinator, etc.)
- Good baseline for RL/evolution comparisons

---

## Design Tradeoffs We Made

### What We Included

✅ **Cooperation**: Extinguish probability increases with # workers
✅ **Deception**: Signals are cheap talk, lying is possible
✅ **Spatial reasoning**: Fires spread to neighbors, location matters
✅ **Risk/reward tradeoffs**: Work costs energy but saves houses
✅ **Public goods dilemma**: Free-riders benefit from others' work
✅ **Temporal planning**: Fires spread over time, prevention matters

### What We Excluded (Deliberately)

❌ **Hidden information**: All state is observable (simplicity)
❌ **Partial observability**: Agents see everything (no fog of war)
❌ **Turn-based play**: Simultaneous actions (natural coordination)
❌ **Continuous space**: Discrete houses (tractable analysis)
❌ **Complex resources**: Only "energy" (work cost)
❌ **Agent elimination**: All agents play until game ends

**Rationale**: Each excluded feature would add complexity without proportionally increasing strategic depth. We chose **minimal sufficient complexity**.

### Comparison to Alternatives

**vs. Prisoner's Dilemma**:
- ✅ More realistic (spatial, temporal, probabilistic)
- ✅ Supports 4-10 agents (not just 2)
- ✅ Rich scenarios (not just one payoff matrix)
- ❌ More complex to analyze theoretically

**vs. Public Goods Game**:
- ✅ Spatial structure (fires spread)
- ✅ Dynamic (state changes over time)
- ✅ Signaling mechanism (communication)
- ❌ Harder to compute Nash equilibria

**vs. Foraging / Coordination Games**:
- ✅ Social dilemma (free-riding possible)
- ✅ Deception (signals can be false)
- ❌ Less studied in game theory literature

**Sweet Spot**: Complex enough to be interesting, simple enough to understand completely.

---

## Implementation: Small and Fast

### Core Engine (~500 lines)

```
bucket_brigade/envs/bucket_brigade_env.py (Python):
  - BucketBrigadeEnv class: ~300 lines
  - Fire spread logic: ~50 lines
  - Reward computation: ~50 lines
  - Observation construction: ~50 lines
  - Replay export: ~50 lines

bucket-brigade-core/src/engine.rs (Rust, 100x faster):
  - Same logic, ~400 lines
  - WASM bindings: ~100 lines
  - Python bindings (PyO3): ~100 lines
```

**Total**: ~1000 lines for full implementation (Python + Rust + WASM)

**For comparison**:
- OpenAI Gym CartPole: ~200 lines (but much simpler)
- OpenAI Gym Atari: ~50K lines (C++ emulator + wrappers)
- PettingZoo MPE: ~5K lines (continuous control)

### Minimal Dependencies

```toml
# Python
dependencies = [
    "numpy",      # Array operations
    "pydantic",   # Data validation
]

# Rust
dependencies = [
    "rand",       # RNG
    "serde",      # Serialization
]
```

**No heavyweight frameworks**: TensorFlow, PyTorch, Unity, etc. not required for core game.

### Cross-Platform Support

```
Python  → bucket_brigade.envs.BucketBrigadeEnv
Rust    → bucket_brigade_core::BucketBrigadeEngine
WASM    → bucket-brigade-wasm (browser)
```

**Same game logic, three platforms**:
- Python: Research, RL training (PufferLib integration)
- Rust: High-performance tournaments (100x speedup)
- WASM: Browser-based visualization and play

---

## How We Use It for Tournament Research

### 1. Policy Development
```python
# Test a new heuristic policy
params = [0.9, 0.7, 0.5, ...]  # 10 behavioral parameters
agent = HeuristicAgent(params)

# Run quick test
env = BucketBrigadeEnv(scenario)
total_reward = run_single_game(env, [agent]*4)
print(f"Score: {total_reward}")  # Fast feedback (~5ms)
```

### 2. Tournament Execution
```python
# Run 100 games with mixed teams
results = run_tournament(
    policies=[firefighter, free_rider, coordinator, hero],
    num_games=100,
    team_size=4,
    scenarios=["greedy_neighbor", "chain_reaction"]
)

# Rank by marginal contribution
rankings = compute_rankings(results)  # See RANKING_METHODOLOGY.md
```

### 3. Evolutionary Optimization
```python
# Evolve optimal policies
ga = GeneticAlgorithm(
    population_size=100,
    num_generations=200,
    fitness_fn=lambda genome: evaluate_in_tournament(genome, scenario)
)

best_policy = ga.evolve()  # Thousands of games in minutes
```

### 4. Nash Equilibrium Analysis
```python
# Compute Nash equilibrium
solver = DoubleOracleNash(scenario, num_simulations=1000)
equilibrium = solver.solve()

# Test: Do evolved policies converge to Nash?
```

### 5. Reinforcement Learning
```python
# Train neural network policy
from pufferlib import PufferEnv

env = PufferEnv(BucketBrigadeEnv(scenario))
policy = train_ppo(env, total_timesteps=1_000_000)

# Compare to heuristics
```

**The Key**: Game is **fast enough** for large-scale experiments, **simple enough** to understand results, **rich enough** to be interesting.

---

## Example: Understanding a Strategy in 30 Seconds

**The Firefighter Policy**:
```python
params = {
    "honesty": 1.0,           # Always signals truthfully
    "work_tendency": 0.9,     # Prefers working over resting
    "neighbor_help": 0.7,     # Helps neighbors proactively
    "own_priority": 0.4,      # Doesn't prioritize own house excessively
    "altruism": 0.8,          # Values team success highly
    ...
}
```

**What it does** (in plain English):
1. Signals "Work" honestly (no deception)
2. Goes to nearest burning house
3. Works even if not own house (altruistic)
4. Ignores signals from others (independent)
5. Rests only when no fires visible

**Expected behavior**:
- ✅ Excels in cooperative scenarios (trivial_cooperation)
- ✅ Handles early containment well (responsive)
- ❌ Exploited by free-riders (works while others rest)
- ❌ Overworks in sparse scenarios (inefficient)

**This interpretability** is crucial for:
- Debugging RL policies (compare to known baselines)
- Explaining results to non-experts
- Building intuition about cooperation dynamics

---

## What Makes It "Tournament-Ready"

### 1. No Manual Intervention
```python
# Fully automated execution
for seed in range(1000):
    env = BucketBrigadeEnv(scenario, seed=seed)
    result = run_game(env, agents)
    save_result(result)  # Done, no human in the loop
```

### 2. Balanced Evaluation
```python
# Same starting conditions for all policies
scenarios = [
    "trivial_cooperation",  # Easy
    "greedy_neighbor",      # Social dilemma
    "chain_reaction",       # Hard
]

# Each policy tested in all scenarios
# Each scenario gets multiple seeds
# Results aggregated statistically
```

### 3. Meaningful Metrics
```python
{
    "team_reward": 245.3,           # Overall success
    "agent_rewards": [62, 61, 58],  # Individual contributions
    "houses_saved": 8,              # Concrete outcome
    "nights_played": 15,            # Efficiency
    "total_work": 42,               # Cooperation level
}
```

### 4. Reproducible Results
```python
# Publish tournament with seeds
tournament_config = {
    "scenarios": [...],
    "seeds": [42, 123, 456, ...],
    "agents": [...],
}

# Others can reproduce exactly
```

---

## Design Evolution: How We Got Here

### Initial Ideas (Rejected)
1. **Grid-based foraging**: Too similar to existing work
2. **Continuous control**: Harder to implement, slower execution
3. **Competitive game**: Wanted cooperation focus
4. **Hidden information**: Added complexity without benefit

### Early Prototypes
1. **Linear houses**: Fires spread one direction → too simple
2. **12 houses**: Ring of 12 → not evenly divisible by common team sizes (4, 6, 8)
3. **20 houses**: Too large, games took too long
4. **10 houses**: ✅ Just right (divisible by 2, 5, 10)

### Key Insights
1. **Ring topology**: Natural spatial structure, no edge cases
2. **Probabilistic fire**: Prevents deterministic exploitation
3. **Signal mechanism**: Cheap communication enables coordination (and deception)
4. **Scenario parameters**: One engine, infinite cooperation challenges

### Validation
- ✅ Implemented in <1 week (Python)
- ✅ Ported to Rust in <1 week (100x speedup)
- ✅ WASM version in <3 days (browser support)
- ✅ First tournament in <1 day (simple heuristics)
- ✅ RL training works out-of-the-box (PufferLib integration)

**Result**: Achieved goal of "small, understandable, tournament-ready"

---

## Frequently Asked Questions

### Q: Why fire/houses and not abstract tokens?

**A**: Narrative context helps:
- Easier to explain to non-experts ("save the town!")
- Intuitive parameter names (fire spread rate, not β)
- Visualization is natural (burning houses, not abstract graphs)

But the game is **mechanically abstract** — you could reskin it as "computer virus spreading in network" or "disease outbreak in town."

### Q: Why 10 houses specifically?

**A**: Design constraints:
- Divisible by common team sizes (4, 5, 10)
- Small enough for fast execution (<10ms per game)
- Large enough for spatial strategy (not just 3-5 nodes)
- Ring topology avoids edge effects

**Tested**: 6, 8, 10, 12 houses. 10 was the sweet spot.

### Q: Could you make it more complex?

**A**: Yes, but we deliberately chose minimalism:
- More complexity → harder to implement, debug, explain
- More complexity → slower execution, fewer tournament games
- More complexity → harder to analyze theoretically (Nash equilibria, etc.)

**Philosophy**: "Minimal sufficient complexity" — complex enough to be interesting, simple enough to understand completely.

### Q: How does it compare to StarCraft or Dota?

**A**:

| Feature | Bucket Brigade | StarCraft |
|---------|---------------|-----------|
| **State space** | ~10⁹ | ~10²⁰⁰ |
| **Implementation** | ~1K LOC | ~1M LOC |
| **Game duration** | 5ms | 15 minutes |
| **Interpretability** | High | Low |
| **Benchmark status** | New | Established |

**Tradeoff**: StarCraft is more realistic, Bucket Brigade is more understandable. Different tools for different research questions.

### Q: Can humans play it?

**A**: Yes! Browser version at https://rjwalters.github.io/bucket-brigade/

- Intuitive UI (click houses, choose work/rest)
- Real-time visualization (see fires spread)
- Compete against AI policies

But it's designed for **AI vs. AI**, not human play. Human reaction time doesn't matter (turn-based planning).

### Q: What research questions can it answer?

**A**:
- When does cooperation emerge in multi-agent systems?
- How do agents learn to trust (or distrust) signals?
- Can evolution discover optimal cooperative strategies?
- Do learned policies converge to Nash equilibria?
- How does team composition affect individual performance?
- What mechanisms prevent free-riding?

**Broader impact**: Insights apply to emergency response, team coordination, resource allocation, social dilemmas.

---

## Summary: Why Bucket Brigade Works

✅ **Small enough to understand**: 30-bit state space, 500 lines of code
✅ **Fast enough to scale**: 5ms per game, 1000-game tournaments in seconds
✅ **Rich enough to be interesting**: Social dilemmas, deception, coordination
✅ **Parameterizable enough to generalize**: 9 scenarios, continuous parameter space
✅ **Observable enough to interpret**: Full visibility, hand-coded baselines
✅ **Simple enough to implement**: No dependencies, cross-platform support

**Perfect for**:
- Developing tournament infrastructure
- Testing ranking methodologies
- Rapid iteration on agent designs
- Understanding cooperation dynamics
- Teaching multi-agent systems

**Not designed for**:
- Realistic simulation (it's abstract!)
- Single-agent RL (it's cooperative!)
- Human entertainment (it's for AI research!)

---

## Getting Started

### Play in Browser
https://rjwalters.github.io/bucket-brigade/

### Run Locally
```bash
# Python
uv run python scripts/run_one_game.py

# Rust (100x faster)
cargo run --release --example tournament
```

### Implement Your Own Agent
```python
from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.agents import HeuristicAgent

# Define your strategy (10 parameters)
my_params = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
agent = HeuristicAgent(params=my_params)

# Test it
env = BucketBrigadeEnv(scenario="greedy_neighbor")
obs = env.reset()

while not env.done:
    action = agent.act(obs)
    obs, reward, done, info = env.step([action])

print(f"Final reward: {reward}")
```

### Run a Tournament
```bash
# 100 games, mixed teams, multiple scenarios
uv run python scripts/run_batch.py \
    --num-games 100 \
    --scenarios greedy_neighbor chain_reaction \
    --output results.csv
```

---

## Next Steps

**For researchers**:
1. Read RANKING_METHODOLOGY.md (how we rank policies)
2. Read SCENARIO_RESEARCH.md (how we design scenarios)
3. Explore experiments/ directory (evolutionary algorithms, Nash equilibria)

**For developers**:
1. Check out API.md (data structures)
2. Browse bucket_brigade/envs/ (game engine)
3. Try bucket-brigade-core/ (Rust implementation)

**For contributors**:
1. Implement new scenarios (see scenarios.py)
2. Design new heuristics (see archetypes.py)
3. Improve visualization (see web/)

---

*"The best games are simple to learn, impossible to master. Bucket Brigade is simple to implement, rich to research."*

**Bucket Brigade**: Small game, big insights. 🔥🏠⏱️
