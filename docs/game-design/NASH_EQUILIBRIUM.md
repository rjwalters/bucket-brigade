# Nash Equilibrium Analysis for Bucket Brigade

## Overview

This document describes approaches for computing Nash equilibrium strategies in the Bucket Brigade game. Nash equilibrium represents a stable strategic configuration where no agent can improve their payoff by unilaterally changing their strategy.

## Game-Theoretic Framework

### Game Classification

Bucket Brigade is a **stochastic Markov game** with the following properties:

- **State Space**: Fire configurations (3^10 states) × agent positions × history
- **Action Space**: Each agent chooses (house_index, mode) where house_index ∈ {0..9}, mode ∈ {WORK, REST}
- **Players**: N agents (typically 4-10)
- **Transitions**: Stochastic fire spread (probability `prob_fire_spreads_to_neighbor`), extinguish (probability `1 - (1-prob_solo_agent_extinguishes_fire)^n`)
- **Payoffs**: Individual rewards based on work cost, rest bonus, ownership bonuses, and team component
- **Time Horizon**: min_nights to 100 nights
- **Information**: Perfect information about current state, imperfect information about other agents' types

### Strategic Properties

**Social Dilemmas**:
- Free-riding incentive: Rest while others fight fires
- Coordination problems: Multiple workers on same fire vs distributed coverage
- Signaling games: Honest vs deceptive communication about intent
- Resource allocation: Balance work (costly) vs rest (free-ride)

**Symmetry**:
- Agents have identical capabilities
- Ownership creates asymmetry (round-robin house assignment)
- Suggests analyzing **symmetric equilibria** (all agents use same strategy)

**Repeated Interactions**:
- Multi-round game allows conditional strategies
- Reputation and punishment mechanisms possible
- Focus on **Markov Perfect Equilibrium** (strategies depend only on current state)

## Equilibrium Concepts

### 1. Markov Perfect Equilibrium (MPE)

**Definition**: Strategy profile σ* = (σ₁*, ..., σₙ*) where each σᵢ*: State → Action is a best response to opponents' strategies, given current state.

**Pros**:
- Most tractable for computational analysis
- State (fires, positions) encodes most relevant information
- Natural for dynamic programming approaches

**Cons**:
- Ignores history-dependent strategies
- May miss equilibria that rely on reputation/punishment

### 2. Symmetric Markov Perfect Equilibrium (SMPE)

**Definition**: All agents use identical strategy σ*: State → Action.

**Pros**:
- Reduces complexity from N-dimensional to 1-dimensional optimization
- Reasonable assumption given agent symmetry
- Easier to compute and verify

**Cons**:
- May not exist in games with strong asymmetries (ownership)
- May be inefficient compared to asymmetric equilibria

### 3. Bayesian Nash Equilibrium (BNE)

**Definition**: Strategy profile where each agent type's strategy maximizes expected payoff given beliefs about opponent types.

**Relevant when**:
- Agents don't know others' heuristic parameters θ
- Population has heterogeneous types (Firefighters, Free Riders, etc.)
- Incomplete information is realistic

### 4. Correlated Equilibrium (CE)

**Definition**: Strategy profile where following a mediator's recommendations is incentive-compatible.

**For Bucket Brigade**:
- Signal mechanism can act as correlation device
- Mediator suggests who should work based on fire state
- May achieve higher social welfare than Nash equilibrium
- The COORDINATOR archetype approximates this concept

## Strategy Representation

### Heuristic Parameter Space

Strategies are represented as 10-dimensional vectors:

```python
θ = [
    honesty_bias,          # [0-1] Probability of truthful signaling
    work_tendency,         # [0-1] Base tendency to work
    neighbor_help_bias,    # [0-1] Help neighbors vs self
    own_house_priority,    # [0-1] Prioritize owned house
    risk_aversion,         # [0-1] Sensitivity to fires
    coordination_weight,   # [0-1] Trust in others' signals
    exploration_rate,      # [0-1] Randomness in decisions
    fatigue_memory,        # [0-1] Inertia to repeat actions
    rest_reward_bias,      # [0-1] Preference for resting
    altruism_factor        # [0-1] Willingness to help others
]
```

**Archetypal Strategies**:
- **Firefighter**: [1.0, 0.9, 0.5, 0.8, 0.5, 0.7, 0.1, 0.0, 0.0, 0.8]
- **Free Rider**: [0.7, 0.2, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0]
- **Hero**: [1.0, 1.0, 1.0, 0.5, 0.1, 0.5, 0.0, 0.9, 0.0, 1.0]
- **Coordinator**: [0.9, 0.6, 0.7, 0.6, 0.8, 1.0, 0.05, 0.0, 0.2, 0.6]
- **Liar**: [0.1, 0.7, 0.0, 0.9, 0.2, 0.8, 0.3, 0.0, 0.4, 0.2]

### Mixed Strategies

Equilibria may involve randomization over pure strategies:

```python
# Mixed strategy: probability distribution over heuristic parameters
mixed_strategy = {
    FIREFIGHTER_PARAMS: 0.60,  # 60% play as Firefighter
    FREE_RIDER_PARAMS: 0.30,   # 30% play as Free Rider
    COORDINATOR_PARAMS: 0.10   # 10% play as Coordinator
}
```

## Computational Approaches

### Algorithm 1: Best Response Dynamics

**Idea**: Iteratively compute best responses until convergence.

```python
def best_response_dynamics(scenario, max_iterations=100, epsilon=0.01):
    """
    Find Nash equilibrium via iterated best response.

    Args:
        scenario: Game scenario with parameters (prob_fire_spreads_to_neighbor,
                  prob_solo_agent_extinguishes_fire, team_reward_house_survives, etc.)
        max_iterations: Maximum iteration count
        epsilon: Convergence threshold

    Returns:
        θ*: Approximate Nash equilibrium strategy
    """
    # Initialize with random strategy
    theta = random_heuristic_params()

    for iteration in range(max_iterations):
        # Compute best response to current strategy
        theta_new = best_response(theta, scenario)

        # Check convergence (fixed point)
        if distance(theta_new, theta) < epsilon:
            return theta_new  # Found Nash equilibrium

        theta = theta_new

    return theta  # Approximate equilibrium

def best_response(theta_opponents, scenario, num_simulations=1000):
    """
    Find best θ_i to play against opponents using θ_opponents.

    Uses scipy.optimize.minimize over 10-dimensional parameter space.
    Payoffs estimated via Monte Carlo simulation.
    """
    def objective(theta_i):
        return -evaluate_payoff(theta_i, theta_opponents, scenario, num_simulations)

    result = scipy.optimize.minimize(
        objective,
        x0=theta_opponents,
        bounds=[(0, 1)] * 10,
        method='L-BFGS-B'
    )

    return result.x
```

**Convergence**: Guaranteed for potential games, may cycle otherwise.

**Performance**: Requires O(iterations × simulations × optimization_steps) game evaluations.

---

### Algorithm 2: Double Oracle

**Idea**: Iteratively build support of equilibrium by adding best responses to a strategy pool.

```python
def double_oracle(scenario, max_iterations=50, epsilon=0.01):
    """
    Find Nash equilibrium via double oracle algorithm.

    Efficiently explores large strategy spaces by maintaining a pool
    of "active" strategies and iteratively adding best responses.

    Args:
        scenario: Game scenario
        max_iterations: Maximum iterations
        epsilon: Improvement threshold

    Returns:
        Distribution over strategy pool (mixed strategy equilibrium)
    """
    # Initialize with archetypal strategies
    strategy_pool = [
        FIREFIGHTER_PARAMS,
        FREE_RIDER_PARAMS,
        HERO_PARAMS,
        COORDINATOR_PARAMS
    ]

    for iteration in range(max_iterations):
        # Solve restricted game (Nash equilibrium over current pool)
        eq_distribution = solve_restricted_game(strategy_pool, scenario)

        # Compute best response to equilibrium distribution
        br = best_response_to_distribution(eq_distribution, scenario)

        # Evaluate improvement from best response
        payoff_br = evaluate_payoff_against_distribution(br, eq_distribution, scenario)
        payoff_eq = expected_equilibrium_payoff(eq_distribution, scenario)

        # If best response improves over equilibrium, add to pool
        if payoff_br > payoff_eq + epsilon:
            strategy_pool.append(br)
        else:
            # No improvement → equilibrium found
            return eq_distribution

    return eq_distribution

def solve_restricted_game(strategy_pool, scenario):
    """
    Find Nash equilibrium over finite strategy pool.

    This reduces to a matrix game solvable via linear programming.
    For symmetric games, we seek a distribution p over strategies
    such that each strategy in the support is a best response.
    """
    K = len(strategy_pool)

    # Build K×K payoff matrix
    payoff_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            # Expected payoff for strategy i vs opponents playing j
            payoff_matrix[i, j] = evaluate_symmetric_payoff(
                strategy_pool[i],
                strategy_pool[j],
                scenario
            )

    # Solve for symmetric Nash equilibrium using linear programming
    distribution = solve_symmetric_nash(payoff_matrix)

    return {strategy_pool[i]: distribution[i] for i in range(K)}
```

**Advantages**:
- Provably finds Nash equilibrium in finite iterations
- Produces interpretable mixed strategies over archetypal agents
- Efficient exploration of large strategy spaces

**Best for**: Initial equilibrium discovery, interpretable results

---

### Algorithm 3: Replicator Dynamics (Evolutionary)

**Idea**: Model evolutionary selection where successful strategies increase in population frequency.

```python
def replicator_dynamics(scenario, population_size=100, generations=1000):
    """
    Find evolutionarily stable strategy via replicator dynamics.

    Simulates evolutionary process where strategies with above-average
    payoff increase in frequency. Converges to evolutionarily stable
    strategies (ESS), a refinement of Nash equilibrium.

    Args:
        scenario: Game scenario
        population_size: Number of agents in population
        generations: Number of evolutionary cycles

    Returns:
        Population distribution (evolutionary equilibrium)
    """
    # Initialize diverse population
    population = [random_heuristic_params() for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness of each strategy
        fitnesses = []
        for theta_i in population:
            # Sample opponents from population
            opponents = random.sample(population, k=N_AGENTS-1)
            fitness = evaluate_payoff_in_population(theta_i, opponents, scenario)
            fitnesses.append(fitness)

        avg_fitness = np.mean(fitnesses)

        # Replication: strategies with above-average fitness reproduce
        new_population = []
        for theta, fitness in zip(population, fitnesses):
            # Probability of survival/reproduction proportional to fitness
            num_offspring = int(population_size * fitness / (avg_fitness * population_size))

            for _ in range(num_offspring):
                # Mutation for exploration
                theta_offspring = mutate(theta, mutation_rate=0.01)
                new_population.append(theta_offspring)

        # Maintain population size
        if len(new_population) > population_size:
            population = random.sample(new_population, population_size)
        else:
            # Refill with random mutations if population shrinks
            while len(new_population) < population_size:
                new_population.append(mutate(random.choice(population), 0.05))
            population = new_population

    return population
```

**Advantages**:
- Naturally handles mixed strategy equilibria (population distribution)
- Robust to local optima (mutation provides exploration)
- Parallelizable (evaluate many strategies simultaneously)

**Convergence**: To evolutionarily stable strategies (ESS), which are Nash equilibria robust to mutations.

---

### Algorithm 4: Fictitious Play

**Idea**: Each agent best-responds to empirical distribution of opponents' historical play.

```python
def fictitious_play(scenario, num_rounds=100):
    """
    Find Nash equilibrium via fictitious play learning.

    Agents iteratively best-respond to the empirical distribution
    of opponents' strategies. Under certain conditions, converges
    to Nash equilibrium.

    Args:
        scenario: Game scenario
        num_rounds: Number of learning iterations

    Returns:
        Empirical distribution over strategies (mixed strategy equilibrium)
    """
    N = scenario.num_agents

    # Initialize strategies
    strategies = [random_heuristic_params() for _ in range(N)]

    # Track empirical frequency
    strategy_history = [[] for _ in range(N)]

    for round_num in range(num_rounds):
        for i in range(N):
            # Compute empirical distribution of opponents' past play
            opponent_distribution = compute_empirical_distribution(
                strategy_history,
                exclude=i
            )

            # Best respond to empirical distribution
            strategies[i] = best_response_to_distribution(
                opponent_distribution,
                scenario
            )

            # Record strategy
            strategy_history[i].append(strategies[i])

    # Return time-averaged strategies (empirical distribution)
    return compute_empirical_distribution(strategy_history)
```

**Convergence**: Guaranteed for certain game classes (e.g., zero-sum games, potential games). May not converge in general.

**Advantages**: Models learning process, can find mixed strategy equilibria.

---

## Payoff Evaluation

Computing equilibrium requires evaluating expected payoffs, which is computationally expensive due to stochastic dynamics.

### Monte Carlo Estimation

```python
def evaluate_payoff(theta_i, theta_opponents, scenario, num_simulations=1000):
    """
    Estimate expected payoff for agent using theta_i playing against
    opponents using theta_opponents.

    Args:
        theta_i: Focal agent's strategy parameters
        theta_opponents: Opponents' strategy parameters (symmetric)
        scenario: Game scenario
        num_simulations: Number of Monte Carlo rollouts

    Returns:
        Average cumulative reward for focal agent
    """
    total_reward = 0.0

    for sim in range(num_simulations):
        env = BucketBrigadeEnv(scenario)

        # Create agents
        agents = [HeuristicAgent(theta_i)] + \
                 [HeuristicAgent(theta_opponents) for _ in range(scenario.num_agents - 1)]

        # Run episode
        observations = env.reset()
        episode_reward = 0
        done = False

        while not done:
            actions = [agent.select_action(obs) for agent, obs in zip(agents, observations)]
            observations, rewards, done, _ = env.step(actions)
            episode_reward += rewards[0]  # Focal agent's reward

        total_reward += episode_reward

    return total_reward / num_simulations
```

### Rust Acceleration

For equilibrium search requiring 10,000+ evaluations, use Rust core:

```rust
// bucket-brigade-core/src/equilibrium.rs
pub fn evaluate_strategy_payoff(
    theta_i: [f64; 10],
    theta_opponents: [f64; 10],
    scenario: &Scenario,
    num_simulations: usize,
) -> f64 {
    let mut total_reward = 0.0;

    for _ in 0..num_simulations {
        let mut env = BucketBrigadeEnv::new(scenario);
        let mut agents = create_agents(theta_i, theta_opponents, scenario.num_agents);

        while !env.is_done() {
            let actions = agents.iter_mut()
                .map(|a| a.select_action(&env.observe()))
                .collect();
            env.step(&actions);
        }

        total_reward += env.get_reward(0);
    }

    total_reward / (num_simulations as f64)
}
```

**Performance**: Rust provides 10-100× speedup over Python simulation.

### Parallel Evaluation

```python
from multiprocessing import Pool

def evaluate_payoffs_parallel(strategy_pairs, scenario, num_workers=8):
    """Evaluate multiple strategy profiles in parallel."""
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(
            evaluate_payoff,
            [(θ_i, θ_j, scenario) for θ_i, θ_j in strategy_pairs]
        )
    return results
```

## Equilibrium Verification

### Nash Equilibrium Test

```python
def verify_nash_equilibrium(theta_star, scenario, epsilon=0.01, num_trials=100):
    """
    Verify that theta_star is a Nash equilibrium.

    Tests whether any deviation improves payoff beyond epsilon threshold.

    Args:
        theta_star: Candidate equilibrium strategy
        scenario: Game scenario
        epsilon: Tolerance for approximate equilibrium
        num_trials: Number of random deviations to test

    Returns:
        (is_equilibrium, max_deviation_gain)
    """
    payoff_equilibrium = evaluate_payoff(theta_star, theta_star, scenario)
    max_gain = 0.0

    # Test random deviations
    for _ in range(num_trials):
        theta_deviation = perturb(theta_star, magnitude=0.1)
        payoff_deviation = evaluate_payoff(theta_deviation, theta_star, scenario)
        gain = payoff_deviation - payoff_equilibrium
        max_gain = max(max_gain, gain)

        if gain > epsilon:
            return False, gain

    # Test best response (strongest test)
    theta_br = best_response(theta_star, scenario)
    payoff_br = evaluate_payoff(theta_br, theta_star, scenario)
    br_gain = payoff_br - payoff_equilibrium
    max_gain = max(max_gain, br_gain)

    is_equilibrium = br_gain <= epsilon
    return is_equilibrium, max_gain
```

### Evolutionary Stability Test

```python
def verify_evolutionary_stability(theta_star, scenario, num_trials=100):
    """
    Test if theta_star is evolutionarily stable (ESS).

    ESS is a stronger concept than Nash equilibrium: the strategy
    must resist invasion by mutants.

    Conditions:
    1. U(θ*, θ*) ≥ U(θ', θ*) for all θ' (Nash condition)
    2. If U(θ*, θ*) = U(θ', θ*), then U(θ*, θ') > U(θ', θ') (stability)

    Returns:
        True if ESS, False otherwise
    """
    payoff_star = evaluate_payoff(theta_star, theta_star, scenario)

    for _ in range(num_trials):
        theta_mutant = mutate(theta_star, mutation_rate=0.05)

        # Mutant's payoff against incumbent population
        payoff_mutant = evaluate_payoff(theta_mutant, theta_star, scenario)

        # ESS condition 1: incumbent does at least as well
        if payoff_mutant > payoff_star + 1e-6:
            # Check secondary condition
            payoff_star_vs_mutant = evaluate_payoff(theta_star, theta_mutant, scenario)
            payoff_mutant_vs_mutant = evaluate_payoff(theta_mutant, theta_mutant, scenario)

            if payoff_mutant_vs_mutant >= payoff_star_vs_mutant:
                return False  # Mutant can invade

    return True
```

## Scenario-Specific Predictions

### Trivial Cooperation (Low prob_fire_spreads_to_neighbor, High prob_solo_agent_extinguishes_fire)

**Expected Equilibrium**: Pure strategy cooperative equilibrium

```python
theta_trivial = [
    1.0,  # honesty_bias (truthful signaling)
    0.9,  # work_tendency (work most nights)
    0.7,  # neighbor_help_bias (help neighbors)
    0.6,  # own_house_priority
    0.5,  # risk_aversion
    0.8,  # coordination_weight (trust signals)
    0.1,  # exploration_rate
    0.0,  # fatigue_memory
    0.0,  # rest_reward_bias
    0.9   # altruism_factor (high cooperation)
]
```

**Intuition**: Cooperation is cheap (easy to extinguish) and profitable (prevents spread). Dominant strategy is to work hard.

**Equilibrium Type**: Pure strategy Nash equilibrium (unique)

---

### Greedy Neighbor (High cost_to_work_one_night, Low prob_solo_agent_extinguishes_fire)

**Expected Equilibrium**: Mixed strategy or population polymorphism

```python
# 60% Cooperators
theta_cooperator = [0.9, 0.7, 0.6, 0.5, 0.5, 0.7, 0.1, 0.0, 0.2, 0.7]

# 40% Free Riders
theta_free_rider = [0.3, 0.2, 0.1, 0.9, 0.2, 0.1, 0.1, 0.0, 0.9, 0.1]

# Mixed equilibrium
equilibrium = {
    theta_cooperator: 0.6,
    theta_free_rider: 0.4
}
```

**Intuition**: Working is expensive, creating free-riding incentive. Equilibrium balances:
- Too many cooperators → free-riding becomes profitable
- Too few cooperators → fires spread, cooperation becomes necessary

**Equilibrium Condition**:
```
U(cooperator | 60% cooperate) = U(free_rider | 60% cooperate)
```
Both strategies earn equal payoff (mixing indifference condition).

**Equilibrium Type**: Mixed strategy Nash equilibrium

---

### Deceptive Calm (prob_house_catches_fire > 0)

**Expected Equilibrium**: Signaling equilibrium with high honesty

```python
theta_honest = [
    0.95,  # honesty_bias (almost always truthful)
    0.6,   # work_tendency
    0.7,   # neighbor_help_bias
    0.5,   # own_house_priority
    0.8,   # risk_aversion (high, due to unpredictability)
    0.9,   # coordination_weight (trust signals highly)
    0.1,   # exploration_rate
    0.0,   # fatigue_memory
    0.2,   # rest_reward_bias
    0.7    # altruism_factor
]
```

**Intuition**: Unpredictable sparks make coordination critical. Honest signaling enables efficient coordination. Deception backfires when signals aren't trusted.

**Equilibrium Type**: Separating equilibrium in signaling game (truth-telling is incentive-compatible)

---

## Implementation Roadmap

### Phase 1: Foundation (1-2 weeks)
1. Implement fast payoff evaluation in Rust
2. Expose via PyO3 bindings to Python
3. Implement best response calculation
4. Unit tests for known scenarios

### Phase 2: Basic Equilibrium Finder (1-2 weeks)
1. Implement double oracle algorithm
2. Implement replicator dynamics
3. Verify on test scenarios (Trivial, Greedy Neighbor, etc.)
4. Visualization tools for equilibrium strategies

### Phase 3: Advanced Analysis (2-4 weeks)
1. Mixed strategy support
2. Bayesian Nash equilibrium (heterogeneous agents)
3. Correlated equilibrium (using signal mechanism)
4. Stability analysis (ESS tests)

### Phase 4: Integration (1 week)
1. Web UI for equilibrium visualization
2. Compare learned (RL) vs theoretical (Nash) strategies
3. Export equilibrium strategies as playable agents

## References

### Game Theory
- Osborne & Rubinstein (1994), *A Course in Game Theory*
- Fudenberg & Tirole (1991), *Game Theory*
- Weibull (1995), *Evolutionary Game Theory*

### Multi-Agent RL
- Littman (1994), "Markov games as a framework for multi-agent reinforcement learning"
- Lanctot et al. (2017), "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning"
- Bansal et al. (2017), "Emergent Complexity via Multi-Agent Competition"

### Computational Methods
- McMahan et al. (2003), "Planning in the Presence of Cost Functions Controlled by an Adversary"
- Bošanský et al. (2014), "Double-oracle algorithm for computing an exact Nash equilibrium"
- Sandholm (2010), "The State of Solving Large Incomplete-Information Games"

## Related Documents

- [GAME_DYNAMICS.md](./GAME_DYNAMICS.md) - Complete game rules
- [SCENARIO_BRAINSTORM.md](./SCENARIO_BRAINSTORM.md) - Scenario designs
- [RANKING_SYSTEM.md](./RANKING_SYSTEM.md) - Agent ranking methodology
- [HEURISTIC_AGENTS.md](./HEURISTIC_AGENTS.md) - Heuristic agent implementation

---

*Last updated: 2025-11-02*
