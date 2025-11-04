"""Fitness evaluation for evolutionary algorithms.

This module provides fitness functions that evaluate agents by running
tournament games and computing performance metrics.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..agents.heuristic_agent import HeuristicAgent
from ..envs.bucket_brigade_env import BucketBrigadeEnv
from ..envs.scenarios import Scenario, default_scenario
from .population import Individual, Population


# ============================================================================
# Fitness Evaluation
# ============================================================================


class FitnessEvaluator:
    """Evaluates agent fitness by running tournament games."""

    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        games_per_individual: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize fitness evaluator.

        Args:
            scenario: Game scenario to use for evaluation (None = default)
            games_per_individual: Number of games to run per agent
            seed: Random seed for reproducibility
        """
        self.scenario = scenario
        self.games_per_individual = games_per_individual
        self.rng = np.random.default_rng(seed)

    def evaluate_individual(self, individual: Individual) -> float:
        """Evaluate a single individual by running tournament games.

        Args:
            individual: Individual to evaluate

        Returns:
            Fitness score (mean reward across games)
        """
        # Create agent from genome
        agent = HeuristicAgent(params=individual.genome, agent_id=0)

        # Run games
        total_reward = 0.0

        for _ in range(self.games_per_individual):
            # Create environment (single-agent evaluation)
            scenario = (
                self.scenario
                if self.scenario is not None
                else default_scenario(num_agents=1)
            )
            env = BucketBrigadeEnv(scenario=scenario)

            # Run game
            obs = env.reset(seed=int(self.rng.integers(0, 2**31)))
            agent.reset()

            max_steps = 1000  # Prevent infinite loops
            steps = 0
            while not env.done and steps < max_steps:
                # Get action from agent
                action = agent.act(obs)

                # Step environment (actions must be numpy array of shape (num_agents, 2))
                actions = np.array([action])  # Wrap in array for single agent
                obs, rewards, done, info = env.step(actions)

                steps += 1
                if done:
                    break

            # Accumulate reward
            total_reward += env.rewards[0]

        # Return mean reward
        return float(total_reward / self.games_per_individual)

    def evaluate_population(
        self, population: Population, parallel: bool = False
    ) -> None:
        """Evaluate all individuals in a population.

        Args:
            population: Population to evaluate
            parallel: If True, use multiprocessing (not implemented yet)
        """
        # TODO: Implement parallel evaluation for speedup
        for individual in population:
            if individual.fitness is None:  # Only evaluate if not already evaluated
                individual.fitness = self.evaluate_individual(individual)


# ============================================================================
# Alternative Fitness Functions
# ============================================================================


def win_rate_fitness(
    individual: Individual, scenario: Optional[Scenario] = None, num_games: int = 20
) -> float:
    """Compute fitness as win rate (fraction of games where team succeeded).

    Win = all fires extinguished before all houses ruined.

    Args:
        individual: Individual to evaluate
        scenario: Game scenario (None = default)
        num_games: Number of games to run

    Returns:
        Win rate (0.0 to 1.0)
    """
    agent = HeuristicAgent(params=individual.genome, agent_id=0)
    wins = 0

    for _ in range(num_games):
        scenario_inst = (
            scenario if scenario is not None else default_scenario(num_agents=1)
        )
        env = BucketBrigadeEnv(scenario=scenario_inst)
        obs = env.reset()
        agent.reset()

        max_steps = 1000  # Prevent infinite loops
        steps = 0
        while not env.done and steps < max_steps:
            action = agent.act(obs)
            actions = np.array([action])  # Wrap in array for single agent
            obs, rewards, done, info = env.step(actions)
            steps += 1
            if done:
                break

        # Check if won (no houses ruined at end)
        if np.sum(env.houses == BucketBrigadeEnv.RUINED) == 0:
            wins += 1

    return float(wins / num_games)


def robustness_fitness(
    individual: Individual, scenarios: list[Scenario], num_games_per_scenario: int = 5
) -> float:
    """Compute fitness as performance across multiple scenarios.

    Robustness = average performance across diverse scenarios.

    Args:
        individual: Individual to evaluate
        scenarios: List of scenarios to test
        num_games_per_scenario: Games per scenario

    Returns:
        Mean reward across all scenarios
    """
    agent = HeuristicAgent(params=individual.genome, agent_id=0)
    total_reward = 0.0
    total_games = 0

    for scenario in scenarios:
        for _ in range(num_games_per_scenario):
            env = BucketBrigadeEnv(scenario=scenario)
            obs = env.reset()
            agent.reset()

            max_steps = 1000  # Prevent infinite loops
            steps = 0
            while not env.done and steps < max_steps:
                action = agent.act(obs)
                actions = np.array([action])  # Wrap in array for single agent
                obs, rewards, done, info = env.step(actions)
                steps += 1
                if done:
                    break

            total_reward += env.rewards[0]
            total_games += 1

    return float(total_reward / total_games) if total_games > 0 else 0.0


def multi_objective_fitness(
    individual: Individual,
    scenario: Optional[Scenario] = None,
    num_games: int = 20,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute multi-objective fitness combining multiple metrics.

    Metrics:
    - reward: Mean reward
    - win_rate: Fraction of wins
    - efficiency: Mean game length (shorter = better)

    Args:
        individual: Individual to evaluate
        scenario: Game scenario (None = default)
        num_games: Number of games to run
        weights: Weight for each objective (default: equal weights)

    Returns:
        Weighted sum of objectives
    """
    if weights is None:
        weights = {"reward": 1.0, "win_rate": 1.0, "efficiency": 0.5}

    agent = HeuristicAgent(params=individual.genome, agent_id=0)

    total_reward = 0.0
    wins = 0
    total_length = 0

    for _ in range(num_games):
        scenario_inst = (
            scenario if scenario is not None else default_scenario(num_agents=1)
        )
        env = BucketBrigadeEnv(scenario=scenario_inst)
        obs = env.reset()
        agent.reset()

        max_steps = 1000  # Prevent infinite loops
        steps = 0
        while not env.done and steps < max_steps:
            action = agent.act(obs)
            actions = np.array([action])  # Wrap in array for single agent
            obs, rewards, done, info = env.step(actions)
            steps += 1
            if done:
                break

        # Collect metrics
        total_reward += env.rewards[0]
        total_length += env.night
        if np.sum(env.houses == BucketBrigadeEnv.RUINED) == 0:
            wins += 1

    # Compute objectives
    mean_reward = total_reward / num_games
    win_rate = wins / num_games
    mean_length = total_length / num_games
    efficiency = 1.0 / (
        1.0 + mean_length / 50.0
    )  # Normalize to [0, 1], shorter is better

    # Weighted sum
    fitness = (
        weights["reward"] * mean_reward
        + weights["win_rate"] * win_rate
        + weights["efficiency"] * efficiency
    )

    return float(fitness)


# ============================================================================
# Fitness Function Factory
# ============================================================================


def create_fitness_function(
    fitness_type: str = "mean_reward",
    scenario: Optional[Scenario] = None,
    num_games: int = 20,
    **kwargs: dict[str, object],
) -> Callable[[Individual], float]:
    """Create a fitness function with specified configuration.

    Args:
        fitness_type: Type of fitness function ("mean_reward", "win_rate", "robustness", "multi_objective")
        scenario: Game scenario to use
        num_games: Number of games per evaluation
        **kwargs: Additional arguments for specific fitness types

    Returns:
        Fitness function that takes an Individual and returns a float score
    """
    if fitness_type == "mean_reward":
        evaluator = FitnessEvaluator(scenario=scenario, games_per_individual=num_games)
        return evaluator.evaluate_individual

    elif fitness_type == "win_rate":
        return lambda ind: win_rate_fitness(ind, scenario=scenario, num_games=num_games)

    elif fitness_type == "robustness":
        scenarios = kwargs.get("scenarios", [default_scenario(num_agents=1)])
        games_per_scenario = kwargs.get(
            "games_per_scenario", num_games // len(scenarios)
        )
        return lambda ind: robustness_fitness(
            ind, scenarios=scenarios, num_games_per_scenario=games_per_scenario
        )

    elif fitness_type == "multi_objective":
        weights = kwargs.get(
            "weights", {"reward": 1.0, "win_rate": 1.0, "efficiency": 0.5}
        )
        return lambda ind: multi_objective_fitness(
            ind, scenario=scenario, num_games=num_games, weights=weights
        )

    else:
        raise ValueError(f"Unknown fitness type: {fitness_type}")
