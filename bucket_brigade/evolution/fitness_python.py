"""Python-based fitness evaluation (fallback when Rust module unavailable).

This is a slower pure-Python implementation used when bucket_brigade_core
Rust module is not built. For production use, build the Rust module with
maturin for 100x speedup.
"""

from __future__ import annotations

from typing import Optional, Any
from multiprocessing import Pool, cpu_count

import numpy as np

from ..envs.scenarios import Scenario, default_scenario
from ..envs.bucket_brigade_env import BucketBrigadeEnv
from .population import Individual


def _heuristic_action(theta: np.ndarray, obs: Any, agent_id: int, rng: Any) -> list[int]:
    """
    Simplified heuristic action selection based on parameters.

    Fast approximation of agent behavior for fitness evaluation.
    """
    # Unpack key parameters
    work_tendency = theta[1]
    own_house_priority = theta[3]
    rest_reward_bias = theta[8]

    # Simple decision: work with probability based on work_tendency
    if rng.random() < work_tendency * (1 - rest_reward_bias):
        # Work - choose which house
        owned_house = agent_id % 10

        # Prioritize owned house if burning
        if obs["houses"][owned_house] == 1 and rng.random() < own_house_priority:
            house = owned_house
        else:
            # Choose a burning house
            burning = [i for i, h in enumerate(obs["houses"]) if h == 1]
            if burning:
                house = rng.choice(burning)
            else:
                house = owned_house

        mode = 1  # WORK
    else:
        # Rest
        house = agent_id % 10
        mode = 0  # REST

    return [house, mode]


class PythonFitnessEvaluator:
    """Pure Python fitness evaluator (slower fallback)."""

    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        games_per_individual: int = 10,
        seed: Optional[int] = None,
        parallel: bool = True,
        num_workers: Optional[int] = None,
    ):
        """
        Initialize Python fitness evaluator.

        Args:
            scenario: Game scenario to use (None = default)
            games_per_individual: Number of games per agent evaluation
            seed: Random seed for reproducibility
            parallel: Whether to use parallel execution
            num_workers: Number of worker processes (default: cpu_count())
        """
        self.scenario = (
            scenario if scenario is not None else default_scenario(num_agents=1)
        )
        self.games_per_individual = games_per_individual
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.parallel = parallel
        self.num_workers = num_workers if num_workers is not None else cpu_count()

    def evaluate_individual(self, individual: Individual) -> float:
        """
        Evaluate a single individual by running games.

        Args:
            individual: Individual to evaluate

        Returns:
            Fitness score (mean scenario payoff across games)
        """
        # Generate seeds for games
        if self.seed is not None:
            seeds = [
                self.rng.randint(0, 2**31 - 1) for _ in range(self.games_per_individual)
            ]
        else:
            seeds = [None] * self.games_per_individual

        episode_payoffs = []
        for game_seed in seeds:
            # Create game
            game = BucketBrigadeEnv(self.scenario)

            # Set seed via numpy random state if provided
            if game_seed is not None:
                np.random.seed(game_seed)

            # Python RNG for heuristic decisions
            game_rng = np.random.RandomState(game_seed)

            # Run episode
            obs = game.reset()
            done = False
            step_count = 0
            max_steps = 100  # Safety limit

            while not done and step_count < max_steps:
                # Get action from heuristic
                obs_dict = {
                    "houses": obs[0]["houses"],
                    "signals": obs[0]["signals"],
                    "locations": obs[0]["locations"],
                }
                action = _heuristic_action(individual.genome, obs_dict, 0, game_rng)

                # Step environment
                obs, rewards, done, info = game.step([action])
                step_count += 1

            # Get final score from game result
            result = game.get_result()
            episode_payoffs.append(result.final_score)

        return float(np.mean(episode_payoffs))

    def evaluate_population(
        self, population: Any, parallel: Optional[bool] = None
    ) -> None:
        """
        Evaluate all individuals in a population.

        Args:
            population: Population to evaluate
            parallel: Override parallel setting (default: use constructor setting)
        """
        for individual in population.individuals:  # type: ignore[attr-defined]
            if individual.fitness is None:  # type: ignore[attr-defined]
                individual.fitness = self.evaluate_individual(individual)  # type: ignore[attr-defined]
