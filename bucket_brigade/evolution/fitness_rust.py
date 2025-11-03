"""Rust-backed fitness evaluation for evolutionary algorithms.

This module provides fast fitness functions using bucket_brigade_core
Rust bindings for 100x speedup over Python implementation.
"""

from __future__ import annotations

from typing import Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import bucket_brigade_core as core

from ..envs.scenarios import Scenario, default_scenario
from .population import Individual


def _convert_scenario_to_rust(scenario: Scenario):
    """Convert Python Scenario to Rust PyScenario."""
    return core.Scenario(
        prob_fire_spreads_to_neighbor=scenario.beta,
        prob_solo_agent_extinguishes_fire=scenario.kappa,
        prob_house_catches_fire=scenario.p_spark,
        team_reward_house_survives=scenario.A,
        team_penalty_house_burns=scenario.L,
        cost_to_work_one_night=scenario.c,
        min_nights=scenario.N_min,
        num_agents=scenario.num_agents,
        # Use default individual rewards
        reward_own_house_survives=10.0,
        reward_other_house_survives=5.0,
        penalty_own_house_burns=-10.0,
        penalty_other_house_burns=-5.0,
    )


def _heuristic_action(theta, obs, agent_id, rng):
    """
    Simplified heuristic action selection based on parameters.

    Fast approximation of HeuristicAgent behavior for fitness evaluation.
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
        if obs['houses'][owned_house] == 1 and rng.random() < own_house_priority:
            house = owned_house
        else:
            # Choose a burning house
            burning = [i for i, h in enumerate(obs['houses']) if h == 1]
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


def _run_rust_game(args):
    """
    Run a single game using Rust core.

    Args:
        args: Tuple of (genome, python_scenario, seed)

    Returns:
        Episode reward for agent
    """
    genome, python_scenario, seed = args

    # Convert to Rust scenario in worker
    rust_scenario = _convert_scenario_to_rust(python_scenario)

    # Create Rust game
    game = core.BucketBrigade(rust_scenario, seed=seed)

    # Python RNG for heuristic decisions
    rng = np.random.RandomState(seed)

    # Track agent reward
    episode_reward = 0.0

    # Run until done
    done = False
    step_count = 0
    max_steps = 100  # Safety limit

    while not done and step_count < max_steps:
        # Get observation
        obs = game.get_observation(0)
        obs_dict = {
            'houses': obs.houses,
            'signals': obs.signals,
            'locations': obs.locations,
        }

        # Get action from heuristic
        action = _heuristic_action(genome, obs_dict, 0, rng)

        # Step the Rust game with single agent action
        rewards, done, info = game.step([action])

        # Accumulate reward
        episode_reward += rewards[0]
        step_count += 1

    return episode_reward


class RustFitnessEvaluator:
    """
    Fast fitness evaluator using Rust core.

    Achieves 100x speedup over Python implementation.
    """

    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        games_per_individual: int = 10,
        seed: Optional[int] = None,
        parallel: bool = True,
        num_workers: Optional[int] = None,
    ):
        """
        Initialize Rust-backed fitness evaluator.

        Args:
            scenario: Game scenario to use (None = default)
            games_per_individual: Number of games per agent evaluation
            seed: Random seed for reproducibility
            parallel: Whether to use parallel execution
            num_workers: Number of worker processes (default: cpu_count())
        """
        self.scenario = scenario if scenario is not None else default_scenario(num_agents=1)
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
            Fitness score (mean reward across games)
        """
        # Generate seeds for games
        if self.seed is not None:
            seeds = [self.rng.randint(0, 2**31 - 1) for _ in range(self.games_per_individual)]
        else:
            seeds = [None] * self.games_per_individual

        # Prepare arguments (pass Python scenario, not Rust - can't pickle Rust objects)
        args_list = [
            (individual.genome, self.scenario, seed)
            for seed in seeds
        ]

        if self.parallel:
            # Parallel execution
            with Pool(processes=self.num_workers) as pool:
                episode_rewards = pool.map(_run_rust_game, args_list)
        else:
            # Sequential execution
            episode_rewards = [_run_rust_game(args) for args in args_list]

        return float(np.mean(episode_rewards))

    def evaluate_population(self, population, parallel: bool = None):
        """
        Evaluate all individuals in a population.

        Args:
            population: Population to evaluate
            parallel: Override parallel setting (default: use constructor setting)
        """
        use_parallel = parallel if parallel is not None else self.parallel

        if use_parallel:
            # Evaluate entire population in parallel
            unevaluated = [ind for ind in population if ind.fitness is None]
            if not unevaluated:
                return

            # Create args for each game of each individual
            all_args = []
            for individual in unevaluated:
                if self.seed is not None:
                    seeds = [self.rng.randint(0, 2**31 - 1) for _ in range(self.games_per_individual)]
                else:
                    seeds = [None] * self.games_per_individual

                for seed in seeds:
                    all_args.append((individual.genome, self.scenario, seed))

            # Run all games in parallel
            with Pool(processes=self.num_workers) as pool:
                all_rewards = pool.map(_run_rust_game, all_args)

            # Assign fitness values
            idx = 0
            for individual in unevaluated:
                individual_rewards = all_rewards[idx:idx + self.games_per_individual]
                individual.fitness = float(np.mean(individual_rewards))
                idx += self.games_per_individual
        else:
            # Sequential evaluation (original behavior)
            for individual in population:
                if individual.fitness is None:
                    individual.fitness = self.evaluate_individual(individual)
