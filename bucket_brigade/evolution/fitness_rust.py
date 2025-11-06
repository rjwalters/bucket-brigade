"""Rust-backed fitness evaluation for evolutionary algorithms.

This module provides fast fitness functions using bucket_brigade_core
Rust bindings for 100x speedup over Python implementation.
"""

from __future__ import annotations

from typing import Optional
from multiprocessing import Pool, cpu_count

import numpy as np
from typing import Any
import bucket_brigade_core as core

from .population import Individual


def _heuristic_action(
    theta: np.ndarray, obs: Any, agent_id: int, rng: Any
) -> list[int]:
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


def _run_rust_game(args: tuple[np.ndarray, str, int, int]) -> float:
    """
    Run a single game using Rust core.

    Args:
        args: Tuple of (genome, scenario_name, num_agents, seed)

    Returns:
        Scenario payoff (team final score)
    """
    genome, scenario_name, num_agents, seed = args

    # Get Rust scenario (single source of truth)
    rust_scenario = core.SCENARIOS[scenario_name]

    # Create Rust game
    game = core.BucketBrigade(rust_scenario, num_agents, seed=seed)

    # Python RNG for heuristic decisions
    rng = np.random.RandomState(seed)

    # Run until done
    done = False
    step_count = 0
    max_steps = 100  # Safety limit

    while not done and step_count < max_steps:
        # Get actions for ALL agents (not just agent 0)
        actions = []
        for agent_id in range(num_agents):
            obs = game.get_observation(agent_id)
            obs_dict = {
                "houses": obs.houses,
                "signals": obs.signals,
                "locations": obs.locations,
            }

            # Get action from heuristic
            action = _heuristic_action(genome, obs_dict, agent_id, rng)
            actions.append(action)

        # Step the Rust game with ALL agent actions
        rewards, done, info = game.step(actions)
        step_count += 1

    # Return scenario final score instead of agent rewards
    # This matches tournament evaluation and provides interpretable fitness
    result = game.get_result()
    return result.final_score


class RustFitnessEvaluator:
    """
    Fast fitness evaluator using Rust core.

    Achieves 100x speedup over Python implementation.
    """

    def __init__(
        self,
        scenario_name: str = "trivial_cooperation",
        num_agents: int = 4,
        games_per_individual: int = 10,
        seed: Optional[int] = None,
        parallel: bool = True,
        num_workers: Optional[int] = None,
    ):
        """
        Initialize Rust-backed fitness evaluator.

        Args:
            scenario_name: Scenario name from core.SCENARIOS
            num_agents: Number of agents in the game
            games_per_individual: Number of games per agent evaluation
            seed: Random seed for reproducibility
            parallel: Whether to use parallel execution
            num_workers: Number of worker processes (default: cpu_count())
        """
        if scenario_name not in core.SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(core.SCENARIOS.keys())}"
            )
        self.scenario_name = scenario_name
        self.num_agents = num_agents
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
            seeds = [
                self.rng.randint(0, 2**31 - 1) for _ in range(self.games_per_individual)
            ]
        else:
            seeds = [None] * self.games_per_individual

        # Prepare arguments
        args_list = [(individual.genome, self.scenario_name, self.num_agents, seed) for seed in seeds]

        if self.parallel:
            # Parallel execution
            with Pool(processes=self.num_workers) as pool:
                episode_rewards = pool.map(_run_rust_game, args_list)
        else:
            # Sequential execution
            episode_rewards = [_run_rust_game(args) for args in args_list]

        return float(np.mean(episode_rewards))

    def evaluate_population(
        self, population: Any, parallel: Optional[bool] = None
    ) -> None:
        """
        Evaluate all individuals in a population.

        Args:
            population: Population to evaluate
            parallel: Override parallel setting (default: use constructor setting)
        """
        use_parallel = parallel if parallel is not None else self.parallel

        if use_parallel:
            # Evaluate entire population in parallel
            unevaluated = [ind for ind in population if ind.fitness is None]  # type: ignore[attr-defined]
            if not unevaluated:
                return

            # Create args for each game of each individual
            all_args = []
            for individual in unevaluated:
                if self.seed is not None:
                    seeds = [
                        self.rng.randint(0, 2**31 - 1)
                        for _ in range(self.games_per_individual)
                    ]
                else:
                    seeds = [None] * self.games_per_individual

                for seed in seeds:
                    all_args.append((individual.genome, self.scenario_name, self.num_agents, seed))  # type: ignore[attr-defined]

            # Run all games in parallel
            with Pool(processes=self.num_workers) as pool:
                all_rewards = pool.map(_run_rust_game, all_args)

            # Assign fitness values
            idx = 0
            for individual in unevaluated:
                individual_rewards = all_rewards[idx : idx + self.games_per_individual]
                individual.fitness = float(np.mean(individual_rewards))  # type: ignore[attr-defined]
                idx += self.games_per_individual
        else:
            # Sequential evaluation (original behavior)
            for individual in population:  # type: ignore[attr-defined]
                if individual.fitness is None:  # type: ignore[attr-defined]
                    individual.fitness = self.evaluate_individual(individual)  # type: ignore[attr-defined]
