"""
Rust-backed payoff evaluation for Nash equilibrium computation.

This version uses the fast Rust core (bucket_brigade_core) for 10-100x speedup.
"""

import numpy as np
from typing import Any, Optional
from multiprocessing import Pool, cpu_count
import bucket_brigade_core as core


def _convert_scenario_to_rust(scenario: Any) -> core.Scenario:
    """Convert Python Scenario to Rust PyScenario."""
    # Map Python scenario parameters to Rust parameter names
    return core.Scenario(
        prob_fire_spreads_to_neighbor=scenario.beta,  # type: ignore[attr-defined]
        prob_solo_agent_extinguishes_fire=scenario.kappa,  # type: ignore[attr-defined]
        prob_house_catches_fire=scenario.p_spark,  # type: ignore[attr-defined]
        team_reward_house_survives=scenario.A,  # type: ignore[attr-defined]
        team_penalty_house_burns=scenario.L,  # type: ignore[attr-defined]
        cost_to_work_one_night=scenario.c,  # type: ignore[attr-defined]
        min_nights=scenario.N_min,  # type: ignore[attr-defined]
        num_agents=scenario.num_agents,  # type: ignore[attr-defined]
        # Use default individual rewards (not in our Python Scenario)
        reward_own_house_survives=10.0,
        reward_other_house_survives=5.0,
        penalty_own_house_burns=-10.0,
        penalty_other_house_burns=-5.0,
    )


def _heuristic_action(
    theta: np.ndarray, obs: Any, agent_id: int, rng: Any
) -> list[int]:
    """
    Simplified heuristic action selection based on parameters.

    This is a fast approximation of HeuristicAgent behavior.
    For Nash equilibrium, we primarily care about work_tendency.
    """
    # Unpack key parameters
    work_tendency = theta[1]
    own_house_priority = theta[3]
    rest_reward_bias = theta[8]

    # Simple decision: work with probability based on work_tendency
    if rng.random() < work_tendency * (1 - rest_reward_bias):  # type: ignore[attr-defined]
        # Work - choose which house
        owned_house = agent_id % 10

        # Prioritize owned house if burning
        if obs["houses"][owned_house] == 1 and rng.random() < own_house_priority:  # type: ignore[index,attr-defined]
            house = owned_house
        else:
            # Choose a burning house
            burning = [i for i, h in enumerate(obs["houses"]) if h == 1]  # type: ignore[index]
            if burning:
                house = rng.choice(burning)  # type: ignore[attr-defined]
            else:
                house = owned_house

        mode = 1  # WORK
    else:
        # Rest
        house = agent_id % 10
        mode = 0  # REST

    return [house, mode]


def _run_rust_simulation(args):
    """
    Run a single simulation using Rust core.

    Args:
        args: Tuple of (theta_focal, theta_opponents, python_scenario, seed)

    Returns:
        Episode reward for focal agent
    """
    theta_focal, theta_opponents, python_scenario, seed = args

    # Convert to Rust scenario in worker
    rust_scenario = _convert_scenario_to_rust(python_scenario)

    # Create Rust game
    game = core.BucketBrigade(rust_scenario, seed=seed)

    # Python RNG for heuristic decisions
    rng = np.random.RandomState(seed)

    # Track focal agent reward
    episode_reward = 0.0

    # Run until done
    done = False
    step_count = 0
    while not done and step_count < 100:  # Safety limit
        # Get observations for all agents
        observations = []
        for agent_id in range(rust_scenario.num_agents):
            obs = game.get_observation(agent_id)
            # Convert to dict format for heuristic
            obs_dict = {
                "houses": obs.houses,
                "signals": obs.signals,
                "locations": obs.locations,
            }
            observations.append(obs_dict)

        # Get actions from heuristics
        actions = []
        for agent_id in range(rust_scenario.num_agents):
            if agent_id == 0:
                theta = theta_focal
            else:
                theta = theta_opponents

            action = _heuristic_action(theta, observations[agent_id], agent_id, rng)
            actions.append(action)

        # Step the Rust game
        rewards, done, info = game.step(actions)

        # Accumulate focal agent reward
        episode_reward += rewards[0]
        step_count += 1

    return episode_reward


def _run_full_rust_simulation(args):
    """
    Run a single simulation entirely in Rust (no Python boundary crossings).

    This is the optimized version that eliminates the ~50-100 Python/Rust
    boundary crossings per episode by running the entire episode in Rust.

    Args:
        args: Tuple of (theta_focal, theta_opponents, python_scenario, seed)

    Returns:
        Episode reward for focal agent
    """
    theta_focal, theta_opponents, python_scenario, seed = args

    # Convert to Rust scenario in worker
    rust_scenario = _convert_scenario_to_rust(python_scenario)

    # Run entire episode in Rust - single function call
    # Use focal-optimized function for Nash equilibrium computation
    return core.run_heuristic_episode_focal(
        rust_scenario,
        theta_focal.tolist(),
        theta_opponents.tolist(),
        seed,
    )


class RustPayoffEvaluator:
    """
    Fast payoff evaluator using Rust core.

    10-100x faster than Python implementation.
    """

    def __init__(
        self,
        scenario,
        num_simulations: int = 1000,
        seed: Optional[int] = None,
        parallel: bool = True,
        num_workers: Optional[int] = None,
        use_full_rust: bool = True,
    ):
        """
        Initialize Rust-backed payoff evaluator.

        Args:
            scenario: Python Scenario object
            num_simulations: Number of Monte Carlo rollouts
            seed: Random seed for reproducibility
            parallel: Whether to use parallel execution
            num_workers: Number of worker processes (default: cpu_count())
            use_full_rust: If True, runs entire episode in Rust (fastest).
                          If False, uses Python heuristic (legacy mode).
        """
        self.scenario = scenario
        self.rust_scenario = _convert_scenario_to_rust(scenario)
        self.num_simulations = num_simulations
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.parallel = parallel
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        self.use_full_rust = use_full_rust

    def evaluate_symmetric_payoff(
        self,
        theta_focal: np.ndarray,
        theta_opponents: np.ndarray,
    ) -> float:
        """
        Evaluate expected payoff using Rust core.

        Args:
            theta_focal: Focal agent's strategy parameters
            theta_opponents: Opponents' strategy parameters

        Returns:
            Average cumulative reward over simulations
        """
        # Generate seeds
        if self.seed is not None:
            seeds = [
                self.rng.randint(0, 2**31 - 1) for _ in range(self.num_simulations)
            ]
        else:
            seeds = [None] * self.num_simulations

        # Prepare arguments (pass Python scenario, not Rust - can't pickle Rust objects)
        args_list = [
            (theta_focal, theta_opponents, self.scenario, seed) for seed in seeds
        ]

        # Choose simulation function based on mode
        sim_func = (
            _run_full_rust_simulation if self.use_full_rust else _run_rust_simulation
        )

        if self.parallel:
            # Parallel execution
            with Pool(processes=self.num_workers) as pool:
                episode_rewards = pool.map(sim_func, args_list)
        else:
            # Sequential execution
            episode_rewards = [sim_func(args) for args in args_list]

        return np.mean(episode_rewards)

    def evaluate_payoff_matrix(
        self,
        strategy_pool: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute payoff matrix for a pool of strategies.

        Args:
            strategy_pool: List of K strategy parameter vectors

        Returns:
            K×K payoff matrix
        """
        K = len(strategy_pool)
        payoff_matrix = np.zeros((K, K))

        for i in range(K):
            for j in range(K):
                payoff_matrix[i, j] = self.evaluate_symmetric_payoff(
                    theta_focal=strategy_pool[i],
                    theta_opponents=strategy_pool[j],
                )

        return payoff_matrix
