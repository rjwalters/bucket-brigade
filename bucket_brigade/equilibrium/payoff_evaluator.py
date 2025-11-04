"""
Payoff evaluation for Nash equilibrium computation.

Provides Monte Carlo estimation of expected payoffs for agents playing
different strategies against opponent strategies in specified scenarios.
"""

import numpy as np
from typing import Optional
from multiprocessing import Pool, cpu_count
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios import Scenario
from bucket_brigade.agents.heuristic_agent import HeuristicAgent


def _run_single_simulation(args: tuple[np.ndarray, np.ndarray, Scenario, int]) -> float:
    """
    Run a single simulation episode. Helper function for parallel execution.

    Args:
        args: Tuple of (theta_focal, theta_opponents, scenario, seed)

    Returns:
        Episode reward for focal agent
    """
    theta_focal, theta_opponents, scenario, seed = args

    # Create environment
    env = BucketBrigadeEnv(scenario)

    # Create agents
    focal_agent = HeuristicAgent(theta_focal, agent_id=0)
    opponent_agents = [
        HeuristicAgent(theta_opponents, agent_id=i + 1)
        for i in range(scenario.num_agents - 1)
    ]
    agents = [focal_agent] + opponent_agents

    # Run episode
    observations = env.reset(seed=seed)
    episode_reward = 0.0
    done = False

    while not done:
        # Get actions from all agents
        actions = []
        for agent in agents:
            action = agent.act(observations)
            actions.append(action)

        # Convert to numpy array
        actions = np.array(actions)

        # Step environment
        observations, rewards, dones, info = env.step(actions)
        done = dones.any()

        # Accumulate focal agent's reward
        episode_reward += rewards[0]

    return episode_reward


class PayoffEvaluator:
    """
    Evaluates expected payoffs for strategies via Monte Carlo simulation.

    This class provides methods to estimate the expected cumulative reward
    for an agent using a given strategy (heuristic parameter vector) when
    playing against opponents using specified strategies.
    """

    def __init__(
        self,
        scenario: Scenario,
        num_simulations: int = 1000,
        seed: Optional[int] = None,
        parallel: bool = True,
        num_workers: Optional[int] = None,
    ):
        """
        Initialize payoff evaluator.

        Args:
            scenario: Game scenario with fire dynamics and reward parameters
            num_simulations: Number of Monte Carlo rollouts for payoff estimation
            seed: Random seed for reproducibility
            parallel: Whether to use parallel execution (default: True)
            num_workers: Number of worker processes (default: cpu_count())
        """
        self.scenario = scenario
        self.num_simulations = num_simulations
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.parallel = parallel
        self.num_workers = num_workers if num_workers is not None else cpu_count()

    def evaluate_symmetric_payoff(
        self,
        theta_focal: np.ndarray,
        theta_opponents: np.ndarray,
    ) -> float:
        """
        Evaluate expected payoff for focal agent against symmetric opponents.

        In a symmetric game, all opponents use the same strategy. This is the
        standard evaluation for finding symmetric Nash equilibria.

        Args:
            theta_focal: Focal agent's strategy parameters (10-dimensional)
            theta_opponents: Opponents' strategy parameters (10-dimensional)

        Returns:
            Average cumulative reward over num_simulations episodes
        """
        if self.parallel:
            # Parallel execution
            # Generate seeds for all simulations
            if self.seed is not None:
                seeds = [
                    self.rng.randint(0, 2**31 - 1) for _ in range(self.num_simulations)
                ]
            else:
                seeds = [None] * self.num_simulations

            # Prepare arguments for parallel execution
            args_list = [
                (theta_focal, theta_opponents, self.scenario, seed) for seed in seeds
            ]

            # Run simulations in parallel
            with Pool(processes=self.num_workers) as pool:
                episode_rewards = pool.map(_run_single_simulation, args_list)

            return float(np.mean(episode_rewards))
        else:
            # Sequential execution (original implementation)
            total_reward = 0.0

            for sim_idx in range(self.num_simulations):
                # Create environment for this simulation
                sim_seed = (
                    self.rng.randint(0, 2**31 - 1) if self.seed is not None else None
                )
                env = BucketBrigadeEnv(self.scenario)

                # Create agents (focal agent + opponents with same strategy)
                focal_agent = HeuristicAgent(theta_focal, agent_id=0)
                opponent_agents = [
                    HeuristicAgent(theta_opponents, agent_id=i + 1)
                    for i in range(self.scenario.num_agents - 1)
                ]
                agents = [focal_agent] + opponent_agents

                # Run episode
                observations = env.reset(seed=sim_seed)
                episode_reward = 0.0
                done = False

                while not done:
                    # Get actions from all agents (all agents receive same observation)
                    actions = []
                    for agent in agents:
                        action = agent.act(observations)
                        actions.append(action)

                    # Convert actions to numpy array
                    actions = np.array(actions)

                    # Step environment
                    observations, rewards, dones, info = env.step(actions)
                    done = dones.any()

                    # Accumulate focal agent's reward
                    episode_reward += rewards[0]

                total_reward += episode_reward

            return total_reward / self.num_simulations

    def evaluate_payoff_matrix(
        self,
        strategy_pool: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute payoff matrix for a pool of strategies.

        For symmetric games, computes K×K matrix where entry (i,j) is the
        expected payoff for an agent using strategy i when opponents use
        strategy j.

        Args:
            strategy_pool: List of K strategy parameter vectors

        Returns:
            K×K numpy array with payoff matrix[i,j] = payoff(strategy_i vs strategy_j)
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

    def evaluate_against_mixture(
        self,
        theta_focal: np.ndarray,
        strategy_mixture: dict[int, float],
        strategy_pool: list[np.ndarray],
    ) -> float:
        """
        Evaluate expected payoff against a mixed strategy distribution.

        Computes E[payoff | opponents sample strategies from mixture].
        This is used to evaluate best responses to mixed equilibria.

        Args:
            theta_focal: Focal agent's strategy parameters
            strategy_mixture: Dictionary mapping strategy indices to probabilities
            strategy_pool: List of available strategies

        Returns:
            Expected payoff when opponents play according to mixture
        """
        expected_payoff = 0.0

        for strategy_idx, prob in strategy_mixture.items():
            if prob > 0:
                theta_opponents = strategy_pool[strategy_idx]
                payoff = self.evaluate_symmetric_payoff(
                    theta_focal=theta_focal,
                    theta_opponents=theta_opponents,
                )
                expected_payoff += prob * payoff

        return expected_payoff

    def evaluate_mixture_vs_mixture(
        self,
        focal_mixture: dict[int, float],
        opponent_mixture: dict[int, float],
        strategy_pool: list[np.ndarray],
    ) -> float:
        """
        Evaluate expected payoff when both focal and opponents use mixed strategies.

        Computes E[payoff | focal and opponents both sample from mixtures].
        This is used to compute the equilibrium payoff in mixed strategy equilibria.

        Args:
            focal_mixture: Focal agent's probability distribution over strategies
            opponent_mixture: Opponents' probability distribution over strategies
            strategy_pool: List of available strategies

        Returns:
            Expected payoff under both mixtures
        """
        expected_payoff = 0.0

        for focal_idx, focal_prob in focal_mixture.items():
            if focal_prob > 0:
                for opponent_idx, opponent_prob in opponent_mixture.items():
                    if opponent_prob > 0:
                        theta_focal = strategy_pool[focal_idx]
                        theta_opponents = strategy_pool[opponent_idx]
                        payoff = self.evaluate_symmetric_payoff(
                            theta_focal=theta_focal,
                            theta_opponents=theta_opponents,
                        )
                        expected_payoff += focal_prob * opponent_prob * payoff

        return expected_payoff
