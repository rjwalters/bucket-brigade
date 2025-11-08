"""
Double Oracle algorithm for Nash equilibrium computation.

Implements the double oracle algorithm, which iteratively builds the support
of a Nash equilibrium by adding best responses to a strategy pool.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from bucket_brigade.envs.scenarios_generated import Scenario
from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
)
from bucket_brigade.equilibrium.payoff_evaluator_rust import RustPayoffEvaluator
from bucket_brigade.equilibrium.best_response import compute_best_response_to_mixture
from bucket_brigade.equilibrium.nash_solver import solve_symmetric_nash


@dataclass
class NashEquilibrium:
    """
    Result of Nash equilibrium computation.

    Attributes:
        strategy_pool: List of strategies in the support
        distribution: Probability distribution over strategies (indices)
        payoff: Expected payoff at equilibrium
        iterations: Number of iterations to converge
        converged: Whether algorithm converged successfully
    """

    strategy_pool: list[np.ndarray]
    distribution: dict[int, float]
    payoff: float
    iterations: int
    converged: bool

    def get_mixed_strategy(self) -> dict[np.ndarray, float]:
        """
        Convert distribution to readable format.

        Returns:
            Dictionary mapping strategy arrays (as tuples) to probabilities
        """
        result = {}
        for idx, prob in self.distribution.items():
            if prob > 1e-6:  # Only include strategies with non-negligible probability
                strategy_tuple = tuple(self.strategy_pool[idx].tolist())
                result[strategy_tuple] = prob
        return result

    def describe(self) -> str:
        """
        Return human-readable description of equilibrium.

        Returns:
            String describing the equilibrium strategy mix
        """
        lines = [
            f"Nash Equilibrium (iterations={self.iterations}, converged={self.converged})",
            f"Expected Payoff: {self.payoff:.2f}",
            f"Support Size: {len([p for p in self.distribution.values() if p > 1e-6])}",
            "",
            "Strategy Distribution:",
        ]

        for idx, prob in sorted(
            self.distribution.items(), key=lambda x: x[1], reverse=True
        ):
            if prob > 1e-6:
                strategy = self.strategy_pool[idx]
                lines.append(f"  Strategy {idx}: {prob:.3f}")
                lines.append(f"    Parameters: {strategy}")

        return "\n".join(lines)


class DoubleOracle:
    """
    Double Oracle algorithm for finding Nash equilibrium.

    Iteratively expands a pool of strategies by computing best responses,
    until no improving best response exists. At each iteration, solves
    the restricted game over the current pool to find the mixed strategy
    equilibrium, then checks if any best response improves upon it.
    """

    def __init__(
        self,
        scenario: Scenario,
        num_simulations: int = 1000,
        max_iterations: int = 50,
        epsilon: float = 0.01,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize Double Oracle solver.

        Args:
            scenario: Game scenario
            num_simulations: Monte Carlo rollouts for payoff estimation
            max_iterations: Maximum iterations before stopping
            epsilon: Convergence threshold (improvement in payoff)
            seed: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.scenario = scenario
        self.num_simulations = num_simulations
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.seed = seed
        self.verbose = verbose
        self.evaluator = RustPayoffEvaluator(
            scenario=scenario,
            num_simulations=num_simulations,
            seed=seed,
            parallel=True,  # Enable parallelization
            use_full_rust=True,  # Use fastest Rust implementation
        )

    def solve(
        self,
        initial_strategies: Optional[list[np.ndarray]] = None,
    ) -> NashEquilibrium:
        """
        Find Nash equilibrium using double oracle algorithm.

        Args:
            initial_strategies: Initial strategy pool (if None, uses archetypes)

        Returns:
            NashEquilibrium object with equilibrium strategy and metadata
        """
        # Initialize strategy pool with archetypes or provided strategies
        if initial_strategies is None:
            strategy_pool = [
                FIREFIGHTER_PARAMS.copy(),
                FREE_RIDER_PARAMS.copy(),
                HERO_PARAMS.copy(),
                COORDINATOR_PARAMS.copy(),
            ]
        else:
            strategy_pool = [s.copy() for s in initial_strategies]

        if self.verbose:
            print(f"Starting Double Oracle with {len(strategy_pool)} strategies")

        # Main loop
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Step 1: Compute payoff matrix for current pool
            if self.verbose:
                print(
                    f"Computing payoff matrix ({len(strategy_pool)}x{len(strategy_pool)})..."
                )

            payoff_matrix = self.evaluator.evaluate_payoff_matrix(strategy_pool)

            # Step 2: Solve restricted game to find equilibrium
            if self.verbose:
                print("Solving restricted game...")

            eq_distribution = solve_symmetric_nash(payoff_matrix)

            # Convert to dictionary format (remove near-zero probabilities)
            eq_dict = {}
            for idx, prob in enumerate(eq_distribution):
                if prob > 1e-6:
                    eq_dict[idx] = prob

            if self.verbose:
                print(f"Equilibrium support size: {len(eq_dict)}")
                for idx, prob in eq_dict.items():
                    print(f"  Strategy {idx}: {prob:.3f}")

            # Step 3: Compute equilibrium payoff
            eq_payoff = self.evaluator.evaluate_mixture_vs_mixture(
                focal_mixture=eq_dict,
                opponent_mixture=eq_dict,
                strategy_pool=strategy_pool,
            )

            if self.verbose:
                print(f"Equilibrium payoff: {eq_payoff:.2f}")

            # Step 4: Compute best response to equilibrium
            if self.verbose:
                print("Computing best response...")

            br_strategy, br_payoff = compute_best_response_to_mixture(
                strategy_mixture=eq_dict,
                strategy_pool=strategy_pool,
                scenario=self.scenario,
                num_simulations=self.num_simulations,
                method="local",  # Use local optimization for 10x speedup
                seed=self.seed + iteration if self.seed is not None else None,
            )

            improvement = br_payoff - eq_payoff

            if self.verbose:
                print(f"Best response payoff: {br_payoff:.2f}")
                print(f"Improvement: {improvement:.4f}")

            # Step 5: Check convergence
            if improvement <= self.epsilon:
                if self.verbose:
                    print("\nConverged! No improving best response found.")

                return NashEquilibrium(
                    strategy_pool=strategy_pool,
                    distribution=eq_dict,
                    payoff=eq_payoff,
                    iterations=iteration + 1,
                    converged=True,
                )

            # Step 6: Add best response to pool
            strategy_pool.append(br_strategy)

            if self.verbose:
                print(f"Added best response to pool (new size: {len(strategy_pool)})")

        # Max iterations reached
        if self.verbose:
            print(f"\nMax iterations ({self.max_iterations}) reached.")

        # Return best equilibrium found so far
        payoff_matrix = self.evaluator.evaluate_payoff_matrix(strategy_pool)
        eq_distribution = solve_symmetric_nash(payoff_matrix)
        eq_dict = {idx: prob for idx, prob in enumerate(eq_distribution) if prob > 1e-6}
        eq_payoff = self.evaluator.evaluate_mixture_vs_mixture(
            focal_mixture=eq_dict,
            opponent_mixture=eq_dict,
            strategy_pool=strategy_pool,
        )

        return NashEquilibrium(
            strategy_pool=strategy_pool,
            distribution=eq_dict,
            payoff=eq_payoff,
            iterations=self.max_iterations,
            converged=False,
        )
