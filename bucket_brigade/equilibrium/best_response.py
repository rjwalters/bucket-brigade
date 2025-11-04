"""
Best response computation for Nash equilibrium finding.

Provides methods to compute best response strategies given opponent strategies.
Uses numerical optimization over the heuristic parameter space.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Optional
from bucket_brigade.envs.scenarios import Scenario
from bucket_brigade.equilibrium.payoff_evaluator import PayoffEvaluator


def compute_best_response(
    theta_opponents: np.ndarray,
    scenario: Scenario,
    num_simulations: int = 1000,
    method: str = "L-BFGS-B",
    x0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, float]:
    """
    Compute best response strategy to opponent strategy.

    Finds the strategy theta_i that maximizes expected payoff against
    opponents playing theta_opponents. Uses scipy.optimize.minimize
    with bounds [0,1] on each parameter.

    Args:
        theta_opponents: Opponent strategy parameters (10-dimensional)
        scenario: Game scenario
        num_simulations: Number of Monte Carlo rollouts for payoff estimation
        method: Optimization method ('L-BFGS-B', 'SLSQP', 'TNC')
        x0: Initial guess for optimization (if None, uses theta_opponents)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (best_response_strategy, expected_payoff)
    """
    evaluator = PayoffEvaluator(
        scenario=scenario,
        num_simulations=num_simulations,
        seed=seed,
    )

    # Objective: minimize negative payoff (maximize payoff)
    def objective(theta_focal):
        payoff = evaluator.evaluate_symmetric_payoff(
            theta_focal=theta_focal,
            theta_opponents=theta_opponents,
        )
        return -payoff

    # Initial guess
    if x0 is None:
        x0 = theta_opponents.copy()

    # Bounds: all parameters in [0, 1]
    bounds = [(0.0, 1.0) for _ in range(10)]

    # Optimize
    result = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        method=method,
        options={"maxiter": 100},
    )

    best_response = result.x
    best_payoff = -result.fun

    return best_response, best_payoff


def compute_best_response_global(
    theta_opponents: np.ndarray,
    scenario: Scenario,
    num_simulations: int = 1000,
    seed: Optional[int] = None,
    maxiter: int = 20,  # Reduced default from 50 to 20
) -> tuple[np.ndarray, float]:
    """
    Compute best response using global optimization.

    Uses differential evolution for more robust global optimization,
    avoiding local optima. Optimized with parallelization and early stopping.

    Args:
        theta_opponents: Opponent strategy parameters
        scenario: Game scenario
        num_simulations: Number of Monte Carlo rollouts
        seed: Random seed for reproducibility
        maxiter: Maximum iterations for differential evolution (default: 20)

    Returns:
        Tuple of (best_response_strategy, expected_payoff)
    """
    evaluator = PayoffEvaluator(
        scenario=scenario,
        num_simulations=num_simulations,
        seed=seed,
    )

    # Objective: minimize negative payoff
    def objective(theta_focal):
        payoff = evaluator.evaluate_symmetric_payoff(
            theta_focal=theta_focal,
            theta_opponents=theta_opponents,
        )
        return -payoff

    # Bounds: all parameters in [0, 1]
    bounds = [(0.0, 1.0) for _ in range(10)]

    # Early stopping callback
    best_values = []

    def convergence_callback(xk, convergence=0):
        """Stop early if no improvement in last 5 iterations."""
        current_value = objective(xk)
        best_values.append(current_value)
        if len(best_values) >= 5:
            recent_improvement = min(best_values[-5:]) - best_values[-1]
            if abs(recent_improvement) < 0.01:
                return True
        return False

    # Global optimization with adaptive early stopping and parallelization
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        polish=True,
        workers=4,  # Parallelize for speed
        updating='deferred',
        callback=convergence_callback,
        atol=0.01,
        tol=0.001,
    )

    best_response = result.x
    best_payoff = -result.fun

    return best_response, best_payoff


def compute_best_response_to_mixture(
    strategy_mixture: dict[int, float],
    strategy_pool: list[np.ndarray],
    scenario: Scenario,
    num_simulations: int = 1000,
    method: str = "global",
    seed: Optional[int] = None,
) -> tuple[np.ndarray, float]:
    """
    Compute best response to a mixed strategy distribution.

    Finds the strategy that maximizes expected payoff when opponents
    sample their strategies from the given mixture.

    Args:
        strategy_mixture: Dictionary mapping strategy indices to probabilities
        strategy_pool: List of available strategies
        scenario: Game scenario
        num_simulations: Number of Monte Carlo rollouts
        method: 'local' for L-BFGS-B or 'global' for differential evolution
        seed: Random seed

    Returns:
        Tuple of (best_response_strategy, expected_payoff)
    """
    evaluator = PayoffEvaluator(
        scenario=scenario,
        num_simulations=num_simulations,
        seed=seed,
    )

    # Objective: minimize negative expected payoff
    def objective(theta_focal):
        payoff = evaluator.evaluate_against_mixture(
            theta_focal=theta_focal,
            strategy_mixture=strategy_mixture,
            strategy_pool=strategy_pool,
        )
        return -payoff

    # Bounds
    bounds = [(0.0, 1.0) for _ in range(10)]

    if method == "global":
        # Early stopping callback for adaptive convergence
        best_values = []

        def convergence_callback(xk, convergence=0):
            """Stop early if no improvement in last 5 iterations."""
            current_value = objective(xk)
            best_values.append(current_value)

            # Check if converged (no improvement in last 5 iterations)
            if len(best_values) >= 5:
                recent_improvement = min(best_values[-5:]) - best_values[-1]
                if abs(recent_improvement) < 0.01:  # Converged
                    return True
            return False

        # Global optimization with adaptive early stopping and parallelization
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=seed,
            maxiter=20,  # Reduced from 50 for faster convergence
            polish=True,
            workers=4,  # Parallelize across 4 cores (48 cores / 12 scenarios)
            updating='deferred',  # Batch updates for better parallelization
            callback=convergence_callback,
            atol=0.01,  # Absolute tolerance for convergence
            tol=0.001,  # Relative tolerance for convergence
        )
    else:
        # Local optimization from multiple starting points
        best_result = None
        best_value = np.inf

        # Try starting from each strategy in the pool
        for strategy_idx, prob in strategy_mixture.items():
            if prob > 0:
                x0 = strategy_pool[strategy_idx]
                result = minimize(
                    objective,
                    x0=x0,
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={"maxiter": 100},
                )
                if result.fun < best_value:
                    best_value = result.fun
                    best_result = result

        result = best_result

    best_response = result.x
    best_payoff = -result.fun

    return best_response, best_payoff


def verify_best_response(
    theta_candidate: np.ndarray,
    theta_opponents: np.ndarray,
    scenario: Scenario,
    num_simulations: int = 1000,
    num_perturbations: int = 20,
    epsilon: float = 0.01,
    seed: Optional[int] = None,
) -> tuple[bool, float]:
    """
    Verify that a strategy is approximately a best response.

    Tests whether random perturbations of the candidate strategy improve
    payoff by more than epsilon.

    Args:
        theta_candidate: Candidate best response strategy
        theta_opponents: Opponent strategy
        scenario: Game scenario
        num_simulations: Number of Monte Carlo rollouts
        num_perturbations: Number of random perturbations to test
        epsilon: Tolerance for approximate best response
        seed: Random seed

    Returns:
        Tuple of (is_best_response, max_improvement)
            is_best_response: True if no perturbation improves by > epsilon
            max_improvement: Maximum improvement found from perturbations
    """
    evaluator = PayoffEvaluator(
        scenario=scenario,
        num_simulations=num_simulations,
        seed=seed,
    )

    # Evaluate candidate
    candidate_payoff = evaluator.evaluate_symmetric_payoff(
        theta_focal=theta_candidate,
        theta_opponents=theta_opponents,
    )

    # Test random perturbations
    rng = np.random.RandomState(seed)
    max_improvement = 0.0

    for _ in range(num_perturbations):
        # Random perturbation
        perturbation = rng.randn(10) * 0.1
        theta_perturbed = np.clip(theta_candidate + perturbation, 0.0, 1.0)

        # Evaluate perturbed strategy
        perturbed_payoff = evaluator.evaluate_symmetric_payoff(
            theta_focal=theta_perturbed,
            theta_opponents=theta_opponents,
        )

        improvement = perturbed_payoff - candidate_payoff
        max_improvement = max(max_improvement, improvement)

    is_best_response = max_improvement <= epsilon

    return is_best_response, max_improvement
