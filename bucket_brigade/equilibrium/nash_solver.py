"""
Nash equilibrium solver for finite matrix games.

Provides methods to solve for symmetric Nash equilibrium in finite
strategy games using linear programming.
"""

import numpy as np
from scipy.optimize import linprog


def solve_symmetric_nash(
    payoff_matrix: np.ndarray,
    method: str = "highs",
) -> np.ndarray:
    """
    Solve for symmetric Nash equilibrium in a symmetric game.

    Given a payoff matrix A where A[i,j] is the payoff for strategy i
    when opponents play strategy j, finds a probability distribution p
    over strategies that forms a symmetric Nash equilibrium.

    Uses linear programming to solve the support LP:
        max_p  min_i  sum_j p[j] * A[i,j]
        s.t.   sum_i p[i] = 1
               p[i] >= 0

    This is equivalent to finding a maxmin strategy, which equals a
    Nash equilibrium in symmetric two-player games.

    Args:
        payoff_matrix: K×K payoff matrix for K strategies
        method: LP solver method ('highs', 'interior-point', 'simplex')

    Returns:
        Probability distribution over strategies (K-dimensional)
    """
    K = payoff_matrix.shape[0]

    # Formulate as LP:
    # Variables: p[0], ..., p[K-1], v
    # Maximize v (the value of the game)
    # Subject to:
    #   For each strategy i: sum_j A[i,j] * p[j] >= v
    #   sum_i p[i] = 1
    #   p[i] >= 0

    # Convert to standard form (minimize -v):
    # Variables: [p[0], ..., p[K-1], v]
    c = np.zeros(K + 1)
    c[-1] = -1.0  # Maximize v <=> minimize -v

    # Inequality constraints: -A^T @ p + v <= 0
    # (equivalent to: A^T @ p >= v for each row)
    A_ub = np.hstack([-payoff_matrix.T, np.ones((K, 1))])
    b_ub = np.zeros(K)

    # Equality constraint: sum p[i] = 1
    A_eq = np.zeros((1, K + 1))
    A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])

    # Bounds: p[i] >= 0, v unbounded
    bounds = [(0, None) for _ in range(K)] + [(None, None)]

    # Solve LP
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
    )

    if not result.success:
        raise ValueError(f"LP solver failed: {result.message}")

    # Extract probability distribution
    p = result.x[:K]

    # Normalize (numerical precision)
    p = np.maximum(p, 0)  # Ensure non-negative
    p = p / p.sum()  # Normalize to sum to 1

    return p


def verify_nash_equilibrium_finite(
    payoff_matrix: np.ndarray,
    distribution: np.ndarray,
    epsilon: float = 1e-4,
) -> tuple[bool, float]:
    """
    Verify that a distribution is a Nash equilibrium for a finite game.

    Checks that no pure strategy deviation improves payoff by more than epsilon.

    Args:
        payoff_matrix: K×K payoff matrix
        distribution: Probability distribution over K strategies
        epsilon: Tolerance for approximate equilibrium

    Returns:
        Tuple of (is_equilibrium, max_improvement)
            is_equilibrium: True if distribution is epsilon-Nash
            max_improvement: Maximum improvement from any deviation
    """
    K = len(distribution)

    # Expected payoff at equilibrium
    eq_payoff = distribution @ payoff_matrix @ distribution

    # Check all pure strategy deviations
    max_improvement = 0.0
    for i in range(K):
        # Payoff from deviating to pure strategy i
        deviation_payoff = payoff_matrix[i] @ distribution
        improvement = deviation_payoff - eq_payoff
        max_improvement = max(max_improvement, improvement)

    is_equilibrium = max_improvement <= epsilon

    return is_equilibrium, max_improvement


def compute_support(
    distribution: np.ndarray,
    threshold: float = 1e-6,
) -> list[int]:
    """
    Compute support of a mixed strategy.

    Args:
        distribution: Probability distribution
        threshold: Minimum probability to be considered in support

    Returns:
        List of strategy indices with probability >= threshold
    """
    return [i for i, p in enumerate(distribution) if p >= threshold]


def compute_expected_payoff(
    payoff_matrix: np.ndarray,
    focal_dist: np.ndarray,
    opponent_dist: np.ndarray,
) -> float:
    """
    Compute expected payoff for focal player against opponent distribution.

    Args:
        payoff_matrix: K×K payoff matrix
        focal_dist: Focal player's probability distribution
        opponent_dist: Opponent's probability distribution

    Returns:
        Expected payoff
    """
    return float(focal_dist @ payoff_matrix @ opponent_dist)
