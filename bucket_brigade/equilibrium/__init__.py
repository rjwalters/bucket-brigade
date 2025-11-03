"""
Nash equilibrium computation for Bucket Brigade game.

This module provides tools for computing and analyzing Nash equilibrium
strategies in the Bucket Brigade game using game-theoretic approaches.

Main components:
- PayoffEvaluator: Monte Carlo estimation of strategy payoffs
- BestResponse: Computation of best response strategies
- DoubleOracle: Nash equilibrium finder using double oracle algorithm
- NashVerifier: Verification of Nash equilibrium properties
"""

from bucket_brigade.equilibrium.payoff_evaluator import PayoffEvaluator
from bucket_brigade.equilibrium.best_response import compute_best_response
from bucket_brigade.equilibrium.double_oracle import DoubleOracle, NashEquilibrium

__all__ = [
    "PayoffEvaluator",
    "compute_best_response",
    "DoubleOracle",
    "NashEquilibrium",
]
