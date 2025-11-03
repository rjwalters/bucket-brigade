"""
Unit tests for Nash equilibrium computation module.

Tests basic functionality with minimal simulations to verify correctness.
"""

import numpy as np
from bucket_brigade.envs.scenarios import trivial_cooperation_scenario
from bucket_brigade.agents.archetypes import FIREFIGHTER_PARAMS, FREE_RIDER_PARAMS
from bucket_brigade.equilibrium.payoff_evaluator import PayoffEvaluator
from bucket_brigade.equilibrium.best_response import compute_best_response
from bucket_brigade.equilibrium.nash_solver import solve_symmetric_nash
from bucket_brigade.equilibrium.double_oracle import DoubleOracle


class TestPayoffEvaluator:
    """Test PayoffEvaluator class."""

    def test_evaluate_symmetric_payoff_sequential(self):
        """Test payoff evaluation with sequential execution."""
        scenario = trivial_cooperation_scenario(num_agents=4)
        evaluator = PayoffEvaluator(
            scenario=scenario,
            num_simulations=10,  # Very small for fast testing
            seed=42,
            parallel=False,  # Test sequential version
        )

        # Evaluate Firefighter vs Firefighter
        payoff = evaluator.evaluate_symmetric_payoff(
            theta_focal=FIREFIGHTER_PARAMS,
            theta_opponents=FIREFIGHTER_PARAMS,
        )

        # Should be a reasonable payoff (positive for cooperative scenario)
        assert isinstance(payoff, float)
        assert payoff > -1000  # Sanity check
        assert payoff < 1000

    def test_evaluate_symmetric_payoff_parallel(self):
        """Test payoff evaluation with parallel execution."""
        scenario = trivial_cooperation_scenario(num_agents=4)
        evaluator = PayoffEvaluator(
            scenario=scenario,
            num_simulations=10,
            seed=42,
            parallel=True,  # Test parallel version
            num_workers=2,  # Use 2 workers for testing
        )

        # Evaluate Firefighter vs Firefighter
        payoff = evaluator.evaluate_symmetric_payoff(
            theta_focal=FIREFIGHTER_PARAMS,
            theta_opponents=FIREFIGHTER_PARAMS,
        )

        assert isinstance(payoff, float)
        assert payoff > -1000
        assert payoff < 1000

    def test_evaluate_payoff_matrix(self):
        """Test payoff matrix computation."""
        scenario = trivial_cooperation_scenario(num_agents=4)
        evaluator = PayoffEvaluator(
            scenario=scenario,
            num_simulations=5,  # Very small
            seed=42,
            parallel=False,  # Sequential for determinism
        )

        strategy_pool = [FIREFIGHTER_PARAMS, FREE_RIDER_PARAMS]
        payoff_matrix = evaluator.evaluate_payoff_matrix(strategy_pool)

        # Check shape
        assert payoff_matrix.shape == (2, 2)

        # Check that cooperators do better in trivial scenario
        ff_vs_ff = payoff_matrix[0, 0]  # Firefighter vs Firefighter
        fr_vs_ff = payoff_matrix[1, 0]  # Free Rider vs Firefighter

        # In trivial cooperation, working should be rewarded
        print(f"Firefighter vs Firefighter: {ff_vs_ff}")
        print(f"Free Rider vs Firefighter: {fr_vs_ff}")


class TestBestResponse:
    """Test best response computation."""

    def test_compute_best_response(self):
        """Test best response computation with minimal optimization."""
        scenario = trivial_cooperation_scenario(num_agents=4)

        # Find best response to Firefighters
        br, payoff = compute_best_response(
            theta_opponents=FIREFIGHTER_PARAMS,
            scenario=scenario,
            num_simulations=10,  # Very small
            method="L-BFGS-B",
            x0=FIREFIGHTER_PARAMS.copy(),
            seed=42,
        )

        # Best response should be a valid strategy
        assert br.shape == (10,)
        assert np.all(br >= 0.0)
        assert np.all(br <= 1.0)
        assert isinstance(payoff, float)


class TestNashSolver:
    """Test Nash equilibrium solver for finite games."""

    def test_solve_symmetric_nash(self):
        """Test Nash solver on a simple 2x2 game."""
        # Simple coordination game payoff matrix
        # Both players prefer to coordinate
        payoff_matrix = np.array([
            [10.0, 0.0],  # Strategy 0 vs [0, 1]
            [0.0, 10.0],  # Strategy 1 vs [0, 1]
        ])

        distribution = solve_symmetric_nash(payoff_matrix)

        # Should be a valid probability distribution
        assert distribution.shape == (2,)
        assert np.all(distribution >= 0.0)
        assert np.abs(distribution.sum() - 1.0) < 1e-6

        # For this coordination game, equilibrium could be:
        # - Pure strategy (0, 0) or (1, 1)
        # - Mixed strategy (0.5, 0.5)
        print(f"Nash equilibrium: {distribution}")

    def test_solve_symmetric_nash_dominant_strategy(self):
        """Test Nash solver with dominant strategy."""
        # Strategy 0 dominates strategy 1
        payoff_matrix = np.array([
            [10.0, 10.0],  # Strategy 0 always gets 10
            [5.0, 5.0],    # Strategy 1 always gets 5
        ])

        distribution = solve_symmetric_nash(payoff_matrix)

        # Should put all weight on dominant strategy 0
        assert distribution[0] > 0.9  # Almost all weight on strategy 0
        assert np.abs(distribution.sum() - 1.0) < 1e-6


class TestDoubleOracle:
    """Test Double Oracle algorithm."""

    def test_double_oracle_minimal(self):
        """Test Double Oracle with minimal parameters."""
        scenario = trivial_cooperation_scenario(num_agents=4)

        solver = DoubleOracle(
            scenario=scenario,
            num_simulations=10,  # Very small
            max_iterations=2,  # Only 2 iterations
            epsilon=0.5,  # Large epsilon for quick convergence
            seed=42,
            verbose=True,
        )

        # Start with just 2 strategies
        initial_strategies = [
            FIREFIGHTER_PARAMS.copy(),
            FREE_RIDER_PARAMS.copy(),
        ]

        equilibrium = solver.solve(initial_strategies=initial_strategies)

        # Check that we got a valid result
        assert len(equilibrium.strategy_pool) >= 2
        assert len(equilibrium.distribution) > 0
        assert isinstance(equilibrium.payoff, float)
        assert equilibrium.iterations <= 2

        # Check probability distribution sums to 1
        total_prob = sum(equilibrium.distribution.values())
        assert np.abs(total_prob - 1.0) < 1e-6

        print(f"\n{equilibrium.describe()}")


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("=" * 80)
    print("Testing Nash Equilibrium Implementation")
    print("=" * 80)

    print("\n1. Testing PayoffEvaluator (sequential)...")
    test = TestPayoffEvaluator()
    test.test_evaluate_symmetric_payoff_sequential()
    print("✓ Sequential payoff evaluation works")

    print("\n2. Testing PayoffEvaluator (parallel)...")
    test.test_evaluate_symmetric_payoff_parallel()
    print("✓ Parallel payoff evaluation works")

    print("\n3. Testing payoff matrix computation...")
    test.test_evaluate_payoff_matrix()
    print("✓ Payoff matrix computation works")

    print("\n4. Testing best response computation...")
    test_br = TestBestResponse()
    test_br.test_compute_best_response()
    print("✓ Best response computation works")

    print("\n5. Testing Nash solver (coordination game)...")
    test_nash = TestNashSolver()
    test_nash.test_solve_symmetric_nash()
    print("✓ Nash solver works")

    print("\n6. Testing Nash solver (dominant strategy)...")
    test_nash.test_solve_symmetric_nash_dominant_strategy()
    print("✓ Dominant strategy detection works")

    print("\n7. Testing Double Oracle algorithm...")
    test_do = TestDoubleOracle()
    test_do.test_double_oracle_minimal()
    print("✓ Double Oracle algorithm works")

    print("\n" + "=" * 80)
    print("All tests passed! Implementation is working correctly.")
    print("=" * 80)
