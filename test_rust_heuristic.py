"""
Quick test to verify Rust heuristic implementation works.
"""

import numpy as np
from bucket_brigade.equilibrium.payoff_evaluator_rust import RustPayoffEvaluator
from bucket_brigade.envs.scenarios import Scenario


def test_rust_vs_python_heuristic():
    """Test that Rust and Python heuristics produce similar results."""
    # Create a simple test scenario
    scenario = Scenario(
        beta=0.15,  # Fire spread probability
        kappa=0.9,  # Extinguish efficiency
        A=100.0,  # Reward per saved house
        L=100.0,  # Penalty per ruined house
        c=0.5,  # Cost per worker
        rho_ignite=0.1,  # Initial burn fraction
        N_min=12,  # Minimum nights
        p_spark=0.01,  # Spark probability
        N_spark=100,  # Spark nights
        num_agents=4,  # Number of agents
    )

    # Create strategy
    theta = np.array([0.0, 0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])

    # Test with Python heuristic
    evaluator_python = RustPayoffEvaluator(
        scenario,
        num_simulations=100,
        seed=42,
        parallel=False,
        use_full_rust=False,
    )
    payoff_python = evaluator_python.evaluate_symmetric_payoff(theta, theta)
    print(f"Python heuristic payoff: {payoff_python:.4f}")

    # Test with Rust heuristic
    evaluator_rust = RustPayoffEvaluator(
        scenario,
        num_simulations=100,
        seed=42,
        parallel=False,
        use_full_rust=True,
    )
    payoff_rust = evaluator_rust.evaluate_symmetric_payoff(theta, theta)
    print(f"Rust heuristic payoff: {payoff_rust:.4f}")

    # They should be close (within Monte Carlo variance)
    diff = abs(payoff_rust - payoff_python)
    print(f"Difference: {diff:.4f}")

    # Allow for some variance due to different RNG implementation
    # but they should be in the same ballpark
    assert diff < 50.0, f"Results too different: {diff:.4f}"

    print("✓ Test passed!")


def test_performance_comparison():
    """Quick performance comparison."""
    import time

    # Create a simple test scenario
    scenario = Scenario(
        beta=0.15,
        kappa=0.9,
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.01,
        N_spark=100,
        num_agents=4,
    )
    theta = np.array([0.0, 0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])

    # Time Python version
    evaluator_python = RustPayoffEvaluator(
        scenario,
        num_simulations=100,
        seed=42,
        parallel=False,
        use_full_rust=False,
    )
    start = time.time()
    _ = evaluator_python.evaluate_symmetric_payoff(theta, theta)
    time_python = time.time() - start
    print(f"Python version: {time_python:.3f}s")

    # Time Rust version
    evaluator_rust = RustPayoffEvaluator(
        scenario,
        num_simulations=100,
        seed=42,
        parallel=False,
        use_full_rust=True,
    )
    start = time.time()
    _ = evaluator_rust.evaluate_symmetric_payoff(theta, theta)
    time_rust = time.time() - start
    print(f"Rust version: {time_rust:.3f}s")

    speedup = time_python / time_rust
    print(f"Speedup: {speedup:.1f}x")

    print("✓ Performance test complete!")


if __name__ == "__main__":
    print("Testing Rust heuristic implementation...\n")
    test_rust_vs_python_heuristic()
    print()
    test_performance_comparison()
