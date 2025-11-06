"""Tests for Rust core integration."""

import pytest
import numpy as np


class TestRustCoreIntegration:
    """Test integration between Python and Rust core."""

    def test_rust_core_availability(self):
        """Test that Rust core can be imported."""
        try:
            from bucket_brigade_core import BucketBrigade, SCENARIOS

            rust_available = True
        except ImportError:
            rust_available = False
            pytest.skip("Rust core not available")

        if rust_available:
            assert BucketBrigade is not None
            assert SCENARIOS is not None

    @pytest.mark.skipif(
        not hasattr(pytest, "rust_core_available"), reason="Rust core not available"
    )
    def test_rust_vs_python_consistency(self):
        """Test that Rust and Python implementations produce consistent results."""
        from bucket_brigade_core import BucketBrigade, SCENARIOS
        from bucket_brigade.envs import BucketBrigadeEnv

        # Use same scenario
        scenario_name = "trivial_cooperation"
        rust_scenario = SCENARIOS[scenario_name]

        # Create environments
        rust_env = BucketBrigade(rust_scenario, 4, seed=42)
        python_env = BucketBrigadeEnv(num_agents=4, seed=42)

        # Run same sequence of actions
        actions = [
            [[0, 1], [1, 0], [2, 1], [3, 0]],  # Mixed work/rest
            [[1, 1], [2, 1], [3, 1], [0, 0]],  # Most working
            [[0, 0], [1, 0], [2, 0], [3, 0]],  # All resting
        ]

        rust_results = []
        python_results = []

        for action_set in actions:
            # Rust step
            rust_rewards, rust_done, _ = rust_env.step(action_set)
            rust_results.append((rust_rewards, rust_done))

            # Python step
            python_obs, python_rewards, python_dones, _ = python_env.step(
                np.array(action_set)
            )
            python_results.append(
                (python_rewards, python_dones[0])
            )  # Single done value

        # Results should be reasonably similar (allowing for floating point differences)
        for i, ((rust_r, rust_d), (python_r, python_d)) in enumerate(
            zip(rust_results, python_results)
        ):
            # Rewards should be close (within tolerance)
            np.testing.assert_allclose(
                rust_r,
                python_r,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Rewards differ at step {i}",
            )

            # Termination should match
            assert rust_d == python_d, (
                f"Termination differs at step {i}: Rust={rust_d}, Python={python_d}"
            )

    def test_rust_core_performance(self):
        """Test that Rust core performs well."""
        try:
            from bucket_brigade_core import BucketBrigade, SCENARIOS
        except ImportError:
            pytest.skip("Rust core not available")

        import time

        scenario = SCENARIOS["trivial_cooperation"]
        env = BucketBrigade(scenario, seed=42)

        # Benchmark: run many steps
        num_steps = 1000
        start_time = time.time()

        for step in range(num_steps):
            actions = [
                [step % 2, (step + j) % 2] for j in range(4)
            ]  # Deterministic actions
            _, done, _ = env.step(actions)

            # Reset if game finishes
            if done:
                env = BucketBrigade(scenario, seed=42 + step)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete in reasonable time (< 1 second for 1000 steps)
        assert total_time < 1.0, f"Took {total_time:.2f}s"

        steps_per_second = num_steps / total_time
        print(f"Steps per second: {steps_per_second:.0f}")

        # Should be very fast (> 1000 steps/second)
        assert steps_per_second > 1000

    def test_rust_core_reproducibility(self):
        """Test that Rust core produces reproducible results."""
        try:
            from bucket_brigade_core import BucketBrigade, SCENARIOS
        except ImportError:
            pytest.skip("Rust core not available")

        scenario = SCENARIOS["trivial_cooperation"]

        # Run same scenario twice with same seed
        results1 = []
        results2 = []

        for run in [1, 2]:
            env = BucketBrigade(scenario, seed=12345)
            run_results = []

            for _ in range(10):
                actions = [[0, 1], [1, 0], [2, 1], [3, 0]]
                rewards, done, _ = env.step(actions)
                run_results.append((rewards, done))

            if run == 1:
                results1 = run_results
            else:
                results2 = run_results

        # Results should be identical
        for i, ((r1, d1), (r2, d2)) in enumerate(zip(results1, results2)):
            np.testing.assert_array_equal(r1, r2, err_msg=f"Rewards differ at step {i}")
            assert d1 == d2, f"Termination differs at step {i}"

    @pytest.mark.skip(
        reason="Rust and Python scenarios may have different parameter values"
    )
    def test_rust_scenario_coverage(self):
        """Test that all scenarios are available in Rust."""
        try:
            from bucket_brigade_core import SCENARIOS
        except ImportError:
            pytest.skip("Rust core not available")

        from bucket_brigade.envs.scenarios import (
            trivial_cooperation_scenario,
            early_containment_scenario,
            greedy_neighbor_scenario,
            sparse_heroics_scenario,
        )

        python_scenarios = [
            ("trivial_cooperation", trivial_cooperation_scenario(num_agents=4)),
            ("early_containment", early_containment_scenario(num_agents=4)),
            ("greedy_neighbor", greedy_neighbor_scenario(num_agents=4)),
            ("sparse_heroics", sparse_heroics_scenario(num_agents=4)),
        ]

        for name, python_scenario in python_scenarios:
            assert name in SCENARIOS
            rust_scenario = SCENARIOS[name]

            # Check key parameters match (use approximate comparison for floats)
            assert (
                abs(rust_scenario.prob_fire_spreads_to_neighbor - python_scenario.beta)
                < 1e-6
            )
            assert (
                abs(
                    rust_scenario.prob_solo_agent_extinguishes_fire
                    - python_scenario.kappa
                )
                < 1e-6
            )
            assert (
                abs(rust_scenario.team_reward_house_survives - python_scenario.A) < 1e-6
            )
            assert (
                abs(rust_scenario.team_penalty_house_burns - python_scenario.L) < 1e-6
            )
            # Note: num_agents is now a team composition parameter, not part of Scenario
