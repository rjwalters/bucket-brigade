"""
Test script for the generalized run_heuristic_episode function.

This demonstrates the new API that supports heterogeneous teams.
"""

import numpy as np


def test_heterogeneous_team():
    """Test the generalized function with different parameters per agent."""
    import bucket_brigade_core as core

    # Get a test scenario
    scenario = core.SCENARIOS["trivial_cooperation"]

    # Create heterogeneous team with different strategies
    num_agents = 4  # Standard team size
    agent_params = []

    for i in range(num_agents):
        # Each agent gets different parameters
        params = np.random.uniform(0, 1, 10)
        # Agent 0: High work tendency
        if i == 0:
            params[1] = 0.9  # work_tendency
        # Agent 1: Low work tendency
        elif i == 1:
            params[1] = 0.3  # work_tendency
        # Other agents: Medium work tendency
        else:
            params[1] = 0.6  # work_tendency

        agent_params.append(params.tolist())

    # Run episode
    seed = 42
    rewards = core.run_heuristic_episode(scenario, 4, agent_params, seed)

    print(f"Number of agents: {num_agents}")
    print(f"Rewards: {rewards}")
    print(f"Total team reward: {sum(rewards):.2f}")
    print(f"Average reward: {np.mean(rewards):.2f}")

    # Verify we got rewards for all agents
    assert len(rewards) == num_agents, (
        f"Expected {num_agents} rewards, got {len(rewards)}"
    )
    print("✓ Heterogeneous team test passed!")


def test_homogeneous_team():
    """Test with all agents having the same parameters."""
    import bucket_brigade_core as core

    scenario = core.SCENARIOS["trivial_cooperation"]
    num_agents = scenario.num_agents

    # All agents get the same parameters
    params = np.array([0.0, 0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    agent_params = [params.tolist()] * num_agents

    # Run episode
    seed = 42
    rewards = core.run_heuristic_episode(scenario, 4, agent_params, seed)

    print("\nHomogeneous team:")
    print(f"Number of agents: {num_agents}")
    print(f"Rewards: {rewards}")
    print(f"Total team reward: {sum(rewards):.2f}")

    assert len(rewards) == num_agents
    print("✓ Homogeneous team test passed!")


def test_focal_wrapper():
    """Test the backward-compatible focal function."""
    import bucket_brigade_core as core

    scenario = core.SCENARIOS["trivial_cooperation"]

    theta_focal = np.array([0.0, 0.9, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    theta_opponents = np.array([0.0, 0.7, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])

    seed = 42
    focal_reward = core.run_heuristic_episode_focal(
        scenario, 4, scenario, theta_focal.tolist(), theta_opponents.tolist(), seed
    )

    print("\nFocal wrapper test:")
    print(f"Focal agent reward: {focal_reward:.2f}")

    # Verify it matches the generalized function
    num_agents = scenario.num_agents
    agent_params = [theta_focal.tolist()] + [theta_opponents.tolist()] * (
        num_agents - 1
    )
    all_rewards = core.run_heuristic_episode(scenario, 4, agent_params, seed)

    assert abs(focal_reward - all_rewards[0]) < 0.001, (
        "Focal wrapper should match generalized function"
    )
    print("✓ Focal wrapper test passed!")


def test_error_handling():
    """Test error handling for invalid inputs."""
    import bucket_brigade_core as core

    scenario = core.SCENARIOS["trivial_cooperation"]

    # Test wrong number of agents
    try:
        wrong_count = [[0.5] * 10] * 10  # Too many agents
        core.run_heuristic_episode(scenario, wrong_count, 42)
        assert False, "Should have raised error for wrong agent count"
    except ValueError as e:
        print(f"\n✓ Correctly caught wrong agent count: {e}")

    # Test wrong parameter count
    try:
        wrong_params = [[0.5] * 5] * scenario.num_agents  # Too few parameters
        core.run_heuristic_episode(scenario, wrong_params, 42)
        assert False, "Should have raised error for wrong parameter count"
    except ValueError as e:
        print(f"✓ Correctly caught wrong parameter count: {e}")


if __name__ == "__main__":
    print("Testing generalized run_heuristic_episode...\n")

    try:
        test_heterogeneous_team()
        test_homogeneous_team()
        test_focal_wrapper()
        test_error_handling()
        print("\n🎉 All tests passed!")
    except ImportError:
        print("⚠️ Cannot import bucket_brigade_core - Rust module not built yet.")
        print("Run 'make build-rust' first to build the module.")
        print(
            "\nTests are documented for future validation once build issues are resolved."
        )
