#!/usr/bin/env python3
"""
Test different agent types in Bucket Brigade.
"""

import numpy as np
import sys
from pathlib import Path

# Add the bucket_brigade package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.envs import BucketBrigadeEnv, default_scenario
from bucket_brigade.agents import create_random_agent, create_archetype_agent, RandomAgent


def test_agent_types():
    """Test different agent archetypes."""
    print("Testing Bucket Brigade Agent Types")
    print("=" * 40)

    # Create environment
    scenario = default_scenario(num_agents=4)
    env = BucketBrigadeEnv(scenario)

    # Create different agent types
    agents = [
        RandomAgent(0, "Random"),
        create_archetype_agent('firefighter', 1),
        create_archetype_agent('free_rider', 2),
        create_archetype_agent('coordinator', 3)
    ]

    print(f"Created {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent}")
    print()

    # Test a few nights
    obs = env.reset(seed=42)

    for night in range(5):
        print(f"Night {night}:")
        print(f"  Houses: {''.join(['□' if h == 0 else '🔥' if h == 1 else '💀' for h in obs['houses']])}")

        actions = []
        for agent in agents:
            action = agent.act(obs)
            actions.append(action)
            mode_name = "WORK" if action[1] else "REST"
            print(f"  {agent.name}: house {action[0]}, {mode_name}")

        actions = np.array(actions)
        obs, rewards, dones, info = env.step(actions)

        print(f"  Rewards: {rewards}")
        print()


def test_random_agents():
    """Test a population of random agents."""
    print("Testing Random Agent Population")
    print("=" * 40)

    # Create environment
    scenario = default_scenario(num_agents=6)
    env = BucketBrigadeEnv(scenario)

    # Create random agents
    agents = [create_random_agent(i) for i in range(6)]

    print(f"Created {len(agents)} random agents")
    print("Parameters shape:", agents[0].params.shape)
    print()

    # Run a short game
    obs = env.reset(seed=123)

    total_rewards = np.zeros(len(agents))

    for night in range(10):
        actions = np.array([agent.act(obs) for agent in agents])
        obs, rewards, dones, info = env.step(actions)
        total_rewards += rewards

        if night == 0:
            print("Initial houses:", ''.join(['□' if h == 0 else '🔥' if h == 1 else '💀' for h in obs['houses']]))
        elif night == 9:
            print("Final houses:  ", ''.join(['□' if h == 0 else '🔥' if h == 1 else '💀' for h in obs['houses']]))

    print("Total rewards:", total_rewards)
    print()


if __name__ == "__main__":
    test_agent_types()
    test_random_agents()
