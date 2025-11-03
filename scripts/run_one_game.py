#!/usr/bin/env python3
"""
Run a single Bucket Brigade game with random agents.
"""

import numpy as np
from pathlib import Path
import sys

# Add the bucket_brigade package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.envs import BucketBrigadeEnv, default_scenario
from bucket_brigade.agents import RandomAgent


def main():
    """Run a single game with random agents."""
    # Create environment
    scenario = default_scenario(num_agents=4)
    env = BucketBrigadeEnv(scenario)

    # Create random agents
    agents = [RandomAgent(i) for i in range(4)]

    # Reset environment
    obs = env.reset(seed=42)
    print("Starting Bucket Brigade game")
    print(f"Scenario: {scenario}")
    print()

    # Run game
    step = 0
    while not env.done:
        print(f"=== Night {step} ===")

        # Get actions from all agents
        actions = []
        for agent in agents:
            action = agent.act(obs)
            actions.append(action)
            print(
                f"{agent.name}: house {action[0]}, mode {'WORK' if action[1] else 'REST'}"
            )

        actions = np.array(actions)

        # Step environment
        obs, rewards, dones, info = env.step(actions)

        # Show results
        print(
            "Houses:",
            "".join(
                ["□" if h == 0 else "🔥" if h == 1 else "💀" for h in obs["houses"]]
            ),
        )
        print("Rewards:", rewards)
        print()

        step += 1
        if step > 50:  # Safety limit
            print("Game didn't terminate, breaking...")
            break

    # Final results
    saved_houses = np.sum(obs["houses"] == 0)
    ruined_houses = np.sum(obs["houses"] == 2)
    total_team_reward = 100 * saved_houses - 100 * ruined_houses

    print("=== Game Over ===")
    print(f"Nights played: {env.night}")
    print(f"Saved houses: {saved_houses}")
    print(f"Ruined houses: {ruined_houses}")
    print(f"Team reward: {total_team_reward}")

    # Save replay
    replay_path = Path("replays") / "test_game.json"
    env.save_replay(str(replay_path))
    print(f"Replay saved to {replay_path}")


if __name__ == "__main__":
    main()
