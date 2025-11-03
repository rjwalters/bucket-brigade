#!/usr/bin/env python3
"""
Test a full game to see if it completes or hangs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.agents import create_archetype_agent

print("Starting full game test...")

# Load scenario
scenario = get_scenario_by_name("greedy_neighbor", num_agents=4)
print(f"Scenario: beta={scenario.beta}, kappa={scenario.kappa}, c={scenario.c}")

# Create environment
env = BucketBrigadeEnv(scenario)

# Create agents
agents = [create_archetype_agent("firefighter", i) for i in range(4)]
print(f"Created {len(agents)} firefighter agents")

# Reset
obs = env.reset(seed=0)
print(f"Game started, night={env.night}")

# Play game
total_rewards = np.zeros(4)
max_nights = 1000  # Safety limit

night = 0
while not env.done and night < max_nights:
    # Get actions
    actions = np.array([agent.act(obs) for agent in agents])

    # Step
    obs, rewards, dones, info = env.step(actions)
    total_rewards += rewards

    night += 1

    # Progress indicator
    if night % 10 == 0:
        print(f"  Night {night}: fires={np.sum(obs['houses'] == 1)}, done={env.done}")

if env.done:
    print(f"\n✅ Game completed successfully!")
    print(f"   Total nights: {env.night}")
    print(f"   Mean reward: {np.mean(total_rewards):.2f}")
    print(f"   Saved houses: {np.sum(obs['houses'] == 0)}")
    print(f"   Ruined houses: {np.sum(obs['houses'] == 2)}")
else:
    print(f"\n⚠️  Game hit max_nights limit ({max_nights})")
