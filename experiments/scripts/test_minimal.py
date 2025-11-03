#!/usr/bin/env python3
"""
Minimal test to debug the hanging issue.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.agents import create_archetype_agent

print("1. Importing modules... ✓")

# Load scenario
scenario = get_scenario_by_name("greedy_neighbor", num_agents=4)
print(f"2. Loaded scenario: {scenario.beta=}, {scenario.kappa=}, {scenario.c=} ✓")

# Create environment
env = BucketBrigadeEnv(scenario)
print("3. Created environment ✓")

# Create agents
agents = [create_archetype_agent("firefighter", i) for i in range(4)]
print(f"4. Created {len(agents)} firefighter agents ✓")

# Reset environment
print("5. Resetting environment...")
obs = env.reset(seed=0)
print(f"   Reset complete! obs keys: {list(obs.keys())} ✓")

# Try one step
print("6. Getting actions from agents...")
try:
    actions = []
    for i, agent in enumerate(agents):
        print(f"   Agent {i} acting...", end=" ", flush=True)
        action = agent.act(obs)
        print(f"action={action} ✓")
        actions.append(action)

    actions = np.array(actions)
    print(f"   All actions collected: {actions} ✓")

except Exception as e:
    print(f"\n   ERROR during agent.act(): {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Try stepping
print("7. Stepping environment...")
try:
    obs, rewards, dones, info = env.step(actions)
    print(f"   Step complete! rewards={rewards} ✓")
except Exception as e:
    print(f"\n   ERROR during env.step(): {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! Environment and agents are working.")
print(f"   Night: {env.night}")
print(f"   Done: {env.done}")
