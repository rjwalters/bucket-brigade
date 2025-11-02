#!/usr/bin/env python3
"""
Test script for the Rust core Bucket Brigade implementation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bucket-brigade-core"))

try:
    from bucket_brigade_core import BucketBrigade, SCENARIOS

    print("✅ Successfully imported Rust core!")

    # Test scenario loading
    scenario = SCENARIOS["trivial_cooperation"]
    print(f"✅ Loaded scenario: {scenario.beta}, {scenario.kappa}")

    # Test environment creation
    env = BucketBrigade(scenario)
    print("✅ Created BucketBrigade environment")

    # Test reset
    env.reset()
    print("✅ Environment reset successful")

    # Test observation
    obs = env.get_observation(0)
    print(f"✅ Got observation: {len(obs.houses)} houses, {len(obs.signals)} signals")

    # Test step - need actions for all agents
    burn_house = obs.houses.index(1) if 1 in obs.houses else 0
    actions = [
        [burn_house, 1],
        [burn_house, 1],
        [burn_house, 1],
        [burn_house, 1],
    ]  # All work on burning house
    rewards, done, info = env.step(actions)
    print(f"✅ Step successful: rewards={rewards}, done={done}")

    # Test result
    result = env.get_result()
    print(
        f"✅ Got result: final_score={result.final_score}, agent_scores={result.agent_scores}"
    )

    print("\n🎉 All tests passed! Rust core is working correctly.")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Note: You may need to build the Python extension first:")
    print(
        "cd bucket-brigade-core && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install -e ."
    )

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback

    traceback.print_exc()
