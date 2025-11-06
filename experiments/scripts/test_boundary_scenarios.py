#!/usr/bin/env python3
"""
Test the universal strategy on extreme boundary scenarios.

Phase 2A: Universality Boundary Testing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.equilibrium import load_evolved_agent, PayoffEvaluator

# Extreme scenarios to test
EXTREME_SCENARIOS = [
    "glacial_spread",      # β=0.02 (very slow)
    "explosive_spread",    # β=0.60 (fast)
    "wildfire",            # β=0.75 (very fast)
    "free_work",           # c=0.05 (very cheap)
    "cheap_work",          # c=0.10 (cheap)
    "expensive_work",      # c=2.0 (expensive)
    "prohibitive_work",    # c=5.0 (very expensive)
    "crisis_cheap",        # β=0.60, c=0.10 (worst case?)
    "calm_expensive",      # β=0.02, c=5.0 (best case)
]

# Phase 1 baseline scenarios for comparison
BASELINE_SCENARIOS = [
    "chain_reaction",
    "sparse_heroics",
    "trivial_cooperation",
]


def main():
    print("=" * 80)
    print("Phase 2A: Universality Boundary Testing")
    print("=" * 80)
    print()

    # Load universal strategy (any Phase 1 agent - they're all identical)
    universal_genome = load_evolved_agent("chain_reaction", version="v4")
    print("Universal Strategy Genome:")
    print(f"  {universal_genome}")
    print()

    results = {}

    print("Testing universal strategy on extreme scenarios...")
    print()

    for scenario_name in EXTREME_SCENARIOS + BASELINE_SCENARIOS:
        print(f"[{scenario_name}]", end=" ", flush=True)

        # Load scenario
        scenario = get_scenario_by_name(scenario_name, num_agents=4)

        # Create evaluator
        evaluator = PayoffEvaluator(
            scenario=scenario,
            num_simulations=2000,
            seed=42,
            parallel=True,
            use_full_rust=True,
        )

        # Evaluate universal strategy in self-play
        payoff = evaluator.evaluate_symmetric_payoff(
            theta_focal=universal_genome,
            theta_opponents=universal_genome,
        )

        results[scenario_name] = {
            "payoff": float(payoff),
            "beta": scenario.beta,
            "c": scenario.c,
            "kappa": scenario.kappa,
        }

        print(f"β={scenario.beta:.2f}, c={scenario.c:.2f} → {payoff:.2f}")

    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()

    # Sort by payoff
    sorted_results = sorted(results.items(), key=lambda x: x[1]["payoff"])

    print("Payoff Rankings (worst to best):")
    print()
    for i, (scenario_name, data) in enumerate(sorted_results, 1):
        beta = data["beta"]
        c = data["c"]
        payoff = data["payoff"]
        tag = " (EXTREME)" if scenario_name in EXTREME_SCENARIOS else " (baseline)"
        print(f"{i:2d}. {scenario_name:20s} β={beta:.2f} c={c:.2f} → {payoff:6.2f}{tag}")

    print()

    # Analyze extremes
    worst = sorted_results[0]
    best = sorted_results[-1]

    print(f"Worst Performance: {worst[0]} (β={worst[1]['beta']:.2f}, c={worst[1]['c']:.2f}) → {worst[1]['payoff']:.2f}")
    print(f"Best Performance:  {best[0]} (β={best[1]['beta']:.2f}, c={best[1]['c']:.2f}) → {best[1]['payoff']:.2f}")
    print(f"Performance Range: {best[1]['payoff'] - worst[1]['payoff']:.2f}")
    print()

    # Check if universal strategy breaks down
    baseline_payoffs = [results[s]["payoff"] for s in BASELINE_SCENARIOS]
    extreme_payoffs = [results[s]["payoff"] for s in EXTREME_SCENARIOS]

    baseline_mean = np.mean(baseline_payoffs)
    extreme_mean = np.mean(extreme_payoffs)

    print(f"Baseline Mean Payoff: {baseline_mean:.2f}")
    print(f"Extreme Mean Payoff:  {extreme_mean:.2f}")
    print(f"Degradation:          {baseline_mean - extreme_mean:.2f} ({(baseline_mean - extreme_mean) / baseline_mean * 100:.1f}%)")
    print()

    # Identify failure scenarios (payoff < 50% of baseline)
    threshold = baseline_mean * 0.5
    failures = [(name, data) for name, data in results.items() if data["payoff"] < threshold]

    if failures:
        print(f"⚠️  BREAKDOWN DETECTED (payoff < {threshold:.2f}):")
        for name, data in failures:
            print(f"  - {name}: {data['payoff']:.2f} (β={data['beta']:.2f}, c={data['c']:.2f})")
    else:
        print(f"✓ Universal strategy remains robust (all payoffs ≥ {threshold:.2f})")

    print()

    # Save results
    output_file = Path("experiments/boundary_testing/universal_strategy_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "universal_genome": universal_genome.tolist(),
            "results": results,
            "summary": {
                "baseline_mean": baseline_mean,
                "extreme_mean": extreme_mean,
                "degradation_pct": (baseline_mean - extreme_mean) / baseline_mean * 100,
                "worst": worst[0],
                "best": best[0],
                "failures": len(failures),
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
