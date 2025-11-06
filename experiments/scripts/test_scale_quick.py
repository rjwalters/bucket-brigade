#!/usr/bin/env python3
"""
Quick scale testing: Test N=4 universal strategy on N=6, 8, 10 scenarios.

This script provides fast validation of whether the universal equilibrium
scales to larger population sizes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.equilibrium import load_evolved_agent, PayoffEvaluator

# Population sizes to test
POPULATION_SIZES = [4, 6, 8, 10]

# Representative scenarios
TEST_SCENARIOS = [
    "chain_reaction",  # Baseline
    "sparse_heroics",  # High performance
    "crisis_cheap",  # Extreme parameters
    "easy_spark_02",  # Optimal p_spark
]


def main():
    print("=" * 80)
    print("Scale Testing: Universal Strategy on N=6, 8, 10")
    print("=" * 80)
    print()

    # Load N=4 universal strategy
    universal_genome = load_evolved_agent("chain_reaction", version="v4")
    print("Universal Strategy (N=4):")
    print(f"  work_tendency: {universal_genome[1]:.4f}")
    print(f"  genome: {universal_genome}")
    print()

    results = {}

    # Test each scenario at each population size
    for scenario_name in TEST_SCENARIOS:
        print(f"Testing {scenario_name}:")
        print()

        scenario_results = {}

        for N in POPULATION_SIZES:
            print(f"  N={N:2d}: ", end="", flush=True)

            # Get scenario with specified number of agents
            scenario = get_scenario_by_name(scenario_name, num_agents=N)

            # Evaluate universal strategy in self-play
            # Using Python evaluation for compatibility across all N
            evaluator = PayoffEvaluator(
                scenario=scenario,
                num_simulations=500,  # Reduced for Python performance
                seed=42,
                parallel=True,
                use_full_rust=False,  # Python-only for compatibility
            )

            payoff = evaluator.evaluate_symmetric_payoff(
                theta_focal=universal_genome,
                theta_opponents=universal_genome,
            )

            scenario_results[f"N{N}"] = {
                "payoff": float(payoff),
                "num_agents": N,
            }

            print(f"{payoff:6.2f}", end="")

            # Calculate degradation vs N=4 baseline
            if N == 4:
                baseline_payoff = payoff
                print(" (baseline)")
            else:
                degradation = (baseline_payoff - payoff) / baseline_payoff * 100
                print(f" ({degradation:+6.2f}% vs N=4)")

        results[scenario_name] = {
            "baseline_N4": scenario_results["N4"]["payoff"],
            "results": scenario_results,
        }

        print()

    # Summary analysis
    print("=" * 80)
    print("Scaling Analysis")
    print("=" * 80)
    print()

    # Calculate average degradation for each N
    for N in [6, 8, 10]:
        degradations = []
        for scenario_name in TEST_SCENARIOS:
            baseline = results[scenario_name]["baseline_N4"]
            actual = results[scenario_name]["results"][f"N{N}"]["payoff"]
            deg = (baseline - actual) / baseline * 100 if baseline > 0 else 0
            degradations.append(deg)

        mean_deg = np.mean(degradations)
        max_deg = max(degradations)
        min_deg = min(degradations)

        print(f"N={N}:")
        print(f"  Mean degradation: {mean_deg:+6.2f}%")
        print(f"  Range: {min_deg:+6.2f}% to {max_deg:+6.2f}%")

        # Flag if significant degradation
        if abs(mean_deg) > 20:
            print("  ⚠️  SIGNIFICANT DEGRADATION (>20%)")
        elif abs(mean_deg) > 10:
            print("  ⚠️  Moderate degradation (>10%)")
        else:
            print("  ✓ Good scaling (<10% degradation)")

        print()

    # Per-scenario analysis
    print("=" * 80)
    print("Per-Scenario Scaling")
    print("=" * 80)
    print()

    for scenario_name in TEST_SCENARIOS:
        print(f"{scenario_name}:")

        baseline = results[scenario_name]["baseline_N4"]
        print(f"  N=4:  {baseline:6.2f} (baseline)")

        scaling_good = True
        for N in [6, 8, 10]:
            payoff = results[scenario_name]["results"][f"N{N}"]["payoff"]
            deg = (baseline - payoff) / baseline * 100
            status = "✓" if abs(deg) < 20 else "✗"
            print(f"  N={N:2d}: {payoff:6.2f} ({deg:+6.2f}%) {status}")

            if abs(deg) > 20:
                scaling_good = False

        if not scaling_good:
            print("  → Consider evolution for this scenario")

        print()

    # Overall assessment
    print("=" * 80)
    print("Overall Assessment")
    print("=" * 80)
    print()

    # Count failures (degradation > 20%)
    failures = []
    for scenario_name in TEST_SCENARIOS:
        baseline = results[scenario_name]["baseline_N4"]
        for N in [6, 8, 10]:
            payoff = results[scenario_name]["results"][f"N{N}"]["payoff"]
            deg = abs((baseline - payoff) / baseline * 100)
            if deg > 20:
                failures.append((scenario_name, N, deg))

    if not failures:
        print("✅ UNIVERSAL STRATEGY SCALES WELL")
        print()
        print("The N=4 universal strategy performs well on N=6, 8, 10.")
        print("Degradation < 20% across all scenarios.")
        print()
        print("Conclusion: Universal equilibrium is population-size invariant.")
        print("No evolution needed for larger N.")
    else:
        print("⚠️  UNIVERSAL STRATEGY SHOWS DEGRADATION")
        print()
        print(f"Found {len(failures)} cases with >20% degradation:")
        for scenario, N, deg in failures:
            print(f"  - {scenario} @ N={N}: {deg:.1f}% degradation")
        print()
        print("Recommendation: Run evolution for:")
        affected_scenarios = set(s for s, _, _ in failures)
        for scenario in affected_scenarios:
            Ns = [N for s, N, _ in failures if s == scenario]
            print(f"  - {scenario}: N={Ns}")
        print()

    # Save results
    output_file = Path("experiments/scale_testing/quick_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "universal_genome": universal_genome.tolist(),
                "results": results,
                "failures": [
                    {"scenario": s, "N": N, "degradation_pct": d}
                    for s, N, d in failures
                ],
                "summary": {
                    "total_tests": len(TEST_SCENARIOS) * len([6, 8, 10]),
                    "failures": len(failures),
                    "success_rate": 1 - len(failures) / (len(TEST_SCENARIOS) * 3),
                },
            },
            f,
            indent=2,
        )

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
