#!/usr/bin/env python3
"""
Investigate the trivial cooperation anomaly.

Phase 2A.1: Testing κ (extinguish rate) and p_spark (ongoing fire generation)
to understand when universal strategy's over-cooperation becomes problematic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.equilibrium import load_evolved_agent, PayoffEvaluator

# κ sweep scenarios (varying extinguish rate, p_spark=0)
KAPPA_SCENARIOS = [
    "easy_kappa_60",  # κ=0.60 (baseline)
    "easy_kappa_70",  # κ=0.70
    "easy_kappa_80",  # κ=0.80
    "easy_kappa_90",  # κ=0.90 (same as trivial_cooperation)
]

# p_spark sweep scenarios (varying ongoing fires, κ=0.90)
SPARK_SCENARIOS = [
    "easy_kappa_90",   # p_spark=0.00 (no ongoing fires - baseline)
    "easy_spark_01",   # p_spark=0.01
    "easy_spark_02",   # p_spark=0.02
    "easy_spark_05",   # p_spark=0.05
]

# Reference scenarios for comparison
REFERENCE_SCENARIOS = [
    "trivial_cooperation",  # κ=0.90, p_spark=0.0 (should match easy_kappa_90)
    "sparse_heroics",       # κ=0.50, p_spark=0.02 (high performance baseline)
]


def main():
    print("=" * 80)
    print("Phase 2A.1: Trivial Cooperation Anomaly Investigation")
    print("=" * 80)
    print()

    # Load universal strategy
    universal_genome = load_evolved_agent("chain_reaction", version="v4")
    print("Universal Strategy Genome:")
    print(f"  {universal_genome}")
    print()

    results = {}

    # Test κ sweep
    print("Testing κ (extinguish rate) sweep (p_spark=0.0)...")
    print()

    for scenario_name in KAPPA_SCENARIOS:
        print(f"[{scenario_name}]", end=" ", flush=True)

        scenario = get_scenario_by_name(scenario_name, num_agents=4)

        evaluator = PayoffEvaluator(
            scenario=scenario,
            num_simulations=2000,
            seed=42,
            parallel=True,
            use_full_rust=True,
        )

        payoff = evaluator.evaluate_symmetric_payoff(
            theta_focal=universal_genome,
            theta_opponents=universal_genome,
        )

        results[scenario_name] = {
            "payoff": float(payoff),
            "beta": scenario.beta,
            "kappa": scenario.kappa,
            "c": scenario.c,
            "p_spark": scenario.p_spark,
        }

        print(f"κ={scenario.kappa:.2f}, p_spark={scenario.p_spark:.2f} → {payoff:.2f}")

    print()
    print("Testing p_spark (ongoing fires) sweep (κ=0.90)...")
    print()

    for scenario_name in SPARK_SCENARIOS:
        if scenario_name in results:
            # Already tested in κ sweep
            continue

        print(f"[{scenario_name}]", end=" ", flush=True)

        scenario = get_scenario_by_name(scenario_name, num_agents=4)

        evaluator = PayoffEvaluator(
            scenario=scenario,
            num_simulations=2000,
            seed=42,
            parallel=True,
            use_full_rust=True,
        )

        payoff = evaluator.evaluate_symmetric_payoff(
            theta_focal=universal_genome,
            theta_opponents=universal_genome,
        )

        results[scenario_name] = {
            "payoff": float(payoff),
            "beta": scenario.beta,
            "kappa": scenario.kappa,
            "c": scenario.c,
            "p_spark": scenario.p_spark,
        }

        print(f"κ={scenario.kappa:.2f}, p_spark={scenario.p_spark:.2f} → {payoff:.2f}")

    print()
    print("Testing reference scenarios...")
    print()

    for scenario_name in REFERENCE_SCENARIOS:
        print(f"[{scenario_name}]", end=" ", flush=True)

        scenario = get_scenario_by_name(scenario_name, num_agents=4)

        evaluator = PayoffEvaluator(
            scenario=scenario,
            num_simulations=2000,
            seed=42,
            parallel=True,
            use_full_rust=True,
        )

        payoff = evaluator.evaluate_symmetric_payoff(
            theta_focal=universal_genome,
            theta_opponents=universal_genome,
        )

        results[scenario_name] = {
            "payoff": float(payoff),
            "beta": scenario.beta,
            "kappa": scenario.kappa,
            "c": scenario.c,
            "p_spark": scenario.p_spark,
        }

        print(f"κ={scenario.kappa:.2f}, p_spark={scenario.p_spark:.2f} → {payoff:.2f}")

    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()

    # κ sweep analysis
    print("κ (Extinguish Rate) Sweep (p_spark=0.0):")
    print()
    kappa_results = [(name, results[name]) for name in KAPPA_SCENARIOS]
    for i, (name, data) in enumerate(kappa_results, 1):
        kappa = data["kappa"]
        payoff = data["payoff"]
        print(f"{i}. {name:20s} κ={kappa:.2f} → {payoff:6.2f}")

    # Check for κ threshold
    kappa_payoffs = [results[name]["payoff"] for name in KAPPA_SCENARIOS]
    kappa_values = [results[name]["kappa"] for name in KAPPA_SCENARIOS]

    print()
    print(f"κ range: {min(kappa_values):.2f} - {max(kappa_values):.2f}")
    print(f"Payoff range: {min(kappa_payoffs):.2f} - {max(kappa_payoffs):.2f}")
    print(f"Performance degradation: {max(kappa_payoffs) - min(kappa_payoffs):.2f} points")
    print()

    # p_spark sweep analysis
    print("p_spark (Ongoing Fire) Sweep (κ=0.90):")
    print()
    spark_results = [(name, results[name]) for name in SPARK_SCENARIOS]
    for i, (name, data) in enumerate(spark_results, 1):
        p_spark = data["p_spark"]
        payoff = data["payoff"]
        print(f"{i}. {name:20s} p_spark={p_spark:.2f} → {payoff:6.2f}")

    # Check for p_spark threshold
    spark_payoffs = [results[name]["payoff"] for name in SPARK_SCENARIOS]
    spark_values = [results[name]["p_spark"] for name in SPARK_SCENARIOS]

    print()
    print(f"p_spark range: {min(spark_values):.2f} - {max(spark_values):.2f}")
    print(f"Payoff range: {min(spark_payoffs):.2f} - {max(spark_payoffs):.2f}")
    print(f"Performance improvement: {max(spark_payoffs) - min(spark_payoffs):.2f} points")
    print()

    # Key findings
    print("=" * 80)
    print("Key Findings")
    print("=" * 80)
    print()

    # Find boundary points
    kappa_best = max(kappa_results, key=lambda x: x[1]["payoff"])
    kappa_worst = min(kappa_results, key=lambda x: x[1]["payoff"])

    spark_best = max(spark_results, key=lambda x: x[1]["payoff"])
    spark_worst = min(spark_results, key=lambda x: x[1]["payoff"])

    print(f"κ Effect:")
    print(f"  Best:  {kappa_best[0]} (κ={kappa_best[1]['kappa']:.2f}) → {kappa_best[1]['payoff']:.2f}")
    print(f"  Worst: {kappa_worst[0]} (κ={kappa_worst[1]['kappa']:.2f}) → {kappa_worst[1]['payoff']:.2f}")
    print(f"  Interpretation: {'Higher κ hurts performance' if kappa_best[1]['kappa'] < kappa_worst[1]['kappa'] else 'Higher κ helps performance'}")
    print()

    print(f"p_spark Effect:")
    print(f"  Best:  {spark_best[0]} (p_spark={spark_best[1]['p_spark']:.2f}) → {spark_best[1]['payoff']:.2f}")
    print(f"  Worst: {spark_worst[0]} (p_spark={spark_worst[1]['p_spark']:.2f}) → {spark_worst[1]['payoff']:.2f}")
    print(f"  Interpretation: {'Ongoing fires help performance' if spark_best[1]['p_spark'] > 0 else 'No ongoing fires is optimal'}")
    print()

    # Verification
    tc_payoff = results["trivial_cooperation"]["payoff"]
    easy_90_payoff = results["easy_kappa_90"]["payoff"]
    print(f"Verification:")
    print(f"  trivial_cooperation: {tc_payoff:.2f}")
    print(f"  easy_kappa_90:       {easy_90_payoff:.2f}")
    print(f"  Match: {'✓' if abs(tc_payoff - easy_90_payoff) < 1.0 else '✗'}")
    print()

    # Save results
    output_file = Path("experiments/boundary_testing/trivial_cooperation_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "universal_genome": universal_genome.tolist(),
            "results": results,
            "analysis": {
                "kappa_effect": {
                    "best": kappa_best[0],
                    "worst": kappa_worst[0],
                    "degradation": max(kappa_payoffs) - min(kappa_payoffs),
                },
                "spark_effect": {
                    "best": spark_best[0],
                    "worst": spark_worst[0],
                    "improvement": max(spark_payoffs) - min(spark_payoffs),
                },
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
