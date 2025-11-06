#!/usr/bin/env python3
"""
Test mechanism design scenarios for cooperation.

Phase 2D: Testing scenarios designed to potentially induce cooperation
and break the universal free-riding equilibrium.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.equilibrium import load_evolved_agent, PayoffEvaluator

# Phase 2D mechanism scenarios
MECHANISM_SCENARIOS = [
    "nearly_free_work",      # c=0.01 (nearly free work)
    "front_loaded_crisis",   # High initial fires, p_spark=0
    "sustained_pressure",    # Very high p_spark=0.10
    "high_stakes",           # A=500, L=500 (high variance)
]

# Reference scenarios for comparison
REFERENCE_SCENARIOS = [
    "chain_reaction",        # Baseline (universal optimum)
    "free_work",            # c=0.05 (Phase 2A extreme)
    "crisis_cheap",         # β=0.60, c=0.10 (Phase 2A extreme)
    "trivial_cooperation",  # p_spark=0 (known failure case)
]


def calculate_genome_work_tendency(genome: np.ndarray) -> float:
    """Extract work_tendency parameter from genome.

    Genome structure (10 parameters):
    [0] honesty, [1] work_tendency, [2] neighbor_help, [3] own_priority,
    [4] risk_aversion, [5] coordination, [6] exploration, [7] fatigue_memory,
    [8] rest_bias, [9] altruism
    """
    return float(genome[1])  # work_tendency is index 1


def main():
    print("=" * 80)
    print("Phase 2D: Mechanism Design for Cooperation")
    print("=" * 80)
    print()

    # Load universal strategy
    universal_genome = load_evolved_agent("chain_reaction", version="v4")
    work_tendency = calculate_genome_work_tendency(universal_genome)

    print("Universal Strategy Genome:")
    print(f"  work_tendency: {work_tendency:.4f}")
    print(f"  Full genome: {universal_genome}")
    print()

    results = {}

    # Test mechanism scenarios
    print("Testing mechanism scenarios...")
    print()

    for scenario_name in MECHANISM_SCENARIOS:
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
            "rho_ignite": scenario.rho_ignite,
            "A": scenario.A,
            "L": scenario.L,
            "type": "mechanism",
        }

        print(f"β={scenario.beta:.2f}, c={scenario.c:.2f}, p_spark={scenario.p_spark:.2f} → {payoff:.2f}")

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
            "rho_ignite": scenario.rho_ignite,
            "A": scenario.A,
            "L": scenario.L,
            "type": "reference",
        }

        print(f"β={scenario.beta:.2f}, c={scenario.c:.2f}, p_spark={scenario.p_spark:.2f} → {payoff:.2f}")

    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()

    # Mechanism scenarios analysis
    print("Mechanism Scenarios Performance:")
    print()
    mechanism_results = [(name, results[name]) for name in MECHANISM_SCENARIOS]
    mechanism_results.sort(key=lambda x: x[1]["payoff"])

    for i, (name, data) in enumerate(mechanism_results, 1):
        payoff = data["payoff"]
        c = data["c"]
        p_spark = data["p_spark"]
        print(f"{i}. {name:25s} c={c:5.2f}, p_spark={p_spark:.2f} → {payoff:6.2f}")

    # Reference scenarios
    print()
    print("Reference Scenarios:")
    print()
    reference_results = [(name, results[name]) for name in REFERENCE_SCENARIOS]
    reference_results.sort(key=lambda x: x[1]["payoff"])

    for i, (name, data) in enumerate(reference_results, 1):
        payoff = data["payoff"]
        c = data["c"]
        p_spark = data["p_spark"]
        print(f"{i}. {name:25s} c={c:5.2f}, p_spark={p_spark:.2f} → {payoff:6.2f}")

    print()
    print("=" * 80)
    print("Key Findings")
    print("=" * 80)
    print()

    # Best and worst mechanism scenarios
    mechanism_payoffs = [results[name]["payoff"] for name in MECHANISM_SCENARIOS]
    reference_payoffs = [results[name]["payoff"] for name in REFERENCE_SCENARIOS]

    mechanism_mean = np.mean(mechanism_payoffs)
    reference_mean = np.mean(reference_payoffs)

    best_mechanism = max(mechanism_results, key=lambda x: x[1]["payoff"])
    worst_mechanism = min(mechanism_results, key=lambda x: x[1]["payoff"])

    print(f"Mechanism Scenarios Mean: {mechanism_mean:.2f}")
    print(f"Reference Scenarios Mean: {reference_mean:.2f}")
    print(f"Difference: {mechanism_mean - reference_mean:+.2f}")
    print()

    print(f"Best Mechanism:")
    print(f"  {best_mechanism[0]}: {best_mechanism[1]['payoff']:.2f}")
    print(f"  c={best_mechanism[1]['c']:.2f}, p_spark={best_mechanism[1]['p_spark']:.2f}")
    print()

    print(f"Worst Mechanism:")
    print(f"  {worst_mechanism[0]}: {worst_mechanism[1]['payoff']:.2f}")
    print(f"  c={worst_mechanism[1]['c']:.2f}, p_spark={worst_mechanism[1]['p_spark']:.2f}")
    print()

    # Check for cooperation induction
    print("Cooperation Analysis:")
    print()
    print(f"Universal strategy work_tendency: {work_tendency:.4f}")
    print()

    # Since we can't measure actual work rate without deeper instrumentation,
    # we can only note that genome parameters remain fixed
    print("Note: Universal strategy genome is fixed. To test if cooperation")
    print("would be beneficial, would need to:")
    print("  1. Evolve new strategies on these scenarios, OR")
    print("  2. Test hand-crafted high-cooperation strategies")
    print()

    # Identify potentially interesting scenarios
    threshold = reference_mean * 0.8  # 80% of reference performance
    weak_scenarios = [(name, data) for name, data in results.items()
                      if data["type"] == "mechanism" and data["payoff"] < threshold]

    if weak_scenarios:
        print(f"⚠️  Scenarios where universal strategy struggles (< {threshold:.2f}):")
        for name, data in weak_scenarios:
            print(f"  - {name}: {data['payoff']:.2f}")
            print(f"    β={data['beta']:.2f}, c={data['c']:.2f}, p_spark={data['p_spark']:.2f}")
        print()
        print("These scenarios might benefit from more cooperative strategies.")
    else:
        print("✓ Universal strategy remains robust across mechanism scenarios.")

    print()

    # Save results
    output_file = Path("experiments/mechanism_design/results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "universal_genome": universal_genome.tolist(),
            "work_tendency": work_tendency,
            "results": results,
            "summary": {
                "mechanism_mean": mechanism_mean,
                "reference_mean": reference_mean,
                "difference": mechanism_mean - reference_mean,
                "best_mechanism": best_mechanism[0],
                "worst_mechanism": worst_mechanism[0],
                "weak_scenarios": [name for name, _ in weak_scenarios],
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
