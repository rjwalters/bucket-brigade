#!/usr/bin/env python3
"""
Evaluate an evolved agent from one scenario in a different scenario.

This script is part of Phase 1.5: Cross-Scenario Generalization Analysis.

Usage:
    # Evaluate chain_reaction agent in greedy_neighbor scenario
    python experiments/scripts/evaluate_cross_scenario.py \
        --agent-scenario chain_reaction \
        --test-scenario greedy_neighbor \
        --simulations 2000

    # Batch evaluation (all 81 combinations)
    python experiments/scripts/evaluate_cross_scenario.py --all
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.equilibrium import (
    load_evolved_agent,
    load_evolved_agent_metadata,
    PayoffEvaluator,
)

# Standard scenarios (9 total)
SCENARIOS = [
    "chain_reaction",
    "deceptive_calm",
    "early_containment",
    "greedy_neighbor",
    "mixed_motivation",
    "overcrowding",
    "rest_trap",
    "sparse_heroics",
    "trivial_cooperation",
]


def evaluate_cross_scenario(
    agent_scenario: str,
    test_scenario: str,
    evolved_version: str = "v4",
    num_simulations: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Evaluate an evolved agent from one scenario in a different scenario.

    Args:
        agent_scenario: Scenario the agent was evolved in
        test_scenario: Scenario to test the agent in
        evolved_version: Evolution version to use (v3, v4, v5)
        num_simulations: Number of Monte Carlo rollouts
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Dictionary with evaluation results
    """

    if verbose:
        print("=" * 80)
        print("Cross-Scenario Agent Evaluation")
        print("=" * 80)
        print()
        print(f"Agent from:  {agent_scenario} ({evolved_version})")
        print(f"Testing in:  {test_scenario}")
        print()

    # Load evolved agent
    agent_genome = load_evolved_agent(agent_scenario, version=evolved_version)

    if agent_genome is None:
        raise ValueError(
            f"No evolved agent found for {agent_scenario} {evolved_version}"
        )

    agent_metadata = load_evolved_agent_metadata(agent_scenario, version=evolved_version)

    if verbose:
        print("Agent Parameters:")
        if agent_metadata and "parameters" in agent_metadata:
            params = agent_metadata["parameters"]
            print(f"  honesty:        {params.get('honesty', 0):.3f}")
            print(f"  work_tendency:  {params.get('work_tendency', 0):.3f}")
            print(f"  neighbor_help:  {params.get('neighbor_help', 0):.3f}")
            print(f"  own_priority:   {params.get('own_priority', 0):.3f}")
        print()

    # Load test scenario
    test_scenario_obj = get_scenario_by_name(test_scenario, num_agents=4)

    if verbose:
        print("Test Scenario Parameters:")
        print(f"  beta (spread):       {test_scenario_obj.beta:.2f}")
        print(f"  kappa (extinguish):  {test_scenario_obj.kappa:.2f}")
        print(f"  c (work cost):       {test_scenario_obj.c:.2f}")
        print()

    # Create Rust evaluator for test scenario
    evaluator = PayoffEvaluator(
        scenario=test_scenario_obj,
        num_simulations=num_simulations,
        seed=seed,
        parallel=True,
        use_full_rust=True,
    )

    if verbose:
        print(f"Running {num_simulations} simulations...")

    # Evaluate agent in self-play (same as evolution fitness evaluation)
    start_time = time.time()
    payoff = evaluator.evaluate_symmetric_payoff(
        theta_focal=agent_genome,
        theta_opponents=agent_genome,  # Self-play
    )
    elapsed_time = time.time() - start_time

    if verbose:
        print(f"  Completed in {elapsed_time:.2f}s")
        print()
        print(f"Average Payoff: {payoff:.4f}")
        print()

    # Prepare results
    results = {
        "agent_scenario": agent_scenario,
        "test_scenario": test_scenario,
        "evolved_version": evolved_version,
        "payoff": float(payoff),
        "num_simulations": num_simulations,
        "seed": seed,
        "elapsed_time": elapsed_time,
        "agent_metadata": agent_metadata,
        "test_scenario_params": {
            "beta": test_scenario_obj.beta,
            "kappa": test_scenario_obj.kappa,
            "c": test_scenario_obj.c,
            "A": test_scenario_obj.A,
            "L": test_scenario_obj.L,
            "num_agents": test_scenario_obj.num_agents,
        },
    }

    # Add on-scenario fitness if available
    if agent_metadata and "fitness" in agent_metadata:
        results["on_scenario_fitness"] = agent_metadata["fitness"]

    return results


def evaluate_all_combinations(
    evolved_version: str = "v4",
    num_simulations: int = 2000,
    seed: int = 42,
    output_file: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """
    Evaluate all 9x9=81 combinations of agents and test scenarios.

    Args:
        evolved_version: Evolution version to use
        num_simulations: Simulations per evaluation
        seed: Random seed
        output_file: Where to save results (default: experiments/generalization/performance_matrix.json)
        verbose: Print detailed progress

    Returns:
        9x9 performance matrix dictionary
    """

    if output_file is None:
        output_file = Path("experiments/generalization/performance_matrix.json")

    print("=" * 80)
    print("Phase 1.5: Cross-Scenario Generalization Analysis")
    print("=" * 80)
    print()
    print(f"Evaluating {len(SCENARIOS)}×{len(SCENARIOS)} = {len(SCENARIOS)**2} combinations")
    print(f"Evolution version: {evolved_version}")
    print(f"Simulations per evaluation: {num_simulations}")
    print()

    # Performance matrix: [agent_scenario][test_scenario] -> payoff
    performance_matrix = {}
    all_results = []

    total_evaluations = len(SCENARIOS) * len(SCENARIOS)
    evaluation_count = 0

    for agent_scenario in SCENARIOS:
        performance_matrix[agent_scenario] = {}

        for test_scenario in SCENARIOS:
            evaluation_count += 1

            print(
                f"[{evaluation_count}/{total_evaluations}] "
                f"{agent_scenario} → {test_scenario}... ",
                end="",
                flush=True,
            )

            try:
                result = evaluate_cross_scenario(
                    agent_scenario=agent_scenario,
                    test_scenario=test_scenario,
                    evolved_version=evolved_version,
                    num_simulations=num_simulations,
                    seed=seed,
                    verbose=verbose,
                )

                payoff = result["payoff"]
                performance_matrix[agent_scenario][test_scenario] = payoff
                all_results.append(result)

                print(f"✓ {payoff:.2f}")

            except Exception as e:
                print(f"✗ ERROR: {e}")
                performance_matrix[agent_scenario][test_scenario] = None

    print()
    print("=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print()

    # Save results
    output_data = {
        "evolved_version": evolved_version,
        "num_simulations": num_simulations,
        "seed": seed,
        "scenarios": SCENARIOS,
        "performance_matrix": performance_matrix,
        "detailed_results": all_results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Print summary table
    print("Performance Matrix Summary:")
    print()
    print("Agent \\ Test    ", end="")
    for test_scenario in SCENARIOS:
        # Abbreviate scenario names
        abbrev = "".join([word[0] for word in test_scenario.split("_")]).upper()
        print(f"{abbrev:>6}", end="")
    print()
    print("-" * 80)

    for agent_scenario in SCENARIOS:
        abbrev = "".join([word[0] for word in agent_scenario.split("_")]).upper()
        print(f"{abbrev:16}", end="")

        for test_scenario in SCENARIOS:
            payoff = performance_matrix[agent_scenario].get(test_scenario)
            if payoff is not None:
                # Highlight diagonal (on-scenario performance)
                if agent_scenario == test_scenario:
                    print(f"\033[1m{payoff:6.1f}\033[0m", end="")
                else:
                    print(f"{payoff:6.1f}", end="")
            else:
                print(f"{'N/A':>6}", end="")
        print()

    print()

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate evolved agents across different scenarios"
    )

    parser.add_argument(
        "--agent-scenario",
        type=str,
        help="Scenario the agent was evolved in",
    )

    parser.add_argument(
        "--test-scenario",
        type=str,
        help="Scenario to test the agent in",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 81 combinations (9 agents × 9 scenarios)",
    )

    parser.add_argument(
        "--evolved-version",
        type=str,
        default="v4",
        choices=["v3", "v4", "v5"],
        help="Evolution version to use (default: v4)",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=2000,
        help="Number of Monte Carlo simulations (default: 2000)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (default: experiments/generalization/performance_matrix.json)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    if args.all:
        # Batch mode: evaluate all combinations
        evaluate_all_combinations(
            evolved_version=args.evolved_version,
            num_simulations=args.simulations,
            seed=args.seed,
            output_file=args.output,
            verbose=args.verbose,
        )

    elif args.agent_scenario and args.test_scenario:
        # Single evaluation mode
        result = evaluate_cross_scenario(
            agent_scenario=args.agent_scenario,
            test_scenario=args.test_scenario,
            evolved_version=args.evolved_version,
            num_simulations=args.simulations,
            seed=args.seed,
            verbose=True,
        )

        # Save single result
        output_file = args.output or Path(
            f"experiments/generalization/{args.agent_scenario}_to_{args.test_scenario}.json"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to: {output_file}")

    else:
        parser.print_help()
        print()
        print("Error: Either --all or both --agent-scenario and --test-scenario required")
        sys.exit(1)


if __name__ == "__main__":
    main()
