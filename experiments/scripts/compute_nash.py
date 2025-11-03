#!/usr/bin/env python3
"""
Compute Nash equilibrium for a scenario using Double Oracle algorithm.

Usage:
    python experiments/scripts/compute_nash.py greedy_neighbor
    python experiments/scripts/compute_nash.py greedy_neighbor --simulations 500
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.equilibrium import DoubleOracle
from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
    LIAR_PARAMS,
)


def classify_strategy(strategy: np.ndarray) -> str:
    """Classify a strategy into an archetype category."""
    # Extract key parameters
    work_tendency = strategy[1]
    altruism = strategy[9]
    own_priority = strategy[3]
    coordination = strategy[5]
    honesty = strategy[0]

    # Classification logic
    if work_tendency > 0.7 and altruism > 0.6:
        return "Firefighter/Hero"
    elif work_tendency < 0.4 and own_priority > 0.7:
        return "Free Rider"
    elif coordination > 0.8:
        return "Coordinator"
    elif honesty < 0.3:
        return "Liar/Deceiver"
    else:
        return "Mixed Strategy"


def strategy_distance_to_archetypes(strategy: np.ndarray) -> dict:
    """Compute distance to known archetypes."""
    archetypes = {
        "Firefighter": FIREFIGHTER_PARAMS,
        "Free Rider": FREE_RIDER_PARAMS,
        "Hero": HERO_PARAMS,
        "Coordinator": COORDINATOR_PARAMS,
        "Liar": LIAR_PARAMS,
    }

    distances = {}
    for name, archetype in archetypes.items():
        distance = float(np.linalg.norm(strategy - archetype))
        distances[name] = distance

    return distances


def compute_nash_equilibrium(
    scenario_name: str,
    output_dir: Path,
    num_simulations: int = 1000,
    max_iterations: int = 50,
    epsilon: float = 0.01,
    seed: Optional[int] = 42,
    verbose: bool = True,
):
    """Compute Nash equilibrium using Double Oracle algorithm."""

    print(f"Computing Nash equilibrium for scenario: {scenario_name}")
    print(f"Output directory: {output_dir}")
    print()

    # Load scenario
    scenario = get_scenario_by_name(scenario_name, num_agents=4)

    print("Scenario Parameters:")
    print(f"  beta (spread):       {scenario.beta:.2f}")
    print(f"  kappa (extinguish):  {scenario.kappa:.2f}")
    print(f"  c (work cost):       {scenario.c:.2f}")
    print(f"  num_agents:          {scenario.num_agents}")
    print()

    print("Double Oracle Configuration:")
    print(f"  Simulations per evaluation: {num_simulations}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Convergence threshold (ε): {epsilon}")
    print(f"  Seed: {seed}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Double Oracle solver
    print("=" * 80)
    print("Running Double Oracle Algorithm")
    print("=" * 80)
    print()

    solver = DoubleOracle(
        scenario=scenario,
        num_simulations=num_simulations,
        max_iterations=max_iterations,
        epsilon=epsilon,
        seed=seed,
        verbose=verbose,
    )

    # Solve for Nash equilibrium
    start_time = time.time()
    equilibrium = solver.solve()
    elapsed_time = time.time() - start_time

    print()
    print("=" * 80)
    print("Nash Equilibrium Results")
    print("=" * 80)
    print()
    print(
        f"Convergence Status: {'CONVERGED' if equilibrium.converged else 'MAX ITERATIONS'}"
    )
    print(f"Iterations: {equilibrium.iterations}")
    print(f"Expected Payoff: {equilibrium.payoff:.2f}")
    print(f"Support Size: {len(equilibrium.distribution)}")
    print(f"Time elapsed: {elapsed_time:.1f}s")
    print()

    # Analyze equilibrium strategies
    print("=" * 80)
    print("Equilibrium Strategy Distribution")
    print("=" * 80)
    print()

    sorted_strategies = sorted(
        equilibrium.distribution.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    strategy_details = []
    param_names = [
        "honesty",
        "work_tendency",
        "neighbor_help",
        "own_priority",
        "risk_aversion",
        "coordination",
        "exploration",
        "fatigue_memory",
        "rest_bias",
        "altruism",
    ]

    for idx, (strategy_idx, probability) in enumerate(sorted_strategies):
        strategy = equilibrium.strategy_pool[strategy_idx]
        classification = classify_strategy(strategy)
        distances = strategy_distance_to_archetypes(strategy)
        closest = min(distances, key=distances.get)

        print(f"Strategy {idx + 1}: Probability = {probability:.3f}")
        print(f"  Classification: {classification}")
        print(f"  Closest archetype: {closest} (distance: {distances[closest]:.3f})")
        print("  Parameters:")

        for name, value in zip(param_names, strategy):
            bar_length = int(value * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"    {name:15s}: {bar} {value:.3f}")

        print()

        strategy_details.append(
            {
                "index": int(strategy_idx),
                "probability": float(probability),
                "classification": classification,
                "closest_archetype": closest,
                "archetype_distance": float(distances[closest]),
                "parameters": {
                    name: float(value) for name, value in zip(param_names, strategy)
                },
                "genome": strategy.tolist(),
            }
        )

    # Game-theoretic interpretation
    print("=" * 80)
    print("Game-Theoretic Interpretation")
    print("=" * 80)
    print()

    if len(equilibrium.distribution) == 1:
        print("Pure Strategy Equilibrium:")
        print("  All agents play the same deterministic strategy.")
        print("  This indicates a dominant strategy exists for this scenario.")
        equilibrium_type = "pure"
    else:
        print("Mixed Strategy Equilibrium:")
        print(f"  Agents randomize over {len(equilibrium.distribution)} strategies.")
        print("  This indicates strategic tension - no single strategy dominates.")
        print()
        print("  Mixing Indifference Condition:")
        print("    All strategies in the support earn equal expected payoff.")
        print("    This balances the incentives between different behaviors.")
        equilibrium_type = "mixed"

    print()

    # Scenario-specific insights
    cooperative_prob = 0.0
    free_riding_prob = 0.0
    for strategy_idx, prob in equilibrium.distribution.items():
        strategy = equilibrium.strategy_pool[strategy_idx]
        if strategy[1] > 0.5:  # work_tendency > 0.5
            cooperative_prob += prob
        else:
            free_riding_prob += prob

    print(f"Cooperative behavior: {cooperative_prob:.1%}")
    print(f"Free-riding behavior: {free_riding_prob:.1%}")

    # Prepare output data
    results = {
        "scenario": scenario_name,
        "parameters": {
            "beta": scenario.beta,
            "kappa": scenario.kappa,
            "c": scenario.c,
            "A": scenario.A,
            "L": scenario.L,
            "rho_ignite": scenario.rho_ignite,
            "num_agents": scenario.num_agents,
        },
        "algorithm": {
            "method": "double_oracle",
            "num_simulations": num_simulations,
            "max_iterations": max_iterations,
            "epsilon": epsilon,
            "seed": seed,
        },
        "equilibrium": {
            "type": equilibrium_type,
            "support_size": len(equilibrium.distribution),
            "expected_payoff": float(equilibrium.payoff),
            "distribution": {
                int(idx): float(prob) for idx, prob in equilibrium.distribution.items()
            },
            "strategy_pool": strategy_details,
        },
        "convergence": {
            "converged": equilibrium.converged,
            "iterations": equilibrium.iterations,
            "elapsed_time": elapsed_time,
        },
        "interpretation": {
            "equilibrium_type": equilibrium_type,
            "cooperation_rate": float(cooperative_prob),
            "free_riding_rate": float(free_riding_prob),
        },
    }

    # Save results
    equilibrium_file = output_dir / "equilibrium.json"
    with open(equilibrium_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"✅ Results saved to: {equilibrium_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute Nash equilibrium")
    parser.add_argument("scenario", type=str, help="Scenario name")
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Simulations per payoff evaluation",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum Double Oracle iterations",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01, help="Convergence threshold"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"experiments/scenarios/{args.scenario}/nash")

    compute_nash_equilibrium(
        args.scenario,
        args.output_dir,
        num_simulations=args.simulations,
        max_iterations=args.max_iterations,
        epsilon=args.epsilon,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
