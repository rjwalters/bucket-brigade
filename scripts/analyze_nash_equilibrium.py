#!/usr/bin/env python3
"""
Nash Equilibrium Analysis Script

Analyzes Nash equilibrium for a specified scenario using the Double Oracle algorithm.
Computes equilibrium strategies, evaluates payoffs, and provides game-theoretic insights.

Usage:
    python scripts/analyze_nash_equilibrium.py --scenario greedy_neighbor --num-agents 4
"""

import argparse
import numpy as np
from multiprocessing import cpu_count
from bucket_brigade.envs.scenarios import get_scenario_by_name, list_scenarios
from bucket_brigade.equilibrium import DoubleOracle
from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
    LIAR_PARAMS,
)


def print_header(text: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_scenario_info(scenario):
    """Print detailed scenario information."""
    print("Scenario Parameters:")
    print(f"  Fire Dynamics:")
    print(f"    beta (spread probability):     {scenario.beta:.2f}")
    print(f"    kappa (extinguish efficiency): {scenario.kappa:.2f}")
    print(f"  Reward Structure:")
    print(f"    A (reward per saved house):    {scenario.A:.2f}")
    print(f"    L (penalty per ruined house):  {scenario.L:.2f}")
    print(f"    c (cost per worker per night): {scenario.c:.2f}")
    print(f"  Initial Conditions:")
    print(f"    rho_ignite (initial burn rate): {scenario.rho_ignite:.2f}")
    print(f"    p_spark (spontaneous ignition): {scenario.p_spark:.2f}")
    print(f"    N_min (minimum nights):         {scenario.N_min}")
    print(f"    num_agents:                     {scenario.num_agents}")


def analyze_strategy(strategy: np.ndarray) -> dict[str, float]:
    """
    Analyze and categorize a strategy based on its parameters.

    Returns dictionary with behavioral characteristics.
    """
    return {
        "honesty": strategy[0],
        "work_tendency": strategy[1],
        "neighbor_help": strategy[2],
        "own_house_priority": strategy[3],
        "risk_aversion": strategy[4],
        "coordination": strategy[5],
        "exploration": strategy[6],
        "fatigue_memory": strategy[7],
        "rest_reward": strategy[8],
        "altruism": strategy[9],
    }


def classify_strategy(strategy: np.ndarray) -> str:
    """
    Classify a strategy into an archetype category.

    Returns human-readable label.
    """
    analysis = analyze_strategy(strategy)

    # High work, high cooperation
    if analysis["work_tendency"] > 0.7 and analysis["altruism"] > 0.6:
        if analysis["work_tendency"] > 0.9:
            return "Hero"
        else:
            return "Firefighter"

    # Low work, high self-interest
    if analysis["work_tendency"] < 0.4 and analysis["own_house_priority"] > 0.7:
        return "Free Rider"

    # High coordination
    if analysis["coordination"] > 0.8:
        return "Coordinator"

    # Low honesty
    if analysis["honesty"] < 0.3:
        return "Liar/Deceiver"

    # Mixed/Balanced
    return "Mixed Strategy"


def print_strategy_details(strategy: np.ndarray, label: str = "Strategy"):
    """Print detailed breakdown of a strategy."""
    analysis = analyze_strategy(strategy)
    classification = classify_strategy(strategy)

    print(f"\n{label}:")
    print(f"  Classification: {classification}")
    print(f"  Parameters:")
    for param_name, value in analysis.items():
        bar_length = int(value * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"    {param_name:20s}: {bar} {value:.3f}")


def compare_to_archetypes(strategy: np.ndarray):
    """Compare a strategy to known archetypes."""
    archetypes = {
        "Firefighter": FIREFIGHTER_PARAMS,
        "Free Rider": FREE_RIDER_PARAMS,
        "Hero": HERO_PARAMS,
        "Coordinator": COORDINATOR_PARAMS,
        "Liar": LIAR_PARAMS,
    }

    print("\n  Distance to Archetypes:")
    distances = {}
    for name, archetype in archetypes.items():
        distance = np.linalg.norm(strategy - archetype)
        distances[name] = distance
        print(f"    {name:15s}: {distance:.3f}")

    closest = min(distances, key=distances.get)
    print(f"  Closest to: {closest}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Nash equilibrium for Bucket Brigade scenarios"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="greedy_neighbor",
        help=f"Scenario name (options: {', '.join(list_scenarios())})",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Number of agents in the game (default: 4)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=1000,
        help="Monte Carlo simulations per payoff evaluation (default: 1000)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum Double Oracle iterations (default: 20)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Convergence threshold for improvement (default: 0.01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    # Print banner
    print_header("Nash Equilibrium Analysis for Bucket Brigade")

    # Load scenario
    print(f"Loading scenario: {args.scenario}")
    try:
        scenario = get_scenario_by_name(args.scenario, args.num_agents)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available scenarios: {', '.join(list_scenarios())}")
        return

    print_scenario_info(scenario)

    # Initialize Double Oracle solver
    print_header("Running Double Oracle Algorithm")

    num_cpus = cpu_count()
    print(f"System CPUs: {num_cpus} (parallelization enabled)")
    print(f"Simulations per evaluation: {args.num_simulations}")
    print()

    solver = DoubleOracle(
        scenario=scenario,
        num_simulations=args.num_simulations,
        max_iterations=args.max_iterations,
        epsilon=args.epsilon,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Solve for Nash equilibrium
    print("Computing Nash equilibrium...")
    print("This computation uses parallel processing across all available CPU cores.")
    equilibrium = solver.solve()

    # Print results
    print_header("Nash Equilibrium Results")

    print(f"Convergence Status: {'CONVERGED' if equilibrium.converged else 'MAX ITERATIONS'}")
    print(f"Iterations: {equilibrium.iterations}")
    print(f"Expected Payoff: {equilibrium.payoff:.2f}")
    print(f"Support Size: {len(equilibrium.distribution)}")

    # Analyze equilibrium strategies
    print_header("Equilibrium Strategy Distribution")

    sorted_strategies = sorted(
        equilibrium.distribution.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for idx, (strategy_idx, probability) in enumerate(sorted_strategies):
        strategy = equilibrium.strategy_pool[strategy_idx]
        print(f"\nStrategy {idx + 1}: Probability = {probability:.3f}")
        print_strategy_details(strategy, label=f"  Details")
        compare_to_archetypes(strategy)

    # Game-theoretic interpretation
    print_header("Game-Theoretic Interpretation")

    if len(equilibrium.distribution) == 1:
        print("Pure Strategy Equilibrium:")
        print("  All agents play the same deterministic strategy.")
        print("  This indicates a dominant strategy exists for this scenario.")
    else:
        print("Mixed Strategy Equilibrium:")
        print(f"  Agents randomize over {len(equilibrium.distribution)} strategies.")
        print("  This indicates strategic tension - no single strategy dominates.")
        print()
        print("  Mixing Indifference Condition:")
        print("    All strategies in the support earn equal expected payoff.")
        print("    This balances the incentives between different behaviors.")

    # Scenario-specific insights
    print_header("Scenario-Specific Insights")

    if args.scenario == "greedy_neighbor":
        print("Greedy Neighbor Analysis:")
        print("  High work cost (c=1.0) creates free-riding incentive.")
        print("  Expected: Mixed equilibrium with cooperators and free riders.")
        print()
        # Count cooperators vs free riders
        cooperative_prob = 0.0
        free_riding_prob = 0.0
        for strategy_idx, prob in equilibrium.distribution.items():
            strategy = equilibrium.strategy_pool[strategy_idx]
            if strategy[1] > 0.5:  # work_tendency > 0.5
                cooperative_prob += prob
            else:
                free_riding_prob += prob
        print(f"  Cooperative behavior: {cooperative_prob:.1%}")
        print(f"  Free-riding behavior: {free_riding_prob:.1%}")

    elif args.scenario == "trivial_cooperation":
        print("Trivial Cooperation Analysis:")
        print("  Low spread + high extinguish rate makes cooperation easy.")
        print("  Expected: Pure cooperative equilibrium.")
        print("  Dominant strategy: Work hard and cooperate.")

    elif args.scenario == "deceptive_calm":
        print("Deceptive Calm Analysis:")
        print("  Spontaneous fires reward honest signaling and coordination.")
        print("  Expected: High honesty + high coordination in equilibrium.")
        print()
        avg_honesty = sum(
            equilibrium.strategy_pool[idx][0] * prob
            for idx, prob in equilibrium.distribution.items()
        )
        avg_coordination = sum(
            equilibrium.strategy_pool[idx][5] * prob
            for idx, prob in equilibrium.distribution.items()
        )
        print(f"  Average honesty: {avg_honesty:.3f}")
        print(f"  Average coordination: {avg_coordination:.3f}")

    # Summary
    print_header("Summary")
    print(f"Scenario: {args.scenario}")
    print(f"Nash Equilibrium Type: {'Pure' if len(equilibrium.distribution) == 1 else 'Mixed'}")
    print(f"Expected Payoff: {equilibrium.payoff:.2f}")
    print(f"Convergence: {equilibrium.iterations} iterations")
    print()
    print("The equilibrium represents a stable strategic configuration where")
    print("no agent can improve their expected payoff by unilaterally changing strategy.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
