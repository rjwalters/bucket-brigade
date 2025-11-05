#!/usr/bin/env python3
"""
Compute Nash equilibrium V2 - includes evolved agents in strategy pool.

This is the V2 Nash analysis that integrates evolved strategies from the
genetic algorithm experiments (V3/V4/V5) into the Double Oracle algorithm.

Usage:
    python experiments/scripts/compute_nash_v2.py chain_reaction
    python experiments/scripts/compute_nash_v2.py chain_reaction --simulations 1000
    python experiments/scripts/compute_nash_v2.py chain_reaction --evolved-versions v4 v5
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.equilibrium import (
    DoubleOracle,
    load_all_evolved_agents,
    load_evolved_agent_metadata,
    get_evolved_agent_description,
)
from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
    LIAR_PARAMS,
)

# Import classification utilities from original script
from compute_nash import (
    classify_strategy,
    strategy_distance_to_archetypes,
)


def compute_nash_v2(
    scenario_name: str,
    output_dir: Path,
    num_simulations: int = 1000,
    max_iterations: int = 50,
    epsilon: float = 0.01,
    seed: Optional[int] = 42,
    evolved_versions: List[str] = ["v4"],
    verbose: bool = True,
):
    """
    Compute Nash equilibrium V2 with evolved agents included.

    Args:
        scenario_name: Scenario to analyze
        output_dir: Where to save results
        num_simulations: Simulations per payoff evaluation
        max_iterations: Maximum Double Oracle iterations
        epsilon: Convergence threshold
        seed: Random seed
        evolved_versions: Which evolution versions to include (e.g., ["v3", "v4", "v5"])
        verbose: Print progress
    """

    print("=" * 80)
    print("Nash Equilibrium V2 - With Evolved Agents")
    print("=" * 80)
    print()
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

    # Load evolved agents
    print(f"Loading evolved agents (versions: {', '.join(evolved_versions)})...")
    evolved_agents = load_all_evolved_agents(scenario_name, versions=evolved_versions)

    if not evolved_agents:
        print("  ⚠️  No evolved agents found!")
        print(
            f"  Looked for: experiments/scenarios/{scenario_name}/evolved_{{v3,v4,v5}}/best_agent.json"
        )
        print()
    else:
        print(f"  ✓ Loaded {len(evolved_agents)} evolved agent(s)")
        for i, version in enumerate(evolved_versions):
            desc = get_evolved_agent_description(scenario_name, version)
            print(f"    {i + 1}. {desc}")
        print()

    # Build initial strategy pool: archetypes + evolved agents
    initial_strategies = [
        FIREFIGHTER_PARAMS.copy(),
        FREE_RIDER_PARAMS.copy(),
        HERO_PARAMS.copy(),
        COORDINATOR_PARAMS.copy(),
        LIAR_PARAMS.copy(),
    ] + evolved_agents

    print(f"Initial Strategy Pool: {len(initial_strategies)} strategies")
    print(
        "  - 5 predefined archetypes (Firefighter, Free Rider, Hero, Coordinator, Liar)"
    )
    print(f"  - {len(evolved_agents)} evolved agent(s)")
    print()

    print("Double Oracle Configuration:")
    print(f"  Simulations per evaluation: {num_simulations}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Convergence threshold (ε): {epsilon}")
    print(f"  Seed: {seed}")
    print("  Evaluator: RustPayoffEvaluator (100x speedup)")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Double Oracle
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

    start_time = time.time()
    equilibrium = solver.solve(initial_strategies=initial_strategies)
    elapsed_time = time.time() - start_time

    print()
    print("=" * 80)
    print("Nash Equilibrium V2 Results")
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

        # Check if this is an evolved strategy
        is_evolved = False
        evolved_info = None
        if strategy_idx >= 5:  # After the 5 archetypes
            evolved_idx = strategy_idx - 5
            if evolved_idx < len(evolved_agents):
                # Check if it matches an evolved agent
                if np.allclose(strategy, evolved_agents[evolved_idx], atol=1e-6):
                    is_evolved = True
                    version = (
                        evolved_versions[evolved_idx]
                        if evolved_idx < len(evolved_versions)
                        else "unknown"
                    )
                    evolved_info = load_evolved_agent_metadata(scenario_name, version)

        print(f"Strategy {idx + 1}: Probability = {probability:.3f}")
        if is_evolved and evolved_info:
            print(f"  🧬 EVOLVED AGENT ({evolved_info['version'].upper()})")
            print(f"     Evolution fitness: {evolved_info['fitness']:.2f}")
            print(f"     Generation: {evolved_info['generation']}")
        else:
            print(f"  Classification: {classification}")
            print(
                f"  Closest archetype: {closest} (distance: {distances[closest]:.3f})"
            )
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
                "is_evolved": is_evolved,
                "evolved_info": evolved_info if is_evolved else None,
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
        equilibrium_type = "pure"
    else:
        print("Mixed Strategy Equilibrium:")
        print(f"  Agents randomize over {len(equilibrium.distribution)} strategies.")
        equilibrium_type = "mixed"

    print()

    # Check if evolved agents are in equilibrium
    evolved_in_equilibrium = [s for s in strategy_details if s["is_evolved"]]
    if evolved_in_equilibrium:
        print("🎯 Evolved Agent(s) in Equilibrium:")
        for s in evolved_in_equilibrium:
            info = s["evolved_info"]
            print(f"  ✓ {info['version'].upper()}: {s['probability']:.1%} probability")
            print(f"    Evolution fitness: {info['fitness']:.2f}")
            print(f"    Nash payoff: {equilibrium.payoff:.2f}")
            gap = abs(info["fitness"] - equilibrium.payoff)
            gap_pct = (gap / equilibrium.payoff) * 100
            print(f"    Gap: {gap:.2f} ({gap_pct:.1f}%)")
        print()
    else:
        print("📊 No evolved agents in equilibrium support")
        print(
            "   This suggests evolved strategies were exploitable by archetypes or best responses"
        )
        print()

    # Cooperation analysis
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
    print()

    # Prepare output data
    results = {
        "scenario": scenario_name,
        "version": "v2",
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
            "method": "double_oracle_v2",
            "num_simulations": num_simulations,
            "max_iterations": max_iterations,
            "epsilon": epsilon,
            "seed": seed,
            "evaluator": "RustPayoffEvaluator",
        },
        "initial_pool": {
            "archetypes": 5,
            "evolved_agents": len(evolved_agents),
            "evolved_versions": evolved_versions,
            "total": len(initial_strategies),
        },
        "equilibrium": {
            "type": equilibrium_type,
            "support_size": len(equilibrium.distribution),
            "expected_payoff": float(equilibrium.payoff),
            "distribution": {
                int(idx): float(prob) for idx, prob in equilibrium.distribution.items()
            },
            "strategy_pool": strategy_details,
            "evolved_in_equilibrium": len(evolved_in_equilibrium),
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
    equilibrium_file = output_dir / "equilibrium_v2.json"
    with open(equilibrium_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"✅ V2 Results saved to: {equilibrium_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute Nash equilibrium V2 with evolved agents"
    )
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
        "--evolved-versions",
        type=str,
        nargs="+",
        default=["v4"],
        help="Evolution versions to include (e.g., v3 v4 v5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"experiments/nash/v2_results/{args.scenario}")

    compute_nash_v2(
        args.scenario,
        args.output_dir,
        num_simulations=args.simulations,
        max_iterations=args.max_iterations,
        epsilon=args.epsilon,
        seed=args.seed,
        evolved_versions=args.evolved_versions,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
