#!/usr/bin/env python3
"""
Compare results across heuristics, evolution, and Nash equilibrium analysis.

Usage:
    python experiments/scripts/run_comparison.py greedy_neighbor
    python experiments/scripts/run_comparison.py greedy_neighbor --evolution-versions evolved evolved_v3
    python experiments/scripts/run_comparison.py greedy_neighbor --evolution-versions all
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.evolution.fitness_rust import (
    _heuristic_action,
    _convert_scenario_to_rust,
)
import bucket_brigade_core as core


def find_evolution_versions(scenario_dir: Path) -> List[str]:
    """Find all available evolution result directories."""
    versions = []
    for path in scenario_dir.iterdir():
        if path.is_dir() and (
            path.name == "evolved" or path.name.startswith("evolved_")
        ):
            if (path / "best_agent.json").exists():
                versions.append(path.name)
    return sorted(versions)


def load_results(
    scenario_dir: Path, evolution_versions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Load all analysis results for a scenario.

    Args:
        scenario_dir: Path to scenario directory
        evolution_versions: List of evolution directories to load (e.g., ['evolved', 'evolved_v3'])
                          If None, loads default 'evolved' directory
                          If ['all'], loads all available evolution versions
    """
    results = {}

    # Load heuristics
    heuristics_file = scenario_dir / "heuristics" / "results.json"
    if heuristics_file.exists():
        with open(heuristics_file) as f:
            results["heuristics"] = json.load(f)
    else:
        results["heuristics"] = None

    # Determine which evolution versions to load
    if evolution_versions is None:
        evolution_versions = ["evolved"]  # Default
    elif evolution_versions == ["all"]:
        evolution_versions = find_evolution_versions(scenario_dir)
        print(f"  Auto-detected evolution versions: {', '.join(evolution_versions)}")

    # Load evolution results
    results["evolution"] = {}
    for version in evolution_versions:
        evolved_file = scenario_dir / version / "best_agent.json"
        if evolved_file.exists():
            with open(evolved_file) as f:
                results["evolution"][version] = json.load(f)
        else:
            print(f"  ⚠️  Evolution version '{version}' not found, skipping...")

    # Load Nash
    nash_file = scenario_dir / "nash" / "equilibrium.json"
    if nash_file.exists():
        with open(nash_file) as f:
            results["nash"] = json.load(f)
    else:
        results["nash"] = None

    return results


def compute_strategy_distance(strategy1: np.ndarray, strategy2: np.ndarray) -> float:
    """Compute Euclidean distance between two strategies."""
    return float(np.linalg.norm(strategy1 - strategy2))


def run_tournament(
    strategies: Dict[str, np.ndarray], scenario_name: str, num_games: int = 20
) -> Dict[str, Dict[str, Any]]:
    """Run head-to-head tournament between strategies using Rust environment.

    This ensures train/test consistency - both use the same Rust implementation.
    """

    print(f"Running tournament with {len(strategies)} strategies...")
    print(f"  Games per matchup: {num_games}")
    print("  Using: Rust environment (single source of truth)")
    print()

    # Get scenario and convert to Rust
    python_scenario = get_scenario_by_name(scenario_name, num_agents=4)
    rust_scenario = _convert_scenario_to_rust(python_scenario)
    num_agents = python_scenario.num_agents

    results = {}

    for name, genome in strategies.items():
        print(f"  Testing {name}...", end=" ", flush=True)

        # Run games using Rust environment
        payoffs = []
        for game_idx in range(num_games):
            # Create Rust game
            game = core.BucketBrigade(rust_scenario, seed=game_idx)

            # Python RNG for heuristic decisions
            rng = np.random.RandomState(game_idx)

            # Run until done
            done = False
            step_count = 0
            max_steps = 100

            while not done and step_count < max_steps:
                # Get actions for all 4 agents using same genome
                actions = []
                for agent_id in range(num_agents):
                    obs = game.get_observation(agent_id)
                    obs_dict = {
                        "houses": obs.houses,
                        "signals": obs.signals,
                        "locations": obs.locations,
                    }
                    action = _heuristic_action(genome, obs_dict, agent_id, rng)
                    actions.append(action)

                # Step the game
                rewards, done, info = game.step(actions)
                step_count += 1

            # Get final result
            result = game.get_result()
            # Use mean of agent scores for consistency with training
            payoffs.append(result.final_score / num_agents)

        mean_payoff = np.mean(payoffs)
        std_payoff = np.std(payoffs)

        results[name] = {
            "mean_payoff": float(mean_payoff),
            "std_payoff": float(std_payoff),
            "payoffs": payoffs,
        }

        print(f"Mean: {mean_payoff:.2f} ± {std_payoff:.2f} ✓")

    print()
    return results


def run_comparison(
    scenario_name: str,
    output_dir: Path,
    num_games: int = 20,
    evolution_versions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare all analysis methods for a scenario.

    Args:
        scenario_name: Name of the scenario
        output_dir: Directory to save results
        num_games: Number of games per strategy
        evolution_versions: List of evolution versions to compare (e.g., ['evolved', 'evolved_v3'])
                          If None, uses default 'evolved'
                          If ['all'], compares all available versions
    """

    print(f"Running comparison analysis for scenario: {scenario_name}")
    print(f"Output directory: {output_dir}")
    print()

    # Scenario directory
    scenario_dir = Path(f"experiments/scenarios/{scenario_name}")

    # Load all results
    print("Loading analysis results...")
    all_results = load_results(scenario_dir, evolution_versions)

    if all_results["heuristics"] is None:
        print("⚠️  No heuristics results found. Run analyze_heuristics.py first.")
        return {}

    if not all_results["evolution"]:
        print("⚠️  No evolution results found. Run run_evolution.py first.")
        return {}

    if all_results["nash"] is None:
        print("⚠️  No Nash results found. Run compute_nash.py first.")
        print("    Continuing with available results...")

    print()

    # Prepare strategies for comparison
    strategies = {}

    # Add best heuristics
    heuristic_results = all_results["heuristics"]["homogeneous_teams"]
    best_heuristic = max(heuristic_results, key=lambda x: x["mean_payoff"])
    strategies["best_heuristic"] = np.array(
        next(
            agent["params"]
            for agent in all_results["heuristics"]["agents"]
            if agent["name"] == best_heuristic["agent_type"]
        )
    )

    # Add evolved agents (all versions)
    for version_name, version_data in all_results["evolution"].items():
        strategies[version_name] = np.array(version_data["genome"])

    # Add Nash strategies (if available)
    if all_results["nash"] is not None:
        nash_strategies = all_results["nash"]["equilibrium"]["strategy_pool"]
        for i, strat in enumerate(nash_strategies):
            strategies[f"nash_strategy_{i + 1}"] = np.array(strat["genome"])

    print("=" * 80)
    print("Strategies in Tournament")
    print("=" * 80)
    for name in strategies.keys():
        print(f"  - {name}")
    print()

    # Run tournament
    print("=" * 80)
    print("Head-to-Head Tournament")
    print("=" * 80)
    print()

    tournament_results = run_tournament(strategies, scenario_name, num_games)

    # Compute strategy distances
    print("=" * 80)
    print("Strategy Distance Matrix")
    print("=" * 80)
    print()

    strategy_names = list(strategies.keys())
    distances = {}

    for i, name1 in enumerate(strategy_names):
        for j, name2 in enumerate(strategy_names):
            if i < j:  # Only compute upper triangle
                dist = compute_strategy_distance(strategies[name1], strategies[name2])
                distances[f"{name1}_vs_{name2}"] = dist
                print(f"  {name1:20s} vs {name2:20s}: {dist:.3f}")

    print()

    # Analysis and insights
    print("=" * 80)
    print("Analysis and Insights")
    print("=" * 80)
    print()

    # Rank by performance
    ranked = sorted(
        tournament_results.items(),
        key=lambda x: float(x[1]["mean_payoff"]),
        reverse=True,  # type: ignore[arg-type,index]
    )

    print("Performance Ranking:")
    for i, (name, result) in enumerate(ranked):
        print(
            f"  {i + 1}. {name:20s}: {float(result['mean_payoff']):.2f} ± {float(result['std_payoff']):.2f}"  # type: ignore[arg-type]
        )

    print()

    # Evolved vs Nash (for each evolution version)
    evolution_vs_nash_insights = {}
    if all_results["nash"] is not None:
        nash_payoffs = [
            float(result["mean_payoff"])  # type: ignore[arg-type]
            for name, result in tournament_results.items()
            if name.startswith("nash_strategy_")
        ]

        if nash_payoffs:
            best_nash_payoff = max(nash_payoffs)

            print("Evolution vs Nash Comparison:")
            for version_name in all_results["evolution"].keys():
                if version_name in tournament_results:
                    evolved_payoff = float(
                        tournament_results[version_name]["mean_payoff"]
                    )  # type: ignore[arg-type]
                    gap = evolved_payoff - best_nash_payoff

                    print(f"  {version_name}:")
                    print(f"    Payoff: {evolved_payoff:.2f}")
                    print(f"    Gap from Nash: {gap:+.2f}")

                    if abs(gap) < 5.0:
                        print("    → Found near-Nash strategy!")
                    elif gap > 0:
                        print("    → Outperformed Nash")
                    else:
                        print("    → Nash outperformed this version")

                    evolution_vs_nash_insights[version_name] = {
                        "evolved_payoff": evolved_payoff,
                        "best_nash_payoff": best_nash_payoff,
                        "gap": gap,
                    }

            print()

    # Strategy similarity for each evolved version
    if all_results["nash"] is not None and len(nash_strategies) > 0:
        print("Strategy Similarity to Nash:")
        for version_name in all_results["evolution"].keys():
            closest_nash = None
            min_distance = float("inf")

            for name, dist in distances.items():
                if name.startswith(f"{version_name}_vs_nash_"):
                    if dist < min_distance:
                        min_distance = dist
                        closest_nash = name.replace(f"{version_name}_vs_", "")

            if closest_nash:
                print(
                    f"  {version_name} is closest to: {closest_nash} (distance: {min_distance:.3f})"
                )

        print()

    # Prepare output
    comparison_results = {
        "scenario": scenario_name,
        "strategies": {name: genome.tolist() for name, genome in strategies.items()},
        "tournament": tournament_results,
        "distances": distances,
        "ranking": [
            {
                "name": name,
                "mean_payoff": result["mean_payoff"],  # type: ignore[typeddict-item]
                "std_payoff": result["std_payoff"],  # type: ignore[typeddict-item]
            }
            for name, result in ranked
        ],
        "insights": {},
    }

    # Add insights
    if evolution_vs_nash_insights:
        comparison_results["insights"]["evolution_vs_nash"] = evolution_vs_nash_insights  # type: ignore[index]

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "comparison.json"

    with open(output_file, "w") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"✅ Results saved to: {output_file}")
    print()

    return comparison_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare analysis methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare default 'evolved' version
  python run_comparison.py greedy_neighbor

  # Compare specific versions
  python run_comparison.py greedy_neighbor --evolution-versions evolved evolved_v3

  # Compare all available versions
  python run_comparison.py greedy_neighbor --evolution-versions all
        """,
    )
    parser.add_argument("scenario", type=str, help="Scenario name")
    parser.add_argument("--num-games", type=int, default=20, help="Games per strategy")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument(
        "--evolution-versions",
        type=str,
        nargs="+",
        default=None,
        help="Evolution versions to compare (e.g., evolved evolved_v3) or 'all' for all versions",
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"experiments/scenarios/{args.scenario}/comparison")

    run_comparison(
        args.scenario, args.output_dir, args.num_games, args.evolution_versions
    )


if __name__ == "__main__":
    main()
