#!/usr/bin/env python3
"""
Compare results across heuristics, evolution, and Nash equilibrium analysis.

Usage:
    python experiments/scripts/run_comparison.py greedy_neighbor
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.agents.heuristic_agent import HeuristicAgent


def load_results(scenario_dir: Path) -> Dict[str, Any]:
    """Load all analysis results for a scenario."""
    results = {}

    # Load heuristics
    heuristics_file = scenario_dir / "heuristics" / "results.json"
    if heuristics_file.exists():
        with open(heuristics_file) as f:
            results["heuristics"] = json.load(f)
    else:
        results["heuristics"] = None

    # Load evolution
    evolved_file = scenario_dir / "evolved" / "best_agent.json"
    if evolved_file.exists():
        with open(evolved_file) as f:
            results["evolved"] = json.load(f)
    else:
        results["evolved"] = None

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
    strategies: Dict[str, np.ndarray], scenario: Any, num_games: int = 20
) -> Dict[str, Dict[str, Any]]:
    """Run head-to-head tournament between strategies."""

    print(f"Running tournament with {len(strategies)} strategies...")
    print(f"  Games per matchup: {num_games}")
    print()

    results = {}

    for name, genome in strategies.items():
        print(f"  Testing {name}...", end=" ", flush=True)

        # Create team of 4 identical agents
        agents = [
            HeuristicAgent(params=genome, agent_id=i, name=f"{name}-{i}")
            for i in range(4)
        ]

        # Run games
        payoffs = []
        for game_idx in range(num_games):
            env = BucketBrigadeEnv(scenario)  # type: ignore[arg-type]
            obs = env.reset(seed=game_idx)

            total_rewards = np.zeros(4)
            while not env.done:
                actions = np.array([agent.act(obs) for agent in agents])
                obs, rewards, dones, info = env.step(actions)
                total_rewards += rewards

            payoffs.append(float(np.mean(total_rewards)))

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
    scenario_name: str, output_dir: Path, num_games: int = 20
) -> Dict[str, Any]:
    """Compare all analysis methods for a scenario."""

    print(f"Running comparison analysis for scenario: {scenario_name}")
    print(f"Output directory: {output_dir}")
    print()

    # Load scenario
    scenario = get_scenario_by_name(scenario_name, num_agents=4)
    scenario_dir = Path(f"experiments/scenarios/{scenario_name}")

    # Load all results
    print("Loading analysis results...")
    all_results = load_results(scenario_dir)

    if all_results["heuristics"] is None:
        print("⚠️  No heuristics results found. Run analyze_heuristics.py first.")
        return

    if all_results["evolved"] is None:
        print("⚠️  No evolution results found. Run run_evolution.py first.")
        return

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

    # Add evolved agent
    strategies["evolved"] = np.array(all_results["evolved"]["genome"])

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

    tournament_results = run_tournament(strategies, scenario, num_games)

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

    # Evolved vs Nash
    if all_results["nash"] is not None:
        evolved_payoff = float(tournament_results["evolved"]["mean_payoff"])  # type: ignore[arg-type]
        nash_payoffs = [
            float(result["mean_payoff"])  # type: ignore[arg-type]
            for name, result in tournament_results.items()
            if name.startswith("nash_strategy_")
        ]

        if nash_payoffs:
            best_nash_payoff = max(nash_payoffs)
            gap = evolved_payoff - best_nash_payoff

            print("Evolved vs Nash:")
            print(f"  Evolved payoff: {evolved_payoff:.2f}")
            print(f"  Best Nash payoff: {best_nash_payoff:.2f}")
            print(f"  Gap: {gap:+.2f}")

            if abs(gap) < 5.0:
                print("  → Evolution found near-Nash strategy!")
            elif gap > 0:
                print(
                    "  → Evolution outperformed Nash (possible exploitation of homogeneous opponents)"
                )
            else:
                print("  → Nash outperformed evolution")

        print()

    # Strategy similarity
    if all_results["nash"] is not None and len(nash_strategies) > 0:
        closest_nash = None
        min_distance = float("inf")

        for name, dist in distances.items():
            if name.startswith("evolved_vs_nash_"):
                if dist < min_distance:
                    min_distance = dist
                    closest_nash = name.replace("evolved_vs_", "")

        if closest_nash:
            print("Strategy Similarity:")
            print(f"  Evolved strategy is closest to: {closest_nash}")
            print(f"  Distance: {min_distance:.3f}")
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
    if all_results["nash"] is not None:
        comparison_results["insights"]["evolution_vs_nash"] = {  # type: ignore[index]
            "evolved_payoff": float(evolved_payoff),
            "best_nash_payoff": float(best_nash_payoff) if nash_payoffs else None,
            "gap": float(gap) if nash_payoffs else None,
        }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "comparison.json"

    with open(output_file, "w") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"✅ Results saved to: {output_file}")
    print()

    return comparison_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare analysis methods")
    parser.add_argument("scenario", type=str, help="Scenario name")
    parser.add_argument("--num-games", type=int, default=20, help="Games per strategy")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"experiments/scenarios/{args.scenario}/comparison")

    run_comparison(args.scenario, args.output_dir, args.num_games)


if __name__ == "__main__":
    main()
