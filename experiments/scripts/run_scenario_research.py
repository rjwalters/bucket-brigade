#!/usr/bin/env python3
"""
Master script to run complete research pipeline for a scenario.

Usage:
    python experiments/scripts/run_scenario_research.py greedy_neighbor
    python experiments/scripts/run_scenario_research.py trivial_cooperation --skip-nash
    python experiments/scripts/run_scenario_research.py --all
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bucket_brigade.envs.scenarios import list_scenarios


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def run_scenario_research(
    scenario_name: str,
    skip_heuristics: bool = False,
    skip_evolution: bool = False,
    skip_nash: bool = False,
    skip_comparison: bool = False,
    heuristic_games: int = 100,
    evolution_pop: int = 100,
    evolution_gens: int = 200,
    evolution_games: int = 20,
    nash_sims: int = 1000,
    nash_iters: int = 50,
    comparison_games: int = 20,
) -> bool:
    """Run full research pipeline for a scenario."""

    print("=" * 80)
    print(f"SCENARIO RESEARCH PIPELINE: {scenario_name}")
    print("=" * 80)
    print()
    print("Steps to run:")
    print(f"  [{'✓' if not skip_heuristics else ' '}] 1. Heuristics Analysis")
    print(f"  [{'✓' if not skip_evolution else ' '}] 2. Evolutionary Optimization")
    print(f"  [{'✓' if not skip_nash else ' '}] 3. Nash Equilibrium")
    print(f"  [{'✓' if not skip_comparison else ' '}] 4. Cross-Method Comparison")
    print()

    scenario_dir = Path(f"experiments/scenarios/{scenario_name}")
    scenario_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # 1. Heuristics
    if not skip_heuristics:
        cmd = [
            ".venv/bin/python",
            "experiments/scripts/analyze_heuristics.py",
            scenario_name,
            "--num-games",
            str(heuristic_games),
        ]
        if not run_command(cmd, "Step 1/4: Heuristics Analysis"):
            success = False

    # 2. Evolution
    if not skip_evolution and success:
        cmd = [
            ".venv/bin/python",
            "experiments/scripts/run_evolution.py",
            scenario_name,
            "--population",
            str(evolution_pop),
            "--generations",
            str(evolution_gens),
            "--games",
            str(evolution_games),
        ]
        if not run_command(cmd, "Step 2/4: Evolutionary Optimization"):
            success = False

    # 3. Nash Equilibrium
    if not skip_nash and success:
        cmd = [
            ".venv/bin/python",
            "experiments/scripts/compute_nash.py",
            scenario_name,
            "--simulations",
            str(nash_sims),
            "--max-iterations",
            str(nash_iters),
        ]
        if not run_command(cmd, "Step 3/4: Nash Equilibrium Computation"):
            success = False

    # 4. Comparison
    if not skip_comparison and success:
        cmd = [
            ".venv/bin/python",
            "experiments/scripts/run_comparison.py",
            scenario_name,
            "--num-games",
            str(comparison_games),
        ]
        if not run_command(cmd, "Step 4/4: Cross-Method Comparison"):
            success = False

    # Summary
    print()
    print("=" * 80)
    if success:
        print(f"✅ RESEARCH COMPLETE: {scenario_name}")
    else:
        print(f"⚠️  RESEARCH INCOMPLETE: {scenario_name} (some steps failed)")
    print("=" * 80)
    print()
    print(f"Results directory: {scenario_dir}")
    print()

    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run complete research pipeline for scenarios"
    )

    # Scenario selection
    parser.add_argument(
        "scenario",
        nargs="?",
        type=str,
        help="Scenario name (or use --all for all scenarios)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run research for all scenarios"
    )

    # Skip options
    parser.add_argument(
        "--skip-heuristics", action="store_true", help="Skip heuristics analysis"
    )
    parser.add_argument("--skip-evolution", action="store_true", help="Skip evolution")
    parser.add_argument(
        "--skip-nash", action="store_true", help="Skip Nash equilibrium"
    )
    parser.add_argument(
        "--skip-comparison", action="store_true", help="Skip comparison"
    )

    # Configuration
    parser.add_argument(
        "--heuristic-games", type=int, default=100, help="Games for heuristics"
    )
    parser.add_argument(
        "--evolution-pop", type=int, default=100, help="Evolution population size"
    )
    parser.add_argument(
        "--evolution-gens", type=int, default=200, help="Evolution generations"
    )
    parser.add_argument(
        "--evolution-games",
        type=int,
        default=20,
        help="Games per individual in evolution",
    )
    parser.add_argument(
        "--nash-sims", type=int, default=1000, help="Simulations per Nash evaluation"
    )
    parser.add_argument(
        "--nash-iters", type=int, default=50, help="Max Nash iterations"
    )
    parser.add_argument(
        "--comparison-games", type=int, default=20, help="Games for comparison"
    )

    # Quick mode (reduced parameters for testing)
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (reduced parameters)"
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.heuristic_games = 20
        args.evolution_pop = 20
        args.evolution_gens = 10
        args.evolution_games = 5
        args.nash_sims = 100
        args.nash_iters = 10
        args.comparison_games = 10

    # Determine scenarios to run
    if args.all:
        scenarios = list_scenarios()
        print(f"Running research for all {len(scenarios)} scenarios:")
        for s in scenarios:
            print(f"  - {s}")
        print()
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        parser.print_help()
        sys.exit(1)

    # Run research for each scenario
    results = {}
    for scenario in scenarios:
        success = run_scenario_research(
            scenario,
            skip_heuristics=args.skip_heuristics,
            skip_evolution=args.skip_evolution,
            skip_nash=args.skip_nash,
            skip_comparison=args.skip_comparison,
            heuristic_games=args.heuristic_games,
            evolution_pop=args.evolution_pop,
            evolution_gens=args.evolution_gens,
            evolution_games=args.evolution_games,
            nash_sims=args.nash_sims,
            nash_iters=args.nash_iters,
            comparison_games=args.comparison_games,
        )
        results[scenario] = success

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    completed = sum(1 for s in results.values() if s)
    total = len(results)

    print(f"Scenarios completed: {completed}/{total}")
    print()

    for scenario, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {scenario}")

    print()


if __name__ == "__main__":
    main()
