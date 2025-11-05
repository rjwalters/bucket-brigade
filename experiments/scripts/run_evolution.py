#!/usr/bin/env python3
"""
Run genetic algorithm to discover optimal strategies for a scenario.

Usage:
    python experiments/scripts/run_evolution.py greedy_neighbor
    python experiments/scripts/run_evolution.py greedy_neighbor --generations 100 --population 50
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
from bucket_brigade.evolution import EvolutionConfig, GeneticAlgorithm


def run_evolution(
    scenario_name: str,
    output_dir: Path,
    population_size: int = 100,
    num_generations: int = 200,
    games_per_individual: int = 20,
    snapshot_interval: int = 10,
    seed: Optional[int] = None,
):
    """Run evolutionary algorithm to discover optimal strategies."""

    print(f"Running evolution for scenario: {scenario_name}")
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

    # Configuration
    config = EvolutionConfig(
        population_size=population_size,
        num_generations=num_generations,
        elite_size=max(5, population_size // 20),
        selection_strategy="tournament",
        tournament_size=3,
        crossover_strategy="uniform",
        crossover_rate=0.7,
        mutation_strategy="gaussian",
        mutation_rate=0.1,
        mutation_scale=0.1,
        fitness_type="mean_reward",
        games_per_individual=games_per_individual,
        maintain_diversity=True,
        min_diversity=0.1,
        early_stopping=False,  # Disable early stopping for research runs
        convergence_generations=10,
        convergence_threshold=0.01,
        seed=seed,
    )

    print("Evolution Configuration:")
    print(f"  Population:       {config.population_size}")
    print(f"  Generations:      {config.num_generations}")
    print(f"  Elite size:       {config.elite_size}")
    print(f"  Selection:        {config.selection_strategy}")
    print(
        f"  Crossover:        {config.crossover_strategy} (rate={config.crossover_rate})"
    )
    print(
        f"  Mutation:         {config.mutation_strategy} (rate={config.mutation_rate})"
    )
    print(
        f"  Fitness:          {config.fitness_type} ({config.games_per_individual} games)"
    )
    print(f"  Seed:             {config.seed}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track evolution history
    evolution_trace = {
        "scenario": scenario_name,
        "config": {
            "population_size": config.population_size,
            "num_generations": config.num_generations,
            "elite_size": config.elite_size,
            "selection_strategy": config.selection_strategy,
            "crossover_strategy": config.crossover_strategy,
            "crossover_rate": config.crossover_rate,
            "mutation_strategy": config.mutation_strategy,
            "mutation_rate": config.mutation_rate,
            "mutation_scale": config.mutation_scale,
            "fitness_type": config.fitness_type,
            "games_per_individual": config.games_per_individual,
            "seed": config.seed,
        },
        "generations": [],
    }

    def progress_callback(generation: int, population) -> None:
        """Print and record progress."""
        stats = population.get_fitness_stats()
        diversity = population.get_diversity()

        print(
            f"Gen {generation:3d} | "
            f"Best: {stats['max']:.2f} | "
            f"Mean: {stats['mean']:.2f} | "
            f"Std: {stats['std']:.2f} | "
            f"Diversity: {diversity:.3f}"
        )

        # Record generation stats
        evolution_trace["generations"].append(
            {
                "generation": generation,
                "best_fitness": float(stats["max"]),
                "mean_fitness": float(stats["mean"]),
                "std_fitness": float(stats["std"]),
                "diversity": float(diversity),
            }
        )

        # Save snapshot every N generations
        if generation % snapshot_interval == 0:
            snapshot_dir = output_dir / f"generation_{generation:04d}"
            snapshot_dir.mkdir(exist_ok=True)

            # Save population snapshot
            snapshot_data = {
                "generation": generation,
                "population_size": len(population.individuals),
                "best_fitness": float(stats["max"]),
                "mean_fitness": float(stats["mean"]),
                "diversity": float(diversity),
                "individuals": [
                    {
                        "genome": ind.genome.tolist(),
                        "fitness": float(ind.fitness)
                        if ind.fitness is not None
                        else None,
                    }
                    for ind in population.individuals[:10]  # Save top 10
                ],
            }

            with open(snapshot_dir / "snapshot.json", "w") as f:
                json.dump(snapshot_data, f, indent=2)

    # Create GA and run evolution
    print("Starting evolution...")
    print()

    ga = GeneticAlgorithm(config)

    start_time = time.time()
    result = ga.evolve(progress_callback=progress_callback)
    elapsed_time = time.time() - start_time

    print()
    print("=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"Time elapsed: {elapsed_time:.1f}s")
    print(
        f"Converged at generation: {result.converged_at if result.converged_at is not None else 'N/A'}"
    )
    print()
    print("Best Individual:")
    print(f"  Fitness: {result.best_individual.fitness:.2f}")
    print(f"  Generation: {result.best_individual.generation}")
    print(
        f"  Genome: [{', '.join([f'{x:.3f}' for x in result.best_individual.genome])}]"
    )
    print()

    # Analyze best strategy
    genome = result.best_individual.genome
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

    print("Best Strategy Parameters:")
    for name, value in zip(param_names, genome):
        bar_length = int(value * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {name:15s}: {bar} {value:.3f}")

    print()

    # Save results
    evolution_trace.update(
        {
            "best_agent": {
                "generation": result.best_individual.generation,
                "fitness": float(result.best_individual.fitness),
                "genome": result.best_individual.genome.tolist(),
            },
            "convergence": {
                "converged": result.converged_at is not None,
                "generation": result.converged_at,
                "elapsed_time": elapsed_time,
            },
        }
    )

    # Save evolution trace
    trace_file = output_dir / "evolution_trace.json"
    with open(trace_file, "w") as f:
        json.dump(evolution_trace, f, indent=2)

    # Save best agent separately
    best_agent_file = output_dir / "best_agent.json"
    with open(best_agent_file, "w") as f:
        json.dump(
            {
                "scenario": scenario_name,
                "fitness": float(result.best_individual.fitness),
                "generation": result.best_individual.generation,
                "genome": result.best_individual.genome.tolist(),
                "parameters": {
                    name: float(value) for name, value in zip(param_names, genome)
                },
            },
            f,
            indent=2,
        )

    print("✅ Results saved:")
    print(f"   Evolution trace: {trace_file}")
    print(f"   Best agent: {best_agent_file}")
    print()

    return evolution_trace


def main():
    parser = argparse.ArgumentParser(description="Run evolutionary optimization")
    parser.add_argument("scenario", type=str, help="Scenario name")
    parser.add_argument("--population", type=int, default=100, help="Population size")
    parser.add_argument(
        "--generations", type=int, default=200, help="Number of generations"
    )
    parser.add_argument("--games", type=int, default=20, help="Games per individual")
    parser.add_argument(
        "--snapshot-interval", type=int, default=10, help="Snapshot every N generations"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"experiments/scenarios/{args.scenario}/evolved")

    run_evolution(
        args.scenario,
        args.output_dir,
        population_size=args.population,
        num_generations=args.generations,
        games_per_individual=args.games,
        snapshot_interval=args.snapshot_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
