#!/usr/bin/env python3
"""
V6 Evolution: Simplified single-scenario evolution with V6 parameters.

This script uses the V6 strategy (larger population, more exploration) but
evolves per-scenario like V4/V5 for compatibility with existing framework.

V6 improvements over V5:
- Larger population: 200 (vs 100)
- More generations: 200 (vs 100)
- Higher mutation rate: 0.15 (vs 0.1)
- Better diversity maintenance

See experiments/evolution/V6_PLAN.md for full strategy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time

from bucket_brigade.evolution import (
    EvolutionConfig,
    GeneticAlgorithm,
    Population,
)
from bucket_brigade.envs.scenarios import get_scenario_by_name, list_scenarios


def progress_callback(generation: int, population: Population, output_dir: Path) -> None:
    """Print evolution progress and save checkpoints."""
    stats = population.get_fitness_stats()
    diversity = population.get_diversity()

    print(
        f"[Gen {generation:4d}] "
        f"Best: {stats['max']:7.2f} | "
        f"Mean: {stats['mean']:7.2f} ± {stats['std']:5.2f} | "
        f"Diversity: {diversity:.3f}"
    )

    # Save checkpoint every 20 generations
    if generation > 0 and generation % 20 == 0:
        checkpoint_path = output_dir / f"checkpoint_gen{generation}.json"
        checkpoint_data = {
            "generation": generation,
            "population": population.to_dict(),
            "best_individual": population.get_best_individual().to_dict(),
            "fitness_stats": stats,
            "diversity": diversity,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"  → Checkpoint saved: {checkpoint_path.name}")


def save_results(result, output_dir: Path, config: EvolutionConfig, scenario: str) -> None:
    """Save evolution results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best agent in standard format
    best_agent_path = output_dir / "best_agent.json"
    best_agent_data = {
        "genome": result.best_individual.genome.tolist(),
        "fitness": float(result.best_individual.fitness),
        "generation": result.best_individual.generation,
        "version": "v6",
        "scenario": scenario,
    }
    with open(best_agent_path, "w") as f:
        json.dump(best_agent_data, f, indent=2)
    print(f"✓ Best agent saved: {best_agent_path}")

    # Save full results
    results_path = output_dir / "evolution_results.json"
    results_data = {
        "version": "v6",
        "scenario": scenario,
        "config": {
            "population_size": config.population_size,
            "num_generations": config.num_generations,
            "mutation_rate": config.mutation_rate,
            "mutation_scale": config.mutation_scale,
            "elite_size": config.elite_size,
            "games_per_individual": config.games_per_individual,
            "fitness_type": config.fitness_type,
            "seed": config.seed,
        },
        "best_individual": result.best_individual.to_dict(),
        "final_population": result.final_population.to_dict(),
        "fitness_history": result.fitness_history,
        "diversity_history": result.diversity_history,
        "converged_at": result.converged_at,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Full results saved: {results_path}")

    # Save summary log
    log_path = output_dir / "evolution_log.txt"
    with open(log_path, "w") as f:
        f.write("V6 Evolution Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Scenario: {scenario}\n")
        f.write(f"Population: {config.population_size}\n")
        f.write(f"Generations: {config.num_generations}\n")
        f.write(f"Mutation Rate: {config.mutation_rate}\n")
        f.write(f"Converged At: {result.converged_at}\n\n")
        f.write("Best Individual:\n")
        f.write(f"  Fitness: {result.best_individual.fitness:.4f}\n")
        f.write(f"  Generation: {result.best_individual.generation}\n")
        f.write(f"  Genome: {result.best_individual.genome.tolist()}\n\n")
        f.write("Final Population Stats:\n")
        final_stats = result.final_population.get_fitness_stats()
        f.write(f"  Best: {final_stats['max']:.4f}\n")
        f.write(f"  Mean: {final_stats['mean']:.4f}\n")
        f.write(f"  Std:  {final_stats['std']:.4f}\n")
        f.write(f"  Diversity: {result.final_population.get_diversity():.4f}\n")
    print(f"✓ Evolution log saved: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="V6 Evolution: Simplified single-scenario evolution with V6 parameters"
    )

    # Core parameters (V6 defaults)
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name (e.g., chain_reaction, trivial_cooperation)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=200,
        help="Population size (default: 200, V6 plan)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Number of generations (default: 200, V6 plan)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.15,
        help="Mutation probability per gene (default: 0.15, V6 plan)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results",
    )

    # Optional
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Number of agents per game (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Validate scenario
    available_scenarios = list_scenarios()
    if args.scenario not in available_scenarios:
        print(f"Error: Scenario '{args.scenario}' not found")
        print(f"Available scenarios: {', '.join(available_scenarios)}")
        sys.exit(1)

    print("=" * 80)
    print("V6 Evolution: Simplified Approach")
    print("=" * 80)
    print(f"Scenario: {args.scenario}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Mutation Rate: {args.mutation_rate}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed if args.seed else 'random'}")
    print()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load scenario
    print(f"Loading scenario: {args.scenario}")
    scenario = get_scenario_by_name(args.scenario, args.num_agents)
    print(f"  Agents: {args.num_agents}")
    print(f"  Fire count: {scenario.fire_count}")
    print(f"  Max steps: {scenario.max_steps}")
    print()

    # Create evolution configuration (V6 parameters)
    config = EvolutionConfig(
        population_size=args.population,
        num_generations=args.generations,
        elite_size=10,  # V6 plan: keep top 10
        selection_strategy="tournament",
        tournament_size=5,  # V6 plan
        crossover_strategy="uniform",
        crossover_rate=0.7,
        mutation_strategy="gaussian",
        mutation_rate=args.mutation_rate,
        mutation_scale=0.1,
        fitness_type="mean_reward",  # Standard fitness with scenario
        games_per_individual=100,  # V6 plan: 100 games per evaluation
        parallel=(args.workers > 1),
        num_workers=args.workers if args.workers > 1 else None,
        maintain_diversity=True,
        min_diversity=0.1,
        early_stopping=False,  # V6: run full 200 generations
        seed=args.seed,
    )

    print("Evolution Configuration:")
    print(f"  Elite size: {config.elite_size}")
    print(f"  Tournament size: {config.tournament_size}")
    print(f"  Games per individual: {config.games_per_individual}")
    print(f"  Parallel: {config.parallel}")
    print(f"  Early stopping: {config.early_stopping}")
    print()

    # Create GA (will use Rust-backed FitnessEvaluator automatically)
    ga = GeneticAlgorithm(config)

    # Run evolution
    print("Starting evolution...")
    print()

    start_time = time.time()

    def progress_with_output(generation: int, population: Population) -> None:
        progress_callback(generation, population, args.output)

    result = ga.evolve(progress_callback=progress_with_output)

    elapsed_time = time.time() - start_time

    # Print results
    print()
    print("=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time / 60:.1f} minutes)")
    print(
        f"Converged at: {result.converged_at if result.converged_at is not None else 'N/A (ran full generations)'}"
    )
    print()
    print("Best Individual:")
    print(f"  Fitness: {result.best_individual.fitness:.4f}")
    print(f"  Generation: {result.best_individual.generation}")
    print()

    # Save results
    save_results(result, args.output, config, args.scenario)

    print()
    print("✓ V6 evolution complete!")
    print()
    print("Next steps:")
    print("  1. Run validation tournaments to compare V6 vs V5 vs V4")
    print("  2. Analyze if larger population + exploration improved robustness")
    print("  3. Compare V6 (evolution) vs PPO (gradient-based RL)")


if __name__ == "__main__":
    main()
