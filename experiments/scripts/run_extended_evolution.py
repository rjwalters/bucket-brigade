#!/usr/bin/env python3
"""
Extended evolution with production hyperparameters.

This script runs longer evolution experiments with larger populations
and more games per individual to find agents that can beat hand-designed heuristics.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from bucket_brigade.algorithms.genetic_algorithm import GeneticAlgorithm, EvolutionConfig
from bucket_brigade.scenarios import get_scenario_by_name, NAMED_SCENARIOS


def run_extended_evolution(
    scenario_name: str,
    output_dir: Path,
    competitive: bool = False,
    warm_start: bool = False,
    population_size: int = 100,
    num_generations: int = 200,
    games_per_individual: int = 50,
    snapshot_interval: int = 20,
    seed: Optional[int] = None,
    resume: bool = False,
):
    """
    Run extended evolution experiment.

    Args:
        scenario_name: Name of the scenario to optimize for
        output_dir: Directory to save results
        competitive: Use competitive co-evolution against best_heuristic
        warm_start: Initialize population with best_heuristic seeds
        population_size: Size of the population
        num_generations: Number of generations to evolve
        games_per_individual: Games to play for fitness evaluation
        snapshot_interval: Save population snapshot every N generations
        seed: Random seed for reproducibility
        resume: Resume from last checkpoint if available
    """
    print(f"Running extended evolution for scenario: {scenario_name}")
    print(f"Output directory: {output_dir}\n")

    # Load scenario
    scenario = get_scenario_by_name(scenario_name, num_agents=4)

    print("Scenario Parameters:")
    print(f"  beta (spread):       {scenario.beta:.2f}")
    print(f"  kappa (extinguish):  {scenario.kappa:.2f}")
    print(f"  c (work cost):       {scenario.c:.2f}")
    print(f"  num_agents:          {scenario.num_agents}")
    print()

    print("Evolution Configuration:")
    print(f"  Population:       {population_size}")
    print(f"  Generations:      {num_generations}")
    print(f"  Elite size:       {population_size // 10}")
    print(f"  Selection:        tournament")
    print(f"  Crossover:        uniform (rate=0.7)")
    print(f"  Mutation:         gaussian (rate=0.1, scale=0.15)")
    print(f"  Fitness:          mean_reward ({games_per_individual} games)")
    print(f"  Competitive:      {competitive}")
    print(f"  Warm start:       {warm_start}")
    print(f"  Seed:             {seed}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure evolution
    config = EvolutionConfig(
        scenario=scenario,
        population_size=population_size,
        num_generations=num_generations,
        elite_size=population_size // 10,
        selection_strategy="tournament",
        tournament_size=3,
        crossover_strategy="uniform",
        crossover_rate=0.7,
        mutation_strategy="gaussian",
        mutation_rate=0.1,
        mutation_scale=0.15,  # Slightly higher than quick mode (0.1)
        fitness_type="mean_reward",
        games_per_individual=games_per_individual,
        seed=seed,
    )

    # Load initial population if warm start
    initial_population = None
    if warm_start:
        # Try to load best_heuristic parameters
        heuristics_file = output_dir.parent.parent / "heuristics" / "results.json"
        if heuristics_file.exists():
            print("🌱 Warm starting from best heuristic...")
            with open(heuristics_file) as f:
                heuristics_data = json.load(f)

            # Find best heuristic genome
            best_heuristic = None
            for team in heuristics_data["homogeneous_teams"]:
                if team["composition"] == heuristics_data["best_homogeneous"]["composition"]:
                    # Map composition to parameters
                    # This is a simplified mapping - you may need to adjust
                    best_heuristic = _heuristic_name_to_params(team["composition"].split()[0].lower())
                    break

            if best_heuristic:
                import numpy as np

                # Initialize population
                initial_population = []

                # 20% exact copies of best heuristic
                for _ in range(population_size // 5):
                    initial_population.append(best_heuristic.copy())

                # 20% mutations of best heuristic
                for _ in range(population_size // 5):
                    mutated = best_heuristic.copy()
                    for i in range(len(mutated)):
                        if np.random.random() < 0.3:  # 30% chance to mutate each gene
                            mutated[i] = np.clip(mutated[i] + np.random.normal(0, 0.15), 0, 1)
                    initial_population.append(mutated)

                print(f"  Seeded {len(initial_population)} individuals from best heuristic")

    # Check for resume checkpoint
    resume_checkpoint = None
    if resume:
        checkpoint_file = output_dir / "checkpoint.json"
        if checkpoint_file.exists():
            print(f"📂 Resuming from checkpoint: {checkpoint_file}")
            with open(checkpoint_file) as f:
                resume_checkpoint = json.load(f)

    # Create genetic algorithm
    ga = GeneticAlgorithm(
        config=config,
        snapshot_dir=output_dir / "snapshots",
        snapshot_interval=snapshot_interval,
    )

    # Set initial population if provided
    if initial_population:
        ga.population = initial_population

    # Progress callback
    def progress_callback(generation: int, best_fitness: float, mean_fitness: float, std_fitness: float, diversity: float):
        print(f"Gen {generation:3d} | Best: {best_fitness:6.2f} | Mean: {mean_fitness:6.2f} | Std: {std_fitness:5.2f} | Diversity: {diversity:5.3f}")

        # Save checkpoint every 10 generations
        if generation % 10 == 0:
            checkpoint = {
                "generation": generation,
                "population": [ind.tolist() for ind in ga.population],
                "fitness_history": ga.fitness_history,
            }
            with open(output_dir / "checkpoint.json", 'w') as f:
                json.dump(checkpoint, f)

    print("Starting evolution...\n")
    start_time = time.time()

    # Run evolution
    result = ga.evolve(
        progress_callback=progress_callback,
        resume_from_generation=resume_checkpoint["generation"] if resume_checkpoint else 0,
    )

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Converged at generation: {result.converged_generation if result.converged else 'N/A'}")
    print()
    print("Best Individual:")
    print(f"  Fitness: {result.best_fitness:.2f}")
    print(f"  Generation: {result.best_generation}")
    print(f"  Genome: {[f'{x:.3f}' for x in result.best_genome]}")
    print()

    # Print parameter names
    param_names = [
        "honesty", "work_tendency", "neighbor_help", "own_priority", "risk_aversion",
        "coordination", "exploration", "fatigue_memory", "rest_bias", "altruism"
    ]
    print("Best Strategy Parameters:")
    for name, value in zip(param_names, result.best_genome):
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {name:14s} : {bar} {value:.3f}")
    print()

    # Save evolution trace
    trace_data = {
        "scenario": scenario_name,
        "config": {
            "population_size": population_size,
            "num_generations": num_generations,
            "elite_size": config.elite_size,
            "selection_strategy": config.selection_strategy,
            "crossover_strategy": config.crossover_strategy,
            "crossover_rate": config.crossover_rate,
            "mutation_strategy": config.mutation_strategy,
            "mutation_rate": config.mutation_rate,
            "mutation_scale": config.mutation_scale,
            "fitness_type": config.fitness_type,
            "games_per_individual": games_per_individual,
            "competitive": competitive,
            "warm_start": warm_start,
            "seed": seed,
        },
        "generations": [
            {
                "generation": gen,
                "best_fitness": float(result.fitness_history[gen]["best"]),
                "mean_fitness": float(result.fitness_history[gen]["mean"]),
                "std_fitness": float(result.fitness_history[gen]["std"]),
                "diversity": float(result.fitness_history[gen]["diversity"]),
            }
            for gen in sorted(result.fitness_history.keys())
        ],
        "best_agent": {
            "generation": result.best_generation,
            "fitness": float(result.best_fitness),
            "genome": [float(x) for x in result.best_genome],
        },
        "convergence": {
            "converged": result.converged,
            "generation": result.converged_generation,
            "elapsed_time": elapsed,
        },
    }

    trace_file = output_dir / "evolution_trace.json"
    with open(trace_file, 'w') as f:
        json.dump(trace_data, f, indent=2)

    # Save best agent
    best_agent_data = {
        "scenario": scenario_name,
        "fitness": float(result.best_fitness),
        "generation": result.best_generation,
        "genome": [float(x) for x in result.best_genome],
        "parameters": {
            name: float(value)
            for name, value in zip(param_names, result.best_genome)
        },
    }

    best_agent_file = output_dir / "best_agent.json"
    with open(best_agent_file, 'w') as f:
        json.dump(best_agent_data, f, indent=2)

    print(f"✅ Results saved:")
    print(f"   Evolution trace: {trace_file}")
    print(f"   Best agent: {best_agent_file}")
    if (output_dir / "snapshots").exists():
        print(f"   Snapshots: {output_dir / 'snapshots'}/")
    print()


def _heuristic_name_to_params(name: str):
    """Map heuristic archetype name to parameter vector."""
    archetypes = {
        "firefighter": [0.7, 0.2, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0],
        "free_rider": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.9, 0.0],
        "hero": [1.0, 0.9, 0.5, 0.8, 0.5, 0.7, 0.1, 0.0, 0.0, 0.8],
        "coordinator": [0.9, 0.6, 0.9, 0.5, 0.3, 0.9, 0.3, 0.5, 0.2, 0.6],
        "liar": [0.0, 0.3, 0.0, 0.9, 0.8, 0.0, 0.5, 0.0, 0.8, 0.0],
    }
    return archetypes.get(name, [0.5] * 10)


def main():
    parser = argparse.ArgumentParser(description="Run extended evolution experiments")
    parser.add_argument("scenario", type=str, nargs="?", help="Scenario name (or use --all)")
    parser.add_argument("--all", action="store_true", help="Run for all scenarios")
    parser.add_argument("--competitive", action="store_true", help="Use competitive co-evolution")
    parser.add_argument("--warm-start", action="store_true", help="Warm start from best heuristic")
    parser.add_argument("--population", type=int, default=100, help="Population size (default: 100)")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations (default: 200)")
    parser.add_argument("--games", type=int, default=50, help="Games per individual (default: 50)")
    parser.add_argument("--snapshot-interval", type=int, default=20, help="Snapshot interval (default: 20)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")

    args = parser.parse_args()

    if args.all:
        scenarios_to_run = list(NAMED_SCENARIOS.keys())
    elif args.scenario:
        scenarios_to_run = [args.scenario]
    else:
        parser.error("Either provide a scenario name or use --all")

    print("=" * 80)
    print("EXTENDED EVOLUTION RESEARCH")
    print("=" * 80)
    print()
    print(f"Scenarios to process: {len(scenarios_to_run)}")
    for scenario in scenarios_to_run:
        print(f"  - {scenario}")
    print()

    # Determine output subdirectory based on mode
    if args.competitive:
        mode = "competitive"
    elif args.warm_start:
        mode = "warm_start"
    else:
        mode = "extended"

    for scenario_name in scenarios_to_run:
        print("=" * 80)
        print(f"Processing: {scenario_name}")
        print("=" * 80)
        print()

        output_dir = Path(f"experiments/scenarios/{scenario_name}/evolved/{mode}")

        try:
            run_extended_evolution(
                scenario_name=scenario_name,
                output_dir=output_dir,
                competitive=args.competitive,
                warm_start=args.warm_start,
                population_size=args.population,
                num_generations=args.generations,
                games_per_individual=args.games,
                snapshot_interval=args.snapshot_interval,
                seed=args.seed,
                resume=args.resume,
            )
        except Exception as e:
            print(f"❌ Error processing {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("=" * 80)
    print("✅ EXTENDED EVOLUTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Scenarios completed: {len(scenarios_to_run)}/{len(scenarios_to_run)}")
    print()
    for scenario in scenarios_to_run:
        print(f"  ✅ {scenario}")


if __name__ == "__main__":
    main()
