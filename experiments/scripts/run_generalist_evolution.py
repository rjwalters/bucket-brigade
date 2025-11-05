#!/usr/bin/env python3
"""
Evolve a generalist agent that performs well across all scenarios.

This agent is trained on a distribution of scenarios and should be robust
to different game dynamics.
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
from bucket_brigade.evolution import EvolutionConfig, GeneticAlgorithm, FitnessEvaluator
from bucket_brigade.evolution.population import Individual
import numpy as np

# All scenarios to train on
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


class CrossScenarioFitnessEvaluator:
    """Fitness evaluator that tests agents across multiple scenarios."""

    def __init__(
        self,
        scenario_names: list[str],
        games_per_scenario: int = 5,
        seed: Optional[int] = None,
    ):
        self.scenario_names = scenario_names
        self.games_per_scenario = games_per_scenario
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Create evaluators for each scenario
        self.evaluators = {}
        for scenario_name in scenario_names:
            scenario = get_scenario_by_name(scenario_name, num_agents=1)
            self.evaluators[scenario_name] = FitnessEvaluator(
                scenario=scenario,
                games_per_individual=games_per_scenario,
                seed=self.rng.randint(0, 2**31 - 1) if seed is not None else None,
            )

    def evaluate_individual(self, individual: Individual) -> float:
        """Evaluate individual across all scenarios."""
        scenario_scores = []

        for scenario_name in self.scenario_names:
            evaluator = self.evaluators[scenario_name]
            score = evaluator.evaluate_individual(individual)
            scenario_scores.append(score)

        # Return mean performance across scenarios
        return float(np.mean(scenario_scores))

    def evaluate_population(self, population, parallel: Optional[bool] = None) -> None:
        """Evaluate all individuals in population."""
        for individual in population:
            if individual.fitness is None:
                individual.fitness = self.evaluate_individual(individual)


def run_generalist_evolution(
    output_dir: Path,
    population_size: int = 100,
    num_generations: int = 200,
    games_per_scenario: int = 5,
    snapshot_interval: int = 10,
    seed: Optional[int] = None,
):
    """Run evolutionary algorithm for generalist agent."""

    print("=" * 80)
    print("GENERALIST AGENT EVOLUTION")
    print("=" * 80)
    print(f"Training on {len(SCENARIOS)} scenarios")
    print(f"Output directory: {output_dir}")
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
        fitness_type="mean_reward",  # Using custom evaluator
        games_per_individual=games_per_scenario * len(SCENARIOS),
        maintain_diversity=True,
        min_diversity=0.1,
        early_stopping=False,
        convergence_generations=10,
        convergence_threshold=0.01,
        seed=seed,
    )

    print("Evolution Configuration:")
    print(f"  Population:       {config.population_size}")
    print(f"  Generations:      {config.num_generations}")
    print(f"  Scenarios:        {len(SCENARIOS)}")
    print(f"  Games/scenario:   {games_per_scenario}")
    print(f"  Total games/eval: {games_per_scenario * len(SCENARIOS)}")
    print(f"  Seed:             {config.seed}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Custom fitness evaluator
    fitness_evaluator = CrossScenarioFitnessEvaluator(
        scenario_names=SCENARIOS,
        games_per_scenario=games_per_scenario,
        seed=seed,
    )

    # Track evolution history
    evolution_trace = {
        "agent_type": "generalist",
        "scenarios": SCENARIOS,
        "config": {
            "population_size": config.population_size,
            "num_generations": config.num_generations,
            "games_per_scenario": games_per_scenario,
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
            f"Best: {stats['max']:7.2f} | "
            f"Mean: {stats['mean']:7.2f} | "
            f"Std: {stats['std']:6.2f} | "
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
                    for ind in population.individuals[:10]
                ],
            }

            with open(snapshot_dir / "snapshot.json", "w") as f:
                json.dump(snapshot_data, f, indent=2)

    # Create GA with custom fitness evaluator
    print("Starting generalist evolution...")
    print()

    ga = GeneticAlgorithm(config, fitness_evaluator=fitness_evaluator)

    start_time = time.time()
    result = ga.evolve(progress_callback=progress_callback)
    elapsed_time = time.time() - start_time

    print()
    print("=" * 80)
    print("Generalist Evolution Complete!")
    print("=" * 80)
    print(f"Time elapsed: {elapsed_time:.1f}s")
    print(f"Converged: {result.converged_at is not None}")
    print()
    print("Best Individual:")
    print(
        f"  Fitness (avg across {len(SCENARIOS)} scenarios): {result.best_individual.fitness:.2f}"
    )
    print(f"  Generation: {result.best_individual.generation}")
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

    print("Best Generalist Strategy Parameters:")
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

    # Save best agent
    best_agent_file = output_dir / "best_agent.json"
    with open(best_agent_file, "w") as f:
        json.dump(
            {
                "agent_type": "generalist",
                "scenarios": SCENARIOS,
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
    parser = argparse.ArgumentParser(
        description="Evolve generalist agent across scenarios"
    )
    parser.add_argument("--population", type=int, default=100, help="Population size")
    parser.add_argument(
        "--generations", type=int, default=200, help="Number of generations"
    )
    parser.add_argument(
        "--games-per-scenario", type=int, default=5, help="Games per scenario per eval"
    )
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
        args.output_dir = Path("experiments/generalist/evolved")

    run_generalist_evolution(
        args.output_dir,
        population_size=args.population,
        num_generations=args.generations,
        games_per_scenario=args.games_per_scenario,
        snapshot_interval=args.snapshot_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
