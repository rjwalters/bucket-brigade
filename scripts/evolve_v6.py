#!/usr/bin/env python3
"""
V6 Evolution: Tournament-based fitness across multiple scenarios.

Key differences from V5:
- Tournament-based fitness (not Nash equilibrium)
- Heterogeneous opponent pool (archetypes + evolved variants + mutations)
- Multi-scenario training for generalization
- Larger population and more exploration

See experiments/evolution/V6_PLAN.md for full strategy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from typing import List, Optional

import numpy as np

from bucket_brigade.evolution import (
    EvolutionConfig,
    GeneticAlgorithm,
    Individual,
    Population,
)
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.agents.heuristic_agent import HeuristicAgent
from bucket_brigade.agents.archetypes import ARCHETYPES
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv


def load_evolved_agent(version: str, scenario: str) -> Optional[np.ndarray]:
    """Load evolved agent parameters from previous version.

    Args:
        version: Agent version (e.g., 'v4', 'v5')
        scenario: Scenario name

    Returns:
        Agent parameters or None if not found
    """
    agent_path = Path(f"experiments/scenarios/{scenario}/evolved_{version}/best_agent.json")
    if not agent_path.exists():
        return None

    with open(agent_path, "r") as f:
        data = json.load(f)

    # Handle different file formats
    if "genome" in data:
        return np.array(data["genome"])
    elif "params" in data:
        return np.array(data["params"])
    elif isinstance(data, list):
        return np.array(data)
    else:
        return None


def create_tournament_fitness(
    scenarios: List[str],
    num_agents: int = 4,
    games_per_opponent: int = 10,
    num_workers: int = 1,
):
    """Create tournament-based fitness function for V6.

    Evaluates individual against diverse opponent pool across multiple scenarios.

    Args:
        scenarios: List of scenario names to train on
        num_agents: Number of agents per game
        games_per_opponent: Games per opponent type
        num_workers: Number of parallel workers (for display only)

    Returns:
        Fitness function
    """
    # Load opponent pool
    opponent_pool = {}

    # 1. Archetypes (5 types)
    for archetype_name, archetype_params in ARCHETYPES.items():
        opponent_pool[f"archetype_{archetype_name}"] = np.array(archetype_params)

    # 2. Evolved agents (V4, V5) - load from first scenario
    first_scenario = scenarios[0]
    for version in ["v4", "v5"]:
        params = load_evolved_agent(version, first_scenario)
        if params is not None:
            opponent_pool[f"evolved_{version}"] = params

    # 3. Random mutations (2 variants for diversity)
    rng = np.random.default_rng(42)
    opponent_pool["mutation_1"] = rng.standard_normal(10) * 0.3
    opponent_pool["mutation_2"] = rng.standard_normal(10) * 0.5

    print(f"Opponent pool: {len(opponent_pool)} types")
    for name in opponent_pool.keys():
        print(f"  - {name}")
    print()

    def fitness_fn(individual: Individual) -> float:
        """Evaluate individual via tournament across all scenarios.

        Args:
            individual: Individual to evaluate

        Returns:
            Average payoff across all scenarios and opponents
        """
        all_payoffs = []
        rng = np.random.default_rng()

        # Evaluate on each scenario
        for scenario_name in scenarios:
            scenario = get_scenario_by_name(scenario_name, num_agents)

            # Play against each opponent type
            for opponent_name, opponent_params in opponent_pool.items():
                # Create teammates using this opponent type
                teammates = [
                    HeuristicAgent(
                        params=opponent_params,
                        agent_id=i + 1,
                        name=f"{opponent_name}_{i}",
                    )
                    for i in range(num_agents - 1)
                ]

                # Run multiple games for stability
                for _ in range(games_per_opponent):
                    try:
                        # Create focal agent
                        focal_agent = HeuristicAgent(
                            params=individual.genome,
                            agent_id=0,
                            name="focal",
                        )

                        # Create environment and run episode
                        env = BucketBrigadeEnv(
                            scenario=scenario,
                            agents=[focal_agent] + teammates,
                        )
                        obs = env.reset(seed=rng.integers(0, 2**31))

                        done = False
                        total_reward = 0.0

                        while not done:
                            actions = []
                            for agent in env.agents:
                                action = agent.get_action(obs)
                                actions.append(action)

                            obs, rewards, terminated, truncated, info = env.step(actions)
                            total_reward += rewards[0]  # Focal agent reward
                            done = terminated or truncated

                        all_payoffs.append(total_reward)

                    except Exception as e:
                        # Penalize failures
                        all_payoffs.append(-200.0)

        # Fitness = average payoff across all games
        return np.mean(all_payoffs)

    return fitness_fn


def progress_callback(generation: int, population: Population, output_dir: Path) -> None:
    """Print evolution progress and save checkpoints.

    Args:
        generation: Current generation number
        population: Current population
        output_dir: Output directory for checkpoints
    """
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


def save_results(
    result,
    output_dir: Path,
    config: EvolutionConfig,
    scenarios: List[str],
) -> None:
    """Save evolution results.

    Args:
        result: EvolutionResult instance
        output_dir: Output directory
        config: Evolution configuration
        scenarios: Scenario names
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best agent in standard format
    best_agent_path = output_dir / "best_agent.json"
    best_agent_data = {
        "genome": result.best_individual.genome.tolist(),
        "fitness": float(result.best_individual.fitness),
        "generation": result.best_individual.generation,
        "version": "v6",
        "scenarios": scenarios,
    }
    with open(best_agent_path, "w") as f:
        json.dump(best_agent_data, f, indent=2)
    print(f"✓ Best agent saved: {best_agent_path}")

    # Save full results
    results_path = output_dir / "evolution_results.json"
    results_data = {
        "version": "v6",
        "scenarios": scenarios,
        "config": {
            "population_size": config.population_size,
            "num_generations": config.num_generations,
            "mutation_rate": config.mutation_rate,
            "mutation_scale": config.mutation_scale,
            "elite_size": config.elite_size,
            "games_per_individual": config.games_per_individual,
            "fitness_type": "tournament",
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
        f.write(f"Scenarios: {', '.join(scenarios)}\n")
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
        description="V6 Evolution: Tournament-based fitness across multiple scenarios"
    )

    # Core parameters (match V6 plan)
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        required=True,
        help="Scenario names to train on (e.g., chain_reaction trivial_cooperation)",
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
        "--games",
        type=int,
        default=10,
        help="Games per opponent type (default: 10, total ~100 games)",
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

    print("=" * 80)
    print("V6 Evolution: Tournament-Based Fitness")
    print("=" * 80)
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Mutation Rate: {args.mutation_rate}")
    print(f"Games per opponent: {args.games}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed if args.seed else 'random'}")
    print()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create tournament fitness function
    print("Creating tournament fitness function...")
    fitness_fn = create_tournament_fitness(
        scenarios=args.scenarios,
        num_agents=args.num_agents,
        games_per_opponent=args.games,
        num_workers=args.workers,
    )

    # Calculate total games per individual for config
    # games_per_opponent * num_opponents * num_scenarios
    # Assuming ~8 opponent types
    total_games = args.games * 8 * len(args.scenarios)

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
        fitness_type="custom",
        games_per_individual=total_games,
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
    print(f"  Total games per individual: ~{total_games}")
    print(f"  Parallel: {config.parallel}")
    print()

    # Create GA with tournament fitness
    ga = GeneticAlgorithm(config)
    ga.fitness_evaluator.fitness_fn = fitness_fn

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
    save_results(result, args.output, config, args.scenarios)

    print()
    print("✓ V6 evolution complete!")
    print()
    print("Next steps:")
    print("  1. Run validation tournaments to compare V6 vs V5 vs V4")
    print("  2. Analyze if tournament fitness solved robustness problem")
    print("  3. Deploy V6 agent if performance improves >5% over V4")


if __name__ == "__main__":
    main()
