#!/usr/bin/env python3
"""
CLI tool for testing policy teams against game scenarios.

Provides a parallel UX to the web demo for controlled testing and development.
Supports testing teams against specific scenarios or random scenario sets.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import typer
import sys
from datetime import datetime
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Add the bucket_brigade package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.agents import HeuristicAgent
from bucket_brigade.envs.scenarios import (
    Scenario,
    random_scenario,
    trivial_cooperation_scenario,
    early_containment_scenario,
    greedy_neighbor_scenario,
    sparse_heroics_scenario,
    rest_trap_scenario,
    chain_reaction_scenario,
    deceptive_calm_scenario,
    overcrowding_scenario,
    mixed_motivation_scenario,
)

app = typer.Typer(help="Test policy teams against game scenarios")

# Agent archetype definitions (from web/src/utils/agentArchetypes.ts)
ARCHETYPES = {
    "firefighter": [1.0, 0.9, 0.7, 0.4, 0.5, 0.7, 0.1, 0.5, 0.1, 0.8],
    "free_rider": [0.7, 0.2, 0.2, 0.9, 0.8, 0.3, 0.2, 0.8, 0.9, 0.1],
    "coordinator": [0.9, 0.6, 0.6, 0.5, 0.5, 1.0, 0.05, 0.4, 0.4, 0.6],
    "liar": [0.1, 0.5, 0.3, 0.7, 0.4, 0.6, 0.3, 0.5, 0.6, 0.2],
    "hero": [1.0, 1.0, 0.9, 0.2, 0.1, 0.5, 0.1, 0.9, 0.0, 1.0],
    "strategist": [0.9, 0.6, 0.5, 0.5, 0.7, 0.9, 0.05, 0.3, 0.5, 0.6],
    "opportunist": [0.6, 0.6, 0.1, 1.0, 0.6, 0.2, 0.2, 0.6, 0.7, 0.0],
    "cautious": [0.9, 0.4, 0.4, 0.7, 0.9, 0.8, 0.05, 0.7, 0.6, 0.4],
    "maverick": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.3, 0.5, 0.5],
    "random": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
}

# Scenario type mapping
SCENARIO_TYPES = {
    "trivial_cooperation": trivial_cooperation_scenario,
    "early_containment": early_containment_scenario,
    "greedy_neighbor": greedy_neighbor_scenario,
    "sparse_heroics": sparse_heroics_scenario,
    "rest_trap": rest_trap_scenario,
    "chain_reaction": chain_reaction_scenario,
    "deceptive_calm": deceptive_calm_scenario,
    "overcrowding": overcrowding_scenario,
    "mixed_motivation": mixed_motivation_scenario,
}


def create_agent_from_archetype(archetype: str, agent_id: int) -> HeuristicAgent:
    """Create an agent from archetype name."""
    archetype_lower = archetype.lower()
    if archetype_lower not in ARCHETYPES:
        raise ValueError(
            f"Unknown archetype: {archetype}. Available: {list(ARCHETYPES.keys())}"
        )

    params = np.array(ARCHETYPES[archetype_lower])

    # Randomize for "random" archetype
    if archetype_lower == "random":
        params = np.random.uniform(0, 1, 10)

    return HeuristicAgent(params, agent_id, name=f"{archetype.title()}-{agent_id}")


def generate_scenario_set(
    num_agents: int,
    count: int,
    scenario_types: Optional[List[str]] = None,
    seed: int = 42,
) -> List[tuple[str, Scenario]]:
    """
    Generate a set of scenarios for testing.

    Args:
        num_agents: Number of agents in the game
        count: Number of scenarios to generate
        scenario_types: List of scenario type names, or None for random
        seed: Random seed

    Returns:
        List of (scenario_name, Scenario) tuples
    """
    np.random.seed(seed)
    scenarios = []

    if scenario_types is None:
        # Generate random scenarios
        for i in range(count):
            scenarios.append(
                (f"random_{i}", random_scenario(num_agents, seed=seed + i))
            )
    else:
        # Generate balanced mix of specified types
        scenarios_per_type = count // len(scenario_types)
        remainder = count % len(scenario_types)

        for scenario_type in scenario_types:
            type_func = SCENARIO_TYPES[scenario_type]
            type_count = scenarios_per_type + (1 if remainder > 0 else 0)
            remainder -= 1

            for i in range(type_count):
                scenarios.append((f"{scenario_type}_{i}", type_func(num_agents)))

    # Shuffle scenarios
    np.random.shuffle(scenarios)
    return scenarios


def run_single_game(
    agents: List[HeuristicAgent],
    scenario: Scenario,
    seed: int = 42,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """
    Run a single game with the given agents and scenario.

    Args:
        agents: List of agents playing the game
        scenario: Game scenario
        seed: Random seed
        max_steps: Maximum number of steps before forcing termination

    Returns:
        Dictionary with game results
    """
    env = BucketBrigadeEnv(scenario)
    obs = env.reset(seed=seed)

    total_rewards = np.zeros(len(agents))
    step_count = 0

    while not env.done and step_count < max_steps:
        actions = np.array([agent.act(obs) for agent in agents])
        obs, rewards, dones, info = env.step(actions)
        total_rewards += rewards
        step_count += 1

    return {
        "team_reward": float(np.sum(total_rewards)),
        "agent_rewards": total_rewards.tolist(),
        "nights_played": env.night,
        "houses_saved": int(np.sum(obs["houses"] == 0)),
        "houses_ruined": int(np.sum(obs["houses"] == 2)),
        "terminated_early": step_count >= max_steps,
    }


def run_tournament(
    team: List[str],
    scenarios: List[tuple[str, Scenario]],
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a tournament with a team against multiple scenarios.

    Args:
        team: List of archetype names
        scenarios: List of (scenario_name, Scenario) tuples
        seed: Random seed
        verbose: Print detailed progress

    Returns:
        Tournament results dictionary
    """
    np.random.seed(seed)
    results = []

    # Create agents
    agents = [
        create_agent_from_archetype(archetype, i) for i, archetype in enumerate(team)
    ]

    # Run games
    if verbose:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Running games", total=len(scenarios))
            for i, (scenario_name, scenario) in enumerate(scenarios):
                game_result = run_single_game(agents, scenario, seed=seed + i)
                game_result["scenario_name"] = scenario_name
                game_result["scenario_index"] = i
                results.append(game_result)
                progress.update(task, advance=1)
    else:
        for i, (scenario_name, scenario) in enumerate(scenarios):
            game_result = run_single_game(agents, scenario, seed=seed + i)
            game_result["scenario_name"] = scenario_name
            game_result["scenario_index"] = i
            results.append(game_result)

    # Calculate statistics
    team_rewards = [r["team_reward"] for r in results]
    houses_saved = [r["houses_saved"] for r in results]
    nights_played = [r["nights_played"] for r in results]

    # Calculate per-agent statistics
    agent_rewards_by_agent = [
        [r["agent_rewards"][i] for r in results] for i in range(len(team))
    ]
    agent_contributions = [
        {
            "archetype": team[i],
            "agent_id": i,
            "mean_reward": float(np.mean(agent_rewards_by_agent[i])),
            "std_reward": float(np.std(agent_rewards_by_agent[i])),
        }
        for i in range(len(team))
    ]

    # Sort by contribution
    agent_contributions.sort(key=lambda x: x["mean_reward"], reverse=True)

    return {
        "team": team,
        "num_games": len(results),
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "team_reward": {
                "mean": float(np.mean(team_rewards)),
                "std": float(np.std(team_rewards)),
                "min": float(np.min(team_rewards)),
                "max": float(np.max(team_rewards)),
                "median": float(np.median(team_rewards)),
            },
            "houses_saved": {
                "mean": float(np.mean(houses_saved)),
                "std": float(np.std(houses_saved)),
                "median": float(np.median(houses_saved)),
            },
            "nights_played": {
                "mean": float(np.mean(nights_played)),
                "std": float(np.std(nights_played)),
            },
            "success_rate": float(np.mean([h >= 5 for h in houses_saved])),
        },
        "agent_contributions": agent_contributions,
        "game_results": results,
    }


@app.command()
def test(
    team: str = typer.Argument(..., help="Comma-separated list of archetype names"),
    scenarios: Optional[str] = typer.Option(
        None,
        "--scenarios",
        "-s",
        help="Comma-separated list of scenario types (or None for random)",
    ),
    count: int = typer.Option(100, "--count", "-c", help="Number of scenarios to test"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (JSON)"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print detailed progress"
    ),
):
    """
    Test a team against scenarios.

    Examples:
        test_team.py firefighter,coordinator,hero
        test_team.py firefighter,hero --scenarios early_containment --count 20
        test_team.py "free_rider,liar,opportunist" --scenarios "greedy_neighbor,rest_trap"
    """
    # Parse team
    team_list = [t.strip() for t in team.split(",")]

    # Validate archetypes
    for archetype in team_list:
        if archetype.lower() not in ARCHETYPES:
            typer.echo(f"Error: Unknown archetype '{archetype}'", err=True)
            typer.echo(
                f"Available archetypes: {', '.join(ARCHETYPES.keys())}", err=True
            )
            raise typer.Exit(1)

    # Parse scenarios
    scenario_types = None
    if scenarios:
        scenario_types = [s.strip() for s in scenarios.split(",")]
        for scenario_type in scenario_types:
            if scenario_type not in SCENARIO_TYPES:
                typer.echo(f"Error: Unknown scenario type '{scenario_type}'", err=True)
                typer.echo(
                    f"Available scenario types: {', '.join(SCENARIO_TYPES.keys())}",
                    err=True,
                )
                raise typer.Exit(1)

    # Generate scenarios
    typer.echo(f"\n🎮 Testing team: {', '.join(team_list)}")
    if scenario_types:
        typer.echo(f"📋 Scenario types: {', '.join(scenario_types)}")
    else:
        typer.echo("📋 Scenario types: Random")
    typer.echo(f"🎲 Number of games: {count}\n")

    scenario_set = generate_scenario_set(len(team_list), count, scenario_types, seed)

    # Run tournament
    results = run_tournament(team_list, scenario_set, seed, verbose)

    # Display results
    stats = results["statistics"]
    typer.echo("\n" + "=" * 60)
    typer.echo("📊 TOURNAMENT RESULTS")
    typer.echo("=" * 60)
    typer.echo(f"\n🏆 Team Performance:")
    typer.echo(
        f"   Mean Team Reward: {stats['team_reward']['mean']:.2f} ± {stats['team_reward']['std']:.2f}"
    )
    typer.echo(f"   Median: {stats['team_reward']['median']:.2f}")
    typer.echo(
        f"   Range: [{stats['team_reward']['min']:.2f}, {stats['team_reward']['max']:.2f}]"
    )

    typer.echo(f"\n🏠 Houses Saved:")
    typer.echo(
        f"   Mean: {stats['houses_saved']['mean']:.2f} ± {stats['houses_saved']['std']:.2f}"
    )
    typer.echo(f"   Median: {stats['houses_saved']['median']:.2f}")
    typer.echo(f"   Success Rate (≥5): {stats['success_rate'] * 100:.1f}%")

    typer.echo(f"\n⏱️  Game Length:")
    typer.echo(
        f"   Mean Nights: {stats['nights_played']['mean']:.1f} ± {stats['nights_played']['std']:.1f}"
    )

    typer.echo(f"\n👥 Agent Contributions (by mean reward):")
    for i, contrib in enumerate(results["agent_contributions"], 1):
        typer.echo(
            f"   {i}. {contrib['archetype']:12s} ({contrib['agent_id']}): "
            f"{contrib['mean_reward']:6.2f} ± {contrib['std_reward']:.2f}"
        )

    # Save results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        typer.echo(f"\n💾 Results saved to: {output_path}")

    typer.echo("\n" + "=" * 60 + "\n")


@app.command()
def list_archetypes():
    """List all available agent archetypes."""
    typer.echo("\n📋 Available Agent Archetypes:\n")
    for name in sorted(ARCHETYPES.keys()):
        typer.echo(f"  • {name}")
    typer.echo()


@app.command()
def list_scenarios():
    """List all available scenario types."""
    typer.echo("\n📋 Available Scenario Types:\n")
    for name in sorted(SCENARIO_TYPES.keys()):
        typer.echo(f"  • {name}")
    typer.echo()


if __name__ == "__main__":
    app()
