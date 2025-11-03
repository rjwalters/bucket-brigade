#!/usr/bin/env python3
"""
CLI tool for comparing multiple team configurations side-by-side.

Useful for validating that preferred strategies succeed on specific scenarios.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import typer
import sys
from datetime import datetime
from rich.console import Console
from rich.table import Table
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

app = typer.Typer(help="Compare multiple team configurations")
console = Console()

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


def generate_scenarios(
    num_agents: int, scenario_types: List[str], count: int, seed: int = 42
) -> List[tuple[str, Scenario]]:
    """Generate balanced set of scenarios."""
    np.random.seed(seed)
    scenarios = []

    scenarios_per_type = count // len(scenario_types)
    remainder = count % len(scenario_types)

    for scenario_type in scenario_types:
        type_func = SCENARIO_TYPES[scenario_type]
        type_count = scenarios_per_type + (1 if remainder > 0 else 0)
        remainder -= 1

        for i in range(type_count):
            scenarios.append((f"{scenario_type}_{i}", type_func(num_agents)))

    return scenarios


def run_single_game(
    agents: List[HeuristicAgent],
    scenario: Scenario,
    seed: int = 42,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """Run a single game with the given agents and scenario."""
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
    }


def run_comparison(
    teams: List[List[str]],
    scenarios: List[tuple[str, Scenario]],
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run comparison between multiple teams on same scenarios."""
    results = {}

    total_tests = len(teams) * len(scenarios)

    if verbose:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Running comparisons", total=total_tests)

            for team_idx, team in enumerate(teams):
                team_name = ", ".join(team)
                team_results = []

                # Create agents for this team
                agents = [
                    create_agent_from_archetype(archetype, i)
                    for i, archetype in enumerate(team)
                ]

                # Run all scenarios
                for i, (scenario_name, scenario) in enumerate(scenarios):
                    game_result = run_single_game(agents, scenario, seed=seed + i)
                    game_result["scenario_name"] = scenario_name
                    team_results.append(game_result)
                    progress.update(task, advance=1)

                results[team_name] = team_results
    else:
        for team_idx, team in enumerate(teams):
            team_name = ", ".join(team)
            team_results = []

            # Create agents for this team
            agents = [
                create_agent_from_archetype(archetype, i)
                for i, archetype in enumerate(team)
            ]

            # Run all scenarios
            for i, (scenario_name, scenario) in enumerate(scenarios):
                game_result = run_single_game(agents, scenario, seed=seed + i)
                game_result["scenario_name"] = scenario_name
                team_results.append(game_result)

            results[team_name] = team_results

    return results


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate statistics from game results."""
    team_rewards = [r["team_reward"] for r in results]
    houses_saved = [r["houses_saved"] for r in results]

    return {
        "mean_reward": float(np.mean(team_rewards)),
        "std_reward": float(np.std(team_rewards)),
        "mean_houses_saved": float(np.mean(houses_saved)),
        "success_rate": float(np.mean([h >= 5 for h in houses_saved])),
    }


def display_comparison_table(
    comparison_results: Dict[str, List[Dict]], statistical_significance: bool = True
):
    """Display comparison results in a formatted table."""
    table = Table(
        title="Team Comparison Results", show_header=True, header_style="bold magenta"
    )

    table.add_column("Team", style="cyan", width=30)
    table.add_column("Mean Reward", justify="right", style="green")
    table.add_column("Std Dev", justify="right")
    table.add_column("Houses Saved", justify="right", style="yellow")
    table.add_column("Success Rate", justify="right", style="blue")

    stats_by_team = {}
    for team_name, results in comparison_results.items():
        stats = calculate_statistics(results)
        stats_by_team[team_name] = stats

        table.add_row(
            team_name,
            f"{stats['mean_reward']:.2f}",
            f"±{stats['std_reward']:.2f}",
            f"{stats['mean_houses_saved']:.2f}",
            f"{stats['success_rate'] * 100:.1f}%",
        )

    console.print(table)

    # Statistical significance test
    if statistical_significance and len(comparison_results) == 2:
        from scipy import stats as scipy_stats

        teams = list(comparison_results.keys())
        rewards1 = [r["team_reward"] for r in comparison_results[teams[0]]]
        rewards2 = [r["team_reward"] for r in comparison_results[teams[1]]]

        t_stat, p_value = scipy_stats.ttest_ind(rewards1, rewards2)

        console.print(f"\n📊 Statistical Significance Test (t-test):")
        console.print(f"   t-statistic: {t_stat:.4f}")
        console.print(f"   p-value: {p_value:.4f}")

        if p_value < 0.05:
            winner = teams[0] if np.mean(rewards1) > np.mean(rewards2) else teams[1]
            console.print(
                f"   ✅ [bold green]Significant difference detected (p < 0.05)[/bold green]"
            )
            console.print(f"   🏆 [bold]{winner}[/bold] performs significantly better")
        else:
            console.print(
                f"   ⚠️  [yellow]No significant difference (p >= 0.05)[/yellow]"
            )


@app.command()
def compare(
    team1: str = typer.Option(
        ..., "--team1", help="First team (comma-separated archetypes)"
    ),
    team2: str = typer.Option(
        ..., "--team2", help="Second team (comma-separated archetypes)"
    ),
    team3: str = typer.Option(None, "--team3", help="Third team (optional)"),
    team4: str = typer.Option(None, "--team4", help="Fourth team (optional)"),
    scenarios: str = typer.Option(
        "early_containment",
        "--scenarios",
        "-s",
        help="Comma-separated scenario types",
    ),
    count: int = typer.Option(50, "--count", "-c", help="Number of games per team"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    output: str = typer.Option(None, "--output", "-o", help="Output file (JSON)"),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", "-v/-q", help="Show progress"
    ),
    stats: bool = typer.Option(
        True, "--stats/--no-stats", help="Run statistical significance tests"
    ),
):
    """
    Compare multiple teams on the same scenarios.

    Examples:
        compare_teams.py --team1 firefighter,hero,coordinator --team2 free_rider,liar,opportunist
        compare_teams.py --team1 hero,hero,hero --team2 free_rider,free_rider,free_rider --scenarios greedy_neighbor --count 100
    """
    # Parse teams
    teams = [
        [t.strip() for t in team.split(",")]
        for team in [team1, team2, team3, team4]
        if team is not None
    ]

    # Validate archetypes
    for team in teams:
        for archetype in team:
            if archetype.lower() not in ARCHETYPES:
                console.print(
                    f"[red]Error: Unknown archetype '{archetype}'[/red]", err=True
                )
                console.print(f"Available: {', '.join(ARCHETYPES.keys())}", err=True)
                raise typer.Exit(1)

    # Ensure all teams have same size
    team_sizes = [len(team) for team in teams]
    if len(set(team_sizes)) > 1:
        console.print(f"[red]Error: All teams must have the same size[/red]", err=True)
        console.print(f"Team sizes: {team_sizes}", err=True)
        raise typer.Exit(1)

    num_agents = team_sizes[0]

    # Parse scenarios
    scenario_types = [s.strip() for s in scenarios.split(",")]
    for scenario_type in scenario_types:
        if scenario_type not in SCENARIO_TYPES:
            console.print(
                f"[red]Error: Unknown scenario type '{scenario_type}'[/red]", err=True
            )
            console.print(f"Available: {', '.join(SCENARIO_TYPES.keys())}", err=True)
            raise typer.Exit(1)

    # Display header
    console.print(f"\n🎮 [bold]Team Comparison[/bold]")
    console.print(f"📋 Scenario types: {', '.join(scenario_types)}")
    console.print(f"🎲 Games per team: {count}")
    console.print(f"👥 Teams to compare: {len(teams)}\n")

    for i, team in enumerate(teams, 1):
        console.print(f"   Team {i}: {', '.join(team)}")
    console.print()

    # Generate scenarios
    scenario_set = generate_scenarios(num_agents, scenario_types, count, seed)

    # Run comparison
    comparison_results = run_comparison(teams, scenario_set, seed, verbose)

    # Display results
    console.print()
    display_comparison_table(
        comparison_results, statistical_significance=stats and len(teams) == 2
    )

    # Save results
    if output:
        output_data = {
            "teams": [", ".join(team) for team in teams],
            "scenario_types": scenario_types,
            "count": count,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "results": comparison_results,
        }

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n💾 Results saved to: {output_path}")

    console.print()


if __name__ == "__main__":
    app()
