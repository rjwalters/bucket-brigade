#!/usr/bin/env python3
"""
Test specific scenarios with known optimal strategies to validate ranking.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import typer

# Add the bucket_brigade package to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.envs import (
    BucketBrigadeEnv,
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
from bucket_brigade.agents import create_random_agent, create_archetype_agent
from bucket_brigade.orchestration import AgentRankingModel


SCENARIO_FUNCTIONS = {
    'trivial_cooperation': trivial_cooperation_scenario,
    'early_containment': early_containment_scenario,
    'greedy_neighbor': greedy_neighbor_scenario,
    'sparse_heroics': sparse_heroics_scenario,
    'rest_trap': rest_trap_scenario,
    'chain_reaction': chain_reaction_scenario,
    'deceptive_calm': deceptive_calm_scenario,
    'overcrowding': overcrowding_scenario,
    'mixed_motivation': mixed_motivation_scenario,
}


def create_test_agents(num_agents: int, scenario_name: str) -> List:
    """
    Create a mix of good and bad agents for testing.

    For each scenario, we create agents with strategies that should perform
    well vs poorly in that specific scenario.
    """
    agents = []

    if scenario_name == 'trivial_cooperation':
        # Good: Honest cooperators (firefighter)
        # Bad: Selfish freeloaders
        for i in range(num_agents):
            if i < num_agents // 2:
                agents.append(create_archetype_agent('firefighter', i))
            else:
                agents.append(create_archetype_agent('free_rider', i))

    elif scenario_name == 'greedy_neighbor':
        # Good: Some cooperation (coordinator)
        # Bad: Pure selfishness (free_rider)
        for i in range(num_agents):
            if i < num_agents // 3:
                agents.append(create_archetype_agent('coordinator', i))
            else:
                agents.append(create_archetype_agent('free_rider', i))

    elif scenario_name == 'sparse_heroics':
        # Good: Willing to work when needed (hero)
        # Bad: Always resting (free_rider)
        for i in range(num_agents):
            if i < num_agents // 4:
                agents.append(create_archetype_agent('hero', i))
            else:
                agents.append(create_archetype_agent('free_rider', i))

    elif scenario_name == 'deceptive_calm':
        # Good: Honest and responsive (firefighter)
        # Bad: Dishonest liars
        for i in range(num_agents):
            if i < num_agents // 2:
                agents.append(create_archetype_agent('firefighter', i))
            else:
                agents.append(create_archetype_agent('liar', i))

    else:
        # Default: Mix of archetypes
        archetypes = ['firefighter', 'coordinator', 'free_rider', 'liar']
        for i in range(num_agents):
            archetype = archetypes[i % len(archetypes)]
            agents.append(create_archetype_agent(archetype, i))

    return agents


def run_scenario_test(scenario_name: str, num_games: int = 20,
                     output_dir: str = "scenario_test") -> None:
    """
    Run a test with a specific scenario and mixed agent types.
    """
    print(f"🧪 Testing Scenario: {scenario_name}")
    print("=" * 50)

    if scenario_name not in SCENARIO_FUNCTIONS:
        print(f"❌ Unknown scenario: {scenario_name}")
        print(f"Available: {list(SCENARIO_FUNCTIONS.keys())}")
        return

    # Create output directory
    output_path = Path(output_dir) / scenario_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Create scenario
    scenario_func = SCENARIO_FUNCTIONS[scenario_name]
    scenario = scenario_func(num_agents=6)  # Fixed at 6 agents for consistency

    print(f"📊 Scenario Parameters:")
    print(f"   β (spread): {scenario.beta:.3f}")
    print(f"   κ (extinguish): {scenario.kappa:.3f}")
    print(f"   Work cost (c): {scenario.c:.2f}")
    print(f"   Initial fires: {scenario.rho_ignite:.2f}")
    print()

    # Create mixed agent pool
    agents = create_test_agents(6, scenario_name)

    print(f"🤖 Agent Composition:")
    agent_types = {}
    for agent in agents:
        agent_type = agent.name.split('-')[0].lower()
        agent_types[agent_type] = agent_types.get(agent_type, 0) + 1

    for agent_type, count in agent_types.items():
        print(f"   {agent_type.title()}: {count}")
    print()

    # Run multiple games with different team compositions
    batch_results = []
    np.random.seed(42)  # For reproducibility

    print(f"🎮 Running {num_games} games...")

    for game_idx in range(num_games):
        # Randomly select 4 agents out of 6 for each game
        team_indices = np.random.choice(6, 4, replace=False)
        team_agents = [agents[i] for i in team_indices]

        # Create environment
        env = BucketBrigadeEnv(scenario)

        # Run game
        obs = env.reset(seed=game_idx)
        total_rewards = np.zeros(4)

        while not env.done:
            actions = np.array([agent.act(obs) for agent in team_agents])
            obs, rewards, dones, info = env.step(actions)
            total_rewards += rewards

        team_reward = float(np.sum(total_rewards))

        # Record result
        result = {
            'game_id': game_idx,
            'scenario_id': 0,
            'team': team_indices.tolist(),
            'agent_params': [[0.5] * 10] * 4,  # Dummy params (not used for ranking)
            'team_reward': team_reward,
            'agent_rewards': total_rewards.tolist(),
            'nights_played': env.night,
            'saved_houses': int(np.sum(obs['houses'] == 0)),
            'ruined_houses': int(np.sum(obs['houses'] == 2)),
            'replay_path': f"replays/game_{game_idx}.json"
        }
        batch_results.append(result)

    print(f"✅ Completed {len(batch_results)} games")

    # Analyze results
    print("
📈 Results Analysis:"    rewards = [r['team_reward'] for r in batch_results]
    print(".2f")
    print(f"   Best game: {max(rewards):.2f}")
    print(f"   Worst game: {min(rewards):.2f}")

    # Fit ranking model
    print("
🤖 Fitting Ranking Model..."    model = AgentRankingModel(regularization_lambda=1.0)
    ranking_result = model.fit(batch_results)

    print("✅ Ranking model fitted")

    # Display agent rankings
    agent_rankings = model.get_agent_rankings(ranking_result)

    print("
🏆 Agent Rankings:"    for ranking in agent_rankings:
        agent_name = agents[ranking['agent_id']].name
        print(f"   #{ranking['rank']} {agent_name}: "
              ".3f"
              ".3f")

    # Save results
    results_file = output_path / "results.json"
    results_data = {
        'scenario': scenario_name,
        'parameters': {
            'beta': scenario.beta,
            'kappa': scenario.kappa,
            'c': scenario.c,
            'rho_ignite': scenario.rho_ignite
        },
        'agents': [{'id': i, 'name': agent.name} for i, agent in enumerate(agents)],
        'games': batch_results,
        'rankings': agent_rankings,
        'summary': {
            'num_games': len(batch_results),
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
        }
    }

    import json
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_file}")

    # Interpretation
    print("
🎯 Scenario Interpretation:"    if scenario_name == 'trivial_cooperation':
        print("   Expected: Firefighters should rank higher than Free Riders")
        firefighters_rank = next((r['rank'] for r in agent_rankings
                                if 'firefighter' in agents[r['agent_id']].name.lower()), None)
        free_riders_rank = next((r['rank'] for r in agent_rankings
                               if 'free_rider' in agents[r['agent_id']].name.lower()), None)
        if firefighters_rank and free_riders_rank and firefighters_rank < free_riders_rank:
            print("   ✅ Ranking correctly identifies cooperative behavior")
        else:
            print("   ⚠️ Ranking may not be capturing the cooperation signal")

    elif scenario_name == 'greedy_neighbor':
        print("   Expected: Mixed results - coordinators may outperform pure free riders")
        coordinators = [r for r in agent_rankings
                       if 'coordinator' in agents[r['agent_id']].name.lower()]
        free_riders = [r for r in agent_rankings
                      if 'free_rider' in agents[r['agent_id']].name.lower()]
        if coordinators and free_riders:
            avg_coord_rank = np.mean([r['rank'] for r in coordinators])
            avg_freeloader_rank = np.mean([r['rank'] for r in free_riders])
            if avg_coord_rank < avg_freeloader_rank:
                print("   ✅ Ranking captures social dilemma dynamics")
            else:
                print("   ⚠️ Results may not reflect cooperation incentives")

    print(f"\n🎉 Scenario {scenario_name} testing complete!")


def main(
    scenario: str = typer.Argument(..., help="Scenario name to test"),
    num_games: int = typer.Option(20, help="Number of games to run"),
    output_dir: str = typer.Option("scenario_test", help="Output directory")
):
    """Test a specific scenario with mixed agent strategies."""
    run_scenario_test(scenario, num_games, output_dir)


if __name__ == "__main__":
    typer.run(main)
