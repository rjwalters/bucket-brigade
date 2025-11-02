#!/usr/bin/env python3
"""
Evaluate trained policies against expert agents.

This script loads trained models and evaluates them in tournament-style
competitions against the expert agents we developed.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add the bucket_brigade package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.envs import make_env
from bucket_brigade.orchestration import AgentRankingModel
import pufferlib.models


class TrainedAgent:
    """Wrapper for a trained PufferLib policy."""

    def __init__(self, policy, device='cpu'):
        self.policy = policy
        self.device = device
        self.id = 0  # Will be set when added to tournament
        self.name = "TrainedAgent"

    def act(self, obs):
        """Get action from trained policy."""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_logits, _ = self.policy(obs_tensor)

            # Sample action (could also use argmax for deterministic)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample().squeeze().numpy()

        # Convert back to [house, mode] format
        house_idx = action // 2  # First dimension (0-9)
        mode = action % 2        # Second dimension (0-1)

        return [int(house_idx), int(mode)]

    def reset(self):
        """Reset agent state (no-op for trained policies)."""
        pass


def load_trained_policy(model_path: str, env, device: str = 'cpu'):
    """Load a trained policy from disk."""
    policy = pufferlib.models.PolicyValueNetwork(
        env.observation_space.shape[0],
        env.action_space.nvec.sum(),
        policy_channels=64,
        value_channels=64,
        policy_layers=3,
        value_layers=3,
    )

    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    return TrainedAgent(policy, device)


def evaluate_policy_tournament(
    model_path: str,
    scenario_name: str = 'default',
    num_games: int = 50,
    num_opponents: int = 3,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Evaluate a trained policy in a tournament against expert agents.

    Returns comprehensive evaluation results.
    """
    print(f"🎯 Evaluating policy: {model_path}")
    print(f"🏟️ Scenario: {scenario_name}")
    print(f"🎮 Games: {num_games}")
    print()

    # Create environment for policy loading
    env = make_env(scenario_name, num_opponents)

    # Load trained policy
    trained_agent = load_trained_policy(model_path, env, device)
    trained_agent.name = f"Trained-{Path(model_path).stem}"

    # Create tournament setup with expert agents
    from bucket_brigade.agents import create_archetype_agent, create_random_agent

    # Create pool of expert agents (excluding the trained one)
    expert_policies = ['firefighter', 'coordinator', 'free_rider', 'liar', 'hero']
    expert_agents = []

    for i, policy in enumerate(expert_policies):
        if policy == 'random':
            agent = create_random_agent(i + 1)
        else:
            agent = create_archetype_agent(policy, i + 1)
        expert_agents.append(agent)

    # Run tournament
    results = []
    trained_agent.id = 0  # Trained agent is always ID 0

    print("🏁 Running tournament...")

    for game_idx in range(num_games):
        # Select random opponents for this game
        opponent_indices = np.random.choice(len(expert_agents), num_opponents, replace=False)
        opponents = [expert_agents[i] for i in opponent_indices]

        # Set agent IDs
        for i, opp in enumerate(opponents):
            opp.id = i + 1

        # Run game
        game_result = run_single_game(env, trained_agent, opponents)
        results.append(game_result)

        if (game_idx + 1) % 10 == 0:
            print(f"  Completed {game_idx + 1}/{num_games} games")

    # Analyze results
    trained_rewards = [r['agent_rewards'][0] for r in results]  # Trained agent is always index 0
    total_scores = [sum(r['agent_rewards']) for r in results]

    analysis = {
        'model_path': model_path,
        'scenario': scenario_name,
        'num_games': num_games,
        'num_opponents': num_opponents,

        # Trained agent performance
        'trained_agent': {
            'name': trained_agent.name,
            'mean_reward': float(np.mean(trained_rewards)),
            'std_reward': float(np.std(trained_rewards)),
            'min_reward': float(np.min(trained_rewards)),
            'max_reward': float(np.max(trained_rewards)),
            'win_rate': float(np.mean([1 if r > 0 else 0 for r in trained_rewards])),
        },

        # Opponent performance summary
        'opponent_performance': {
            'mean_total_reward': float(np.mean(total_scores)),
            'std_total_reward': float(np.std(total_scores)),
        },

        # Individual game results
        'game_results': results,

        # Ranking analysis
        'ranking_analysis': analyze_agent_rankings(results, trained_agent.name)
    }

    print("
📊 Results Summary:"    print(".3f"    print(".3f"    print(".3f"    print(".1f"    print(".3f"
    return analysis


def run_single_game(env, trained_agent, opponents):
    """Run a single game with the trained agent and opponents."""
    # Reset all agents
    trained_agent.reset()
    for opp in opponents:
        opp.reset()

    # Reset environment
    obs = env.reset()

    # Track rewards
    agent_rewards = np.zeros(len(opponents) + 1)

    while not env.done:
        # Get trained agent action
        trained_obs = env.get_observation(0)  # Trained agent is ID 0
        trained_action = trained_agent.act(trained_obs)

        # Get opponent actions
        all_actions = [trained_action]
        for opp in opponents:
            opp_obs = env.get_observation(opp.id)
            opp_action = opp.act(opp_obs)
            all_actions.append(opp_action)

        # Step environment
        obs, rewards, dones, info = env.step(np.array(all_actions))

        # Accumulate rewards
        for i, reward in enumerate(rewards):
            agent_rewards[i] += reward

    return {
        'scenario': env.scenario.__dict__,
        'agent_rewards': agent_rewards.tolist(),
        'nights_played': env.night,
        'final_houses': obs['houses'].tolist(),
        'opponent_types': [opp.name for opp in opponents]
    }


def analyze_agent_rankings(results, trained_agent_name):
    """Analyze how the trained agent ranks against opponents."""
    # Create batch results format for ranking analysis
    batch_results = []

    for game_idx, result in enumerate(results):
        agent_rewards = result['agent_rewards']

        # Create fake agent parameters (not used in ranking)
        agent_params = [[0.5] * 10 for _ in agent_rewards]

        batch_results.append({
            'game_id': game_idx,
            'scenario_id': 0,
            'team': list(range(len(agent_rewards))),
            'agent_params': agent_params,
            'team_reward': sum(agent_rewards),
            'agent_rewards': agent_rewards,
            'nights_played': result['nights_played'],
            'saved_houses': sum(1 for h in result['final_houses'] if h == 0),
            'ruined_houses': sum(1 for h in result['final_houses'] if h == 2),
            'replay_path': f"replays/game_{game_idx}.json"
        })

    # Fit ranking model
    model = AgentRankingModel(regularization_lambda=1.0)
    ranking_result = model.fit(batch_results)

    # Get rankings
    agent_rankings = model.get_agent_rankings(ranking_result)

    # Find trained agent ranking
    trained_ranking = next((r for r in agent_rankings if r['rank'] == 1), None)  # Assume first is trained

    return {
        'rankings': agent_rankings,
        'trained_agent_rank': trained_ranking['rank'] if trained_ranking else None,
        'trained_agent_score': trained_ranking['skill_estimate'] if trained_ranking else None,
        'ranking_model_log_likelihood': ranking_result.log_likelihood
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Bucket Brigade policy')
    parser.add_argument('model_path', help='Path to trained model file')
    parser.add_argument('--scenario', type=str, default='default',
                       choices=['default', 'trivial_cooperation', 'early_containment',
                               'greedy_neighbor', 'sparse_heroics'],
                       help='Evaluation scenario')
    parser.add_argument('--num-games', type=int, default=50,
                       help='Number of evaluation games')
    parser.add_argument('--num-opponents', type=int, default=3,
                       help='Number of opponent agents per game')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run evaluation on')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON)')

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)

    # Run evaluation
    results = evaluate_policy_tournament(
        args.model_path,
        args.scenario,
        args.num_games,
        args.num_opponents,
        args.device
    )

    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output}")

    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
