#!/usr/bin/env python3
"""
Evaluate a trained Bucket Brigade policy.

This script loads a trained policy and evaluates its performance
over multiple episodes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from train_simple import PolicyNetwork
from bucket_brigade.envs.puffer_env import PufferBucketBrigade


def evaluate_policy(policy, env, num_episodes=100, render=False):
    """Evaluate policy over multiple episodes."""

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.FloatTensor(obs)

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            with torch.no_grad():
                actions, _, _ = policy.get_action(obs.unsqueeze(0), deterministic=True)
                actions_list = [a.item() for a in actions]

            obs, reward, terminated, truncated, info = env.step(actions_list)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            obs = torch.FloatTensor(obs)

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward (last 10): {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.1f}"
            )

    return episode_rewards, episode_lengths


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate trained Bucket Brigade policy"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/bucket_brigade_policy.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--num-opponents", type=int, default=3, help="Number of opponent agents"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--scenario",
        type=str,
        default="default",
        help="Scenario name for evaluation (e.g., 'default', 'trivial_cooperation')",
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    print(f"📂 Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path)

    obs_dim = checkpoint["obs_dim"]
    action_dims = checkpoint["action_dims"]
    hidden_size = checkpoint["hidden_size"]

    policy = PolicyNetwork(obs_dim, action_dims, hidden_size=hidden_size)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dims: {action_dims}")
    print(f"   Hidden size: {hidden_size}")

    # Create environment with scenario
    from bucket_brigade.envs import get_scenario_by_name

    print(f"\n🎮 Creating environment")
    print(f"   Scenario: {args.scenario}")
    print(f"   Number of opponents: {args.num_opponents}")

    scenario = get_scenario_by_name(args.scenario, num_agents=args.num_opponents + 1)
    env = PufferBucketBrigade(scenario=scenario, num_opponents=args.num_opponents)

    # Evaluate
    print(f"\n🔍 Evaluating for {args.num_episodes} episodes...")
    episode_rewards, episode_lengths = evaluate_policy(
        policy, env, num_episodes=args.num_episodes, render=args.render
    )

    # Print statistics
    print(f"\n📊 Evaluation Results:")
    print(
        f"   Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(f"   Min Reward: {np.min(episode_rewards):.2f}")
    print(f"   Max Reward: {np.max(episode_rewards):.2f}")
    print(
        f"   Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )


if __name__ == "__main__":
    main()
