#!/usr/bin/env python3
"""
Simple training script for Bucket Brigade using vanilla PPO.

This script trains a policy to play Bucket Brigade using a basic PPO implementation
that works with the current environment setup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time

from bucket_brigade.envs.puffer_env import PufferBucketBrigade


class PolicyNetwork(nn.Module):
    """Simple policy network for discrete action spaces."""

    def __init__(self, obs_dim, action_dims, hidden_size=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Separate heads for each action dimension
        self.action_heads = nn.ModuleList(
            [nn.Linear(hidden_size, dim) for dim in action_dims]
        )

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.shared(x)

        # Get logits for each action dimension
        action_logits = [head(features) for head in self.action_heads]

        # Get value
        value = self.value_head(features)

        return action_logits, value

    def get_action(self, x, deterministic=False):
        action_logits, value = self.forward(x)

        actions = []
        log_probs = []

        for logits in action_logits:
            probs = torch.softmax(logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))

            actions.append(action)

        return actions, torch.stack(log_probs).sum(0) if log_probs else None, value


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return advantages


def train_ppo(
    env,
    policy,
    optimizer,
    num_steps=1000000,
    batch_size=2048,
    num_epochs=4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    eval_interval=10000,
):
    """Train the policy using PPO."""

    obs, _ = env.reset()
    obs = torch.FloatTensor(obs)

    episode_rewards = deque(maxlen=100)
    current_episode_reward = 0

    global_step = 0

    print(f"🚀 Starting training for {num_steps:,} steps")
    print(f"📊 Batch size: {batch_size}, Epochs: {num_epochs}")

    start_time = time.time()

    while global_step < num_steps:
        # Collect trajectories
        observations = []
        actions_taken = []
        log_probs_old = []
        rewards = []
        dones = []
        values = []

        for _ in range(batch_size):
            with torch.no_grad():
                actions, log_prob, value = policy.get_action(obs.unsqueeze(0))
                actions_list = [a.item() for a in actions]

            observations.append(obs)
            actions_taken.append(actions_list)
            log_probs_old.append(log_prob)
            values.append(value.squeeze())

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions_list)
            done = terminated or truncated

            rewards.append(reward)
            dones.append(done)
            current_episode_reward += reward

            obs = torch.FloatTensor(next_obs)
            global_step += 1

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                obs, _ = env.reset()
                obs = torch.FloatTensor(obs)

        # Compute advantages
        values_np = torch.stack(values).detach().numpy()
        advantages = compute_gae(rewards, values_np, dones)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values_np)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_batch = torch.stack(observations)
        actions_batch = torch.LongTensor(actions_taken)
        log_probs_old = torch.stack(log_probs_old)

        # PPO update
        for epoch in range(num_epochs):
            # Forward pass
            action_logits, values_pred = policy(obs_batch)

            # Compute log probs for taken actions
            log_probs_new = []
            entropies = []

            for i, logits in enumerate(action_logits):
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs_new.append(dist.log_prob(actions_batch[:, i]))
                entropies.append(dist.entropy())

            log_probs_new = torch.stack(log_probs_new).sum(0)
            entropy = torch.stack(entropies).mean()

            # PPO loss
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values_pred.squeeze(), returns)

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

        # Logging
        if global_step % eval_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0

            print(
                f"Step {global_step:,}/{num_steps:,} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"FPS: {fps:.0f} | "
                f"Episodes: {len(episode_rewards)}"
            )

    print(
        f"\n✅ Training complete! Total time: {(time.time() - start_time)/60:.1f} minutes"
    )

    return policy


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Bucket Brigade policy with PPO")
    parser.add_argument(
        "--num-opponents", type=int, default=2, help="Number of opponent agents"
    )
    parser.add_argument(
        "--num-steps", type=int, default=100000, help="Total training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for training"
    )
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/simple_policy.pt",
        help="Path to save trained model",
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    print(f"🎮 Creating environment with {args.num_opponents} opponents")
    env = PufferBucketBrigade(num_opponents=args.num_opponents)

    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()

    print(f"🧠 Creating policy network")
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dims: {action_dims}")

    policy = PolicyNetwork(obs_dim, action_dims, hidden_size=args.hidden_size)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Train
    policy = train_ppo(
        env=env,
        policy=policy,
        optimizer=optimizer,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
    )

    # Save model
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "action_dims": action_dims,
            "hidden_size": args.hidden_size,
        },
        args.save_path,
    )

    print(f"💾 Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
