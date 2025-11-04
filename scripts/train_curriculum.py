#!/usr/bin/env python3
"""
Train agents with a curriculum of increasingly difficult scenarios.

This implements curriculum learning where agents start with simple scenarios
and progressively learn more complex strategies. The curriculum automatically
adapts based on performance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from bucket_brigade.envs import get_scenario_by_name
from bucket_brigade.envs.puffer_env import PufferBucketBrigade
from bucket_brigade.training import CurriculumTrainer, PolicyNetwork


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


def evaluate_policy(policy, env, num_episodes=10):
    """Evaluate policy performance."""
    total_reward = 0
    policy.eval()

    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.FloatTensor(obs)
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                actions, _, _ = policy.get_action(obs.unsqueeze(0), deterministic=True)
                actions_list = [a.item() for a in actions]

            obs, reward, terminated, truncated, _ = env.step(actions_list)
            episode_reward += reward
            done = terminated or truncated
            obs = torch.FloatTensor(obs)

        total_reward += episode_reward

    policy.train()
    return total_reward / num_episodes


def train_stage_ppo(
    env,
    policy,
    optimizer,
    num_steps,
    batch_size=2048,
    num_epochs=4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    writer=None,
    global_step_offset=0,
):
    """Train the policy using PPO for a single curriculum stage."""

    obs, _ = env.reset()
    obs = torch.FloatTensor(obs)

    episode_rewards = deque(maxlen=100)
    current_episode_reward = 0

    global_step = global_step_offset

    while global_step - global_step_offset < num_steps:
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
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

        # TensorBoard logging
        if writer is not None and global_step % 100 == 0:
            writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("train/value_loss", value_loss.item(), global_step)
            writer.add_scalar("train/entropy", entropy.item(), global_step)
            writer.add_scalar("train/total_loss", loss.item(), global_step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
            if episode_rewards:
                writer.add_scalar(
                    "episode/mean_reward", np.mean(episode_rewards), global_step
                )
                writer.add_scalar(
                    "episode/max_reward", np.max(episode_rewards), global_step
                )
                writer.add_scalar(
                    "episode/min_reward", np.min(episode_rewards), global_step
                )

    return global_step, np.mean(episode_rewards) if episode_rewards else 0


def main():
    parser = argparse.ArgumentParser(
        description="Train Bucket Brigade agent with curriculum learning"
    )

    # Curriculum settings
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run (default: auto-generated)",
    )
    parser.add_argument(
        "--custom-curriculum",
        action="store_true",
        help="Use custom curriculum (modify in code)",
    )

    # Model architecture
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden layer size",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for PPO",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=4,
        help="PPO epochs per update",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        help="PPO clip coefficient",
    )
    parser.add_argument(
        "--value-coef",
        type=float,
        default=0.5,
        help="Value loss coefficient",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create and run curriculum trainer
    trainer = CurriculumTrainer(args)
    trainer.train_curriculum()


if __name__ == "__main__":
    main()
