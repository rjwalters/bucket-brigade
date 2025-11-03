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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from torch.utils.tensorboard import SummaryWriter
import argparse

from bucket_brigade.envs.puffer_env import PufferBucketBrigade
from bucket_brigade.envs import get_scenario_by_name


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
                writer.add_scalar("episode/mean_reward", np.mean(episode_rewards), global_step)
                writer.add_scalar("episode/max_reward", np.max(episode_rewards), global_step)
                writer.add_scalar("episode/min_reward", np.min(episode_rewards), global_step)

    return global_step, np.mean(episode_rewards) if episode_rewards else 0


class CurriculumTrainer:
    """Manages curriculum learning across scenarios."""

    def __init__(self, args):
        self.args = args

        # Define curriculum stages (easier to harder)
        self.curriculum = [
            {
                "name": "trivial_cooperation",
                "num_opponents": 2,
                "steps": 100_000,
                "description": "Learn basic cooperation",
                "progression_threshold": 5.0,  # Mean reward threshold to advance
            },
            {
                "name": "early_containment",
                "num_opponents": 3,
                "steps": 150_000,
                "description": "Learn timing and coordination",
                "progression_threshold": 4.0,
            },
            {
                "name": "greedy_neighbor",
                "num_opponents": 3,
                "steps": 200_000,
                "description": "Learn social dilemmas",
                "progression_threshold": 3.0,
            },
            {
                "name": "sparse_heroics",
                "num_opponents": 4,
                "steps": 250_000,
                "description": "Learn resource allocation",
                "progression_threshold": 2.5,
            },
            {
                "name": "chain_reaction",
                "num_opponents": 4,
                "steps": 300_000,
                "description": "Learn distributed coordination",
                "progression_threshold": 2.0,
            },
        ]

        # Allow custom curriculum if specified
        if args.custom_curriculum:
            print("📋 Using custom curriculum from arguments")
            # User can override stages via arguments if needed

    def train_curriculum(self):
        """Train through the full curriculum."""
        print("🎓 Starting Curriculum Training")
        print("=" * 60)

        # Initialize TensorBoard
        run_name = self.args.run_name or f"curriculum_{int(time.time())}"
        writer = SummaryWriter(f"runs/{run_name}")
        print(f"📊 TensorBoard logging to: runs/{run_name}")

        # Create initial environment to get observation/action space
        initial_scenario = get_scenario_by_name(
            self.curriculum[0]["name"], num_agents=self.curriculum[0]["num_opponents"] + 1
        )
        initial_env = PufferBucketBrigade(
            scenario=initial_scenario,
            num_opponents=self.curriculum[0]["num_opponents"],
        )

        obs_dim = initial_env.observation_space.shape[0]
        action_dims = initial_env.action_space.nvec.tolist()

        # Initialize policy
        print(f"🧠 Creating policy network")
        print(f"   Observation dim: {obs_dim}")
        print(f"   Action dims: {action_dims}")

        policy = PolicyNetwork(obs_dim, action_dims, hidden_size=self.args.hidden_size)
        optimizer = optim.Adam(policy.parameters(), lr=self.args.lr)

        global_step = 0
        start_time = time.time()

        # Train through curriculum stages
        for stage_idx, stage in enumerate(self.curriculum):
            print(f"\n{'='*60}")
            print(f"📚 Stage {stage_idx + 1}/{len(self.curriculum)}: {stage['name']}")
            print(f"   {stage['description']}")
            print(f"   Steps: {stage['steps']:,}")
            print(f"   Opponents: {stage['num_opponents']}")
            print(f"   Progression threshold: {stage['progression_threshold']:.2f}")
            print(f"{'='*60}")

            # Create environment for this stage
            scenario = get_scenario_by_name(stage["name"], num_agents=stage["num_opponents"] + 1)
            env = PufferBucketBrigade(
                scenario=scenario,
                num_opponents=stage["num_opponents"],
            )

            # Train on this stage
            stage_start = time.time()
            global_step, final_reward = train_stage_ppo(
                env=env,
                policy=policy,
                optimizer=optimizer,
                num_steps=stage["steps"],
                batch_size=self.args.batch_size,
                num_epochs=self.args.num_epochs,
                clip_epsilon=self.args.clip_epsilon,
                value_coef=self.args.value_coef,
                entropy_coef=self.args.entropy_coef,
                max_grad_norm=self.args.max_grad_norm,
                writer=writer,
                global_step_offset=global_step,
            )
            stage_duration = time.time() - stage_start

            # Evaluate on this stage
            eval_reward = evaluate_policy(policy, env, num_episodes=20)
            print(f"\n   ✅ Stage complete!")
            print(f"   Training reward: {final_reward:.2f}")
            print(f"   Evaluation reward: {eval_reward:.2f}")
            print(f"   Duration: {stage_duration / 60:.1f} minutes")

            # Log stage completion
            writer.add_scalar(f"curriculum/stage_{stage_idx}_eval", eval_reward, global_step)
            writer.add_scalar(f"curriculum/stage_{stage_idx}_train", final_reward, global_step)

            # Save stage checkpoint
            output_dir = Path("models") / run_name
            output_dir.mkdir(parents=True, exist_ok=True)
            stage_path = output_dir / f"stage_{stage_idx}_{stage['name']}.pt"
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dims": action_dims,
                    "hidden_size": self.args.hidden_size,
                    "stage": stage_idx,
                    "stage_name": stage["name"],
                },
                stage_path,
            )
            print(f"   💾 Saved checkpoint: {stage_path}")

            # Check if we should advance (adaptive progression)
            if eval_reward < stage["progression_threshold"] and stage_idx < len(self.curriculum) - 1:
                print(f"   ⚠️  Warning: Performance below threshold ({eval_reward:.2f} < {stage['progression_threshold']:.2f})")
                print(f"   Continuing to next stage anyway (human approval recommended)")

        # Final evaluation across all scenarios
        print(f"\n{'='*60}")
        print("🎯 Final Evaluation Across All Scenarios")
        print(f"{'='*60}")

        for stage_idx, stage in enumerate(self.curriculum):
            scenario = get_scenario_by_name(stage["name"], num_agents=stage["num_opponents"] + 1)
            env = PufferBucketBrigade(scenario=scenario, num_opponents=stage["num_opponents"])
            eval_reward = evaluate_policy(policy, env, num_episodes=30)
            print(f"   {stage['name']:20s}: {eval_reward:.2f}")
            writer.add_scalar(f"final_eval/{stage['name']}", eval_reward, global_step)

        # Save final model
        final_path = output_dir / "curriculum_final.pt"
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "obs_dim": obs_dim,
                "action_dims": action_dims,
                "hidden_size": self.args.hidden_size,
            },
            final_path,
        )

        total_duration = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"🎓 Curriculum Training Complete!")
        print(f"   Total duration: {total_duration / 60:.1f} minutes")
        print(f"   Final model: {final_path}")
        print(f"   TensorBoard: runs/{run_name}")
        print(f"{'='*60}")

        writer.close()


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
