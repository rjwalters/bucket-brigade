"""
GPU Policy Learner for Population-Based Training.

This module implements the GPU-side policy learner that:
1. Receives experience batches from CPU game simulator
2. Trains agent policy using PPO
3. Sends updated policy weights back to CPU simulator
4. Maintains replay buffer for on-policy learning

Architecture:
    CPU Simulator → Experience Queue → GPU Learner (this)
    GPU Learner (this) → Policy Update Queue → CPU Simulator
"""

import multiprocessing as mp
from typing import Optional, List, Tuple
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from bucket_brigade.training import PolicyNetwork, compute_gae


class PolicyLearner:
    """
    GPU-based policy learner for population training.

    Each learner trains one agent's policy using PPO with on-policy learning.
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dims: List[int],
        hidden_size: int = 512,
        learning_rate: float = 3e-4,
        device: str = "cuda",
        experience_queue: Optional[mp.Queue] = None,
        policy_update_queue: Optional[mp.Queue] = None,
        # PPO hyperparameters
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        # Training parameters
        batch_size: int = 256,
        num_epochs: int = 4,
        update_interval: int = 100,  # Send policy to CPU every N batches
    ):
        """
        Initialize the policy learner.

        Args:
            agent_id: ID of the agent this learner is training
            obs_dim: Observation dimension
            action_dims: List of action dimensions [num_houses, num_modes]
            hidden_size: Hidden layer size
            learning_rate: Learning rate
            device: Device to use ("cuda" or "cpu")
            experience_queue: Queue for receiving experiences from CPU
            policy_update_queue: Queue for sending policy updates to CPU
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Max gradient norm for clipping
            batch_size: Batch size for training
            num_epochs: Number of PPO epochs per batch
            update_interval: Send policy update every N batches
        """
        self.agent_id = agent_id
        self.device = torch.device(device)

        # Queues
        self.experience_queue = experience_queue
        self.policy_update_queue = policy_update_queue

        # Policy network
        self.policy = PolicyNetwork(
            obs_dim=obs_dim,
            action_dims=action_dims,
            hidden_size=hidden_size,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Training parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.update_interval = update_interval

        # Experience buffer (for on-policy learning)
        self.experience_buffer = deque(maxlen=batch_size * 2)

        # Statistics
        self.total_batches = 0
        self.total_updates = 0
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []

    def get_policy_state(self) -> dict:
        """Get current policy state dict (for sending to CPU)."""
        return self.policy.state_dict()

    def collect_experiences(self, min_size: int) -> bool:
        """
        Collect experiences from queue until buffer has at least min_size.

        Args:
            min_size: Minimum number of experiences to collect

        Returns:
            True if enough experiences collected, False if timeout
        """
        timeout = 10.0  # seconds
        start = time.time()

        while len(self.experience_buffer) < min_size:
            if time.time() - start > timeout:
                return False

            if self.experience_queue and not self.experience_queue.empty():
                agent_id, experience = self.experience_queue.get()

                # Only accept experiences for this agent (on-policy)
                if agent_id == self.agent_id:
                    self.experience_buffer.append(experience)
            else:
                time.sleep(0.01)  # Small sleep to prevent busy waiting

        return True

    def prepare_batch(self) -> Tuple[torch.Tensor, ...]:
        """
        Prepare a batch of experiences for training.

        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones, old_logprobs)
        """
        # Sample batch from buffer
        batch_size = min(self.batch_size, len(self.experience_buffer))
        batch = [self.experience_buffer.popleft() for _ in range(batch_size)]

        # Convert to tensors
        observations = torch.FloatTensor(np.array([exp['obs'] for exp in batch])).to(self.device)
        actions = torch.LongTensor(np.array([exp['action'] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([float(exp['done']) for exp in batch]).to(self.device)
        old_logprobs = torch.FloatTensor([exp['logprob'] for exp in batch]).to(self.device)

        # Handle next observations (may be None if done)
        next_observations = []
        for exp in batch:
            if exp['next_obs'] is not None:
                next_observations.append(exp['next_obs'])
            else:
                next_observations.append(np.zeros_like(exp['obs']))
        next_observations = torch.FloatTensor(np.array(next_observations)).to(self.device)

        return observations, actions, rewards, next_observations, dones, old_logprobs

    def compute_ppo_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PPO loss.

        Args:
            observations: Batch of observations
            actions: Batch of actions [house, mode]
            old_logprobs: Old log probabilities
            advantages: Computed advantages
            returns: Computed returns

        Returns:
            Tuple of (total_loss, policy_loss, value_loss, entropy)
        """
        # Forward pass
        action_logits, values = self.policy(observations)
        values = values.squeeze(-1)

        # Calculate new log probabilities
        house_probs = torch.softmax(action_logits[0], dim=-1)
        mode_probs = torch.softmax(action_logits[1], dim=-1)

        house_dist = torch.distributions.Categorical(house_probs)
        mode_dist = torch.distributions.Categorical(mode_probs)

        house_logprobs = house_dist.log_prob(actions[:, 0])
        mode_logprobs = mode_dist.log_prob(actions[:, 1])
        new_logprobs = house_logprobs + mode_logprobs

        # PPO policy loss
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)

        # Entropy bonus (for exploration)
        entropy = (house_dist.entropy() + mode_dist.entropy()).mean()

        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        return total_loss, policy_loss, value_loss, entropy

    def train_batch(self) -> dict:
        """
        Train on a batch of experiences.

        Returns:
            Dictionary of training statistics
        """
        # Prepare batch
        observations, actions, rewards, next_observations, dones, old_logprobs = self.prepare_batch()

        # Compute values for GAE
        with torch.no_grad():
            _, values = self.policy(observations)
            values = values.squeeze(-1)

            _, next_values = self.policy(next_observations)
            next_values = next_values.squeeze(-1)
            next_values = next_values * (1 - dones)  # Zero out if done

        # Compute advantages using GAE
        advantages = torch.tensor(
            compute_gae(
                rewards.cpu().tolist(),
                values.cpu().tolist(),
                dones.cpu().tolist(),
                self.gamma,
                self.gae_lambda,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []

        for epoch in range(self.num_epochs):
            loss, policy_loss, value_loss, entropy = self.compute_ppo_loss(
                observations, actions, old_logprobs, advantages, returns
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Record
            epoch_losses.append(loss.item())
            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())
            epoch_entropies.append(entropy.item())

        self.total_batches += 1

        # Record statistics
        mean_policy_loss = np.mean(epoch_policy_losses)
        mean_value_loss = np.mean(epoch_value_losses)
        mean_entropy = np.mean(epoch_entropies)

        self.policy_loss_history.append(mean_policy_loss)
        self.value_loss_history.append(mean_value_loss)
        self.entropy_history.append(mean_entropy)

        return {
            'batch': self.total_batches,
            'policy_loss': mean_policy_loss,
            'value_loss': mean_value_loss,
            'entropy': mean_entropy,
            'mean_reward': rewards.mean().item(),
        }

    def send_policy_update(self):
        """Send updated policy weights to CPU simulator."""
        if self.policy_update_queue:
            state_dict = self.get_policy_state()
            self.policy_update_queue.put((self.agent_id, state_dict))
            self.total_updates += 1

    def train(self, max_batches: Optional[int] = None):
        """
        Main training loop.

        Args:
            max_batches: Maximum number of batches to train (None for infinite)
        """
        print(f"\n{'='*60}")
        print(f"🧠 Starting Policy Learner for Agent {self.agent_id}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"PPO epochs: {self.num_epochs}")
        print(f"Update interval: {self.update_interval}")
        print(f"{'='*60}\n")

        start_time = time.time()
        last_report = start_time

        while max_batches is None or self.total_batches < max_batches:
            # Collect experiences
            if not self.collect_experiences(self.batch_size):
                print(f"[Agent {self.agent_id}] Timeout waiting for experiences")
                time.sleep(1.0)
                continue

            # Train on batch
            stats = self.train_batch()

            # Periodically send policy update to CPU
            if self.total_batches % self.update_interval == 0:
                self.send_policy_update()

                # Report progress
                elapsed = time.time() - last_report
                batches_per_sec = self.update_interval / elapsed

                print(f"[Agent {self.agent_id}] Batch {stats['batch']:,} | "
                      f"Policy Loss: {stats['policy_loss']:.4f} | "
                      f"Value Loss: {stats['value_loss']:.4f} | "
                      f"Entropy: {stats['entropy']:.4f} | "
                      f"Reward: {stats['mean_reward']:.3f} | "
                      f"Speed: {batches_per_sec:.1f} batch/s")

                last_report = time.time()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Training Complete for Agent {self.agent_id}!")
        print(f"Total batches: {self.total_batches:,}")
        print(f"Total policy updates sent: {self.total_updates}")
        print(f"Time: {total_time/60:.1f} minutes")
        print(f"Speed: {self.total_batches/total_time:.1f} batches/s")
        print(f"{'='*60}\n")

    def get_statistics(self) -> dict:
        """Get training statistics."""
        return {
            'total_batches': self.total_batches,
            'total_updates': self.total_updates,
            'policy_loss_history': self.policy_loss_history,
            'value_loss_history': self.value_loss_history,
            'entropy_history': self.entropy_history,
        }


def learner_process(
    agent_id: int,
    obs_dim: int,
    action_dims: List[int],
    hidden_size: int,
    learning_rate: float,
    device: str,
    experience_queue: mp.Queue,
    policy_update_queue: mp.Queue,
    max_batches: Optional[int] = None,
    **kwargs,
):
    """
    Entry point for running a learner in a separate process.

    Args:
        agent_id: ID of agent to train
        obs_dim: Observation dimension
        action_dims: Action dimensions
        hidden_size: Hidden layer size
        learning_rate: Learning rate
        device: Device to use
        experience_queue: Queue for receiving experiences
        policy_update_queue: Queue for sending policy updates
        max_batches: Maximum batches to train
        **kwargs: Additional PolicyLearner parameters
    """
    learner = PolicyLearner(
        agent_id=agent_id,
        obs_dim=obs_dim,
        action_dims=action_dims,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        device=device,
        experience_queue=experience_queue,
        policy_update_queue=policy_update_queue,
        **kwargs,
    )

    learner.train(max_batches=max_batches)
