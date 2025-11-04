"""Policy network architectures for reinforcement learning.

This module provides neural network architectures for multi-discrete action
spaces, commonly used in reinforcement learning scenarios.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Actor-critic policy network with multi-discrete action space.

    This network architecture uses shared feature extraction layers with
    separate heads for each action dimension and a value estimation head.
    It's designed for environments with multi-discrete action spaces where
    each action dimension can have different sizes.

    Architecture:
        - Shared feature extraction: 2 fully-connected layers with ReLU
        - Action heads: One linear layer per action dimension
        - Value head: Single linear layer for state value estimation

    Args:
        obs_dim: Dimension of the observation space
        action_dims: List of action space sizes for each action dimension
        hidden_size: Size of hidden layers (default: 64)

    Example:
        >>> # Create policy for 42-dim observation, 2 action dimensions
        >>> policy = PolicyNetwork(obs_dim=42, action_dims=[10, 2], hidden_size=128)
        >>> obs_batch = torch.randn(16, 42)  # Batch of 16 observations
        >>> action_logits, value = policy(obs_batch)
        >>> len(action_logits)  # 2 action heads
        2
        >>> action_logits[0].shape  # First action dimension
        torch.Size([16, 10])
        >>> value.shape  # State value estimates
        torch.Size([16, 1])
    """

    def __init__(
        self, obs_dim: int, action_dims: List[int], hidden_size: int = 64
    ) -> None:
        super().__init__()

        # Shared feature extraction layers
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

        # Value estimation head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Observation tensor of shape (batch_size, obs_dim)

        Returns:
            Tuple containing:
                - List of action logits for each action dimension
                - Value estimates of shape (batch_size, 1)
        """
        features = self.shared(x)

        # Get logits for each action dimension
        action_logits = [head(features) for head in self.action_heads]

        # Get value estimate
        value = self.value_head(features)

        return action_logits, value

    def get_action(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Sample or select actions from the policy.

        Args:
            x: Observation tensor of shape (batch_size, obs_dim)
            deterministic: If True, select argmax action. If False, sample
                from categorical distribution (default: False)

        Returns:
            Tuple containing:
                - List of action tensors (one per action dimension)
                - Log probabilities of selected actions (None if deterministic)
                - Value estimates of shape (batch_size, 1)

        Example:
            >>> policy = PolicyNetwork(obs_dim=42, action_dims=[10, 2])
            >>> obs = torch.randn(1, 42)
            >>> actions, log_prob, value = policy.get_action(obs, deterministic=True)
            >>> len(actions)  # One action per dimension
            2
            >>> log_prob is None  # No log_prob in deterministic mode
            True
        """
        action_logits, value = self.forward(x)

        actions = []
        log_probs = []

        for logits in action_logits:
            probs = torch.softmax(logits, dim=-1)

            if deterministic:
                # Select action with highest probability
                action = torch.argmax(probs, dim=-1)
            else:
                # Sample from categorical distribution
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))

            actions.append(action)

        # Sum log probabilities across action dimensions (independent actions)
        log_prob = torch.stack(log_probs).sum(0) if log_probs else None

        return actions, log_prob, value

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions and values in batch-compatible format.

        This method is designed for batch processing during training, particularly
        for GPU-accelerated environments. It computes actions, log probabilities,
        entropy, and values in a format compatible with vectorized training loops.

        Args:
            x: Observation tensor of shape (batch_size, obs_dim)
            action: Optional action tensor of shape (batch_size, num_action_dims)
                for computing log probabilities of given actions. If None, samples
                new actions (default: None)

        Returns:
            Tuple containing:
                - Actions tensor of shape (batch_size, num_action_dims)
                - Log probabilities of shape (batch_size,)
                - Entropy of shape (batch_size,)
                - Value estimates of shape (batch_size,)

        Example:
            >>> policy = PolicyNetwork(obs_dim=42, action_dims=[10, 2])
            >>> obs_batch = torch.randn(16, 42)
            >>> actions, log_probs, entropy, values = policy.get_action_and_value(obs_batch)
            >>> actions.shape
            torch.Size([16, 2])
            >>> log_probs.shape
            torch.Size([16,])
        """
        action_logits, value = self.forward(x)

        actions = []
        log_probs = []
        entropies = []

        for i, logits in enumerate(action_logits):
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if action is None:
                # Sample new actions
                a = dist.sample()
            else:
                # Use provided actions
                a = action[:, i]

            actions.append(a)
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())

        return (
            torch.stack(actions, dim=1),  # [batch_size, num_action_dims]
            torch.stack(log_probs, dim=1).sum(dim=1),  # [batch_size]
            torch.stack(entropies, dim=1).mean(dim=1),  # [batch_size]
            value.squeeze(-1),  # [batch_size]
        )


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> List[float]:
    """Compute Generalized Advantage Estimation.

    GAE provides a way to reduce variance in policy gradient estimates while
    maintaining low bias. It's a technique for computing advantage estimates
    that balances between Monte Carlo and TD estimates.

    Args:
        rewards: List of rewards for each timestep
        values: List of value estimates for each timestep
        dones: List of done flags indicating episode termination
        gamma: Discount factor for future rewards (default: 0.99)
        gae_lambda: Lambda parameter controlling bias-variance tradeoff
            (default: 0.95). λ=0 gives TD advantage, λ=1 gives Monte Carlo

    Returns:
        List of advantage estimates, same length as rewards

    Example:
        >>> rewards = [1.0, 2.0, 3.0]
        >>> values = [0.5, 1.5, 2.5]
        >>> dones = [False, False, True]
        >>> advantages = compute_gae(rewards, values, dones)
        >>> len(advantages)
        3

    Reference:
        "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
        Schulman et al., 2016
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        # Compute TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # Compute GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return advantages
