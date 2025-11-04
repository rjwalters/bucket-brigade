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
