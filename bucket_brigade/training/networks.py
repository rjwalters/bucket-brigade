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

    def encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Return the shared trunk's output (Ẑ_i in the Slepian-Wolf paper).

        Used by the joint multi-agent trainer to compute cross-agent
        representational redundancy penalties. The gradient flows back into
        the shared trunk through this tap.

        Args:
            x: Observation tensor of shape (batch_size, obs_dim)

        Returns:
            Feature tensor of shape (batch_size, hidden_size).
        """
        return self.shared(x)

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


class CentralizedCritic(nn.Module):
    """Centralized state-value network for MAPPO (issue #208).

    A separate value network used by :class:`JointPPOTrainer` when
    ``centralized_critic=True``. The critic consumes the **global** part of
    the observation (i.e. the per-agent flat obs with the identity one-hot
    tail stripped) and produces a single scalar value estimate that is
    shared across all agents at training time. Per-agent advantages are
    still computed off the per-agent reward streams, so each agent gets a
    distinct learning signal (the value baseline is shared, the targets
    are not).

    This is the conservative MAPPO formulation from Yu et al. 2021,
    adapted to this codebase's particular setup where the env already
    returns a global observation (see ``joint_trainer.flatten_dict_obs``)
    --- no per-agent obs concatenation is needed.

    Architecture mirrors :class:`PolicyNetwork`'s shared trunk: a 2-layer
    MLP with ReLU, then a linear scalar head. Default hidden size matches
    the actor trunk so the critic has comparable capacity.

    Args:
        obs_dim: Dimension of the global observation input (i.e. the
            actor-side ``obs_dim`` minus the identity-one-hot tail length).
        hidden_size: Size of the hidden layers (default: 64, matching
            :class:`PolicyNetwork`).

    Example:
        >>> critic = CentralizedCritic(obs_dim=42, hidden_size=64)
        >>> obs = torch.randn(16, 42)
        >>> value = critic(obs)
        >>> value.shape
        torch.Size([16, 1])
    """

    def __init__(self, obs_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Global observation tensor of shape ``(batch_size, obs_dim)``.

        Returns:
            Value estimates of shape ``(batch_size, 1)``.
        """
        return self.value_head(self.shared(x))


class CentralizedQCritic(nn.Module):
    """Centralized action-value Q-critic for COMA (issue #284).

    A separate Q-network used by :class:`JointPPOTrainer` when ``coma=True``.
    Distinct from :class:`CentralizedCritic` (which is a V-function): COMA
    needs ``Q(s, a)`` so the counterfactual baseline

    ::

        A_i(s, a) = Q(s, a)_i - sum_u  pi_i(u | s) * Q(s, (a_{-i}, u))_i

    can be evaluated by enumerating over agent i's per-agent action space.

    Architecture (mirrors :class:`CentralizedCritic` capacity):

    - **Input**: concatenation of
        - ``global_obs`` of length ``global_obs_dim`` (the per-agent obs
          with the identity one-hot tail stripped — same convention as
          ``CentralizedCritic``),
        - ``joint_action_one_hot`` of length ``N * action_dim_total`` where
          ``action_dim_total = prod(action_dims)`` is the joint-per-agent
          action dim (recommended in the COMA spec: enumerate over the
          ``[10, 2, 2]`` = 40 joint actions),
        - ``agent_id_one_hot`` of length ``N``.
    - **Trunk**: 2-layer MLP with ReLU, hidden size ``hidden_size``.
    - **Output**: vector of length ``action_dim_total`` — the action-value
      head returns Q for every possible per-agent action holding the other
      agents' actions fixed. This lets the COMA baseline be evaluated by
      a single forward pass per agent.

    The "joint-per-agent" action representation collapses the multi-discrete
    factorization ``[10, 2, 2]`` to a flat 40-way one-hot. This is option 2
    from the curator's spec — exact (no factorization bias) and tractable
    (40 entries × 4 agents × T steps is well within budget).

    Args:
        global_obs_dim: Dimension of the global observation (identity tail
            stripped).
        num_agents: Number of agents N.
        action_dim_total: Joint per-agent action dim (e.g. ``prod([10,2,2]) = 40``).
        hidden_size: Hidden layer size (default 64, matching the actor trunk).

    Example:
        >>> critic = CentralizedQCritic(
        ...     global_obs_dim=42, num_agents=4, action_dim_total=40
        ... )
        >>> global_obs = torch.randn(8, 42)
        >>> joint_action_oh = torch.zeros(8, 4 * 40)  # N=4 one-hot actions
        >>> agent_id_oh = torch.zeros(8, 4); agent_id_oh[:, 0] = 1.0
        >>> q = critic(global_obs, joint_action_oh, agent_id_oh)
        >>> q.shape
        torch.Size([8, 40])
    """

    def __init__(
        self,
        global_obs_dim: int,
        num_agents: int,
        action_dim_total: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.global_obs_dim = global_obs_dim
        self.num_agents = num_agents
        self.action_dim_total = action_dim_total
        self.hidden_size = hidden_size

        input_dim = global_obs_dim + num_agents * action_dim_total + num_agents
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden_size, action_dim_total)

    def forward(
        self,
        global_obs: torch.Tensor,
        joint_action_one_hot: torch.Tensor,
        agent_id_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            global_obs: ``(batch, global_obs_dim)`` — the global state input.
            joint_action_one_hot: ``(batch, N * action_dim_total)`` — flat
                one-hot encoding of the joint per-agent action. The slot
                belonging to ``agent_id_one_hot`` is conventionally zeroed
                by the caller (it does not matter for COMA because the
                baseline averages over agent i's possible actions; the
                gradient through it would be invalid since Q must not be
                conditioned on the action we are evaluating against).
            agent_id_one_hot: ``(batch, N)`` — one-hot agent identity. Lets
                a single Q-network share parameters across agents while
                still routing per-agent computation.

        Returns:
            ``(batch, action_dim_total)`` — vector of Q-values, one per
            possible per-agent action for the indicated agent given the
            other agents' actions.
        """
        x = torch.cat([global_obs, joint_action_one_hot, agent_id_one_hot], dim=-1)
        return self.q_head(self.shared(x))


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


class TransformerPolicyNetwork(nn.Module):
    """Transformer-based actor-critic policy for multi-agent coordination.

    This network uses self-attention to process house observations, allowing
    the agent to reason about spatial relationships and coordination needs.
    Designed for ~200-500K parameters to better utilize GPU resources.

    Architecture:
        - Input embedding: Linear projection of flattened observation
        - Positional encoding: Learned positional embeddings for houses
        - Transformer encoder: Multi-head self-attention layers
        - Action/value heads: Separate MLPs for actions and value estimation

    Args:
        obs_dim: Total dimension of flattened observation (e.g., 42 for 10 houses x 4 features + 2 global)
        action_dims: List of action space sizes for each action dimension
        num_houses: Number of houses in the environment (default: 10)
        house_features: Number of features per house (default: 4: fire_level, bucket_count, distance, threatened)
        d_model: Transformer embedding dimension (default: 256)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 3)
        dim_feedforward: Dimension of feedforward network in transformer (default: 512)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> # Create policy for bucket brigade (10 houses, 42-dim obs)
        >>> policy = TransformerPolicyNetwork(
        ...     obs_dim=42,
        ...     action_dims=[10, 2],  # house selection + mode
        ...     num_houses=10,
        ...     d_model=256,
        ...     num_layers=3
        ... )
        >>> # Approximately 350K parameters
        >>> sum(p.numel() for p in policy.parameters())
        ~350000
    """

    def __init__(
        self,
        obs_dim: int,
        action_dims: List[int],
        num_houses: int = None,  # Auto-detect if not provided
        house_features: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.house_features = house_features
        self.d_model = d_model

        # Auto-detect number of houses if not provided
        # Assume all features are house features (common case)
        if num_houses is None:
            if obs_dim % house_features == 0:
                num_houses = obs_dim // house_features
                self.global_dim = 0
            else:
                # If not evenly divisible, assume remainder is global features
                num_houses = obs_dim // house_features
                self.global_dim = obs_dim % house_features
        else:
            # Calculate global features dimension (everything except houses)
            self.global_dim = obs_dim - (num_houses * house_features)

        self.num_houses = num_houses

        # Input projection: house features -> d_model
        self.house_embed = nn.Linear(house_features, d_model)

        # Global features embedding
        if self.global_dim > 0:
            self.global_embed = nn.Linear(self.global_dim, d_model)

        # Learnable positional encoding for houses
        self.pos_embed = nn.Parameter(torch.randn(1, num_houses, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Action heads (2-layer MLPs)
        self.action_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, dim),
                )
                for dim in action_dims
            ]
        )

        # Value head (2-layer MLP)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using scaled initialization for transformers."""
        # Use Xavier/Glorot initialization for better gradient flow
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through transformer network.

        Args:
            x: Observation tensor of shape (batch_size, obs_dim)

        Returns:
            Tuple containing:
                - List of action logits for each action dimension
                - Value estimates of shape (batch_size, 1)
        """
        batch_size = x.shape[0]

        # Split observation into house features and global features
        # Assuming obs layout: [house_0_features..., house_1_features..., ..., global_features]
        house_obs = x[:, : self.num_houses * self.house_features]
        house_obs = house_obs.reshape(batch_size, self.num_houses, self.house_features)

        # Embed house observations
        house_embeds = self.house_embed(house_obs)  # (batch, num_houses, d_model)

        # Add positional encoding
        house_embeds = house_embeds + self.pos_embed

        # If we have global features, embed and prepend as [CLS] token
        if self.global_dim > 0:
            global_obs = x[:, self.num_houses * self.house_features :]
            global_embed = self.global_embed(global_obs).unsqueeze(
                1
            )  # (batch, 1, d_model)

            # Concatenate: [global, house_0, house_1, ...]
            tokens = torch.cat([global_embed, house_embeds], dim=1)
        else:
            tokens = house_embeds

        # Apply transformer
        features = self.transformer(tokens)  # (batch, num_tokens, d_model)

        # Use [CLS] token (first token) for policy/value
        cls_features = features[:, 0, :]  # (batch, d_model)

        # Generate action logits
        action_logits = [head(cls_features) for head in self.action_heads]

        # Generate value estimate
        value = self.value_head(cls_features)

        return action_logits, value

    def get_action(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Sample or select actions from the policy.

        Args:
            x: Observation tensor of shape (batch_size, obs_dim)
            deterministic: If True, select argmax action. If False, sample

        Returns:
            Tuple containing:
                - List of action tensors (one per action dimension)
                - Log probabilities of selected actions (None if deterministic)
                - Value estimates
        """
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

        log_prob = torch.stack(log_probs).sum(0) if log_probs else None

        return actions, log_prob, value

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions and values in batch-compatible format.

        Args:
            x: Observation tensor of shape (batch_size, obs_dim)
            action: Optional action tensor for computing log probabilities

        Returns:
            Tuple containing:
                - Actions tensor of shape (batch_size, num_action_dims)
                - Log probabilities of shape (batch_size,)
                - Entropy of shape (batch_size,)
                - Value estimates of shape (batch_size,)
        """
        action_logits, value = self.forward(x)

        actions = []
        log_probs = []
        entropies = []

        for i, logits in enumerate(action_logits):
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if action is None:
                a = dist.sample()
            else:
                a = action[:, i]

            actions.append(a)
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())

        return (
            torch.stack(actions, dim=1),
            torch.stack(log_probs, dim=1).sum(dim=1),
            torch.stack(entropies, dim=1).mean(dim=1),
            value.squeeze(-1),
        )
