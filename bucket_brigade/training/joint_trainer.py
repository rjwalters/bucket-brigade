"""Joint multi-agent PPO trainer for the Slepian-Wolf MARL P3 experiment.

One process, N ``PolicyNetwork`` instances trained on shared synchronized
rollout batches. This is independent-learners PPO (IPPO) reorganized so that
all agents see the same minibatch at training time --- which makes
cross-agent regularizers (the conditional-MI redundancy penalty proposed in
``slepian-wolf-marl/paper/slepian-wolf-marl.4/paper.tex`` Section 5.3) a
single batch-level operation rather than a cross-process synchronization
problem.

At inference time the agents are still strictly independent: each
``PolicyNetwork`` consumes the global observation and outputs an action.
The redundancy penalty only couples them through gradients during
``update()``.

Typical use::

    from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv

    def env_fn():
        return BucketBrigadeEnv(num_agents=4)

    trainer = JointPPOTrainer(
        env_fn=env_fn,
        num_agents=4,
        obs_dim=42,
        action_dims=[10, 2],
        redundancy_coef=0.01,
    )
    for epoch in range(num_epochs):
        rollout = trainer.collect_rollout(num_steps=2048)
        stats = trainer.update(rollout)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bucket_brigade.training.networks import PolicyNetwork, compute_gae


__all__ = ["JointPPOTrainer", "RolloutBuffer", "flatten_dict_obs"]


def flatten_dict_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten the dict observation returned by ``BucketBrigadeEnv`` into
    a 1-D float32 array suitable for ``PolicyNetwork``.

    Layout: ``[houses(10), signals(N), locations(N), last_actions(2N),
    scenario_info(10)]``.
    """
    return np.concatenate(
        [
            np.asarray(obs["houses"], dtype=np.float32),
            np.asarray(obs["signals"], dtype=np.float32),
            np.asarray(obs["locations"], dtype=np.float32),
            np.asarray(obs["last_actions"], dtype=np.float32).flatten(),
            np.asarray(obs["scenario_info"], dtype=np.float32),
        ]
    )


@dataclass
class RolloutBuffer:
    """Synchronized multi-agent rollout produced by :meth:`JointPPOTrainer.collect_rollout`.

    Shapes (with ``T`` = num_steps, ``N`` = num_agents, ``A`` = ``len(action_dims)``):

    - ``observations``: ``[T, obs_dim]`` --- same global obs for every agent at each step.
    - ``actions[i]``:   ``[T, A]`` --- agent i's actions.
    - ``log_probs[i]``: ``[T]``   --- log probability of the action under the
      rollout-time policy (used for PPO's old-policy ratio).
    - ``values[i]``:    ``[T]``   --- value estimate at the rollout-time policy.
    - ``rewards[i]``:   ``[T]``   --- per-agent reward.
    - ``dones``:        ``[T]``   --- shared episode termination flag.
    """

    observations: torch.Tensor
    actions: Dict[int, torch.Tensor]
    log_probs: Dict[int, torch.Tensor]
    values: Dict[int, torch.Tensor]
    rewards: Dict[int, torch.Tensor]
    dones: torch.Tensor


class JointPPOTrainer:
    """Synchronized IPPO with a cross-agent redundancy penalty.

    Args:
        env_fn: Zero-argument factory that returns a fresh environment with a
            ``reset(seed=...)`` method returning a dict observation and a
            ``step(actions)`` method taking a ``[N, A]`` int array and returning
            ``(obs_dict, rewards[N], dones[N], info)``.
        num_agents: Number of agents N.
        obs_dim: Flattened observation dimension (see :func:`flatten_dict_obs`).
        action_dims: List of per-dimension action sizes (e.g. ``[10, 2]``).
        hidden_size: Hidden size of the shared trunk in each ``PolicyNetwork``.
        lr: Adam learning rate.
        gamma, gae_lambda: Discount and GAE coefficients.
        clip_epsilon: PPO clip range.
        value_coef, entropy_coef: PPO loss weights.
        max_grad_norm: Per-policy gradient clip.
        redundancy_coef: λ_red. If 0, the redundancy penalty is skipped entirely
            (no extra forward through the encoder tap).
        ppo_epochs: Number of PPO update passes per rollout.
        minibatch_size: Number of samples used per epoch (one minibatch per epoch;
            split into multiple chunks in a future revision if needed).
        device: ``"cpu"`` or ``"cuda"``.
        seed: Optional torch + env seed for reproducibility.
    """

    def __init__(
        self,
        env_fn: Callable[[], object],
        num_agents: int,
        obs_dim: int,
        action_dims: List[int],
        hidden_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        redundancy_coef: float = 0.0,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.env_fn = env_fn
        self.env = env_fn()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.hidden_size = hidden_size

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.redundancy_coef = redundancy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.policies = nn.ModuleList(
            [
                PolicyNetwork(
                    obs_dim=obs_dim,
                    action_dims=action_dims,
                    hidden_size=hidden_size,
                )
                for _ in range(num_agents)
            ]
        ).to(self.device)

        # Per-agent optimizers. Each only updates its own params, but
        # ``total_loss.backward()`` is called once across all of them so the
        # redundancy penalty's gradient can flow into each encoder.
        self.optimizers = [optim.Adam(p.parameters(), lr=lr) for p in self.policies]

        self._reset_env(seed=seed)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _reset_env(self, seed: Optional[int] = None) -> None:
        obs = self.env.reset(seed=seed)
        self._last_obs = flatten_dict_obs(obs)

    @torch.no_grad()
    def _act_all(
        self, obs_t: torch.Tensor
    ) -> Tuple[np.ndarray, List[torch.Tensor], List[torch.Tensor]]:
        """Sample one action per agent given the same observation.

        Returns ``(joint_action [N, A], per-agent log_prob list, per-agent value list)``.
        """
        joint_action = np.zeros(
            (self.num_agents, len(self.action_dims)), dtype=np.int64
        )
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        for i, policy in enumerate(self.policies):
            actions, lp, _, v = policy.get_action_and_value(obs_t)
            joint_action[i] = actions[0].cpu().numpy()
            log_probs.append(lp[0])
            values.append(v[0])
        return joint_action, log_probs, values

    def collect_rollout(self, num_steps: int) -> RolloutBuffer:
        """Run ``num_steps`` synchronized transitions, auto-resetting on done."""
        observations = np.zeros((num_steps, self.obs_dim), dtype=np.float32)
        actions = np.zeros(
            (self.num_agents, num_steps, len(self.action_dims)), dtype=np.int64
        )
        log_probs = np.zeros((self.num_agents, num_steps), dtype=np.float32)
        values = np.zeros((self.num_agents, num_steps), dtype=np.float32)
        rewards = np.zeros((self.num_agents, num_steps), dtype=np.float32)
        dones = np.zeros(num_steps, dtype=np.float32)

        for t in range(num_steps):
            obs_t = torch.from_numpy(self._last_obs).unsqueeze(0).to(self.device)
            joint_action, lps, vs = self._act_all(obs_t)

            next_obs_dict, rew, done_arr, _ = self.env.step(joint_action)
            done = bool(np.asarray(done_arr).any())

            observations[t] = self._last_obs
            for i in range(self.num_agents):
                actions[i, t] = joint_action[i]
                log_probs[i, t] = lps[i].item()
                values[i, t] = vs[i].item()
                rewards[i, t] = rew[i]
            dones[t] = float(done)

            if done:
                self._reset_env()
            else:
                self._last_obs = flatten_dict_obs(next_obs_dict)

        return RolloutBuffer(
            observations=torch.from_numpy(observations).to(self.device),
            actions={
                i: torch.from_numpy(actions[i]).to(self.device)
                for i in range(self.num_agents)
            },
            log_probs={
                i: torch.from_numpy(log_probs[i]).to(self.device)
                for i in range(self.num_agents)
            },
            values={
                i: torch.from_numpy(values[i]).to(self.device)
                for i in range(self.num_agents)
            },
            rewards={
                i: torch.from_numpy(rewards[i]).to(self.device)
                for i in range(self.num_agents)
            },
            dones=torch.from_numpy(dones).to(self.device),
        )

    # ------------------------------------------------------------------
    # Redundancy penalty (differentiable surrogate)
    # ------------------------------------------------------------------

    @staticmethod
    def redundancy_penalty(encoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Differentiable surrogate for cross-agent representational redundancy.

        For each ordered pair ``(i, j)`` with ``i < j``, compute the Frobenius
        norm of the cross-correlation matrix between the standardized encoder
        outputs ``Ẑ_i`` and ``Ẑ_j`` across the minibatch, and average over
        pairs and features.

        This is a *linear* proxy for ``I(Ẑ_i; Ẑ_j)``: it equals zero when the
        per-feature cross-correlations all vanish, and saturates when any
        feature-pair is perfectly correlated. It captures the same intuition
        as the plug-in MI estimator in
        :mod:`bucket_brigade.analysis.info_theory` but is differentiable
        end-to-end and cheap to compute.

        The plug-in CMI from ``info_theory`` is the reporting quantity (no
        gradients); this is the training quantity. Iterating to a MINE or
        InfoNCE neural estimator is a follow-up if results are
        estimator-bound.

        Args:
            encoder_outputs: List of N tensors of shape ``[B, d]``.

        Returns:
            Scalar tensor in ``[0, 1]`` (under per-feature unit-variance
            standardization).
        """
        n = len(encoder_outputs)
        if n < 2:
            return encoder_outputs[0].new_tensor(0.0)

        standardized = []
        for z in encoder_outputs:
            c = z - z.mean(dim=0, keepdim=True)
            std = c.std(dim=0, keepdim=True).clamp_min(1e-6)
            standardized.append(c / std)

        batch_size = standardized[0].shape[0]
        d = standardized[0].shape[1]

        total = encoder_outputs[0].new_tensor(0.0)
        num_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                cross = standardized[i].t() @ standardized[j] / batch_size
                total = total + (cross**2).sum()
                num_pairs += 1
        return total / (num_pairs * d * d)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        adv = torch.tensor(
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
        ret = adv + values
        return adv, ret

    def update(self, rollout: RolloutBuffer) -> Dict[str, float]:
        """Run ``ppo_epochs`` PPO updates against the rollout.

        Returns a dict of mean per-epoch metrics, including per-agent policy,
        value, and entropy losses plus the redundancy penalty and total loss.
        """
        t_total = rollout.observations.shape[0]

        # Per-agent advantages and returns from GAE.
        advantages: Dict[int, torch.Tensor] = {}
        returns: Dict[int, torch.Tensor] = {}
        for i in range(self.num_agents):
            adv, ret = self._compute_advantages(
                rollout.rewards[i], rollout.values[i], rollout.dones
            )
            advantages[i] = (adv - adv.mean()) / (adv.std() + 1e-8)
            returns[i] = ret

        stats: Dict[str, float] = {
            f"policy_loss/agent_{i}": 0.0 for i in range(self.num_agents)
        }
        stats.update({f"value_loss/agent_{i}": 0.0 for i in range(self.num_agents)})
        stats.update({f"entropy/agent_{i}": 0.0 for i in range(self.num_agents)})
        stats["redundancy_loss"] = 0.0
        stats["total_loss"] = 0.0

        mb_size = min(self.minibatch_size, t_total)

        for _epoch in range(self.ppo_epochs):
            idx = torch.randperm(t_total, device=self.device)[:mb_size]
            obs_mb = rollout.observations[idx]

            encoder_outputs: List[torch.Tensor] = []
            per_agent_losses: List[torch.Tensor] = []

            for i, policy in enumerate(self.policies):
                actions_mb = rollout.actions[i][idx]
                old_lp_mb = rollout.log_probs[i][idx]
                adv_mb = advantages[i][idx]
                ret_mb = returns[i][idx]

                # One forward pass through the trunk; reuse features for
                # action heads, value head, and the redundancy penalty.
                features = policy.encoder_output(obs_mb)
                encoder_outputs.append(features)

                action_logits = [head(features) for head in policy.action_heads]
                values = policy.value_head(features).squeeze(-1)

                new_log_probs = []
                entropies = []
                for k, logits in enumerate(action_logits):
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs.append(dist.log_prob(actions_mb[:, k]))
                    entropies.append(dist.entropy())
                new_log_prob = torch.stack(new_log_probs, dim=1).sum(dim=1)
                entropy = torch.stack(entropies, dim=1).mean(dim=1).mean()

                ratio = torch.exp(new_log_prob - old_lp_mb)
                surr1 = ratio * adv_mb
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * adv_mb
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_mb)
                agent_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                per_agent_losses.append(agent_loss)

                stats[f"policy_loss/agent_{i}"] += policy_loss.item() / self.ppo_epochs
                stats[f"value_loss/agent_{i}"] += value_loss.item() / self.ppo_epochs
                stats[f"entropy/agent_{i}"] += entropy.item() / self.ppo_epochs

            if self.redundancy_coef > 0:
                red_pen = self.redundancy_penalty(encoder_outputs)
            else:
                red_pen = encoder_outputs[0].new_tensor(0.0)
            stats["redundancy_loss"] += red_pen.item() / self.ppo_epochs

            total_loss = (
                torch.stack(per_agent_losses).sum() + self.redundancy_coef * red_pen
            )
            stats["total_loss"] += total_loss.item() / self.ppo_epochs

            for opt in self.optimizers:
                opt.zero_grad()
            total_loss.backward()
            for policy in self.policies:
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            for opt in self.optimizers:
                opt.step()

        return stats

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encoder_outputs_batch(self, observations: torch.Tensor) -> List[torch.Tensor]:
        """Run every policy's encoder tap on a shared batch of observations.

        Returns a list of ``[B, hidden_size]`` tensors (no gradients). Use this
        to feed the plug-in conditional-MI estimator in
        :mod:`bucket_brigade.analysis.info_theory` for reporting.
        """
        return [p.encoder_output(observations) for p in self.policies]
