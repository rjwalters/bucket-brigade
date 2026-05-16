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

    # obs_dim now includes a per-agent identity one-hot tail (issue #204).
    # For N=4 agents and the default 10-house layout, this is 42 + 4 = 46.
    # In practice, derive it from a probe:
    #   obs = env.reset(); obs_dim = flatten_dict_obs(obs, agent_id=0,
    #       num_agents=4).shape[0]
    trainer = JointPPOTrainer(
        env_fn=env_fn,
        num_agents=4,
        obs_dim=46,
        action_dims=[10, 2, 2],  # [house, mode, signal] (issue #235)
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

from bucket_brigade.training.networks import (
    CentralizedCritic,
    PolicyNetwork,
    compute_gae,
)
from bucket_brigade.training.normalizers import RunningMeanStd


__all__ = ["JointPPOTrainer", "RolloutBuffer", "flatten_dict_obs"]


def flatten_dict_obs(
    obs: Dict[str, np.ndarray],
    agent_id: Optional[int] = None,
    num_agents: Optional[int] = None,
) -> np.ndarray:
    """Flatten the dict observation returned by ``BucketBrigadeEnv`` into
    a 1-D float32 array suitable for ``PolicyNetwork``.

    Layout when ``agent_id`` is ``None`` (legacy, no per-agent identity):
        ``[houses(10), signals(N), locations(N), last_actions(2N),
        scenario_info(10)]``.

    Layout when ``agent_id`` is given (issue #204 — per-agent
    differentiation via one-hot identity tail):
        ``[houses(10), signals(N), locations(N), last_actions(2N),
        scenario_info(10), identity_one_hot(num_agents)]``.

    Args:
        obs: Dict observation from ``BucketBrigadeEnv._get_observation``.
        agent_id: If provided, append a one-hot identity slot of length
            ``num_agents`` so per-agent flattened observations are
            distinct. This is the latent-bug fix described in #204 —
            the Python flatteners previously dropped the ``agent_id``
            field that ``AgentObservation`` already carries, which
            caused ``collect_rollout`` to feed identical input to every
            agent.
        num_agents: Length of the identity one-hot. Required when
            ``agent_id`` is provided. Typically the env's ``num_agents``.

    Raises:
        ValueError: If ``agent_id`` is given without ``num_agents``, or
            ``agent_id`` is outside ``[0, num_agents)``.
    """
    base = [
        np.asarray(obs["houses"], dtype=np.float32),
        np.asarray(obs["signals"], dtype=np.float32),
        np.asarray(obs["locations"], dtype=np.float32),
        np.asarray(obs["last_actions"], dtype=np.float32).flatten(),
        np.asarray(obs["scenario_info"], dtype=np.float32),
    ]
    if agent_id is None:
        return np.concatenate(base)
    if num_agents is None:
        raise ValueError(
            "flatten_dict_obs: num_agents must be provided when agent_id is given"
        )
    if not 0 <= agent_id < num_agents:
        raise ValueError(
            f"flatten_dict_obs: agent_id {agent_id} out of range [0, {num_agents})"
        )
    identity = np.zeros(num_agents, dtype=np.float32)
    identity[agent_id] = 1.0
    base.append(identity)
    return np.concatenate(base)


@dataclass
class RolloutBuffer:
    """Synchronized multi-agent rollout produced by :meth:`JointPPOTrainer.collect_rollout`.

    Shapes (with ``T`` = num_steps, ``N`` = num_agents, ``A`` = ``len(action_dims)``):

    - ``observations``: ``[T, N, obs_dim]`` --- **per-agent** flattened
      observation. Each agent sees the same global world state but a
      different identity one-hot in the tail slot (issue #204). Prior to
      #204 this was ``[T, obs_dim]`` and the same row was reused for every
      agent, which made identical-policy convergence mathematically forced.
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
            Must include the per-agent identity one-hot tail (i.e. it is the
            length of ``flatten_dict_obs(obs, agent_id=i, num_agents=N)``,
            *not* the legacy length without identity).
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
        normalize_returns: If True, maintain a running mean/std of returns
            (Welford / Chan) shared across all agents and divide returns by
            ``sqrt(var + 1e-8)`` before the value-loss MSE. Default ``False``
            preserves prior behavior. See issue #159 for the motivating
            value-loss-magnitude diagnostics.
        centralized_critic: If True, enable MAPPO (issue #208). A separate
            :class:`CentralizedCritic` consumes the **global** portion of
            each observation (the per-agent flat obs with the identity
            one-hot tail stripped) and produces a single shared value
            estimate used for advantage computation across all agents.
            Per-agent rewards still drive per-agent advantages; only the
            value baseline is shared. The per-agent policy networks'
            value heads are bypassed during rollout *and* update. Default
            ``False`` preserves the independent-PPO (IPPO) path bit-for-bit.

            **Incompatible with ``redundancy_coef > 0``** — the redundancy
            penalty couples the per-agent actor trunks; combining it with
            a centralized critic would entangle two unrelated experiments.
            Raises ``ValueError`` if both are set.
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
        normalize_returns: bool = False,
        centralized_critic: bool = False,
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

        # Issue #208: MAPPO / centralized-critic flag. Stored before the env
        # reset and policy construction so downstream wiring (advantage
        # computation, optimizer setup) can branch on it.
        self.centralized_critic = centralized_critic
        if centralized_critic and redundancy_coef > 0:
            raise ValueError(
                "centralized_critic=True is incompatible with redundancy_coef>0: "
                "the redundancy penalty couples the per-agent actor trunks, "
                "while the centralized critic replaces the per-agent value "
                "heads. Combining them entangles two unrelated experiments. "
                "Pick one."
            )

        # Issue #159: optional running mean/std of returns to shrink the
        # value-loss MSE target to O(1) (Phase 1 diagnostics showed
        # value_term/policy_term ~ 10^6-10^7).
        self.normalize_returns = normalize_returns
        self._return_rms: Optional[RunningMeanStd] = (
            RunningMeanStd(shape=()) if normalize_returns else None
        )

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

        # Issue #208: optional MAPPO centralized critic. The critic consumes
        # the **global** portion of the observation (the per-agent flat obs
        # with the identity one-hot tail stripped, length
        # ``obs_dim - num_agents``). It has its own optimizer so its Adam
        # state never co-mingles with the actor optimizers.
        if self.centralized_critic:
            self._global_obs_dim = obs_dim - num_agents
            if self._global_obs_dim <= 0:
                raise ValueError(
                    f"centralized_critic=True requires obs_dim ({obs_dim}) > "
                    f"num_agents ({num_agents}) so the identity-tail-stripped "
                    "global obs has positive length."
                )
            self.critic: Optional[CentralizedCritic] = CentralizedCritic(
                obs_dim=self._global_obs_dim,
                hidden_size=hidden_size,
            ).to(self.device)
            self.critic_optimizer: Optional[optim.Optimizer] = optim.Adam(
                self.critic.parameters(), lr=lr
            )
        else:
            self._global_obs_dim = obs_dim
            self.critic = None
            self.critic_optimizer = None

        self._reset_env(seed=seed)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _flatten_per_agent(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten one dict observation into a per-agent ``[N, obs_dim]`` array.

        Fixes the latent bug in #204 — every agent now gets its own row with
        a distinct identity one-hot tail, instead of all sharing one global row.
        """
        rows = [
            flatten_dict_obs(obs_dict, agent_id=i, num_agents=self.num_agents)
            for i in range(self.num_agents)
        ]
        return np.stack(rows, axis=0)

    def _reset_env(self, seed: Optional[int] = None) -> None:
        obs = self.env.reset(seed=seed)
        # ``_last_obs`` is now ``[N, obs_dim]`` — one row per agent.
        self._last_obs = self._flatten_per_agent(obs)

    def _global_obs_from_per_agent(self, obs_per_agent: torch.Tensor) -> torch.Tensor:
        """Strip the per-agent identity one-hot tail to recover the global obs.

        Used only when ``centralized_critic=True``. The global obs is the
        prefix shared by all agents at each timestep — i.e. each row of
        ``[N, obs_dim]`` minus its last ``num_agents`` features. We pick
        agent 0's row by convention (any row works: per-row prefixes are
        identical by the #204 contract; see
        ``tests/test_joint_trainer.py::test_flatten_dict_obs_per_agent_differs``).

        Args:
            obs_per_agent: Tensor of shape ``[..., N, obs_dim]`` or
                ``[N, obs_dim]``.

        Returns:
            Tensor of shape ``[..., global_obs_dim]`` --- the per-agent
            leading dim is collapsed (agent 0 selected).
        """
        # Take agent 0's row, then drop the identity one-hot tail.
        if obs_per_agent.dim() == 3:  # [T, N, obs_dim] -> [T, obs_dim]
            agent0 = obs_per_agent[:, 0, :]
        elif obs_per_agent.dim() == 2:  # [N, obs_dim] -> [obs_dim]
            agent0 = obs_per_agent[0]
        else:
            raise ValueError(
                f"_global_obs_from_per_agent: expected 2D or 3D input, got "
                f"shape {tuple(obs_per_agent.shape)}"
            )
        return agent0[..., : self._global_obs_dim]

    @torch.no_grad()
    def _act_all(
        self, obs_per_agent_t: torch.Tensor
    ) -> Tuple[np.ndarray, List[torch.Tensor], List[torch.Tensor]]:
        """Sample one action per agent given each agent's own observation row.

        Args:
            obs_per_agent_t: Tensor of shape ``[N, obs_dim]`` — agent ``i``
                consumes row ``i``. This is the per-agent obs from #204;
                previously the same row was passed to all N policies.

        Returns ``(joint_action [N, A], per-agent log_prob list, per-agent value list)``.

        When ``centralized_critic=True`` (issue #208), the per-agent value
        from ``policy.get_action_and_value`` is discarded and a single
        shared value from :attr:`critic` (applied to the identity-tail-
        stripped global obs) is broadcast to all N agents' value slots,
        so the existing GAE machinery downstream stays unchanged.
        """
        joint_action = np.zeros(
            (self.num_agents, len(self.action_dims)), dtype=np.int64
        )
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        for i, policy in enumerate(self.policies):
            obs_i = obs_per_agent_t[i : i + 1]  # [1, obs_dim]
            actions, lp, _, v = policy.get_action_and_value(obs_i)
            joint_action[i] = actions[0].cpu().numpy()
            log_probs.append(lp[0])
            values.append(v[0])

        if self.centralized_critic:
            # Replace per-agent values with one shared centralized estimate.
            # The global obs is the same for every agent (identity tail
            # stripped), so a single forward suffices.
            global_obs = self._global_obs_from_per_agent(obs_per_agent_t).unsqueeze(0)
            shared_value = self.critic(global_obs).squeeze(-1).squeeze(0)
            values = [shared_value for _ in range(self.num_agents)]
        return joint_action, log_probs, values

    def collect_rollout(self, num_steps: int) -> RolloutBuffer:
        """Run ``num_steps`` synchronized transitions, auto-resetting on done.

        Issue #204: ``observations`` is now ``[T, N, obs_dim]`` — each agent
        gets its own row per step with a distinct identity one-hot tail.
        The previous shape ``[T, obs_dim]`` was a latent bug that fed
        identical input to every agent.
        """
        observations = np.zeros(
            (num_steps, self.num_agents, self.obs_dim), dtype=np.float32
        )
        actions = np.zeros(
            (self.num_agents, num_steps, len(self.action_dims)), dtype=np.int64
        )
        log_probs = np.zeros((self.num_agents, num_steps), dtype=np.float32)
        values = np.zeros((self.num_agents, num_steps), dtype=np.float32)
        rewards = np.zeros((self.num_agents, num_steps), dtype=np.float32)
        dones = np.zeros(num_steps, dtype=np.float32)

        for t in range(num_steps):
            obs_t = torch.from_numpy(self._last_obs).to(self.device)  # [N, obs_dim]
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
                self._last_obs = self._flatten_per_agent(next_obs_dict)

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

        # Issue #159: optional return normalization. Update the shared running
        # mean/std from the concatenation of all agents' returns (commensurate
        # since rewards live on the same scale), then scale each agent's
        # returns by 1/sqrt(var + eps). This shrinks the MSE target to O(1)
        # so that vf_coef=0.5 isn't dwarfing the policy gradient. Advantages
        # are already standardized above and are NOT rescaled here (they enter
        # the policy loss, not the value loss).
        if self._return_rms is not None:
            all_returns = (
                torch.cat([returns[i] for i in range(self.num_agents)])
                .detach()
                .cpu()
                .numpy()
            )
            self._return_rms.update(all_returns)
            scale = float(np.sqrt(self._return_rms.var + 1e-8))
            for i in range(self.num_agents):
                returns[i] = returns[i] / scale

        stats: Dict[str, float] = {
            f"policy_loss/agent_{i}": 0.0 for i in range(self.num_agents)
        }
        stats.update({f"value_loss/agent_{i}": 0.0 for i in range(self.num_agents)})
        stats.update({f"entropy/agent_{i}": 0.0 for i in range(self.num_agents)})
        stats["redundancy_loss"] = 0.0
        stats["total_loss"] = 0.0

        mb_size = min(self.minibatch_size, t_total)

        # Issue #208: when the centralized critic is on, the value-loss is
        # computed once against the mean across-agents return (the shared
        # MAPPO regression target). Pre-compute it here so it can be sliced
        # by minibatch index alongside the per-agent returns.
        if self.centralized_critic:
            mean_returns = torch.stack(
                [returns[i] for i in range(self.num_agents)], dim=0
            ).mean(dim=0)
            stats["critic_value_loss"] = 0.0

        for _epoch in range(self.ppo_epochs):
            idx = torch.randperm(t_total, device=self.device)[:mb_size]
            # Issue #204: ``rollout.observations`` is now ``[T, N, obs_dim]``.
            # Each agent reads its own slice ``[:, i, :]`` so the identity
            # one-hot tail differs per policy. Pre-#204 this was ``[T, obs_dim]``
            # and a single shared ``obs_mb`` was passed to every encoder.
            obs_mb_all = rollout.observations[idx]  # [mb, N, obs_dim]

            encoder_outputs: List[torch.Tensor] = []
            per_agent_losses: List[torch.Tensor] = []

            for i, policy in enumerate(self.policies):
                actions_mb = rollout.actions[i][idx]
                old_lp_mb = rollout.log_probs[i][idx]
                adv_mb = advantages[i][idx]
                ret_mb = returns[i][idx]
                obs_mb = obs_mb_all[:, i, :]  # [mb, obs_dim] — agent i's view

                # One forward pass through the trunk; reuse features for
                # action heads, value head, and the redundancy penalty.
                features = policy.encoder_output(obs_mb)
                encoder_outputs.append(features)

                action_logits = [head(features) for head in policy.action_heads]

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

                if self.centralized_critic:
                    # Per-agent value head is bypassed; the centralized
                    # critic handles all value learning below. Record
                    # value_loss=0 to keep the stats schema consistent.
                    value_loss = policy_loss.new_tensor(0.0)
                    agent_loss = policy_loss - self.entropy_coef * entropy
                else:
                    values = policy.value_head(features).squeeze(-1)
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

            if self.centralized_critic:
                # One critic forward on the global obs minibatch (agent 0's
                # row, identity-tail stripped). MSE against the shared
                # mean-across-agents returns.
                global_obs_mb = self._global_obs_from_per_agent(obs_mb_all)
                critic_values = self.critic(global_obs_mb).squeeze(-1)
                critic_value_loss = F.mse_loss(critic_values, mean_returns[idx])
                stats["critic_value_loss"] += critic_value_loss.item() / self.ppo_epochs
                total_loss = (
                    torch.stack(per_agent_losses).sum()
                    + self.value_coef * critic_value_loss
                )
            else:
                total_loss = (
                    torch.stack(per_agent_losses).sum() + self.redundancy_coef * red_pen
                )
            stats["total_loss"] += total_loss.item() / self.ppo_epochs

            for opt in self.optimizers:
                opt.zero_grad()
            if self.centralized_critic:
                self.critic_optimizer.zero_grad()
            total_loss.backward()
            for policy in self.policies:
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            if self.centralized_critic:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            for opt in self.optimizers:
                opt.step()
            if self.centralized_critic:
                self.critic_optimizer.step()

        return stats

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encoder_outputs_batch(self, observations: torch.Tensor) -> List[torch.Tensor]:
        """Run every policy's encoder tap on its per-agent observation slice.

        Issue #221: after PR #216, ``observations`` is ``[T, N, obs_dim]`` (one
        row per agent, each carrying its own identity one-hot tail). Each
        policy ``i`` consumes its own row ``observations[:, i, :]`` of shape
        ``[T, obs_dim]`` so the encoder output is the 2-D ``[T, hidden_size]``
        that downstream consumers (``_measure_information`` →
        ``quantize_uniform``) expect.

        Legacy 2-D ``[B, obs_dim]`` input is still accepted — every policy
        sees the same batch — to preserve callers that haven't been migrated.

        Returns a list of ``[B, hidden_size]`` tensors (no gradients). Use this
        to feed the plug-in conditional-MI estimator in
        :mod:`bucket_brigade.analysis.info_theory` for reporting.
        """
        if observations.dim() == 3:
            # ``[T, N, obs_dim]`` → per-agent ``[T, obs_dim]`` slice.
            return [
                p.encoder_output(observations[:, i, :])
                for i, p in enumerate(self.policies)
            ]
        return [p.encoder_output(observations) for p in self.policies]
