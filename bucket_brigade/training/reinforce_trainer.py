"""Vanilla REINFORCE trainer for the P3 specialization diagnostic (issue #273).

REINFORCE = the classical Monte-Carlo policy-gradient algorithm
(Williams 1992). For each rolled-out trajectory, agent ``i``'s gradient
estimator is

    g_i  =  E_{τ ~ π_i}  [  Σ_t  ∇_{θ_i} log π_i(a_{i,t} | s_t) ⋅ G_{i,t}  ]

where ``G_{i,t} = Σ_{k ≥ t} γ^{k-t} r_{i,k}`` is the Monte-Carlo discounted
return-to-go for agent ``i``. **No value baseline, no GAE, no clip, no
multiple-epoch update, no critic.** Single gradient step per rollout.

This is the diagnostic positive control for issue #273: vanilla policy
gradient on the same bucket-brigade env that PPO and MAPPO both plateau
on. Three possible outcomes:

- **Same plateau as PPO** → the failure is RL-general, not PPO-specific;
  doubling down on env-side / CTDE / intrinsic-reward interventions is
  justified.
- **REINFORCE > PPO** → PPO's clip / GAE is hurting; revisit clip schedule
  and GAE lambda.
- **REINFORCE < PPO** → PPO is mitigating real variance; the plateau is
  genuinely "best PPO can do" and orthogonal axes (env, reward, init) are
  the only path forward.

The trainer mirrors :class:`JointPPOTrainer`'s public API (``__init__``,
``collect_rollout``, ``update``, ``encoder_outputs_batch``) so the rest of
``experiments/p3_specialization`` machinery can swap one for the other
behind a CLI flag (``--algorithm reinforce``).

**Mutual exclusion** (mirrors the guards at
:class:`LolaTrainer` ``__init__`` and ``joint_trainer.py:236``): REINFORCE
rejects ``centralized_critic``, ``coma``, ``influence_coef > 0``,
``redundancy_coef > 0``, ``advantage_estimator != "gae"`` (HCA), and
``lola_dice``. The diagnostic's value comes from being clean vanilla PG;
combining it with any of these would conflate the algorithmic and
architectural sides of the experiment.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bucket_brigade.training.joint_trainer import (
    RolloutBuffer,
    flatten_dict_obs,
)
from bucket_brigade.training.networks import PolicyNetwork


__all__ = ["REINFORCETrainer"]


class REINFORCETrainer:
    """Vanilla REINFORCE trainer mirroring :class:`JointPPOTrainer`'s API.

    Args:
        env_fn: Zero-argument factory; same contract as ``JointPPOTrainer``.
        num_agents: Number of agents.
        obs_dim: Flattened observation dim (must include the per-agent
            identity one-hot tail).
        action_dims: Per-dimension action sizes.
        hidden_size: Trunk hidden size for each per-agent
            :class:`PolicyNetwork`.
        lr: Adam learning rate for each per-agent actor.
        gamma: Discount factor. Same convention as ``JointPPOTrainer``.
        max_grad_norm: Per-policy gradient clip.
        normalize_returns: If True, standardize per-agent discounted
            returns to zero-mean unit-variance within each rollout before
            multiplying by the log-prob. This is **not** a value baseline
            — it is the classical "return normalization" variance-reduction
            trick (orthogonal to advantage estimation) and is the natural
            ablation to expose vanilla REINFORCE's high-variance behavior.
            Default ``True`` so smoke runs are usable; the sweep covers
            both settings.
        device: ``"cpu"`` or ``"cuda"``.
        seed: Optional torch + numpy seed.

        # --- Mutual-exclusion knobs (all must be at their no-op default) ---
        centralized_critic: Must be False. REINFORCE is plain on-policy PG;
            combining it with a shared value baseline (MAPPO) defeats the
            point of the diagnostic. Validated at construction.
        coma: Must be False. Same rationale as ``centralized_critic`` —
            COMA replaces the per-agent advantage with a counterfactual
            Q-value, which is exactly the kind of structure the
            REINFORCE control is meant to *strip away*.
        redundancy_coef: Must be 0.0. The redundancy penalty couples the
            per-agent actor trunks; REINFORCE's diagnostic value depends
            on independent per-agent updates.
        advantage_estimator: Must be ``"gae"`` (the no-op sentinel). HCA
            installs hindsight networks that re-introduce a learned
            baseline-like quantity; not compatible with the "no critic, no
            advantage estimator" framing.
        influence_coef: Must be 0.0. Social-influence intrinsic reward
            modifies the per-step reward signal upstream of the advantage
            computation; conflates with the REINFORCE return estimate.
        lola_dice: Must be False. LOLA-DiCE installs a second-order
            opponent-shaping surrogate; the whole point of REINFORCE is to
            run *without* opponent awareness.
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
        max_grad_norm: float = 0.5,
        normalize_returns: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
        # Mutual-exclusion knobs (all must be at their no-op default).
        centralized_critic: bool = False,
        coma: bool = False,
        redundancy_coef: float = 0.0,
        advantage_estimator: str = "gae",
        influence_coef: float = 0.0,
        lola_dice: bool = False,
    ):
        # Mutual-exclusion guards. The diagnostic's interpretation
        # depends on running clean vanilla PG; combining with any of the
        # below would entangle two experiments.
        if centralized_critic:
            raise ValueError(
                "REINFORCETrainer is incompatible with centralized_critic=True "
                "(MAPPO). REINFORCE is plain on-policy policy gradient with no "
                "value baseline; combining it with a shared critic would "
                "conflate two unrelated experiments. Pick one."
            )
        if coma:
            raise ValueError(
                "REINFORCETrainer is incompatible with coma=True. COMA "
                "installs a centralized Q-critic and replaces the per-agent "
                "advantage with a counterfactual Q-value, which is exactly "
                "the kind of structure the REINFORCE control is meant to "
                "strip away. Pick one."
            )
        if redundancy_coef > 0:
            raise ValueError(
                "REINFORCETrainer is incompatible with redundancy_coef>0. "
                "The redundancy penalty couples the per-agent actor trunks "
                "via a shared loss, while REINFORCE assumes strictly "
                "independent per-agent updates. Pick one."
            )
        if advantage_estimator != "gae":
            raise ValueError(
                f"REINFORCETrainer is incompatible with "
                f"advantage_estimator={advantage_estimator!r}. REINFORCE "
                "uses raw Monte-Carlo discounted returns as the score-"
                "function multiplier — installing any advantage estimator "
                "(GAE/HCA) defeats the point of the diagnostic. Leave "
                "advantage_estimator at its default ('gae' sentinel)."
            )
        if influence_coef > 0:
            raise ValueError(
                "REINFORCETrainer is incompatible with influence_coef>0. "
                "Social-influence intrinsic reward modifies the per-step "
                "reward upstream of the return computation, conflating "
                "with the REINFORCE estimate. Pick one."
            )
        if lola_dice:
            raise ValueError(
                "REINFORCETrainer is incompatible with lola_dice=True. "
                "LOLA-DiCE installs a second-order opponent-shaping "
                "surrogate; REINFORCE is the no-opponent-awareness "
                "control. Pick one."
            )

        self.env_fn = env_fn
        self.env = env_fn()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.normalize_returns = normalize_returns

        # Stored so the cell-config / metrics introspection can read them
        # back as no-ops.
        self.centralized_critic = False
        self.coma = False
        self.redundancy_coef = 0.0
        self.advantage_estimator = "gae"
        self.influence_coef = 0.0
        self.lola_dice = False

        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Per-agent policies. Use the same :class:`PolicyNetwork` as IPPO
        # so a fair baseline comparison only varies the objective. The
        # value head is constructed but unused — REINFORCE has no critic.
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
        self.optimizers = [optim.Adam(p.parameters(), lr=lr) for p in self.policies]

        self._reset_env(seed=seed)

    # ------------------------------------------------------------------
    # Per-agent flat-obs helpers (copy of the JointPPOTrainer wiring).
    # ------------------------------------------------------------------

    def _flatten_per_agent(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        rows = [
            flatten_dict_obs(obs_dict, agent_id=i, num_agents=self.num_agents)
            for i in range(self.num_agents)
        ]
        return np.stack(rows, axis=0)

    def _reset_env(self, seed: Optional[int] = None) -> None:
        obs = self.env.reset(seed=seed)
        self._last_obs = self._flatten_per_agent(obs)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, num_steps: int) -> RolloutBuffer:
        """Collect a :class:`RolloutBuffer`-compatible rollout.

        The buffer schema matches :meth:`JointPPOTrainer.collect_rollout`
        so the existing ``experiments/p3_specialization/train.py`` info-
        theory hook (which consumes a :class:`RolloutBuffer`) keeps
        working unchanged. Values are filled with zeros — REINFORCE has
        no critic — but the field is preserved for schema parity.

        Sampling-time log-probs are stored as detached scalars; the
        actual gradient flows in :meth:`update` by re-evaluating each
        sampled action under the current policy. This mirrors the PPO
        protocol and keeps the collect/update split clean.
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

        with torch.no_grad():
            for t in range(num_steps):
                obs_t = torch.from_numpy(self._last_obs).to(self.device)  # [N, obs_dim]
                joint_action = np.zeros(
                    (self.num_agents, len(self.action_dims)), dtype=np.int64
                )
                for i, policy in enumerate(self.policies):
                    obs_i = obs_t[i : i + 1]
                    a, lp, _ent, _v = policy.get_action_and_value(obs_i)
                    joint_action[i] = a[0].cpu().numpy()
                    log_probs[i, t] = lp[0].item()
                    # ``values`` left at 0.0 — REINFORCE has no critic.

                next_obs_dict, rew, done_arr, _ = self.env.step(joint_action)
                done = bool(np.asarray(done_arr).any())
                observations[t] = self._last_obs
                for i in range(self.num_agents):
                    actions[i, t] = joint_action[i]
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
    # Discounted returns
    # ------------------------------------------------------------------

    def _discounted_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Per-step discounted return-to-go for a single agent.

        Args:
            rewards: ``[T]`` per-step rewards.
            dones:   ``[T]`` shared episode termination flag.

        Returns:
            ``[T]`` per-step return ``G_t = Σ_{k ≥ t} γ^{k-t} r_k``
            with the discount reset to 0 on done boundaries (so an
            episode boundary at step ``t`` makes ``G_t = r_t``).
        """
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        running = rewards.new_tensor(0.0)
        for t in reversed(range(T)):
            # On a done step, the post-done future has zero contribution.
            running = rewards[t] + self.gamma * (1.0 - dones[t]) * running
            returns[t] = running
        return returns

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(self, rollout: Optional[RolloutBuffer] = None) -> Dict[str, float]:
        """Run one REINFORCE update.

        The update is a **single** gradient step per rollout per agent —
        no PPO-style multiple epochs, no minibatching, no importance
        sampling, no clip. For each agent ``i`` we:

        1. Re-evaluate the log-probabilities of the rolled-out actions
           under the **current** policy parameters. (Strictly equivalent
           to using the sampling-time log-probs because no parameter
           updates have happened between collect and update — but
           re-evaluating keeps the autograd graph attached.)
        2. Compute Monte-Carlo discounted returns ``G_t`` from the
           per-step rewards, resetting on episode boundaries.
        3. Optionally standardize ``G_t`` within the rollout (zero-mean,
           unit-variance) as the classical variance-reduction trick.
           This is **not** a value baseline — it does not bias the
           gradient asymptotically because it subtracts a constant from
           each return.
        4. Form the policy-gradient loss ``-Σ_t log π_i(a_{i,t}) ⋅ G_t``
           (mean over the rollout), backprop, take one Adam step.

        Returns:
            Dict of per-agent and total policy losses plus metrics-schema-
            compatible zero entries for ``value_loss/agent_i``,
            ``entropy/agent_i``, ``redundancy_loss`` so the existing
            analyzer pipelines parse a REINFORCE cell's
            ``metrics.json`` without choking on missing keys.
        """
        if rollout is None:
            raise ValueError(
                "REINFORCETrainer.update() requires a rollout argument. "
                "Call collect_rollout(num_steps) first."
            )

        T = rollout.observations.shape[0]

        per_agent_losses: List[torch.Tensor] = []
        stats: Dict[str, float] = {}

        for i, policy in enumerate(self.policies):
            # Re-evaluate log-probs under the current policy params.
            obs_i = rollout.observations[:, i, :]  # [T, obs_dim]
            actions_i = rollout.actions[i]  # [T, A]
            _a, log_probs_i, _entropy, _v = policy.get_action_and_value(
                obs_i, action=actions_i
            )  # [T]

            # Per-agent Monte-Carlo discounted returns.
            returns_i = self._discounted_returns(rollout.rewards[i], rollout.dones)
            returns_i = returns_i.detach()

            if self.normalize_returns and T > 1:
                # Standardize within the rollout. This is the classical
                # reward-/return-normalization trick (Sutton & Barto §13.3
                # variance-reduction note); subtracting a state-
                # independent constant from each return is gradient-
                # neutral asymptotically.
                returns_i = (returns_i - returns_i.mean()) / (returns_i.std() + 1e-8)

            # REINFORCE loss: -E_t [ log π(a_t | s_t) ⋅ G_t ]. Mean (not
            # sum) keeps the loss scale comparable across rollout
            # lengths.
            loss_i = -(log_probs_i * returns_i).mean()
            per_agent_losses.append(loss_i)
            stats[f"policy_loss/agent_{i}"] = float(loss_i.detach().item())

        # Single backward pass across all per-agent losses. Each
        # optimizer touches only its own parameters, so summing the
        # per-agent losses before .backward() is equivalent to N
        # separate backprops but avoids constructing N autograd graphs
        # from scratch (the trunks are independent so the saving is in
        # graph bookkeeping, not in matmul work).
        total_loss = torch.stack(per_agent_losses).sum()
        stats["policy_loss/total"] = float(total_loss.detach().item())
        stats["total_loss"] = stats["policy_loss/total"]

        for opt in self.optimizers:
            opt.zero_grad()
        total_loss.backward()
        for policy in self.policies:
            nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
        for opt in self.optimizers:
            opt.step()

        # Diagnostic / metrics-schema-compatible zero scalars. The
        # existing ``experiments/p3_specialization/analyze_*.py`` parsers
        # expect these keys on every cell; populating them as zero avoids
        # an n-of-1 analyzer fork.
        for i in range(self.num_agents):
            stats[f"value_loss/agent_{i}"] = 0.0
            stats[f"entropy/agent_{i}"] = 0.0
        stats["redundancy_loss"] = 0.0

        # Per-step team reward, for parity with JointPPOTrainer-driven cells.
        team_reward_sum = float(
            torch.stack([rollout.rewards[i].sum() for i in range(self.num_agents)])
            .sum()
            .item()
        )
        stats["reinforce/mean_step_reward_team"] = team_reward_sum / float(T)
        stats["reinforce/normalize_returns"] = float(self.normalize_returns)
        stats["reinforce/gamma"] = float(self.gamma)
        return stats

    # ------------------------------------------------------------------
    # Encoder tap parity (for the info-theory hook in train.py).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encoder_outputs_batch(self, observations: torch.Tensor) -> List[torch.Tensor]:
        """Per-agent encoder taps for the plug-in conditional-MI estimator.

        Mirrors :meth:`JointPPOTrainer.encoder_outputs_batch` so the
        existing info-theory hook in ``experiments/p3_specialization/train.py``
        works against either trainer without branching.
        """
        if observations.dim() == 3:
            return [
                p.encoder_output(observations[:, i, :])
                for i, p in enumerate(self.policies)
            ]
        return [p.encoder_output(observations) for p in self.policies]
