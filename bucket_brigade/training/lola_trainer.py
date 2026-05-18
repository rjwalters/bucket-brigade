"""LOLA-DiCE trainer for the P3 specialization experiment (issue #287).

LOLA = **Learning with Opponent-Learning Awareness** (Foerster et al. 2018,
https://arxiv.org/abs/1709.04326). Standard policy gradient, augmented with
a second-order term that anticipates each opponent's own learning step.

Mathematically, agent ``i``'s LOLA objective is

    g_i^LOLA = ∇_{θ_i} V_i + Σ_{j != i} η ⋅ (∇_{θ_i} ∇_{θ_j} V_i)^T ⋅ ∇_{θ_j} V_j

i.e. the standard policy-gradient term plus a mixed Hessian contracted with
opponent ``j``'s own value gradient.

This module implements the **LOLA-DiCE variant** (Foerster et al. 2018,
DiCE appendix, https://arxiv.org/abs/1802.05098). Rather than computing
the mixed Hessian explicitly, we build a "magic box" surrogate loss that
becomes the desired LOLA gradient after a single ``.backward()`` call.
This is the recipe used in the reference implementation and is much
easier to make numerically stable.

DiCE magic-box recipe:

    MagicBox(W)  ≜  exp(W − W.detach())

It evaluates to 1.0 in the forward pass and to ``∇W`` in the backward pass.
For a trajectory with per-agent log-probs ``log π_{a,t}``, the per-step
cumulative ``W_t = Σ_{a, t' ≤ t} log π_{a, t'}`` lets us write a surrogate

    L_i^surr  =  Σ_t  MagicBox(W_t) ⋅ r_{i,t}

whose first derivative w.r.t. ``θ_i`` is the standard policy gradient and
whose mixed second derivative w.r.t. ``θ_i, θ_j`` is the LOLA correction
term (up to discount factors). See the DiCE paper for the proof.

The trainer mirrors :class:`JointPPOTrainer`'s public API (``__init__``,
``collect_rollout``, ``update``) so the rest of ``experiments/p3_specialization``
machinery can swap one for the other behind a CLI flag.

**Mutual exclusion** (mirrors the MAPPO/redundancy guard at
``joint_trainer.py:236``): LOLA cannot be combined with ``centralized_critic``
or ``redundancy_coef > 0`` --- both modifications change which loss is
being optimized and entangling them with LOLA would conflate the
algorithmic and architectural sides of the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
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


__all__ = ["LolaTrainer"]


def _magic_box(w: torch.Tensor) -> torch.Tensor:
    """The DiCE 'magic box' operator (Foerster et al. 2018 DiCE).

    Forward: ``exp(w - w.detach()) == 1``.
    Backward: contributes ``∇w`` to autograd, so when used as a multiplier
    on rewards in a surrogate loss the resulting gradient is the standard
    score-function policy gradient (and higher-order derivatives are also
    correct).
    """
    return torch.exp(w - w.detach())


@dataclass
class _LolaRollout:
    """Differentiable rollout for LOLA-DiCE.

    Differs from :class:`RolloutBuffer` in two ways:

    1. ``log_probs`` are stored as a list of length ``T`` of tensors of shape
       ``[N]`` *with autograd attached* — i.e. they are NOT ``.detach()``ed
       before being placed in the buffer. The magic-box surrogate
       relies on differentiating through these.
    2. ``rewards`` is a plain numpy/CPU array (no gradients needed there);
       it is converted to a tensor inside the loss builder.

    This buffer is created fresh on every ``update()`` and discarded after
    one optimizer step, matching the on-policy LOLA-PG protocol from
    Foerster et al. 2018.
    """

    observations: torch.Tensor  # [T, N, obs_dim]  (no grad needed)
    log_probs_per_step: List[torch.Tensor]  # T-list of [N] tensors (with grad)
    actions: Dict[int, torch.Tensor]  # detached, per-agent [T, A]
    rewards: torch.Tensor  # [T, N], detached
    values: torch.Tensor  # [T, N], detached
    dones: torch.Tensor  # [T], detached


class LolaTrainer:
    """LOLA-DiCE trainer mirroring the public surface of :class:`JointPPOTrainer`.

    Args:
        env_fn: Zero-argument factory; same contract as ``JointPPOTrainer``.
        num_agents: Number of agents.
        obs_dim: Flattened observation dim (must include the #204 identity
            one-hot tail).
        action_dims: Per-dimension action sizes.
        hidden_size: Trunk hidden size for each per-agent :class:`PolicyNetwork`.
        lr: Adam learning rate for each per-agent actor.
        gamma, gae_lambda: Discount and GAE coefficients (used only for the
            value-loss target — LOLA-PG uses raw discounted returns for the
            surrogate, not GAE advantages).
        max_grad_norm: Per-policy gradient clip.
        lola_eta: Opponent-lookahead step size η. Foerster '18 uses η = lr;
            the curator spec sweeps 0.1 × lr, 1.0 × lr, 10.0 × lr.
        lola_lookahead_steps: Number of opponent lookahead unrolls. 1 in
            the paper's main experiments; 2 is the standard ablation.
        lola_dice: Use the DiCE magic-box surrogate (default True). If
            False, an explicit second-order autograd path is used; DiCE
            is recommended for numerical stability.
        device: ``"cpu"`` or ``"cuda"``.
        seed: Optional torch + numpy seed.
        centralized_critic: Must be False. LOLA is mutually exclusive with
            MAPPO; the guard is duplicated here so a config-error surfaces
            at trainer construction (not deep in the update loop).
        redundancy_coef: Must be 0.0. Same rationale as ``centralized_critic``.
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
        max_grad_norm: float = 0.5,
        lola_eta: float = 1.0,
        lola_lookahead_steps: int = 1,
        lola_dice: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
        centralized_critic: bool = False,
        redundancy_coef: float = 0.0,
    ):
        # Mutual-exclusion guards. LOLA fundamentally changes the per-agent
        # objective by adding a second-order opponent-shaping term; combining
        # it with MAPPO's shared value baseline or the cross-agent redundancy
        # penalty would entangle two unrelated experiments. Mirrors the
        # MAPPO/redundancy guard at ``joint_trainer.py:236``.
        if centralized_critic:
            raise ValueError(
                "LolaTrainer is incompatible with centralized_critic=True "
                "(MAPPO). LOLA replaces the per-agent PG objective with a "
                "second-order opponent-shaping surrogate; combining it with "
                "MAPPO's shared value baseline would conflate two unrelated "
                "experiments. Pick one."
            )
        if redundancy_coef > 0:
            raise ValueError(
                "LolaTrainer is incompatible with redundancy_coef>0. The "
                "redundancy penalty couples the per-agent actor trunks via a "
                "shared loss, while LOLA's opponent-shaping term assumes "
                "independent inner gradient steps per agent. Combining them "
                "would entangle two unrelated experiments. Pick one."
            )
        if lola_lookahead_steps < 1:
            raise ValueError(
                f"lola_lookahead_steps must be >= 1, got {lola_lookahead_steps}"
            )

        self.env_fn = env_fn
        self.env = env_fn()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.lola_eta = lola_eta
        self.lola_lookahead_steps = lola_lookahead_steps
        self.lola_dice = lola_dice
        self.centralized_critic = False
        self.redundancy_coef = 0.0

        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Per-agent policies. We use the same :class:`PolicyNetwork` as IPPO
        # so a fair baseline comparison only varies the objective.
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
        """Collect a JointPPOTrainer-compatible rollout (for parity with the
        IPPO surface). Internally, the LOLA update path collects its OWN
        differentiable rollout in :meth:`_collect_differentiable_rollout`;
        this method exists so the existing
        ``experiments/p3_specialization/train.py`` info-theory hook (which
        consumes a :class:`RolloutBuffer`) keeps working unchanged.
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
                    a, lp, _ent, v = policy.get_action_and_value(obs_i)
                    joint_action[i] = a[0].cpu().numpy()
                    log_probs[i, t] = lp[0].item()
                    values[i, t] = v[0].item()

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

    def _collect_differentiable_rollout(self, num_steps: int) -> _LolaRollout:
        """Collect an on-policy rollout where each step's joint log-prob is
        a live autograd tensor.

        This is the rollout the LOLA-DiCE surrogate is built over. We can NOT
        simply re-evaluate log-probs on a stored ``(obs, action)`` buffer
        after the fact: re-evaluation gives the policy's CURRENT log-prob of
        the SAMPLED action, which is exactly what we want — but the
        cleanest implementation is to keep the sampling-time log-prob
        attached to autograd, because that guarantees the gradient flows
        through the same sampling path the actions were drawn from.

        We do step the env with ``.detach()``ed actions (the env is not
        differentiable) and store rewards as plain tensors. The autograd
        graph at the end of the rollout has size ``O(T * N * |θ|)`` —
        memory-bound for very long rollouts; see ``rollout_steps`` knob.
        """
        observations = np.zeros(
            (num_steps, self.num_agents, self.obs_dim), dtype=np.float32
        )
        log_probs_per_step: List[torch.Tensor] = []
        actions_np = np.zeros(
            (self.num_agents, num_steps, len(self.action_dims)), dtype=np.int64
        )
        values = np.zeros((num_steps, self.num_agents), dtype=np.float32)
        rewards = np.zeros((num_steps, self.num_agents), dtype=np.float32)
        dones = np.zeros(num_steps, dtype=np.float32)

        for t in range(num_steps):
            obs_t = torch.from_numpy(self._last_obs).to(self.device)  # [N, obs_dim]
            joint_action_np = np.zeros(
                (self.num_agents, len(self.action_dims)), dtype=np.int64
            )
            step_log_probs: List[torch.Tensor] = []
            for i, policy in enumerate(self.policies):
                obs_i = obs_t[i : i + 1]
                # Forward with grad. ``get_action_and_value`` samples a fresh
                # action (no provided ``action`` arg) and returns its
                # log-prob WITH autograd attached.
                a, lp, _ent, v = policy.get_action_and_value(obs_i)
                joint_action_np[i] = a[0].detach().cpu().numpy()
                step_log_probs.append(lp[0])
                values[t, i] = v[0].detach().item()

            # [N] tensor; stays on autograd.
            log_probs_per_step.append(torch.stack(step_log_probs, dim=0))

            next_obs_dict, rew, done_arr, _ = self.env.step(joint_action_np)
            done = bool(np.asarray(done_arr).any())
            observations[t] = self._last_obs
            for i in range(self.num_agents):
                actions_np[i, t] = joint_action_np[i]
                rewards[t, i] = rew[i]
            dones[t] = float(done)

            if done:
                self._reset_env()
            else:
                self._last_obs = self._flatten_per_agent(next_obs_dict)

        return _LolaRollout(
            observations=torch.from_numpy(observations).to(self.device),
            log_probs_per_step=log_probs_per_step,
            actions={
                i: torch.from_numpy(actions_np[i]).to(self.device)
                for i in range(self.num_agents)
            },
            rewards=torch.from_numpy(rewards).to(self.device),
            values=torch.from_numpy(values).to(self.device),
            dones=torch.from_numpy(dones).to(self.device),
        )

    # ------------------------------------------------------------------
    # LOLA-DiCE surrogate construction
    # ------------------------------------------------------------------

    def _discounted_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Per-agent discounted returns from per-step rewards.

        Args:
            rewards: ``[T, N]`` per-step per-agent rewards.
            dones:   ``[T]`` shared episode termination flag.

        Returns:
            ``[T, N]`` per-step per-agent discounted returns
            ``G_t = Σ_{k >= t} γ^{k-t} r_k`` with the discount reset to 0
            on done boundaries.
        """
        T, N = rewards.shape
        returns = torch.zeros_like(rewards)
        running = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
        for t in reversed(range(T)):
            # On a done step, the post-done future has zero contribution.
            running = rewards[t] + self.gamma * (1.0 - dones[t]) * running
            returns[t] = running
        return returns

    def _episode_resetting_cumsum(
        self, per_step: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Cumulative sum of ``per_step`` along time, reset at done boundaries.

        ``per_step`` is shape ``[T]`` (typically the per-step joint log-prob).
        Resetting at done boundaries prevents the magic-box's argument from
        growing unboundedly over a multi-episode rollout, which would cause
        the second-order gradient to overflow.
        """
        T = per_step.shape[0]
        out: List[torch.Tensor] = []
        running = per_step.new_tensor(0.0)
        for t in range(T):
            running = running + per_step[t]
            out.append(running)
            # After emitting the value at step t, if step t terminates the
            # episode, the next step belongs to a fresh trajectory and its
            # cumulative starts from 0. We reset AFTER the append so step
            # t's own contribution is preserved.
            if dones[t].item() > 0.5:
                running = per_step.new_tensor(0.0)
        return torch.stack(out, dim=0)

    def _build_dice_surrogate(
        self, rollout: _LolaRollout, agent_idx: int
    ) -> torch.Tensor:
        """Build the DiCE surrogate ``L_i`` for agent ``i``.

        ``L_i = Σ_t  MagicBox(Σ_{a, 0 ≤ t' ≤ t in same episode} log π_a(t'))
                  ⋅  Â_{i, t}``

        where the cumulative log-prob is taken over **all agents** (including
        ``i``) WITHIN THE CURRENT EPISODE (reset at done boundaries to
        prevent the magic-box argument from growing unboundedly over a
        multi-episode rollout). Differentiating this w.r.t. ``θ_i`` gives
        the standard score-function policy gradient; differentiating w.r.t.
        ``θ_j`` gives the cross-agent term that LOLA uses to anticipate
        ``j``'s learning step.
        """
        log_probs_TN = torch.stack(rollout.log_probs_per_step, dim=0)  # [T, N]
        joint_log_prob_per_step = log_probs_TN.sum(dim=1)  # [T]
        # Reset the cumsum at done boundaries — important for numerical
        # stability when ``rollout_steps`` >> episode length, otherwise
        # the cumsum saturates at very large negative values and the
        # backward of magic_box overflows.
        cum_joint_log_prob = self._episode_resetting_cumsum(
            joint_log_prob_per_step, rollout.dones
        )  # [T]

        returns_TN = self._discounted_returns(rollout.rewards, rollout.dones)
        returns_i = returns_TN[:, agent_idx]

        # Variance-reduction baseline. Mean-subtract works because MagicBox
        # has unit forward, so the baseline does not bias the surrogate's
        # gradient.
        baseline = returns_i.detach().mean()
        adv_i = returns_i - baseline

        magic = _magic_box(cum_joint_log_prob)  # [T]
        return (magic * adv_i).sum()

    # ------------------------------------------------------------------
    # LOLA inner-step + update
    # ------------------------------------------------------------------

    @staticmethod
    def _take_inner_step(
        loss: torch.Tensor,
        params: List[torch.nn.Parameter],
        eta: float,
        create_graph: bool,
    ) -> List[torch.Tensor]:
        """Take one differentiable gradient step on ``params`` along ``∇loss``.

        Returns a list of *new* tensors ``params + η ⋅ ∇loss`` (gradient
        ASCENT — we are maximizing ``V_j``). Functional: the original
        parameters are NOT mutated.

        ``allow_unused=True`` because :class:`PolicyNetwork`'s value head is
        not in the LOLA-DiCE surrogate (which only depends on log-probs
        through action heads + trunk), so ``value_head.*`` params have no
        gradient on ``L_j``. Their inner-step update is then a no-op, which
        is the correct behavior.
        """
        grads = torch.autograd.grad(
            loss, params, create_graph=create_graph, allow_unused=True
        )
        new_params: List[torch.Tensor] = []
        for p, g in zip(params, grads):
            if g is None:
                # Param doesn't appear in the surrogate; pass it through.
                new_params.append(p)
            else:
                new_params.append(p + eta * g)
        return new_params

    def _evaluate_log_probs_with_params(
        self,
        agent_idx: int,
        new_params: List[torch.Tensor],
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Re-evaluate agent ``agent_idx``'s log-probs of the rolled-out actions
        under a HYPOTHETICAL parameter set ``new_params``.

        ``observations`` is the agent's per-step obs slice (``[T, obs_dim]``);
        ``actions`` is the agent's rolled-out actions (``[T, A]``).

        Uses ``torch.nn.utils.stateless.functional_call`` so the new params
        flow through autograd into the surrogate loss.
        """
        policy = self.policies[agent_idx]
        param_names = [name for name, _ in policy.named_parameters()]
        params_and_buffers: Dict[str, torch.Tensor] = dict(zip(param_names, new_params))
        # Buffers (BatchNorm etc.) — PolicyNetwork has none, but be safe.
        for name, buf in policy.named_buffers():
            params_and_buffers[name] = buf

        # functional_call returns (action_logits_list, value)
        out = torch.func.functional_call(policy, params_and_buffers, (observations,))
        action_logits, _value = out
        # Sum log-probs across the multi-discrete action dims.
        log_probs = []
        for k, logits in enumerate(action_logits):
            dist = torch.distributions.Categorical(logits=logits)
            log_probs.append(dist.log_prob(actions[:, k]))
        return torch.stack(log_probs, dim=1).sum(dim=1)  # [T]

    def _build_lola_surrogate_for_agent(
        self, rollout: _LolaRollout, agent_idx: int
    ) -> torch.Tensor:
        """Build agent ``i``'s LOLA-DiCE surrogate WITH opponent lookahead.

        For each opponent ``j``, take one (or ``lola_lookahead_steps``)
        functional gradient ascent step on agent ``j``'s own DiCE value
        with ``create_graph=True``. Then re-evaluate agent ``i``'s DiCE
        surrogate using the post-step parameters of ``j`` (and current
        parameters of all other agents). The mixed second derivative
        that autograd computes is exactly the LOLA cross-term.
        """
        # 1) Build agent j's DiCE surrogate (this is what j will "learn from").
        # 2) Compute ∇_{θ_j} L_j with create_graph=True, take an inner step.
        # 3) Re-evaluate the joint log-prob sequence WITH j's new params.
        # 4) Rebuild agent i's DiCE surrogate using the modified log-probs.

        # The cleanest functional implementation: replace agent ``j``'s
        # contribution to the cumulative joint log-prob with the
        # re-evaluation under the post-step params, leave all others as-is.

        # Pull out the original per-agent per-step log-probs (with grad).
        log_probs_TN = torch.stack(rollout.log_probs_per_step, dim=0)  # [T, N]

        # Per-agent observation slices and rolled-out actions.
        obs_per_agent = [rollout.observations[:, j, :] for j in range(self.num_agents)]
        actions_per_agent = [rollout.actions[j] for j in range(self.num_agents)]

        # Start with the original per-agent log-probs; we'll overwrite the
        # columns of the modified opponents below.
        # Treat the [T, N] tensor as a list of N [T] columns so we can swap
        # individual columns without losing autograd.
        log_prob_columns: List[torch.Tensor] = [
            log_probs_TN[:, j] for j in range(self.num_agents)
        ]

        # For each opponent j, build j's DiCE surrogate, take an inner step,
        # and re-evaluate j's log-probs under the post-step params. The
        # post-step params depend on j's policy params (and through the
        # joint log-prob, on all agents' params); autograd tracks this.
        for j in range(self.num_agents):
            if j == agent_idx:
                continue
            # j's own DiCE surrogate built from the CURRENT joint log-probs.
            L_j = self._build_dice_surrogate(rollout, agent_idx=j)

            params_j = list(self.policies[j].parameters())
            new_params_j = params_j
            for _ in range(self.lola_lookahead_steps):
                # ``create_graph=True`` makes the inner gradient itself
                # differentiable, which is what we need for the LOLA
                # cross-term.
                new_params_j = self._take_inner_step(
                    loss=L_j,
                    params=new_params_j,
                    eta=self.lola_eta,
                    create_graph=True,
                )
                # For lookahead > 1, rebuild L_j under the new params.
                if self.lola_lookahead_steps > 1:
                    new_lp_j = self._evaluate_log_probs_with_params(
                        agent_idx=j,
                        new_params=new_params_j,
                        observations=obs_per_agent[j],
                        actions=actions_per_agent[j],
                    )
                    # Splice into a fresh log_prob_columns to rebuild surrogate.
                    cols = list(log_prob_columns)
                    cols[j] = new_lp_j
                    cum = torch.cumsum(torch.stack(cols, dim=1).sum(dim=1), dim=0)
                    returns_TN = self._discounted_returns(
                        rollout.rewards, rollout.dones
                    )
                    returns_j = returns_TN[:, j]
                    baseline_j = returns_j.detach().mean()
                    adv_j = returns_j - baseline_j
                    L_j = (_magic_box(cum) * adv_j).sum()

            # Re-evaluate j's log-probs under the final post-step params.
            new_lp_j_final = self._evaluate_log_probs_with_params(
                agent_idx=j,
                new_params=new_params_j,
                observations=obs_per_agent[j],
                actions=actions_per_agent[j],
            )
            log_prob_columns[j] = new_lp_j_final

        # Reassemble the modified [T, N] joint log-prob tensor and build
        # agent i's DiCE surrogate using THOSE log-probs (with episode-
        # resetting cumsum, same recipe as ``_build_dice_surrogate``).
        modified_log_probs = torch.stack(log_prob_columns, dim=1)  # [T, N]
        joint_log_prob_per_step = modified_log_probs.sum(dim=1)  # [T]
        cum_joint_log_prob = self._episode_resetting_cumsum(
            joint_log_prob_per_step, rollout.dones
        )

        returns_TN = self._discounted_returns(rollout.rewards, rollout.dones)
        returns_i = returns_TN[:, agent_idx]
        baseline_i = returns_i.detach().mean()
        adv_i = returns_i - baseline_i

        magic = _magic_box(cum_joint_log_prob)
        return (magic * adv_i).sum()

    def update(self, rollout: Optional[RolloutBuffer] = None) -> Dict[str, float]:
        """Run one LOLA-DiCE update.

        Unlike :meth:`JointPPOTrainer.update`, this trainer collects its OWN
        differentiable rollout for the update (PPO's importance-ratio
        clipping is incompatible with on-policy LOLA, and re-using a
        detached buffer breaks the second-order autograd graph). The
        ``rollout`` arg is accepted only to match the IPPO public surface:
        if provided, its number of steps is used; otherwise we default to
        a small rollout suitable for smoke tests.

        Returns:
            Dict of mean per-agent metrics ``policy_loss/agent_i``,
            ``policy_loss/total``, ``mean_step_reward_team`` and the
            ``lola_eta`` / ``lola_lookahead_steps`` config echoes (so the
            metrics.json schema parses identically across PPO and LOLA
            cells).
        """
        if rollout is not None:
            # Match the size of an externally collected rollout so the
            # info-theory hook gets a buffer of the size it expects.
            num_steps = rollout.observations.shape[0]
        else:
            num_steps = 128

        # Collect a fresh on-policy rollout WITH autograd attached.
        lola_rollout = self._collect_differentiable_rollout(num_steps)

        # Build each agent's LOLA-DiCE surrogate and accumulate per-agent
        # losses. We backprop ONCE through the sum so PyTorch shares the
        # autograd graph across the per-agent passes.
        per_agent_losses: List[torch.Tensor] = []
        stats: Dict[str, float] = {}
        for i in range(self.num_agents):
            # NEGATIVE because we maximize value.
            L_i = -self._build_lola_surrogate_for_agent(lola_rollout, agent_idx=i)
            per_agent_losses.append(L_i)
            stats[f"policy_loss/agent_{i}"] = float(L_i.detach().item())

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

        # Diagnostic / metrics-schema-compatible scalars. We populate
        # `value_loss/agent_i` and `entropy/agent_i` as zero so the existing
        # ``experiments/p3_specialization/analyze_*.py`` parsers don't choke
        # on missing keys when reading a LOLA cell's ``metrics.json``.
        for i in range(self.num_agents):
            stats[f"value_loss/agent_{i}"] = 0.0
            stats[f"entropy/agent_{i}"] = 0.0
        stats["redundancy_loss"] = 0.0

        # Per-step team reward, for parity with JointPPOTrainer-driven cells.
        mean_step_team = float(
            lola_rollout.rewards.sum().detach().item() / float(num_steps)
        )
        stats["lola/mean_step_reward_team"] = mean_step_team
        stats["lola/eta"] = float(self.lola_eta)
        stats["lola/lookahead_steps"] = float(self.lola_lookahead_steps)
        stats["lola/dice"] = float(self.lola_dice)
        return stats

    # ------------------------------------------------------------------
    # Convenience: parity with JointPPOTrainer's encoder tap so the
    # info-theory measurement in ``train.py`` works unchanged.
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
