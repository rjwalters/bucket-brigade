"""Single-agent joint-action wrapper over ``BucketBrigadeEnv`` (issue #291).

This wrapper collapses the multi-agent decomposition of
:class:`BucketBrigadeEnv` into a *single* controller that emits the joint
action for all ``num_agents`` sub-agents each step. The underlying environment
mechanics, scenario, observation contents, episode length, fire dynamics, and
per-agent reward components are **unchanged**; only the question of "who
decides the actions" changes.

This is the scope-determining experiment for the misaligned-gradient thesis
(see ``research_notebook/2026-05-17_thesis_misaligned_gradients.md`` open
question #1 and issue #291): does the basin trap survive when multi-agent
credit assignment is removed as a confound?

NB on scope:

- This is **not** :issue:`286` (macro-action wrapper). :issue:`286` coarsens
  a *single* agent into temporally-extended options; this issue removes the
  multi-agent decomposition entirely so that one controller emits the joint
  action vector for *all* 4 sub-agents simultaneously. Different mechanic.

- This is **not** :issue:`292` (toy 2-agent iterated dilemma). :issue:`292`
  reduces the env itself; this wrapper keeps ``minimal_specialization``
  intact and reduces only the *training* decomposition.

API:

- :meth:`SingleAgentJointWrapper.reset` returns the concatenated per-agent
  observation as a single flat vector (the same flattened representation
  used by :func:`bucket_brigade.training.joint_trainer.flatten_dict_obs`,
  one row per sub-agent stacked end-to-end).

- :meth:`SingleAgentJointWrapper.step` accepts a joint action either as a
  flat ``[N*A]`` vector or a ``[N, A]`` array. The flat layout matches the
  factorized joint head produced by a single :class:`PolicyNetwork` with
  ``action_dims = [10, 2, 2] * num_agents``. The wrapper splits the joint
  action into per-agent ``[N, A]`` chunks, steps the underlying env, and
  returns ``(obs, sum_team_reward, all_done, info)``.

The single returned obs is a 2-D ``[N, obs_dim_per_agent]`` array (matching
the shape that :class:`JointPPOTrainer` already expects), so the existing
trainer can consume the wrapper unchanged when configured with
``num_agents=1`` and a sub-agent-aware ``flatten`` step (see the companion
training script ``experiments/p3_specialization/train_single_agent.py``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.training.joint_trainer import flatten_dict_obs


__all__ = ["SingleAgentJointWrapper"]


# Per-sub-agent action layout. Mirrors the post-#235 ``[house, mode, signal]``
# vector consumed by ``BucketBrigadeEnv.step``. Hard-coded here (rather than
# pulled from the env) because the wrapper exposes a flat joint action space
# whose shape is fixed at construction time.
_ACTION_DIM_PER_AGENT = 3


class SingleAgentJointWrapper:
    """Wrap a multi-agent ``BucketBrigadeEnv`` as a single-agent joint-action env.

    The wrapper holds one :class:`BucketBrigadeEnv` instance. From the
    controller's perspective:

    - One observation per step: the per-agent flattened observation rows
      stacked into a ``[num_agents, obs_dim_per_agent]`` array. The per-agent
      flattening uses the existing :func:`flatten_dict_obs` with the
      ``agent_id`` one-hot tail (issue #204), so each sub-agent row remains
      distinguishable to a controller that processes them jointly.

    - One scalar reward per step: the **team sum** ``sum(per_agent_rewards)``.
      This eliminates the per-agent credit-assignment signal that IPPO would
      otherwise route to per-agent advantage estimators; a single controller
      training against the team sum cannot blame "the other agent" for low
      reward.

    - One done flag per step: ``True`` iff the underlying env signaled done
      on any agent (the env signals done uniformly; this is the standard
      convention).

    Args:
        env: A :class:`BucketBrigadeEnv` instance to wrap. The wrapper takes
            ownership of stepping/resetting it. Construct with the desired
            scenario before passing in.

    Attributes:
        env: The wrapped multi-agent env.
        num_agents: Cached sub-agent count from the wrapped env.
        action_dims_per_agent: The post-#235 ``[house, mode, signal]`` layout
            for a single sub-agent. Always ``[num_houses, 2, 2]``.
        joint_action_dims: The flat factorized joint action layout consumed by
            a single-controller :class:`PolicyNetwork`. Length
            ``num_agents * 3``: ``[house_0, mode_0, signal_0, house_1, ...]``.
        joint_action_size: The product of ``joint_action_dims`` — the
            cardinality of the joint discrete action space. For
            ``minimal_specialization`` (10 houses, 4 agents) this is
            ``(10 * 2 * 2) ** 4 = 2_560_000``.
        obs_dim_per_agent: Length of one sub-agent's flattened observation
            (with the identity one-hot tail).
    """

    def __init__(self, env: BucketBrigadeEnv) -> None:
        self.env = env
        self.num_agents = int(env.num_agents)
        # Per-sub-agent multi-discrete layout: [house, mode, signal].
        # ``num_houses`` is scenario-dependent (10 for minimal_specialization,
        # 2 for v2_minimal); ``mode`` and ``signal`` are binary.
        self.action_dims_per_agent: List[int] = [int(env.num_houses), 2, 2]
        # Flat joint factorized action layout: concat of per-agent dims.
        self.joint_action_dims: List[int] = (
            list(self.action_dims_per_agent) * self.num_agents
        )
        # Cardinality of the joint discrete action space (informational; the
        # factorized head doesn't enumerate it).
        self.joint_action_size: int = int(np.prod(self.action_dims_per_agent)) ** int(
            self.num_agents
        )
        # Probe obs_dim from a single reset so callers can size their policy
        # network without re-deriving the flatten layout. We use a fixed seed
        # for the probe and rely on the caller to re-seed via ``reset(seed=)``
        # before training begins.
        probe_obs = env.reset(seed=0)
        self.obs_dim_per_agent: int = flatten_dict_obs(
            probe_obs, agent_id=0, num_agents=self.num_agents
        ).shape[0]

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the wrapped env and return the flat per-agent observation stack.

        Args:
            seed: Optional seed forwarded to the underlying env.

        Returns:
            Observation array of shape ``[num_agents, obs_dim_per_agent]``.
            Row ``i`` is the post-#204 per-agent flattened obs for sub-agent
            ``i`` (with the identity one-hot tail). The controller is
            expected to flatten further (e.g., concatenate rows) before
            feeding to a single policy network.
        """
        obs_dict = self.env.reset(seed=seed)
        return self._flatten_per_agent(obs_dict)

    def step(
        self, joint_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, np.ndarray]]:
        """Step the wrapped env with a joint action.

        Args:
            joint_action: Either a flat ``[N * 3]`` ``int`` array
                (``[house_0, mode_0, signal_0, house_1, ...]``) or a
                ``[N, 3]`` ``int`` array. Both shapes split into the same
                per-agent action tensor before being passed to the env.

        Returns:
            Tuple ``(obs, team_reward, done, info)`` where:

            - ``obs`` is shape ``[N, obs_dim_per_agent]`` (same layout as
              :meth:`reset`).
            - ``team_reward`` is the scalar ``sum(per_agent_rewards)``
              produced by the wrapped env this step. This is the
              credit-assignment-free reward signal the single controller
              trains against.
            - ``done`` is a Python ``bool`` — ``True`` iff the wrapped env
              flagged any agent done (the env signals done uniformly).
            - ``info`` is a small dict echoing the per-agent rewards
              vector (``"per_agent_rewards"``) so analyzers can decompose
              the team reward if they wish.
        """
        per_agent_action = self._split_joint_action(joint_action)
        obs_dict, rewards, dones, _info = self.env.step(per_agent_action)
        team_reward = float(np.asarray(rewards, dtype=np.float64).sum())
        done = bool(np.asarray(dones).any())
        next_obs = self._flatten_per_agent(obs_dict)
        info: Dict[str, np.ndarray] = {
            "per_agent_rewards": np.asarray(rewards, dtype=np.float32).copy()
        }
        return next_obs, team_reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_per_agent(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack per-agent flattened obs rows into ``[N, obs_dim_per_agent]``.

        Each row uses the post-#204 per-agent flattening (with the identity
        one-hot tail) so a downstream controller that flattens the stack
        further still sees per-sub-agent distinguishability.
        """
        rows = [
            flatten_dict_obs(obs_dict, agent_id=i, num_agents=self.num_agents)
            for i in range(self.num_agents)
        ]
        return np.stack(rows, axis=0).astype(np.float32)

    def _split_joint_action(self, joint_action: np.ndarray) -> np.ndarray:
        """Split a joint action vector into a ``[N, 3]`` per-agent array.

        Accepts both flat ``[N * 3]`` and pre-shaped ``[N, 3]`` inputs.
        Validates shape and dtype; raises ``ValueError`` on mismatch so
        upstream bugs surface immediately instead of silently corrupting
        per-agent action assignment.
        """
        arr = np.asarray(joint_action)
        if arr.ndim == 1:
            expected = self.num_agents * _ACTION_DIM_PER_AGENT
            if arr.size != expected:
                raise ValueError(
                    f"flat joint_action has size {arr.size}, expected "
                    f"{expected} = num_agents({self.num_agents}) * "
                    f"action_dim_per_agent({_ACTION_DIM_PER_AGENT})"
                )
            arr = arr.reshape(self.num_agents, _ACTION_DIM_PER_AGENT)
        elif arr.ndim == 2:
            if arr.shape != (self.num_agents, _ACTION_DIM_PER_AGENT):
                raise ValueError(
                    f"2-D joint_action has shape {tuple(arr.shape)}, "
                    f"expected ({self.num_agents}, {_ACTION_DIM_PER_AGENT})"
                )
        else:
            raise ValueError(f"joint_action must be 1-D or 2-D, got ndim={arr.ndim}")
        return arr.astype(np.int64, copy=False)
