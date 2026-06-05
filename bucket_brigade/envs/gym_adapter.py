"""Gymnasium-compatible adapter for Bucket Brigade (issue #369).

This module exposes :class:`BucketBrigadeGymEnv`, a thin
:class:`gymnasium.Env` wrapper around the existing multi-agent
:class:`~bucket_brigade.envs.bucket_brigade_env.BucketBrigadeEnv`. The
adapter collapses the per-agent decomposition into a single-controller
joint-action view, mirroring the design of
:class:`~bucket_brigade.envs.single_agent_wrapper.SingleAgentJointWrapper`
but staying self-contained (no dependency on the Rust ``bucket_brigade_core``
module — see issue #369 for why this matters for the Gym surface).

The adapter is intentionally **thin**: it does not own any game logic, it
just maps the joint-controller API to Gymnasium's ``(obs, reward,
terminated, truncated, info)`` convention with proper ``observation_space``
and ``action_space`` declarations.

Why single-agent (and not PettingZoo Parallel)?
-----------------------------------------------

The first slice of issue #365 (issue #369) picks **single-agent joint
control** because:

1. It is the simplest surface that satisfies the Gym/Gymnasium contract
   exactly, with zero new mechanics — every existing P3 / specialization
   diagnostic already uses :class:`SingleAgentJointWrapper`.
2. Stable-Baselines3, CleanRL, Tianshou, and similar single-agent
   pipelines can consume the result with no special handling.
3. A PettingZoo Parallel surface remains valuable but is a separate
   slice (out of scope per the issue body).

The trade-off: external multi-agent researchers who want per-agent reward
streams get the team sum here, but the per-agent rewards are surfaced
via ``info["per_agent_rewards"]`` (the wrapper preserves them verbatim).

Gymnasium compliance
--------------------

- ``reset(seed=...) -> (obs, info)`` — info dict is empty on reset.
- ``step(action) -> (obs, reward, terminated, truncated, info)`` —
  ``truncated`` is always ``False`` (the env signals termination via
  ``terminated`` only; episode length is bounded by ``min_nights`` +
  burn-out, not a time cap).
- ``observation_space`` is a :class:`gymnasium.spaces.Box` matching the
  flattened ``[num_agents * obs_dim_per_agent]`` layout produced by the
  wrapper (further flattened from its native ``[N, D]`` shape so the
  space declaration is a simple 1-D Box).
- ``action_space`` is a :class:`gymnasium.spaces.MultiDiscrete` matching
  the joint factorized layout
  ``[num_houses, 2, 2, num_houses, 2, 2, ...]``
  (length ``num_agents * 3``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bucket_brigade_env import BucketBrigadeEnv
from .scenarios_generated import Scenario


__all__ = ["BucketBrigadeGymEnv"]


# Per-sub-agent action layout used by ``BucketBrigadeEnv.step``: the
# post-#235 ``[house, mode, signal]`` vector. Hard-coded here so the
# adapter is self-contained — the equivalent constant in
# :class:`SingleAgentJointWrapper` is the canonical source of truth.
_ACTION_DIM_PER_AGENT = 3


def _flatten_per_agent_obs(
    obs: Dict[str, Any], agent_id: int, num_agents: int
) -> np.ndarray:
    """Per-agent flat observation with identity one-hot tail (#204 layout).

    Replicates the relevant portion of
    :func:`bucket_brigade.training.joint_trainer.flatten_dict_obs` so the
    Gym adapter does NOT need to import from ``bucket_brigade.training``
    (which transitively requires the Rust ``bucket_brigade_core`` module).
    Keeping the adapter self-contained means a user who only wants the
    Gym surface does not pay the cost of building the Rust extension.

    Layout matches the post-#204, pre-#252 default produced by the
    canonical flattener with ``include_round1_signals=False``:
        ``[houses, signals(N), locations(N), last_actions(2N),
        scenario_info, identity_one_hot(N)]``.
    """
    base = [
        np.asarray(obs["houses"], dtype=np.float32),
        np.asarray(obs["signals"], dtype=np.float32),
        np.asarray(obs["locations"], dtype=np.float32),
        np.asarray(obs["last_actions"], dtype=np.float32).flatten(),
        np.asarray(obs["scenario_info"], dtype=np.float32),
    ]
    identity = np.zeros(num_agents, dtype=np.float32)
    identity[agent_id] = 1.0
    base.append(identity)
    return np.concatenate(base)


class BucketBrigadeGymEnv(gym.Env):
    """Gymnasium :class:`Env` wrapping a frozen Bucket Brigade scenario.

    The env is constructed from a :class:`Scenario` (typically produced by
    :func:`bucket_brigade.envs.registry.get_scenario_by_id`) and a
    ``num_agents`` count. Internally it owns one
    :class:`SingleAgentJointWrapper` instance and forwards
    ``reset``/``step`` to it after Gymnasium-style shape adaptation.

    Args:
        scenario: A :class:`Scenario` instance to drive the env.
        scenario_id: Optional frozen versioned ID (e.g.
            ``"minimal_specialization-v1"``) for traceability. Stored on
            ``self.metadata["scenario_id"]`` and exposed via
            ``info["scenario_id"]`` on every ``reset``/``step``.

    Attributes:
        observation_space: 1-D ``Box`` of length
            ``num_agents * obs_dim_per_agent``, dtype ``float32``.
        action_space: :class:`MultiDiscrete` of shape ``(num_agents * 3,)``
            with per-position dims ``[num_houses, 2, 2] * num_agents``.
        metadata: Standard Gymnasium metadata dict; includes
            ``"scenario_id"`` when constructed via :func:`bucket_brigade.make`.

    Example:
        >>> import bucket_brigade
        >>> env = bucket_brigade.make("minimal_specialization-v1")
        >>> obs, info = env.reset(seed=0)
        >>> obs.shape == env.observation_space.shape
        True
        >>> action = env.action_space.sample()
        >>> obs2, reward, terminated, truncated, info = env.step(action)
    """

    # Gymnasium standard metadata. ``render_modes`` is empty because the
    # underlying env exposes a ``render_text()`` print-to-stdout helper
    # but no first-class Gym render mode; we can add modes in a follow-up.
    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario,
        scenario_id: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Build the underlying multi-agent env. We do a probe ``reset(seed=0)``
        # to size the observation space; the probe seed is overwritten on
        # every user-facing ``reset(seed=...)`` call so it does not leak
        # into reproducibility.
        self._mae = BucketBrigadeEnv(scenario=scenario)

        self.num_agents: int = int(self._mae.num_agents)
        self.num_houses: int = int(self._mae.num_houses)
        self.scenario: Scenario = scenario

        # Stash the frozen ID (if known) for traceability.
        self._scenario_id = scenario_id
        # Copy metadata so per-instance overrides don't mutate the class
        # attribute shared by all instances.
        self.metadata = dict(self.metadata)
        if scenario_id is not None:
            self.metadata["scenario_id"] = scenario_id

        # Per-agent multi-discrete layout: [num_houses, 2, 2].
        # Mirrors ``SingleAgentJointWrapper.action_dims_per_agent``.
        self._action_dims_per_agent = [self.num_houses, 2, 2]
        joint_action_dims = list(self._action_dims_per_agent) * self.num_agents
        self.action_space = spaces.MultiDiscrete(
            np.asarray(joint_action_dims, dtype=np.int64)
        )

        # Probe obs to size the observation space. We call reset(seed=0)
        # then re-call it on every user-facing reset() so the probe is
        # not visible to callers.
        probe_obs = self._mae.reset(seed=0)
        self._obs_dim_per_agent: int = int(
            _flatten_per_agent_obs(
                probe_obs, agent_id=0, num_agents=self.num_agents
            ).shape[0]
        )
        flat_dim = self.num_agents * self._obs_dim_per_agent
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

        # Track whether reset() has been called so step() can fail
        # loudly on a pre-reset call (Gymnasium convention).
        self._reset_called: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the env to a fresh episode.

        Args:
            seed: Optional seed forwarded to the underlying env. If
                ``None`` the previous seed (or the wrapper's probe seed
                of 0) continues to advance.
            options: Unused (reserved for future use; ignored).

        Returns:
            Tuple ``(obs, info)`` where ``obs`` is a 1-D ``float32`` array
            of shape ``self.observation_space.shape`` and ``info`` is a
            small dict (includes ``"scenario_id"`` when known).
        """
        # Honor Gymnasium's super().reset(seed=...) contract so the parent
        # class's np_random gets seeded too; we don't actually rely on
        # ``self.np_random`` but external consumers might.
        super().reset(seed=seed)

        obs_dict = self._mae.reset(seed=seed)
        obs_flat = self._stack_and_flatten(obs_dict)

        self._reset_called = True
        info: Dict[str, Any] = {}
        if self._scenario_id is not None:
            info["scenario_id"] = self._scenario_id
        return obs_flat, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take one joint-action step.

        Args:
            action: An array conforming to ``self.action_space`` —
                shape ``(num_agents * 3,)``, dtype int-like, with per
                position bounds matching the ``MultiDiscrete`` dims. The
                underlying wrapper also accepts a ``(num_agents, 3)``
                reshape; we coerce to the flat layout for the
                ``action_space.contains`` check to pass cleanly.

        Returns:
            Tuple ``(obs, reward, terminated, truncated, info)``:

            - ``obs`` shape/dtype matches :attr:`observation_space`.
            - ``reward`` is the team-sum scalar (single-controller view).
            - ``terminated`` is ``True`` iff the underlying env signaled
              done (all fires out or all houses ruined past
              ``min_nights``).
            - ``truncated`` is always ``False`` — episode length is
              dynamics-driven, not time-capped.
            - ``info`` carries ``"per_agent_rewards"`` (verbatim from the
              wrapper) and ``"scenario_id"`` when known.
        """
        if not self._reset_called:
            raise RuntimeError(
                "BucketBrigadeGymEnv.step() called before reset(). "
                "Gymnasium requires reset() before the first step()."
            )

        # Coerce to [N, 3] for the underlying env. Accept both flat
        # ``[N*3]`` (the canonical action_space layout) and pre-shaped
        # ``[N, 3]`` so callers that already pre-shape don't pay an
        # unnecessary reshape.
        action_arr = np.asarray(action, dtype=np.int64)
        if action_arr.ndim == 1:
            expected = self.num_agents * _ACTION_DIM_PER_AGENT
            if action_arr.size != expected:
                raise ValueError(
                    f"flat action has size {action_arr.size}, expected "
                    f"{expected} = num_agents({self.num_agents}) * "
                    f"action_dim_per_agent({_ACTION_DIM_PER_AGENT})"
                )
            action_arr = action_arr.reshape(self.num_agents, _ACTION_DIM_PER_AGENT)

        obs_dict, rewards, dones, _ = self._mae.step(action_arr)
        team_reward = float(np.asarray(rewards, dtype=np.float64).sum())
        terminated = bool(np.asarray(dones).any())
        truncated = False

        obs_flat = self._stack_and_flatten(obs_dict)
        info: Dict[str, Any] = {
            "per_agent_rewards": np.asarray(rewards, dtype=np.float32).copy(),
        }
        if self._scenario_id is not None:
            info["scenario_id"] = self._scenario_id
        return obs_flat, team_reward, terminated, truncated, info

    def render(self) -> None:
        """Render the env. ``render_modes`` is empty so this is a no-op.

        Use ``self._mae.render_text()`` directly for the existing
        text-based render helper.
        """
        # Gymnasium 0.28+ no longer requires render() to do anything when
        # render_modes is empty; we keep the method for explicit
        # discoverability.
        return None

    def close(self) -> None:
        """Release resources. The env holds only Python state, so this
        is a no-op; provided for Gymnasium-API completeness."""
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stack_and_flatten(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Build the flat per-agent obs stack from an env obs dict.

        Stacks per-agent rows produced by :func:`_flatten_per_agent_obs`
        (each carrying the identity one-hot tail from #204) and
        flattens to 1-D ``float32``. The Gymnasium
        ``observation_space`` is declared 1-D for maximum compatibility
        with off-the-shelf RL libraries (SB3, CleanRL, etc.); callers
        that want the per-agent view can call
        ``obs.reshape(num_agents, -1)``.
        """
        rows = [
            _flatten_per_agent_obs(obs_dict, agent_id=i, num_agents=self.num_agents)
            for i in range(self.num_agents)
        ]
        return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32).reshape(
            -1
        )
