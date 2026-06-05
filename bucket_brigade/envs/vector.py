"""Synchronous vectorized env wrapper for Bucket Brigade (issue #370).

This module exposes :class:`SyncVectorEnv` and the public factory
:func:`make_vec`, providing a Gymnasium-compatible vectorized view over
:class:`~bucket_brigade.envs.gym_adapter.BucketBrigadeGymEnv`.

Design choices
--------------

- **Sync / in-process only.** No multiprocessing, no asyncio. Bucket
  Brigade dynamics are cheap (pure-Python with optional Rust acceleration
  in the underlying env) so the per-step cross-process IPC overhead would
  dominate. A subprocess backend is a follow-up if benchmark numbers
  demand it (see issue #365 parent epic).
- **Gymnasium auto-reset convention.** When a sub-env reports
  ``terminated`` or ``truncated``, this wrapper resets that sub-env
  in-place and surfaces the **terminal** observation/info under the
  ``final_observation`` / ``final_info`` keys of the batched ``info``
  dict, while the batched ``observations`` slot for that sub-env is
  the **post-reset** observation. This matches
  :class:`gymnasium.vector.SyncVectorEnv` exactly so SB3 / CleanRL
  consumers can swap in without code changes.
- **Per-sub-env seed plumbing.** ``reset(seed=base)`` seeds sub-env ``i``
  with ``base + i`` so each lane gets a distinct deterministic stream
  while preserving reproducibility across constructions. Passing a list
  of seeds is also supported (one seed per sub-env).
- **Single shared scenario, num_envs independent envs.** Each sub-env
  is constructed via :func:`bucket_brigade.make` with the same ``id`` and
  ``num_agents`` kwargs, so they share the frozen scenario contract but
  hold independent state. The auto-reset semantics rely on this — each
  sub-env is fully isolated.

See :func:`make_vec` for the public entry point.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from gymnasium import spaces

from .gym_adapter import BucketBrigadeGymEnv


__all__ = ["SyncVectorEnv", "make_vec"]


# A seed argument to ``reset()`` can be either a single ``int`` (we expand
# to ``[seed, seed+1, ..., seed+num_envs-1]`` so each lane is distinct) or
# an explicit per-sub-env sequence. ``None`` means "don't re-seed".
_SeedArg = Optional[Union[int, Sequence[Optional[int]]]]


class SyncVectorEnv:
    """Synchronous vectorized wrapper over N independent Bucket Brigade envs.

    The API matches :class:`gymnasium.vector.SyncVectorEnv`:

    - :meth:`reset` returns ``(obs, info)`` where ``obs`` is a stacked
      ``(num_envs, *obs_shape)`` ndarray and ``info`` is a dict of
      batched / list-valued sub-env info entries.
    - :meth:`step` returns ``(obs, rewards, terminated, truncated, info)``
      with batched ndarray fields plus a Gymnasium-convention info dict.
      Sub-envs that report done are auto-reset in-place; their terminal
      observation and info are surfaced under ``info["final_observation"]``
      and ``info["final_info"]``.

    The wrapper is **synchronous** — every ``step()`` call iterates over
    sub-envs in order in the current process. This is the cheapest backend
    and is sufficient for the workloads in this repo (CPU-bound dynamics,
    small per-step cost). A multiprocessing backend may be added later if
    PPO rollout throughput becomes a bottleneck.

    Attributes:
        num_envs: Number of sub-envs.
        envs: List of underlying :class:`BucketBrigadeGymEnv` instances.
        single_observation_space: ``Box`` matching a single sub-env's
            observation_space (NOT batched).
        single_action_space: ``MultiDiscrete`` matching a single sub-env's
            action_space (NOT batched).
        observation_space: Batched ``Box`` of shape
            ``(num_envs, *single_observation_space.shape)``.
        action_space: Batched ``MultiDiscrete`` of shape
            ``(num_envs, *single_action_space.shape)``.
        metadata: Forwarded from the first sub-env.

    Example:
        >>> import bucket_brigade
        >>> vec = bucket_brigade.make_vec("minimal_specialization-v1", num_envs=8)
        >>> obs, info = vec.reset(seed=0)
        >>> obs.shape[0] == 8
        True
        >>> actions = vec.action_space.sample()
        >>> obs, rewards, terminated, truncated, info = vec.step(actions)
        >>> rewards.shape
        (8,)
    """

    def __init__(self, envs: Sequence[BucketBrigadeGymEnv]) -> None:
        if len(envs) == 0:
            raise ValueError("SyncVectorEnv requires at least one sub-env.")
        self.envs: List[BucketBrigadeGymEnv] = list(envs)
        self.num_envs: int = len(self.envs)

        # Sanity: every sub-env must share the same observation/action
        # space shape. Different shapes would break the batched ndarray
        # contract.
        e0 = self.envs[0]
        self.single_observation_space: spaces.Box = e0.observation_space
        self.single_action_space: spaces.MultiDiscrete = e0.action_space
        for i, e in enumerate(self.envs[1:], start=1):
            if e.observation_space.shape != self.single_observation_space.shape:
                raise ValueError(
                    f"Sub-env {i} observation_space.shape "
                    f"{e.observation_space.shape} != "
                    f"{self.single_observation_space.shape}"
                )
            if not np.array_equal(
                np.asarray(e.action_space.nvec),
                np.asarray(self.single_action_space.nvec),
            ):
                raise ValueError(
                    f"Sub-env {i} action_space.nvec disagrees with sub-env 0."
                )

        # Batched spaces. Gymnasium represents these as Box / MultiDiscrete
        # with a leading num_envs axis. We mirror that convention so
        # ``self.action_space.sample()`` returns a ``(num_envs, *action_shape)``
        # batched action ready for ``step()``.
        obs_shape = (self.num_envs,) + tuple(self.single_observation_space.shape)
        self.observation_space = spaces.Box(
            low=float(self.single_observation_space.low.min()),
            high=float(self.single_observation_space.high.max()),
            shape=obs_shape,
            dtype=self.single_observation_space.dtype,
        )
        # Tile MultiDiscrete nvec across the leading num_envs axis. This
        # yields the same per-position bounds for every lane.
        per_lane_nvec = np.asarray(self.single_action_space.nvec, dtype=np.int64)
        batched_nvec = np.tile(per_lane_nvec, (self.num_envs, 1))
        self.action_space = spaces.MultiDiscrete(batched_nvec)

        # Forward metadata from sub-env 0 (typically carries ``scenario_id``).
        self.metadata: Dict[str, Any] = dict(e0.metadata)
        self.metadata["num_envs"] = self.num_envs

        # Whether reset() has been called — Gymnasium convention is to
        # require it before the first step().
        self._reset_called: bool = False

    # ------------------------------------------------------------------
    # Gymnasium VectorEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: _SeedArg = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset every sub-env and return a batched ``(obs, info)``.

        Args:
            seed: Either ``None`` (no re-seed), a single ``int`` (sub-env
                ``i`` is seeded with ``seed + i``), or a sequence of
                ``num_envs`` ints / ``None`` values (one per sub-env).
            options: Forwarded verbatim to each sub-env. Currently unused
                by :class:`BucketBrigadeGymEnv`.

        Returns:
            Tuple ``(obs, info)``:

            - ``obs`` is a ``(num_envs, *single_obs_shape)`` ndarray
              matching ``self.observation_space.dtype``.
            - ``info`` is a dict whose values are length-``num_envs``
              lists (one entry per sub-env) — this matches Gymnasium's
              "list-of-info" convention for the sync backend.
        """
        seeds = self._expand_seeds(seed)

        obs_list: List[np.ndarray] = []
        infos: List[Dict[str, Any]] = []
        for env, s in zip(self.envs, seeds):
            o, info = env.reset(seed=s, options=options)
            obs_list.append(o)
            infos.append(info)

        self._reset_called = True
        return self._stack_obs(obs_list), self._batch_info(infos)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step every sub-env with one row of ``actions`` each.

        Args:
            actions: A ``(num_envs, *single_action_shape)`` ndarray. Each
                row is forwarded to the corresponding sub-env's
                ``step()``. The flat ``(num_envs * action_dim,)`` shape
                is also accepted and auto-reshaped for caller convenience.

        Returns:
            Tuple ``(obs, rewards, terminated, truncated, info)``:

            - ``obs`` is a ``(num_envs, *obs_shape)`` ndarray. For
              sub-envs that hit done, the row is the **post-reset**
              observation; the **terminal** observation appears under
              ``info["final_observation"][i]``.
            - ``rewards`` is a ``(num_envs,)`` float32 ndarray.
            - ``terminated`` is a ``(num_envs,)`` bool ndarray reflecting
              the pre-auto-reset done flags.
            - ``truncated`` is a ``(num_envs,)`` bool ndarray (always
              all-False under the current Bucket Brigade dynamics, but
              we surface the field for forward compatibility).
            - ``info`` is a dict of list-of-sub-env-info values plus the
              Gymnasium auto-reset bookkeeping keys
              ``"final_observation"`` (list, ``None`` for non-done lanes)
              and ``"final_info"`` (list, ``None`` for non-done lanes).
        """
        if not self._reset_called:
            raise RuntimeError(
                "SyncVectorEnv.step() called before reset(). Call reset() "
                "before the first step()."
            )

        actions = np.asarray(actions)
        # Accept flat ``(num_envs * D,)`` for convenience (mirrors the
        # single-env adapter's reshape path).
        single_action_size = int(np.prod(self.single_action_space.shape))
        if actions.ndim == 1 and actions.size == self.num_envs * single_action_size:
            actions = actions.reshape(self.num_envs, *self.single_action_space.shape)

        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"actions has leading dim {actions.shape[0]}, expected "
                f"num_envs={self.num_envs}"
            )

        obs_list: List[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = []
        # Per-Gymnasium convention: for non-done lanes these entries are
        # ``None`` so callers can ``zip``-iterate safely.
        final_observations: List[Optional[np.ndarray]] = [None] * self.num_envs
        final_infos: List[Optional[Dict[str, Any]]] = [None] * self.num_envs

        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(actions[i])
            rewards[i] = reward
            terminated[i] = term
            truncated[i] = trunc

            if term or trunc:
                # Auto-reset: stash the terminal observation/info, then
                # roll the sub-env into a fresh episode. The batched
                # ``obs[i]`` row carries the post-reset observation,
                # matching gymnasium.vector.SyncVectorEnv semantics.
                final_observations[i] = obs
                final_infos[i] = info
                reset_obs, reset_info = env.reset()
                obs_list.append(reset_obs)
                infos.append(reset_info)
            else:
                obs_list.append(obs)
                infos.append(info)

        batched_info = self._batch_info(infos)
        batched_info["final_observation"] = final_observations
        batched_info["final_info"] = final_infos
        return (
            self._stack_obs(obs_list),
            rewards,
            terminated,
            truncated,
            batched_info,
        )

    def close(self) -> None:
        """Close every sub-env. Safe to call multiple times."""
        for env in self.envs:
            env.close()

    def render(self) -> None:
        """Render is a no-op (matching the single-env adapter)."""
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_seeds(self, seed: _SeedArg) -> List[Optional[int]]:
        """Normalize the ``seed=`` kwarg into a length-``num_envs`` list.

        - ``None`` → ``[None] * num_envs`` (no explicit re-seed).
        - ``int`` → ``[seed, seed+1, ..., seed+num_envs-1]`` so each lane
          gets a distinct deterministic stream.
        - sequence → returned as-is after length validation.
        """
        if seed is None:
            return [None] * self.num_envs
        if isinstance(seed, (int, np.integer)):
            return [int(seed) + i for i in range(self.num_envs)]
        seeds = list(seed)
        if len(seeds) != self.num_envs:
            raise ValueError(
                f"Got {len(seeds)} seeds for {self.num_envs} sub-envs; "
                "lengths must match."
            )
        return seeds

    def _stack_obs(self, obs_list: Sequence[np.ndarray]) -> np.ndarray:
        """Stack sub-env observations into a single batched ndarray."""
        return np.ascontiguousarray(
            np.stack(obs_list, axis=0),
            dtype=self.single_observation_space.dtype,
        )

    @staticmethod
    def _batch_info(infos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert a list of per-sub-env info dicts into a dict of lists.

        Matches Gymnasium's sync-vector convention: keys are the union of
        all per-sub-env keys; values are length-``num_envs`` lists with
        ``None`` filling slots where a given sub-env did not produce that
        key. Callers that prefer a plain "list of dicts" can also access
        ``info["_per_env"]`` which is exactly that.
        """
        keys = set()
        for d in infos:
            keys.update(d.keys())
        batched: Dict[str, Any] = {k: [d.get(k, None) for d in infos] for k in keys}
        # Stash the raw list-of-dicts under a private-ish key so callers
        # that prefer that shape don't have to re-zip.
        batched["_per_env"] = list(infos)
        return batched


def make_vec(
    id: str,
    num_envs: int,
    *,
    num_agents: Optional[int] = None,
) -> SyncVectorEnv:
    """Construct a synchronous vectorized Bucket Brigade env.

    Args:
        id: A versioned scenario ID registered in
            :data:`bucket_brigade.envs.registry.SCENARIO_VERSIONS`
            (e.g. ``"minimal_specialization-v1"``). Use
            :func:`bucket_brigade.list_envs` to enumerate.
        num_envs: Number of independent parallel sub-envs.
        num_agents: Optional override for the number of agents in each
            sub-env. Defaults to
            :data:`~bucket_brigade.envs.registry.DEFAULT_NUM_AGENTS`. As
            with :func:`bucket_brigade.make`, overriding here means the
            env is no longer covered by the frozen ID's reproducibility
            guarantee and downstream consumers should record the override
            in experiment metadata.

    Returns:
        A :class:`SyncVectorEnv` wrapping ``num_envs`` independent
        :class:`bucket_brigade.envs.gym_adapter.BucketBrigadeGymEnv`
        instances.

    Raises:
        KeyError: If ``id`` is not a registered frozen scenario ID.
        ValueError: If ``num_envs < 1``.

    Example:
        >>> import bucket_brigade
        >>> vec = bucket_brigade.make_vec("minimal_specialization-v1", num_envs=4)
        >>> obs, info = vec.reset(seed=0)
        >>> obs.shape[0]
        4
    """
    # Local import to avoid a circular ``bucket_brigade.__init__`` <->
    # ``bucket_brigade.envs.vector`` dependency at module-import time.
    from .. import make as _make_single

    if int(num_envs) < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    envs = [_make_single(id, num_agents=num_agents) for _ in range(int(num_envs))]
    return SyncVectorEnv(envs)
