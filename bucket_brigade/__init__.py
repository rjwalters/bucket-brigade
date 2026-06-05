"""Bucket Brigade — multi-agent firefighting environment.

Public entry points:

- :func:`make` — Gym/Gymnasium-compatible factory keyed on a versioned
  scenario ID (e.g. ``"minimal_specialization-v1"``). See
  :mod:`bucket_brigade.envs.registry` for the version-bump policy.
- :func:`make_vec` — synchronous vectorized factory returning a
  :class:`~bucket_brigade.envs.vector.SyncVectorEnv` (issue #370).
- :func:`list_envs` — list every registered frozen scenario ID.

Example:

    >>> import bucket_brigade
    >>> env = bucket_brigade.make("minimal_specialization-v1")
    >>> obs, info = env.reset(seed=0)
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from .envs.registry import (
    DEFAULT_NUM_AGENTS,
    SCENARIO_VERSIONS,
    get_scenario_by_id,
    list_versioned_scenarios,
    parse_scenario_id,
)


if (
    TYPE_CHECKING
):  # pragma: no cover - type hint only, avoid hard gym import at import time
    from .envs.gym_adapter import BucketBrigadeGymEnv
    from .envs.vector import SyncVectorEnv


__all__ = [
    "make",
    "make_vec",
    "list_envs",
    "DEFAULT_NUM_AGENTS",
    "SCENARIO_VERSIONS",
    "get_scenario_by_id",
    "list_versioned_scenarios",
    "parse_scenario_id",
]


def make(
    id: str,
    num_agents: Optional[int] = None,
) -> "BucketBrigadeGymEnv":
    """Construct a Gymnasium-compatible Bucket Brigade env from a frozen ID.

    Args:
        id: A versioned scenario ID registered in
            :data:`bucket_brigade.envs.registry.SCENARIO_VERSIONS`,
            e.g. ``"minimal_specialization-v1"``. Use
            :func:`list_envs` to enumerate.
        num_agents: Optional override for the number of agents. Defaults
            to :data:`DEFAULT_NUM_AGENTS` (4). Overriding produces an env
            that is NOT covered by the frozen ID's reproducibility
            guarantee — the override is reflected in the underlying
            scenario but downstream consumers should record it in their
            experiment metadata.

    Returns:
        A :class:`bucket_brigade.envs.gym_adapter.BucketBrigadeGymEnv`
        instance — a fully Gymnasium-compatible ``gym.Env`` with proper
        ``observation_space`` (``Box``) and ``action_space``
        (``MultiDiscrete``).

    Raises:
        KeyError: If ``id`` is not a registered frozen scenario ID.

    Example:
        >>> import bucket_brigade
        >>> env = bucket_brigade.make("minimal_specialization-v1")
        >>> env.action_space.shape
        (12,)
        >>> obs, info = env.reset(seed=0)
        >>> obs.shape == env.observation_space.shape
        True
    """
    # Local import so a missing ``gymnasium`` install only blows up when
    # somebody actually calls ``make()``, not at package import time.
    # (gymnasium is a hard dep per pyproject.toml today, but keeping the
    # import lazy means ``from bucket_brigade import list_envs`` stays
    # cheap and robust.)
    from .envs.gym_adapter import BucketBrigadeGymEnv

    n = int(num_agents) if num_agents is not None else DEFAULT_NUM_AGENTS
    scenario = get_scenario_by_id(id, num_agents=n)
    return BucketBrigadeGymEnv(scenario=scenario, scenario_id=id)


def make_vec(
    id: str,
    num_envs: int,
    *,
    num_agents: Optional[int] = None,
) -> "SyncVectorEnv":
    """Construct a synchronous vectorized Bucket Brigade env (issue #370).

    Returns a :class:`bucket_brigade.envs.vector.SyncVectorEnv` wrapping
    ``num_envs`` independent :class:`BucketBrigadeGymEnv` instances built
    from the same versioned scenario ``id``. The vectorized env exposes
    the Gymnasium-style batched ``step`` / ``reset`` API with auto-reset
    semantics (terminal observations surfaced via ``info["final_observation"]``
    / ``info["final_info"]``).

    Args:
        id: A versioned scenario ID — see :func:`make` for the contract.
        num_envs: Number of parallel sub-envs (>= 1).
        num_agents: Optional per-sub-env override; same caveat as
            :func:`make` regarding reproducibility guarantees.

    Returns:
        A :class:`bucket_brigade.envs.vector.SyncVectorEnv` instance.

    Raises:
        KeyError: If ``id`` is not a registered frozen scenario ID.
        ValueError: If ``num_envs < 1``.

    Example:
        >>> import bucket_brigade
        >>> vec = bucket_brigade.make_vec("minimal_specialization-v1", num_envs=8)
        >>> obs, info = vec.reset(seed=0)
        >>> obs.shape[0]
        8
    """
    # Local import for the same reason as ``make``: keep top-level import
    # of ``bucket_brigade`` cheap and avoid a hard gymnasium dependency
    # until the user actually constructs an env.
    from .envs.vector import make_vec as _make_vec

    return _make_vec(id, num_envs, num_agents=num_agents)


def list_envs() -> List[str]:
    """Return a sorted list of frozen scenario IDs accepted by :func:`make`.

    Returns:
        Sorted list of versioned IDs, e.g.
        ``["chain_reaction-v1", "default-v1", ...]``.

    Example:
        >>> import bucket_brigade
        >>> "minimal_specialization-v1" in bucket_brigade.list_envs()
        True
    """
    return list_versioned_scenarios()
