"""Bucket Brigade — multi-agent firefighting environment.

Public entry points:

- :func:`make` — Gym/Gymnasium-compatible factory keyed on a versioned
  scenario ID (e.g. ``"minimal_specialization-v1"``). See
  :mod:`bucket_brigade.envs.registry` for the version-bump policy.
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


__all__ = [
    "make",
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
