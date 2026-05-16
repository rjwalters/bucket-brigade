"""Hand-coded specialist baseline policy.

The specialist policy is dead-simple per agent:

* If any house I own is currently BURNING, work the lowest-index burning
  owned house.
* Otherwise, rest (any house, mode=REST).

This is intended as a reference policy for scenarios where per-agent ownership
is the dominant gradient signal (e.g. :data:`minimal_specialization`, issue
#199). On scenarios where the dominant signal is the shared team reward this
policy will under-perform a free-rider / generalist policy by design.

House ownership follows the canonical round-robin assignment used by
:class:`bucket_brigade.envs.bucket_brigade_env.BucketBrigadeEnv`:
``house_owners = np.arange(num_houses) % num_agents``. The baseline is kept
deliberately decoupled from the env class so it can be invoked against a raw
observation dict (which is what trainers and evaluators see).

Reusable across scenarios. No experiment-specific assumptions.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# House state codes, mirrored from BucketBrigadeEnv. Kept as module constants
# rather than imported so this module has zero dependency on the env class,
# which keeps it cheap to import in tests and notebooks.
_SAFE = 0
_BURNING = 1
_RUINED = 2

# Action mode codes, mirrored from BucketBrigadeEnv.
_REST = 0
_WORK = 1


def _owned_houses(agent_id: int, num_agents: int, num_houses: int) -> np.ndarray:
    """Return the indices of houses owned by ``agent_id`` under round-robin
    ownership (``np.arange(num_houses) % num_agents == agent_id``).

    For ``num_agents=4, num_houses=10``:
        agent 0 -> [0, 4, 8]
        agent 1 -> [1, 5, 9]
        agent 2 -> [2, 6]
        agent 3 -> [3, 7]
    """
    owners = np.arange(num_houses) % num_agents
    return np.where(owners == agent_id)[0]


def specialist_action(
    observation: Mapping[str, np.ndarray],
    agent_id: int,
    num_agents: int,
    num_houses: int = 10,
) -> np.ndarray:
    """Compute the specialist action for a single agent.

    Parameters
    ----------
    observation
        Observation dict from :class:`BucketBrigadeEnv`. Must contain a
        ``"houses"`` key holding a length-``num_houses`` int array with the
        per-house state code (``0=SAFE, 1=BURNING, 2=RUINED``). All other keys
        are ignored.
    agent_id
        Which agent we are computing the action for (``0 <= agent_id < num_agents``).
    num_agents
        Total number of agents in the scenario (controls round-robin ownership).
    num_houses
        Number of houses on the ring. Defaults to 10 to match the env.

    Returns
    -------
    np.ndarray
        Length-2 ``int64`` array ``[house_idx, mode]`` matching the
        ``MultiDiscrete([num_houses, 2])`` action space.

    Policy
    ------
    1. Look at the ``"houses"`` field of the observation.
    2. Find houses I own that are currently ``BURNING``.
    3. If any, pick the lowest-index one and ``WORK`` it.
    4. Otherwise, pick my lowest-index owned house and ``REST``. (Choice of
       house under REST is irrelevant to the env, since REST does not target a
       house. We return an owned-house index for determinism.)
    """
    if not 0 <= agent_id < num_agents:
        raise ValueError(
            f"agent_id={agent_id} out of range for num_agents={num_agents}"
        )

    houses = np.asarray(observation["houses"])
    if houses.shape != (num_houses,):
        raise ValueError(
            f"observation['houses'] has shape {houses.shape}, expected ({num_houses},)"
        )

    owned = _owned_houses(agent_id, num_agents, num_houses)
    if owned.size == 0:
        # Defensive: with round-robin ownership and num_agents <= num_houses
        # every agent owns at least one house. Fall back to house 0 + REST
        # rather than raise, so this function never errors on a well-formed obs.
        return np.array([0, _REST], dtype=np.int64)

    burning_owned = owned[houses[owned] == _BURNING]
    if burning_owned.size > 0:
        return np.array([int(burning_owned[0]), _WORK], dtype=np.int64)
    return np.array([int(owned[0]), _REST], dtype=np.int64)


def specialist_action_joint(
    observation: Mapping[str, np.ndarray],
    num_agents: int,
    num_houses: int = 10,
) -> np.ndarray:
    """Compute joint specialist actions for all agents.

    Returns an ``(num_agents, 2)`` ``int64`` array suitable for passing
    directly to :meth:`BucketBrigadeEnv.step`.
    """
    return np.stack(
        [
            specialist_action(observation, a, num_agents, num_houses)
            for a in range(num_agents)
        ],
        axis=0,
    )


class SpecialistPolicy:
    """Lightweight callable wrapper around :func:`specialist_action_joint`.

    Useful when an evaluator expects a ``policy(obs) -> actions`` callable,
    matching the shape of trained policies. Pure-functional otherwise — no
    state, no learning.
    """

    def __init__(self, num_agents: int, num_houses: int = 10) -> None:
        self.num_agents = num_agents
        self.num_houses = num_houses

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return specialist_action_joint(observation, self.num_agents, self.num_houses)
