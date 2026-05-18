"""Hand-coded specialists for the minimal-dilemma env (issue #292).

Two policies the BC pipeline can clone:

- :func:`always_cooperate` — unconditional cooperation. Simplest possible
  cooperative basin signal; if BC clones this and PPO-continuation holds,
  that's the basin-trap verdict in the toy.
- :func:`tit_for_tat` — cooperate on step 0, then copy opponent's previous
  action. Matches the LOLA paper's testbed specialist.

Both functions mirror the signature of
:func:`bucket_brigade.baselines.specialist_action_joint` — they take a dict
observation and return a ``[num_agents, action_dims_count]`` int64 array —
so the BC pipeline (``bc_init.py``) can swap them in with zero plumbing
changes.

Self-play reference rewards (per-step, per-agent, mutual play, m=1.6):

    always_cooperate vs always_cooperate → 0.6   (cooperative upper bound)
    tit_for_tat      vs tit_for_tat      → 0.6   (both cooperate forever)
    always_cooperate vs always_defect    → -0.2  (the cooperator's loss)
    tit_for_tat      vs always_defect    → ~0.0  (only step 0 is C; rest D)
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from experiments.p3_specialization.minimal_dilemma.env import (
    ACTION_COOPERATE,
    ACTION_DEFECT,
    NUM_AGENTS,
)


def always_cooperate(
    obs: Mapping[str, np.ndarray], num_agents: int = NUM_AGENTS
) -> np.ndarray:
    """Return the all-cooperate joint action regardless of observation.

    Signature mirrors :func:`bucket_brigade.baselines.specialist_action_joint`:
    obs is the dict returned by :meth:`MinimalDilemmaEnv.reset`/``step``,
    output is a ``[num_agents, 1]`` int64 array (one column because the env's
    ``action_dims = [2]``).
    """
    if num_agents != NUM_AGENTS:
        raise ValueError(
            f"always_cooperate expects num_agents={NUM_AGENTS}; got {num_agents}"
        )
    return np.full((num_agents, 1), ACTION_COOPERATE, dtype=np.int64)


def always_defect(
    obs: Mapping[str, np.ndarray], num_agents: int = NUM_AGENTS
) -> np.ndarray:
    """Return the all-defect joint action regardless of observation.

    Included as a baseline arm (mutual-defect = 0.0 per-step reward floor).
    Same signature as :func:`always_cooperate`.
    """
    if num_agents != NUM_AGENTS:
        raise ValueError(
            f"always_defect expects num_agents={NUM_AGENTS}; got {num_agents}"
        )
    return np.full((num_agents, 1), ACTION_DEFECT, dtype=np.int64)


def tit_for_tat(
    obs: Mapping[str, np.ndarray], num_agents: int = NUM_AGENTS
) -> np.ndarray:
    """Cooperate on step 0; otherwise copy opponent's previous action.

    Reads ``obs["last_actions"]`` (shape ``[2, 2]``, one-hot rows). On step 0
    every row is all-zeros (no previous action), which we interpret as
    "no signal yet → cooperate" (the canonical TFT starting move).

    Returns ``[num_agents, 1]`` int64.
    """
    if num_agents != NUM_AGENTS:
        raise ValueError(
            f"tit_for_tat expects num_agents={NUM_AGENTS}; got {num_agents}"
        )
    last_actions = np.asarray(obs["last_actions"], dtype=np.float32)
    if last_actions.shape != (NUM_AGENTS, 2):
        raise ValueError(
            f"tit_for_tat: obs['last_actions'] must have shape (2, 2); "
            f"got {last_actions.shape}"
        )

    joint = np.empty((num_agents, 1), dtype=np.int64)
    # On step 0 every row is all-zeros → ``argmax`` returns 0, but that's
    # "defect". Detect the start-of-episode condition explicitly and emit
    # cooperate instead.
    is_reset = bool(last_actions.sum() == 0.0)
    for i in range(num_agents):
        opponent = 1 - i
        if is_reset:
            joint[i, 0] = ACTION_COOPERATE
        else:
            # Copy opponent's previous action. ``argmax`` over the one-hot row
            # recovers the action index in {0=D, 1=C}.
            joint[i, 0] = int(np.argmax(last_actions[opponent]))
    return joint


def always_random(
    obs: Mapping[str, np.ndarray],
    num_agents: int = NUM_AGENTS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform random action for each agent. Useful sanity baseline.

    Per-step reward in expectation: E[r] = (m/2)*E[a_0 + a_1] - E[a_i]
    = (m/2)*1 - 0.5 = 0.8 - 0.5 = 0.3 (Option A, m=1.6).

    Not used by BC (random policies aren't worth cloning) but included for
    the baseline table.
    """
    if num_agents != NUM_AGENTS:
        raise ValueError(
            f"always_random expects num_agents={NUM_AGENTS}; got {num_agents}"
        )
    gen = rng if rng is not None else np.random.default_rng()
    return gen.integers(0, 2, size=(num_agents, 1), dtype=np.int64)
