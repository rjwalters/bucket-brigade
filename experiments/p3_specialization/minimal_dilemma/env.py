"""2-player iterated public-goods dilemma (issue #292, Option A).

Tiny env that quacks like ``bucket_brigade.envs.BucketBrigadeEnv`` enough to
plug into :class:`bucket_brigade.training.joint_trainer.JointPPOTrainer` with
**zero core modifications**. The trick: we return a dict observation whose
keys (``houses``, ``signals``, ``locations``, ``last_actions``,
``scenario_info``) match what :func:`flatten_dict_obs` reads, but we size
each slot so the concatenated obs is exactly the 5-dimensional vector
specified in the issue (previous joint action one-hot ⊕ normalized step).

Game (Option A — iterated 2-player public goods):

- Two agents, each with action space ``Discrete(2)`` (``0=defect/keep``,
  ``1=cooperate/contribute``).
- Per-step per-agent reward: ``r_i = (m / 2) * (a_0 + a_1) - a_i`` with
  multiplier ``m = 1.6``. Payoff matrix (per agent, per step):

      (C, C) → (0.6, 0.6)
      (C, D) → (-0.2, 0.8)
      (D, C) → (0.8, -0.2)
      (D, D) → (0.0, 0.0)

  Mutual-C is socially optimal (sum 1.2 > 0); mutual-D is the dominant-strategy
  equilibrium (each agent has a strict incentive to defect at every step).
  Classic public-goods dilemma — see Foerster et al. 2018 (LOLA) for the same
  family of testbeds.

- Fixed 50-step episodes (no early termination). The dones flag fires
  only at step 50, after which ``JointPPOTrainer.collect_rollout`` auto-resets.

Observation (dict, agent-symmetric):

    - ``houses``         : shape (0,)  — unused, kept for the flatten contract.
    - ``signals``        : shape (0,)  — unused.
    - ``locations``      : shape (0,)  — unused.
    - ``last_actions``   : shape (2, 2) — one-hot of each agent's previous
      action. On reset (step 0), both rows are zeros.
    - ``scenario_info``  : shape (1,) — normalized step counter in [0, 1].

After :func:`flatten_dict_obs` flattens ``last_actions`` and concatenates,
the base obs vector is length 4 + 1 = 5; the per-agent identity one-hot
(#204) brings it to 5 + 2 = 7 for actual trainer consumption. The
"State dim = 5" in the issue spec refers to the pre-identity base vector.

Step contract (mirrors ``BucketBrigadeEnv``):

    obs = env.reset(seed=...)                       # dict
    obs, rewards, done_arr, info = env.step(act)    # act: [2, 1] int64
        rewards: np.ndarray shape (2,) float32
        done_arr: np.ndarray shape (2,) bool — both entries identical
        info: dict (currently empty)

The "[2, 1]" action shape matches ``JointPPOTrainer.collect_rollout``'s
``joint_action`` (one row per agent, one column per action dim).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# Game constants. Public so analyzer/tests can pin reference values.
MULTIPLIER: float = 1.6
NUM_AGENTS: int = 2
NUM_ACTIONS: int = 2
EPISODE_LENGTH: int = 50

ACTION_DEFECT: int = 0
ACTION_COOPERATE: int = 1

# Analytic per-step rewards (Option A, m=1.6). Used by tests and the analyzer.
REWARD_MUTUAL_COOPERATE: float = 0.6
REWARD_MUTUAL_DEFECT: float = 0.0
REWARD_UNILATERAL_COOPERATE: float = -0.2  # agent who plays C while other plays D
REWARD_UNILATERAL_DEFECT: float = 0.8  # agent who plays D while other plays C


def step_reward(
    a0: int, a1: int, multiplier: float = MULTIPLIER
) -> Tuple[float, float]:
    """Analytic per-step reward for joint action ``(a0, a1)``.

    ``r_i = (m / 2) * (a_0 + a_1) - a_i``. Pure function; used by both the env
    step and the unit tests to pin the payoff matrix.
    """
    contribution = 0.5 * multiplier * (a0 + a1)
    return float(contribution - a0), float(contribution - a1)


class MinimalDilemmaEnv:
    """2-player iterated public-goods dilemma compatible with JointPPOTrainer.

    The env shape contract matches BucketBrigadeEnv as consumed by
    ``JointPPOTrainer.collect_rollout``:

    - ``reset(seed=...)`` returns a dict obs.
    - ``step(joint_action)`` takes a ``[N, A]`` int64 array (here ``[2, 1]``)
      and returns ``(obs_dict, rewards[N], dones[N], info)``.
    - ``self.done`` exposes the current termination flag (used by evaluators
      that loop until done, e.g. the BC eval driver).

    The env is fully deterministic given the joint action sequence — the
    ``seed`` argument is accepted for API parity but has no effect on the
    state transition (there is no exogenous randomness in this game). Per-
    episode seeding still affects ``JointPPOTrainer``'s policy sampling
    upstream of the env.
    """

    def __init__(
        self,
        multiplier: float = MULTIPLIER,
        episode_length: int = EPISODE_LENGTH,
    ) -> None:
        if multiplier <= 1.0 or multiplier >= 2.0:
            # The dilemma requires 1 < m < 2: mutual-C beats mutual-D
            # (m > 1) but unilateral-D beats mutual-C for the defector
            # (m < 2). Reject silly values defensively.
            raise ValueError(
                f"multiplier must be in (1, 2) for a public-goods dilemma; got {multiplier}"
            )
        if episode_length <= 0:
            raise ValueError(f"episode_length must be positive; got {episode_length}")

        self.multiplier = float(multiplier)
        self.episode_length = int(episode_length)
        self.num_agents = NUM_AGENTS

        # Internal state.
        self._step: int = 0
        self._last_actions: np.ndarray = np.zeros(
            (NUM_AGENTS, NUM_ACTIONS), dtype=np.float32
        )
        self._done: bool = False

    # ------------------------------------------------------------------
    # Env API (mirrors BucketBrigadeEnv)
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset to step 0. Seed is accepted for API parity (no-op internally)."""
        # ``seed`` is consumed for API parity with ``BucketBrigadeEnv.reset``.
        # The env has no exogenous randomness, so the seed has no effect on
        # state transitions. We *do* avoid asserting here so callers that
        # pass seeds for stochastic-policy parity don't hit a surprise.
        _ = seed
        self._step = 0
        self._last_actions = np.zeros((NUM_AGENTS, NUM_ACTIONS), dtype=np.float32)
        self._done = False
        return self._observation()

    def step(
        self, joint_action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, object]]:
        """Apply joint action and advance one step.

        Args:
            joint_action: ``[2, 1]`` int64 array. ``joint_action[i, 0]`` is
                agent i's action in {0, 1}.

        Returns:
            ``(obs_dict, rewards, dones, info)``:
                - ``rewards``: shape (2,) float32.
                - ``dones``: shape (2,) bool — both entries identical
                  (shared termination); set to True on the last step.
                - ``info``: empty dict.
        """
        if self._done:
            raise RuntimeError(
                "MinimalDilemmaEnv.step called on a terminated episode; "
                "call reset() first"
            )

        ja = np.asarray(joint_action, dtype=np.int64)
        if ja.shape != (NUM_AGENTS, 1):
            raise ValueError(
                f"joint_action must have shape ({NUM_AGENTS}, 1); got {ja.shape}"
            )
        a0 = int(ja[0, 0])
        a1 = int(ja[1, 0])
        if a0 not in (0, 1) or a1 not in (0, 1):
            raise ValueError(f"actions must be in {{0, 1}}; got ({a0}, {a1})")

        r0, r1 = step_reward(a0, a1, self.multiplier)
        rewards = np.array([r0, r1], dtype=np.float32)

        # Update transition state.
        self._last_actions = np.zeros((NUM_AGENTS, NUM_ACTIONS), dtype=np.float32)
        self._last_actions[0, a0] = 1.0
        self._last_actions[1, a1] = 1.0

        self._step += 1
        if self._step >= self.episode_length:
            self._done = True

        dones = np.array([self._done, self._done], dtype=bool)
        return self._observation(), rewards, dones, {}

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        """Whether the current episode has terminated. Mirrors BucketBrigadeEnv."""
        return self._done

    def _observation(self) -> Dict[str, np.ndarray]:
        """Build the dict observation consumed by ``flatten_dict_obs``.

        Layout (base vector after flatten = 5-d):
            houses (0,) | signals (0,) | locations (0,) |
            last_actions (2, 2) → flat 4-d | scenario_info (1,)

        The per-agent identity one-hot (#204) is appended downstream by
        ``flatten_dict_obs(obs, agent_id=i, num_agents=2)``, yielding the
        7-d vector the policy consumes.
        """
        # Normalized step counter in [0, 1). Episode_length steps after reset,
        # we report ``self._step / episode_length`` ∈ {0/L, 1/L, ..., (L-1)/L}.
        # (At the moment of terminal observation we'd report L/L = 1.0, but the
        # trainer doesn't query the obs after a done is registered: it auto-
        # resets and the next reset reports 0.0 again.)
        step_norm = np.array(
            [self._step / float(self.episode_length)], dtype=np.float32
        )
        return {
            "houses": np.zeros(0, dtype=np.float32),
            "signals": np.zeros(0, dtype=np.float32),
            "locations": np.zeros(0, dtype=np.float32),
            "last_actions": self._last_actions.copy(),
            "scenario_info": step_norm,
        }
